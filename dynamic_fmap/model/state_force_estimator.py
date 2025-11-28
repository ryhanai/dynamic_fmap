import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
#  ViTベースのフレームエンコーダ
# =========================

class PatchEmbedding(nn.Module):
    """画像をパッチに分割して線形埋め込みする部分."""
    def __init__(self, image_size=224, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        # Conv2dでパッチ分割 + 線形変換をまとめてやる
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        return: (B, N, D)  N=num_patches, D=embed_dim
        """
        x = self.proj(x)                    # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)    # (B, N, D)
        return x


class ViTFrameEncoder(nn.Module):
    """1フレームのRGB画像をViTで1つのベクトルにエンコード."""
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # [CLS] トークンと位置埋め込み
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # (B, S, D) で扱う
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """
        x: (B, C, H, W)
        return: (B, D)  フレームごとの埋め込み (CLS token)
        """
        B = x.size(0)
        x = self.patch_embed(x)      # (B, N, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls_tokens, x], dim=1)          # (B, 1+N, D)

        # 位置埋め込みを加算
        x = x + self.pos_embed[:, : x.size(1), :]
        x = self.dropout(x)

        x = self.encoder(x)          # (B, 1+N, D)
        cls = x[:, 0]                # (B, D)
        return cls


# =========================
#  メインネットワーク
# =========================

class ForceStateTransformer(nn.Module):
    """
    画像列 + 関節角度列 → Force埋め込み列 + 状態ベクトル列
    - 画像: フレームごとにViTでエンコード
    - 関節角度: フレームごとにMLPでエンコード
    - 時系列方向にTransformer
    - 各フレームごとにMLPで force / state をデコード
    """
    def __init__(
        self,
        # 画像関連
        image_size=224,
        patch_size=16,
        image_embed_dim=384,

        # 角度関連
        joint_input_dim=7,          # 例: 7自由度ロボット
        joint_embed_dim=128,

        # 時系列Transformer関連
        model_dim=512,
        num_layers=4,
        num_heads=8,
        temporal_dropout=0.1,

        # 出力の次元
        force_embed_dim=256,        # 別の force encoder の出力次元に合わせる
        state_dim=128,              # 位置や速度など状態ベクトルの次元
    ):
        super().__init__()

        # --- フレームごとの画像エンコーダ (ViT) ---
        self.image_encoder = ViTFrameEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=image_embed_dim,
            depth=6,
            num_heads=8,
            mlp_ratio=4.0,
            dropout=0.1,
        )

        # --- フレームごとの関節角度エンコーダ ---
        self.joint_encoder = nn.Sequential(
            nn.Linear(joint_input_dim, joint_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(joint_embed_dim, joint_embed_dim),
            nn.ReLU(inplace=True),
        )

        # --- 画像 + 角度の結合 → 時系列用の埋め込み次元へ ---
        fusion_dim = image_embed_dim + joint_embed_dim
        self.fusion_proj = nn.Linear(fusion_dim, model_dim)

        # --- 時系列Transformer (Temporal Transformer) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=temporal_dropout,
            activation="gelu",
            batch_first=True,  # (B, T, D)
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # --- フレームごとのデコーダ (force / state) ---
        self.force_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(inplace=True),
            nn.Linear(model_dim, force_embed_dim),
        )

        self.state_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(inplace=True),
            nn.Linear(model_dim, state_dim),
        )

    def forward(self, images, joint_angles):
        """
        images: (B, T, 3, H, W)
        joint_angles: (B, T, joint_input_dim)

        return:
          force_embeddings: (B, T, force_embed_dim)
          state_vectors:    (B, T, state_dim)
        """
        B, T, C, H, W = images.shape
        _, _, joint_dim = joint_angles.shape

        # ------- フレームごとの画像エンコード -------
        # (B*T, C, H, W) にまとめてから ViT
        images_flat = images.view(B * T, C, H, W)
        img_emb = self.image_encoder(images_flat)       # (B*T, image_embed_dim)
        img_emb = img_emb.view(B, T, -1)                # (B, T, image_embed_dim)

        # ------- フレームごとの関節角度エンコード -------
        joint_flat = joint_angles.view(B * T, joint_dim)
        joint_emb = self.joint_encoder(joint_flat)      # (B*T, joint_embed_dim)
        joint_emb = joint_emb.view(B, T, -1)            # (B, T, joint_embed_dim)

        # ------- 画像 + 関節を結合 -------
        fused = torch.cat([img_emb, joint_emb], dim=-1) # (B, T, image_embed_dim+joint_embed_dim)
        fused = self.fusion_proj(fused)                 # (B, T, model_dim)

        # ------- 時系列Transformer -------
        # batch_first=True なので (B, T, D) のまま渡せる
        temporal_feat = self.temporal_transformer(fused)  # (B, T, model_dim)

        # ------- 各フレームから force / state をデコード -------
        force_embeddings = self.force_head(temporal_feat)  # (B, T, force_embed_dim)
        state_vectors = self.state_head(temporal_feat)      # (B, T, state_dim)

        return force_embeddings, state_vectors


# =========================
#  動作チェック例
# =========================
if __name__ == "__main__":
    B = 2
    T = 8
    C, H, W = 3, 224, 224
    joint_dim = 7

    images = torch.randn(B, T, C, H, W)
    joint_angles = torch.randn(B, T, joint_dim)

    model = ForceStateTransformer(
        image_size=H,
        patch_size=16,
        image_embed_dim=384,
        joint_input_dim=joint_dim,
        joint_embed_dim=128,
        model_dim=512,
        num_layers=4,
        num_heads=8,
        force_embed_dim=256,
        state_dim=128,
    )

    force_emb, state_vec = model(images, joint_angles)
    print("force_emb:", force_emb.shape)   # (B, T, 256)
    print("state_vec:", state_vec.shape)   # (B, T, 128)