# set_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- ユーティリティ：点群の中心化＋スケール正規化 --------
def normalize_points(x, eps=1e-6):
    """
    x: [B, N, 3]  3D点群
    返り値: x_norm [B,N,3], center [B,1,3], scale [B,1,1]
    """
    center = x.mean(dim=1, keepdim=True)          # 重心で中心化
    x0 = x - center
    scale = x0.norm(dim=-1, keepdim=True).amax(dim=1, keepdim=True) + eps  # 最大半径でスケーリング
    x_norm = x0 / scale
    return x_norm, center, scale

# -------- 基本のFFN（Transformerブロックの中身） --------
class FFN(nn.Module):
    def __init__(self, dim, hidden=128, drop=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )
    def forward(self, x):
        return self.net(x)

# -------- 小規模Set Transformerエンコーダ --------
class SmallSetTransformerEncoder(nn.Module):
    """
    入力:
      x_pts: [B, N, 3]   3D点群（Nは数点〜数十点）
      x_feats (任意): [B, N, F]  追加特徴（RGB/法線など）
    出力:
      z: [B, 128] グローバル特徴ベクトル
    """
    def __init__(self, in_feat_dim=0, model_dim=128, heads=4, blocks=2, out_dim=128, use_dist_feat=True, drop=0.0):
        super().__init__()
        self.use_dist_feat = use_dist_feat
        # 入力次元 = 3D座標 + (中心距離1ch) + 追加特徴F
        in_dim = 3 + (1 if use_dist_feat else 0) + in_feat_dim
        self.fc_in = nn.Linear(in_dim, model_dim)

        self.attn_blocks = nn.ModuleList([])
        self.norm1 = nn.ModuleList([])
        self.norm2 = nn.ModuleList([])
        self.ffns  = nn.ModuleList([])
        for _ in range(blocks):
            self.attn_blocks.append(nn.MultiheadAttention(embed_dim=model_dim, num_heads=heads, batch_first=True))
            self.norm1.append(nn.LayerNorm(model_dim))
            self.ffns.append(FFN(model_dim, hidden=model_dim*2, drop=drop))
            self.norm2.append(nn.LayerNorm(model_dim))

        self.fc_out = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.ReLU(inplace=True),
            nn.Linear(model_dim, out_dim)
        )

    def forward(self, x_pts, x_feats=None):
        """
        x_pts:   [B,N,3]
        x_feats: [B,N,F] or None
        """
        B, N, _ = x_pts.shape
        feats = [x_pts]
        if self.use_dist_feat:
            # 各点の中心からの距離を追加（安定性UP）
            r = x_pts.norm(dim=-1, keepdim=True)        # [B,N,1]
            feats.append(r)
        if x_feats is not None:
            feats.append(x_feats)

        x = torch.cat(feats, dim=-1)                     # [B,N,in_dim]
        x = self.fc_in(x)                                # [B,N,D]

        # --- Transformer ブロック（Self-Attn + FFN, Residual + LN）---
        for attn, ln1, ffn, ln2 in zip(self.attn_blocks, self.norm1, self.ffns, self.norm2):
            x2, _ = attn(x, x, x)                        # Self-Attention
            x = ln1(x + x2)                              # 残差 + LayerNorm
            x2 = ffn(x)
            x = ln2(x + x2)                              # 残差 + LayerNorm

        # 順序不変な集約（mean pooling）でグローバル化
        z = x.mean(dim=1)                                # [B,D]
        z = self.fc_out(z)                               # [B,out_dim]
        return z

# --------------------- 実行デモ ---------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, N = 4, 10                       # バッチ4、各10点（←少点数）
    pts = torch.randn(B, N, 3) * 0.2 + torch.tensor([0.5, -0.3, 1.2])  # ダミー点群
    # 追加特徴（例: RGB）を使うなら用意（0..1）
    rgb = torch.rand(B, N, 3)          # 使わない場合は None

    encoder = SmallSetTransformerEncoder(
        in_feat_dim=3,     # 追加特徴RGBを入れるので3
        model_dim=128,
        heads=4,
        blocks=2,
        out_dim=128,
        use_dist_feat=True,
        drop=0.0
    )

    z = encoder(pts, rgb)              # [B,128]
    print("Encoded shape:", z.shape)   # -> torch.Size([4, 128])
    print("Sample vector (first row, first 8 dims):", z[0, :8])
