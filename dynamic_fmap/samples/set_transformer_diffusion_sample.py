"""
Sample usage: encode a point cloud with the Set Transformer encoder and feed the
result into the ManiSkill diffusion policy UNet as part of the global condition.
"""

import sys
from pathlib import Path

import torch

from dynamic_fmap.model.set_transformer import SmallSetTransformerEncoder

# The ConditionalUnet1D lives under ManiSkill/examples/baselines/diffusion_policy.
THIS_DIR = Path(__file__).resolve()
PROJECT_ROOT = THIS_DIR.parents[4]
DIFFUSION_POLICY_DIR = PROJECT_ROOT / "ManiSkill" / "examples" / "baselines" / "diffusion_policy"
if str(DIFFUSION_POLICY_DIR) not in sys.path:
    sys.path.append(str(DIFFUSION_POLICY_DIR))

from diffusion_policy.conditional_unet1d import ConditionalUnet1D  # noqa: E402


def encode_point_cloud_sequence(encoder: SmallSetTransformerEncoder, point_clouds: torch.Tensor) -> torch.Tensor:
    """
    Runs the Set Transformer encoder on a sequence of point clouds.

    Args:
        encoder: SmallSetTransformerEncoder instance.
        point_clouds: Tensor with shape [B, T, N, 3] (batch, observation horizon, points, xyz).

    Returns:
        Tensor with shape [B, T, D] where D is the encoder output dimension.
    """
    b, t, n, c = point_clouds.shape
    assert c == 3, f"Expected last dim == 3 (xyz), got {c}"
    latents = []
    for step in range(t):
        latent = encoder(point_clouds[:, step])  # [B, D]
        latents.append(latent)
    return torch.stack(latents, dim=1)


#def main():
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Fake observation batch -------------------------------------------------
batch_size = 4
obs_horizon = 2
num_points = 128
state_dim = 32  # e.g. proprioceptive measurements per step

# Dummy point cloud history [B, T, N, 3]
point_cloud_seq = torch.randn(batch_size, obs_horizon, num_points, 3, device=device)
# Dummy low-dimensional state history [B, T, state_dim]
state_seq = torch.randn(batch_size, obs_horizon, state_dim, device=device)

# --- Encode point clouds ----------------------------------------------------
encoder = SmallSetTransformerEncoder(
    in_feat_dim=0,
    model_dim=128,
    heads=4,
    blocks=2,
    out_dim=64,
    use_dist_feat=True,
    drop=0.0,
).to(device)
encoder.eval()

pc_latent_seq = encode_point_cloud_sequence(encoder, point_cloud_seq)  # [B, T, 64]

# Fuse proprio state and point cloud features per step, then flatten over horizon.
fused_obs_seq = torch.cat([state_seq, pc_latent_seq], dim=-1)  # [B, T, state_dim + 64]
global_cond = fused_obs_seq.reshape(batch_size, -1)  # [B, T * (state_dim + 64)]

# --- Diffusion policy UNet setup -------------------------------------------
action_dim = 7
act_horizon = 8
diffusion_step_embed_dim = 64

diffusion_model = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=global_cond.shape[-1],
    diffusion_step_embed_dim=diffusion_step_embed_dim,
    down_dims=[64, 128, 256],
    kernel_size=5,
    n_groups=8,
).to(device)
diffusion_model.eval()

# Simulate a noisy action trajectory sample that the UNet will denoise.
noisy_action_traj = torch.randn(batch_size, act_horizon, action_dim, device=device)
diffusion_step = torch.randint(
    low=0,
    high=1000,
    size=(batch_size,),
    device=device,
    dtype=torch.long,
)

with torch.no_grad():
    noise_pred = diffusion_model(
        sample=noisy_action_traj,
        timestep=diffusion_step,
        global_cond=global_cond,
    )

print(f"Global condition shape: {global_cond.shape}")
print(f"Predicted noise shape: {noise_pred.shape}")

    # Example output:
    # Global condition shape: torch.Size([4, 192])
    # Predicted noise shape: torch.Size([4, 8, 7])


# if __name__ == "__main__":
#     main()
