import torch


def normalize_range(x, source_range, target_range):
    source_min, source_max = source_range
    target_min, target_max = target_range
    assert source_max > source_min and target_max > target_min
    x_clipped = torch.clamp(x, min=source_min, max=source_max)
    return (x_clipped - source_min) * (target_max - target_min) / (source_max - source_min) + target_min


# def normalize_point_forces(point_forces, force_range=[-1, 1]):
#     coords = point_forces[..., :3]
#     coords = normalize_range(coords, [-0.2, 0.2], [0.1, 0.9])  # (B, obs_horizon, F*k, 3)            
#     force_magnitudes = point_forces[..., 3:6]
#     force_magnitudes = normalize_range(force_magnitudes, force_range, [0.1, 0.9])  # (B, obs_horizon, F*k, 3)            
#     return torch.cat([coords, force_magnitudes], dim=-1)


def normalize_point_forces(point_forces):
    def to_log_scale(x):
        return 0.25 * torch.log10(1 + 100 * x)

    coords = point_forces[..., :3]
    coords = normalize_range(coords, [-0.2, 0.2], [0.1, 0.9])  # (B, obs_horizon, F*k, 3)            
    force_vectors = point_forces[..., 3:6]
    force_vectors = torch.abs(force_vectors)  # sign-less vectors
    force_vectors = to_log_scale(force_vectors)
    force_vectors = normalize_range(force_vectors, [0, 1.0], [0.1, 0.9])  # (B, obs_horizon, F*k, 3)            
    return torch.cat([coords, force_vectors], dim=-1)
