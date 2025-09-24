from cProfile import label
import torch
import torch.nn as nn

# import torch.nn.functional as F


# from fmdev.forcemap import GridForceMap
# fmap = GridForceMap('maniskill_table')


class VoxelFieldL2Loss(nn.Module):
    def __init__(self, voxel_centers, sigma=0.04, reduction='mean', device='cuda'):
        """
        voxel_centers: (N_voxels, 3) ボクセルの中心座標
        sigma: Gaussianの幅
        """
        super().__init__()
        self.sigma = sigma
        self.reduction = reduction
        self.device = device

        self.voxel_centers = torch.from_numpy(voxel_centers).float()

    def l(self, point_weight, point_cloud, voxel_weight):
        """
        voxel_density_pred: (B, N_voxels) ネットワークの出力
        point_cloud: (B, N_points, 3) 正解点群
        point_weights: (B, N_points) 点の重み
        """
        # B, N_voxels = voxel_weight.shape
        # _, N_points, _ = point_cloud.shape

        # voxel_centers を (1, N_voxels, 3) に拡張
        batch_size = point_weight.size(0)
        voxel_centers = self.voxel_centers.unsqueeze(0).expand(batch_size, -1, -1).to(self.device)

        # 点群 -> voxel field に Gaussian で変換
        # (B, N_voxels, N_points)
        diff = voxel_centers[:, :, None, :] - point_cloud[:, None, :, :]  # voxel-center - point
        dist2 = torch.sum(diff**2, dim=-1)
        gaussian = torch.exp(-dist2 / (2 * self.sigma**2))  # (B, N_voxels, N_points)

        # 重み付き sum over points
        field_gt = torch.sum(gaussian * point_weight[:, None, :], dim=-1)  # (B, N_voxels)

        # 正規化しても良い（任意）
        # field_gt = field_gt / (field_gt.sum(dim=1, keepdim=True) + 1e-8)

        # Field L2
        diff_field = voxel_weight - field_gt
        loss_per_batch = torch.sum(diff_field ** 2, dim=1)  # (B,)
        
        if self.reduction == 'mean':
            return loss_per_batch.mean()
        elif self.reduction == 'sum':
            return loss_per_batch.sum()
        else:
            return loss_per_batch

    def forward(self, y_hat, y):
        point_cloud = y[:,:,:3]
        point_weight_x = y[:,:,3]
        point_weight_y = y[:,:,4]
        point_weight_z = y[:,:,5]

        y_hat = y_hat.reshape(y_hat.size(0), y_hat.size(1), -1)

        loss = self.l(point_weight_x, point_cloud, y_hat[:,0]) \
            + self.l(point_weight_y, point_cloud, y_hat[:,1]) \
            + self.l(point_weight_z, point_cloud, y_hat[:,2])

        return torch.sum(loss)


# loss_fn = VoxelFieldL2Loss(fmap)
# predicted_batch = torch.rand((2, 3, 60, 120, 120), requires_grad=True)
# w = predicted_batch.reshape((2, 3, -1)).to('cuda')
# label_point_cloud = torch.rand((2, 32, 6), requires_grad=True)
# label_point_cloud_g = label_point_cloud.to('cuda')
# loss = loss_fn(w, label_point_cloud_g)



# # ======================
# # 1. Gaussian kernel 3D
# # ======================
# def gaussian_kernel3d(kernel_size=5, sigma=1.0, device="cpu"):
#     """
#     Returns a 3D Gaussian kernel for conv3d
#     """
#     ax = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2.0
#     xx, yy, zz = torch.meshgrid(ax, ax, ax, indexing="ij")
#     kernel = torch.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
#     kernel = kernel / kernel.sum()
#     return kernel.unsqueeze(0).unsqueeze(0)  # shape (1,1,k,k,k)

# # ======================
# # 2. Rasterize point cloud into voxel grid
# # ======================
# def rasterize_point_cloud(points, weights, grid_size, bounds, device="cpu"):
#     """
#     points: (N,3) tensor of xyz coords
#     weights: (N,) tensor
#     grid_size: int (assuming cubic grid)
#     bounds: (min_xyz, max_xyz), each (3,)
#     """
#     N = points.shape[0]
#     min_xyz, max_xyz = bounds
#     # normalize to voxel indices
#     normed = (points - min_xyz) / (max_xyz - min_xyz + 1e-9)  # [0,1]
#     idx = (normed * (grid_size - 1)).long()  # (N,3)
#     idx = torch.clamp(idx, 0, grid_size - 1)

#     grid = torch.zeros((grid_size, grid_size, grid_size), device=device)
#     flat_idx = idx[:, 0] * grid_size * grid_size + idx[:, 1] * grid_size + idx[:, 2]
#     grid_flat = grid.view(-1)
#     grid_flat.index_add_(0, flat_idx, weights)
#     return grid

# # ======================
# # 3. Field L2 distance
# # ======================
# def field_L2(voxel_grid, points, weights, bounds, sigma=1.0, kernel_size=5):
#     """
#     voxel_grid: (D,H,W) tensor (float)
#     points: (N,3) tensor
#     weights: (N,) tensor
#     bounds: (min_xyz, max_xyz)
#     """
#     device = voxel_grid.device
#     D, H, W = voxel_grid.shape
#     grid_size = D  # assume cubic

#     # normalize voxel mass
#     voxel_grid = voxel_grid / (voxel_grid.sum() + 1e-9)

#     # rasterize points
#     pc_grid = rasterize_point_cloud(points, weights, grid_size, bounds, device)
#     pc_grid = pc_grid / (pc_grid.sum() + 1e-9)

#     # smoothing
#     kernel = gaussian_kernel3d(kernel_size=kernel_size, sigma=sigma, device=device)
#     voxel_sm = F.conv3d(voxel_grid.unsqueeze(0).unsqueeze(0), kernel, padding=kernel_size//2).squeeze()
#     pc_sm    = F.conv3d(pc_grid.unsqueeze(0).unsqueeze(0), kernel, padding=kernel_size//2).squeeze()

#     # L2 distance
#     diff = voxel_sm - pc_sm
#     l2 = torch.sqrt((diff**2).sum())
#     return l2

# # ======================
# # Example usage
# # ======================
# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     # Fake voxel grid (100³)
#     grid_size = 100
#     voxel = torch.rand((grid_size, grid_size, grid_size), device=device)

#     # Fake weighted point cloud (32 points)
#     N = 32
#     points = torch.rand((N, 3), device=device) * 1.0  # assume coords in [0,1]
#     weights = torch.rand(N, device=device)
#     weights = weights / weights.sum()

#     # Bounds of voxel space (min,max) in world coordinates
#     bounds = (torch.tensor([0,0,0], device=device),
#               torch.tensor([1,1,1], device=device))

#     # Compute Field L2
#     l2_value = field_L2(voxel, points, weights, bounds, sigma=2.0, kernel_size=7)
#     print("Field L2 distance:", l2_value.item())
