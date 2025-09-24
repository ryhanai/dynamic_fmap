import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss



# 学習時のlabelをsamplingするときに，単純なrandom samplingでなく，
# Gumbel-Softmax（連続緩和）か重み付きリスト（stratified sampling)を使った方が良いのではないか？
# random samplingだと密度が変わってしまう

# Sinkhorn lossの計算に時間がかかるようなら，先にvoxelにtop-k filteringをかける
# SinkhornはメモリO(N^2?)で，voxelの全点数を扱えない．時間もかかりすぎる（256くらいが妥当か？）
# もしくはlabel smoothingをloss


# from pytorch3d.loss import chamfer_distance as p3d_chamfer

# class WeightedChamferLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, y, y_hat):
#         xyz = y[:,:,:3]
#         wx = y[:,:,3]
#         wy = y[:,:,4]
#         wz = y[:,:,5]
#         xyz_hat = y_hat[:,:,:3]
#         wx_hat = y_hat[:,:,3]
#         wy_hat = y_hat[:,:,4]
#         wz_hat = y_hat[:,:,5]

#         # x: (B, N, 3), y: (B, M, 3), B=batch size, N=number of points

#         batch_loss = self._loss_fn(fx, xyz, fx_hat, xyz_hat) \
#             + self._loss_fn(fy, xyz, fy_hat, xyz_hat) \
#             + self._loss_fn(fz, xyz, fz_hat, xyz_hat)
#         return torch.sum(batch_loss)


# x = torch.randn(2, 1024, 3, requires_grad=True)
# y = torch.randn(2, 1024, 3, requires_grad=True)

# loss_chamfer, loss_chamfer_per_point = p3d_chamfer(x, y)
# # loss_chamfer: scalar (averaged), loss_chamfer_per_point: (B, N) symmetric per point loss
# print("pytorch3d chamfer:", loss_chamfer.item())
# loss_chamfer.backward()



# def topk_voxels(voxels, voxel_centers, K=8192):
#     # voxels: (B, D,H,W) -> flatten
#     B = voxels.shape[0]
#     v = voxels.view(B, -1)  # (B, M)
#     vals, idx = torch.topk(v, K, dim=1)
#     centers = voxel_centers.view(1, -1, 3).expand(B, -1, -1)
#     top_centers = torch.gather(centers, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
#     return vals, top_centers


# def sample_points(vals, centers, N=32, tau=0.5, hard=True):
#     B, K = vals.shape
#     logits = vals.unsqueeze(1).expand(B, N, K)  # (B, N, K)
#     # gumbel-softmax
#     gumbels = -torch.log(-torch.log(torch.rand_like(logits)))
#     y_soft = F.softmax((logits + gumbels) / tau, dim=-1)
#     if hard:
#         y_hard = torch.zeros_like(y_soft).scatter_(-1, y_soft.argmax(dim=-1, keepdim=True), 1.0)
#         y = (y_hard - y_soft).detach() + y_soft
#     else:
#         y = y_soft
#     pts = torch.bmm(y, centers)  # (B, N, 3)
#     return pts


class SinkhornLoss(nn.Module):
    def __init__(self, voxel_centers, batch_size=2, p=2, blur=0.05, K=128, device='cuda'):
        super().__init__()
        self._loss_fn = SamplesLoss("sinkhorn", p=p, blur=blur, scaling=0.5, backend="multiscale")
        self.K = K
        
        centers = torch.from_numpy(voxel_centers).float()
        centers = centers[torch.newaxis, :, :]
        centers = torch.tile(centers, (batch_size, 1, 1))
        self.voxel_centers = centers.to(device=device)

    def l(self, pc_w, pc_xyz, voxel_densities):
        sample_weights, indices = torch.topk(voxel_densities, self.K)
        expanded_indices = indices.unsqueeze(-1).expand(-1, -1, 3)
        sample_points = torch.gather(self.voxel_centers, dim=1, index=expanded_indices)
        return self._loss_fn(pc_w, pc_xyz, sample_weights, sample_points)

    def forward(self, y_hat, y):
        # print(f"SinkhornLoss:, {y_hat.shape}, {y.shape}")
        xyz = y[:,:,:3]
        fx = y[:,:,3]
        fy = y[:,:,4]
        fz = y[:,:,5]

        y_hat = y_hat.reshape(y_hat.size(0), y_hat.size(1), -1)

        batch_loss = self.l(fx, xyz, y_hat[:,0]) \
            + self.l(fy, xyz, y_hat[:,1]) \
            + self.l(fz, xyz, y_hat[:,2])
        return torch.sum(batch_loss)


# loss_fn = SinkhornLoss(fmap)

# predicted_batch = torch.rand((2, 3, 60, 120, 120), requires_grad=True)
# w = predicted_batch.reshape((2, 3, -1))
# label_point_cloud = torch.rand((2, 32, 6), requires_grad=True)
# loss = loss_fn(label_point_cloud, w)


# predicted_batch = torch.rand((2, 40, 40, 20, 3), requires_grad=True)
# w = predicted_batch.reshape((2, -1, 3))
# label_point_cloud = torch.rand((2, 32, 6), requires_grad=True)
# loss = loss_fn(label_point_cloud, w)

# loss.backward()



# # ---- Step 1: Top-k voxel候補を作る（今回はランダムで代用） ----
# B = 1      # batch size
# K = 128    # 候補点数（本当はtopkで絞る）
# N = 32     # サンプリングしたい点の数

# # voxel density（logits用）
# vals = torch.rand(B, K)  # (B, K)

# # voxelの中心座標（今回は [-1,1] の中にランダムに配置）
# centers = torch.rand(B, K, 3) * 2 - 1  # (B, K, 3)


# # ---- Step 3: 実行 ----
# points = sample_points(vals, centers, N=N, tau=0.7, hard=True)

# print("Sampled points shape:", points.shape)
# print(points[0, :5])  # 最初の5点を表示