# import numpy as np
# import open3d as o3d
# from scipy.stats import multivariate_normal
# import matplotlib.pyplot as plt

# # 1. パラメータ設定
# mean = np.array([0.0, 0.0, 0.0])
# cov = np.diag([1.0, 1.0, 1.0])  # 単位分散の等方性ガウス

# # 2. ランダムに点を生成
# num_points = 10000
# points = np.random.multivariate_normal(mean, cov * 2, size=num_points)

# # 3. 各点に対してガウス密度を計算
# rv = multivariate_normal(mean=mean, cov=cov)
# densities = rv.pdf(points)

# # 4. 密度を[0,1]に正規化して、色（緑〜赤）に変換
# densities_normalized = (densities - densities.min()) / (densities.max() - densities.min())
# colors = plt.cm.get_cmap('RdYlGn_r')(densities_normalized)[:, :3]  # 赤(高密度)〜緑(低密度)

# # 5. 点群をOpen3Dで可視化
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)
# pcd.colors = o3d.utility.Vector3dVector(colors)

# o3d.visualization.draw_geometries([pcd])


import open3d as o3d
import numpy as np

# PointCloud 1: 赤、半透明
pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(np.random.randn(1000, 3))
pcd1.colors = o3d.utility.Vector3dVector(np.tile([[1, 0, 0]], (1000, 1)))

material1 = o3d.visualization.rendering.MaterialRecord()
material1.shader = "defaultUnlit"
material1.base_color = [1, 1, 1, 0.4]  # 半透明赤
material1.point_size = 10.0

# PointCloud 2: 青、不透明
pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(np.random.randn(1000, 3) + [2, 0, 0])
pcd2.colors = o3d.utility.Vector3dVector(np.tile([[0, 0, 1]], (1000, 1)))

material2 = o3d.visualization.rendering.MaterialRecord()
material2.shader = "defaultUnlit"
material2.base_color = [1, 1, 1, 1.0]  # 不透明青
material2.point_size = 10.0

# 複数点群を draw() に渡す
o3d.visualization.draw([
    {"name": "Red Semi", "geometry": pcd1, "material": material1},
    {"name": "Blue Opaque", "geometry": pcd2, "material": material2}
])
