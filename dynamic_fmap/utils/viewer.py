import numpy as np
import open3d as o3d
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


# カメラ位置（eye）と注視点（lookat）を指定して extrinsic を自前で計算
def get_extrinsic_look_at(eye, lookat, up):
    forward = (lookat - eye)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    R = np.vstack([right, -up, forward])
    T = -R @ eye
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = T
    return extrinsic


class Viewer:
    def __init__(self):
        # Create visualizer
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(window_name="Force Map")

        self._obs_pcd = o3d.geometry.PointCloud()
        self._fmap_pcd = o3d.geometry.PointCloud()
        self._vis.add_geometry(self._obs_pcd)
        self._vis.add_geometry(self._fmap_pcd)

        self.set_camera_pose()
        self._vis.poll_events()
        self._vis.update_renderer()

    def set_camera_pose(self,
                        lookat=np.array([0.0, 0.0, 0.0]),  # 注視点
                        eye=np.array([1.0, 0.0, 0.5]),     # カメラ位置
                        up=np.array([0.0, 0.0, 1.0]),      # 上方向
                        ):
        # カメラパラメータを取得して変更
        ctr = self._vis.get_view_control()
   
        param = ctr.convert_to_pinhole_camera_parameters()
        param.extrinsic = get_extrinsic_look_at(eye, lookat, up)
        ctr.convert_from_pinhole_camera_parameters(param)        

    def update_fmap(self, point_forces):
        cov = np.diag([0.0001, 0.0001, 0.0001])
        min_draw = 1500.

        # sample random points
        # num_points = 1000
        # points = np.random.multivariate_normal(mean, cov * 2, size=num_points)

        # sample grid points
        # 各軸の座標範囲
        x = np.linspace(-0.4, 0.4, 80)  # x方向に10点
        y = np.linspace(-0.4, 0.4, 80)  # y方向に10点
        z = np.linspace(0, 0.4, 40)  # z方向に10点

        # 格子点を生成
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  # ijはx, y, zの順を保つ

        # 座標を(N, 3)の形に変換
        points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

        # ガウス密度を計算
        densities = np.zeros(points.shape[0])
        for position, force, normal in point_forces:
            force_mag = np.linalg.norm(force)
            if force_mag < 1e-5:
                continue

            rv = multivariate_normal(mean=position, cov=cov)
            point_densities = rv.pdf(points)
            densities += point_densities

        # 表示する点を閾値でfiltering
        indices = np.where(densities > min_draw)[0]
        points = points[indices]
        densities = densities[indices]

        if len(densities) == 0:
            return

        print(np.max(densities), np.min(densities))

        # 4. 密度を[0,1]に正規化して、色（緑〜赤）に変換
        densities_normalized = (densities - densities.min()) / (densities.max() - densities.min())
        colors = plt.cm.get_cmap('RdYlGn_r')(densities_normalized)[:, :3]  # 赤(高密度)〜緑(低密度)    

        self._fmap_pcd.points = o3d.utility.Vector3dVector(points)
        self._fmap_pcd.colors = o3d.utility.Vector3dVector(colors)
        self._vis.add_geometry(self._fmap_pcd)

    def update_observed_pointcloud(self, obs):
        pts = obs['pointcloud']['xyzw'][0, ..., :3].cpu().numpy()
        col = obs['pointcloud']['rgb'][0].cpu().numpy()
        col = col / 255.
        pcd = o3d.geometry.PointCloud()
        self._obs_pcd.points = o3d.utility.Vector3dVector(pts)
        self._obs_pcd.colors = o3d.utility.Vector3dVector(col)
        self._vis.add_geometry(self._obs_pcd)

    def draw_pointclouds(self, obs_cloud, fmap_cloud):
        # 2. マテリアル設定（RGBA）
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.base_color = [1, 1, 1, 0.8]  # RGBA（A=0.3で半透明）
        material.point_size = 4.0  # 点の大きさを調整

        material2 = o3d.visualization.rendering.MaterialRecord()
        material2.shader = "defaultUnlit"
        material2.base_color = [1, 1, 1, 0.2]
        material2.point_size = 1.0

        # 3. 描画（新しいGUIで半透明対応）
        o3d.visualization.draw([
            {"name": "obs_cloud", "geometry": obs_cloud, "material": material},
            {"name": "force_cloud", "geometry": fmap_cloud, "material": material2},
            ])

        # o3d.visualization.draw_geometries([obs_cloud, fmap_cloud])

    def update(self, obs_cloud, point_forces):
        if obs_cloud != None:
            self.update_observed_pointcloud(obs_cloud)
        self.update_fmap(point_forces)
        self.set_camera_pose()
        self._vis.poll_events()
        self._vis.update_renderer()

    def __del__(self):
        self._vis.destroy_window()