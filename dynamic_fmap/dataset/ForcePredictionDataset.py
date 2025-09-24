from pathlib import Path
import cv2
import numpy as np
import torch
from core.utils import print_info, normalization
from torch.utils.data import Dataset
import h5py
import random


train_tasks = [
    "DrawTriangle-v1",
    "LiftPegUpright-v1",
    # "PegInsertionSide-v1", This worked in Aug.
    "PickCube-v1",
    "PlugCharger-v1",
    "PokeCube-v1",
    "PullCube-v1",
#    "PullCubeTool-v1",
    "PushCube-v1",
    "PushT-v1",
#    "RollBall-v1",
#    "TwoRobotPickCube-v1",
#     "TwoRobotStackCube-v1",
]

test_tasks = [
    "StackCube-v1",
]

all_tasks = train_tasks + test_tasks


def get_task_demonstration_files(task_name: str,
                                 demo_root: Path) -> list[str]:
    task_root = demo_root / task_name
    rl_traj_files = list(task_root.glob("rl/trajectory*.h5"))
    mp_traj_files = list(task_root.glob("motionplanning/trajectory*.h5"))
    return [str(p) for p in rl_traj_files] + [str(p) for p in mp_traj_files]


def get_all_task_demonstation_files(task_names: list[str],
                                    demo_root: Path) -> list[str]:
    nested =[get_task_demonstration_files(task_name, demo_root) for task_name in task_names]
    return [x for sublist in nested for x in sublist]


def resample_point_cloud(pc, n_points=1024):
    if pc.shape[0] >= n_points:
        idx = torch.randperm(pc.shape[0])[:n_points]
    else:
        idx = torch.randint(0, pc.shape[0], (n_points,))
    return pc[idx]


class ForcePredictionDataset(Dataset):
    """
        Dataset for force prediction

    Args:
        data_split (string):        Set the data split (train/test) .
        minmax (float, optional):  Set normalization range, default is [0.1,0.9].
        root_dir (string, optional):   Root directory of the data set.
    """

    def __init__(
        self,
        data_split,
        minmax=[0.1, 0.9],
        fminmax=[0.0, 100.0],
        root_dir=Path.home() / 'Dataset' / 'dynamic_forcemap' / '250923',
        dataset_name="250826",
        seed=42,
    ):
        self.data_split = data_split
        self.minmax = minmax
        self.fminmax = np.array(fminmax)
        self.root_dir = root_dir
        self.dataset_name = dataset_name

        np.random.seed(seed)  # data split should be the same for all runs
        self._scan_dataset()
        
    # def get_data(self, device=None):
    #     return self.images.to(device), self.forces.to(device)

    def _scan_dataset(self):
        """
        This is called at the time of initialization, and generate the list of episodes.
        """
        def update_range(rng, new_vals):
            if rng is None:
                return (new_vals[0], new_vals[1])
            else:
                return (np.minimum(rng[0], new_vals[0]), np.maximum(rng[1], new_vals[1]))

        if self.data_split == 'train':
            tasks = train_tasks
        elif self.data_split == 'test':
            tasks = test_tasks

        demo_files = get_all_task_demonstation_files(task_names=tasks, demo_root=self.root_dir)
        self._episode_ids = []

        for demo_file in demo_files:
            with h5py.File(demo_file) as f:
                n_traj = len(f)
                if n_traj == 0:
                    print_info(f" no trajectory included in {demo_file}")
                    continue

                for traj_number, traj in enumerate(f.values()):
                    point_forces = traj['obs/sensor_data/force_camera/point_forces']
                    force_range = [np.inf, -np.inf]

                    for cloud in point_forces.values():  # contact forces at each time step
                        forces = cloud[:, 3:6]
                        range_of_force = (forces.min(axis=0), forces.max(axis=0))
                        force_range = update_range(force_range, range_of_force)

                    episode_id = (demo_file, traj_number, force_range)
                    self._episode_ids.append(episode_id)

        random.shuffle(self._episode_ids)

    def _normalization(self, data, bounds):
        return normalization(data, bounds, self.minmax)

    def load_image_and_force(self, demo_file, traj_number):
        # these values shoud be taken from forcemap definition
        xyz_range = (-0.3, 0.3)
        x_range = xyz_range
        y_range = xyz_range
        z_range = xyz_range

        with h5py.File(demo_file, 'r') as f:
            imgs = np.array(f[f'traj_{traj_number}/obs/sensor_data/base_camera/rgb'])
            tm = np.random.choice(imgs.shape[0])
            # cv2.cvtColor needed?
            rgb = imgs[tm].transpose(2, 0, 1).astype('float32')
            rgb = self._normalization(rgb, (0.0, 255.0))

            pfs = f[f'traj_{traj_number}/obs/sensor_data/force_camera/point_forces/t_{tm}']

            # position vectors and force vectors are normalized differently
            xyz = pfs[:, :3]
            idx = np.where(
                (x_range[0] <= xyz[:, 0]) & (xyz[:, 0] <= x_range[1]) &
                (y_range[0] <= xyz[:, 1]) & (xyz[:, 1] <= y_range[1]) &
                (z_range[0] <= xyz[:, 2]) & (xyz[:, 2] <= z_range[1])
            )
            xyz = xyz[idx]
            xyz = self._normalization(xyz, xyz_range)

            fxyz = pfs[:, 3:6][idx]
            fxyz = np.clip(fxyz, self.fminmax[0], self.fminmax[1])
            fxyz = np.log(1. + fxyz)
            fxyz = self._normalization(fxyz, np.log(1. + self.fminmax))

            if xyz.shape[0] <= 0:
                # print(f'no point force in the defined forcemap, {demo_file}, {traj_number}')
                xyz = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
                fxyz = np.array([[0.1, 0.1, 0.1]], dtype=np.float32)

            normalized_pfs = np.concatenate([xyz, fxyz], axis=1)
            resampled_pfs = resample_point_cloud(normalized_pfs, n_points=32) # XYZRGB

            assert resampled_pfs.shape == (32, 6), f"Unexpected shape [force]: {resampled_pfs.shape}, {demo_file}, {traj_number}"
            assert rgb.shape == (3, 128, 128), f"Unexpected shape [rgb]: {rgb.shape}, {demo_file}, {traj_number}"

            return rgb, resampled_pfs

    def __len__(self):
        return len(self._episode_ids)  # This is different from the number of data samples

    def __getitem__(self, indices):
        imgs = []
        forces = []

        # print(indices)

        for idx in indices:
            demo_file, traj_number, _ = self._episode_ids[idx]
            rgb, pfs = self.load_image_and_force(demo_file, traj_number)
            imgs.append(rgb)
            forces.append(pfs)

        imgs = np.array(imgs)
        x_img = torch.from_numpy(imgs).float()
        forces = np.array(forces)
        y_force = torch.from_numpy(forces).float()

        return x_img, y_force
