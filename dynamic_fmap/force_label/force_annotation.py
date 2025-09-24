from typing import Union
from pathlib import Path

# from matplotlib.pyplot import annotate
from dynamic_fmap.benchmarks.maniskill import Replayer


all_tasks = [
    # "DrawTriangle-v1",
    "LiftPegUpright-v1",
    # "PegInsertionSide-v1", This worked in Aug.
    "PickCube-v1",
    "PlugCharger-v1",
    "PokeCube-v1",
    "PullCube-v1",
#    "PullCubeTool-v1",
    "PushCube-v1",
    "PushT-v1",
    "RollBall-v1",
    "StackCube-v1",
#    "TwoRobotPickCube-v1",
#     "TwoRobotStackCube-v1",
]


# output_dir = Path('/home/ryo/Dataset/dynamic_forcemap/250826')
output_dir = Path('/home/ryo/Downloads/250923')


def get_task_demonstration_files(task_name: str,
                                 demo_root: Path = Path.home() / '.maniskill' / 'demos') -> list[str]:
    task_root = demo_root / task_name
    rl_traj_files = list(task_root.glob("rl/trajectory*.h5"))
    mp_traj_files = list(task_root.glob("motionplanning/trajectory*.h5"))
    return [str(p) for p in rl_traj_files] + [str(p) for p in mp_traj_files]


def annotate_tasks(task: Union[str, None] = None, count: Union[int, None] = None, visualize_existing_annotation=False):
    replayer = Replayer()    
    tasks = all_tasks if task is None else [task]

    for task in tasks:
        print(f"Task: {task}")
        traj_paths = get_task_demonstration_files(task)
        print(f"Number of demonstration files: {len(traj_paths)}")
        for traj_path in traj_paths:
            print(f"Processing trajectory file: {traj_path}")
            replayer.replay(traj_path,
                            count=count,
                            visualize_existing_annotation=visualize_existing_annotation,
                            output_dir = str(output_dir),
                            )


import numpy as np
import h5py

def count_trajectories(task: Union[str, None] = None):
    total_count = 0
    tasks = all_tasks if task is None else [task]    
    
    for task in tasks:
        traj_paths = get_task_demonstration_files(task)
        for i, traj_path in enumerate(traj_paths):
            total_count += i
            with h5py.File(traj_path) as f:
                print(f"{i},{total_count}: {traj_path}, {len(f.keys())}")

def analyze_trajectories(demo_root: Path, task: Union[str, None] = None):
    def update_range(rng, new_vals):
        if rng is None:
            return (new_vals[0], new_vals[1])
        else:
            return (np.minimum(rng[0], new_vals[0]), np.maximum(rng[1], new_vals[1]))

    tasks = all_tasks if task is None else [task]    

    contact_range = [np.inf, -np.inf]
    force_range = [np.inf, -np.inf]         
    all_points = []
    all_forces = []

    for task in tasks:
        print(f"Task: {task}")

        traj_paths = get_task_demonstration_files(task, demo_root=demo_root)
        for i, traj_path in enumerate(traj_paths):
            print(f"{i}: {traj_path}")
            with h5py.File(traj_path) as f:
                if len(f) == 0:
                    print(" no trajectory included")
                    continue

                traj = f['traj_0']
                camera_frames = {}
                params = traj['obs/sensor_param']
                for k2, param in params.items():
                    if 'extrinsic_cv' in param:
                        camera_frames[k2] = param['extrinsic_cv'][0]
                # print(f" # of camera: {len(camera_frames)}")
                # print(f" camera frames: {camera_frames}")

                for traj in f.values():
                    print(f" # of steps: {len(traj['actions'])}")
                    point_forces = traj['obs/sensor_data/force_camera/point_forces']
                    for cloud in point_forces.values():
                        points = cloud[:, :3]
                        range_of_contact_points = (points.min(axis=0), points.max(axis=0))
                        forces = cloud[:, 3:6]
                        # magnitudes = (forces**2).sum(axis=1)**0.5
                        range_of_force = (forces.min(axis=0), forces.max(axis=0))

                        contact_range = update_range(contact_range, range_of_contact_points)
                        force_range = update_range(force_range, range_of_force)

                        all_points.append(points)
                        all_forces.append(forces)

    print(f"range of contact points: {contact_range}")
    print(f"range of force: {force_range}")
    return np.concatenate(all_points, axis=0), np.concatenate(all_forces, axis=0)


# if __name__ == '__main__':
#     # annotate_tasks(count=2)
#     # count_trajectories()
#     analyze_trajectories(Path.home() / 'Dataset' / 'dynamic_forcemap' / '250826')

