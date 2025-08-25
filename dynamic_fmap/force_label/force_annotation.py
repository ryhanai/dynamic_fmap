from typing import Union
from pathlib import Path

from matplotlib.pyplot import annotate
from dynamic_fmap.benchmarks.maniskill import Replayer


all_tasks = [
    "DrawTriangle-v1",
    "LiftPegUpright-v1",
    "PegInsertionSide-v1",
    "PickCube-v1",
    "PlugCharger-v1",
    "PokeCube-v1",
    "PullCube-v1",
    "PullCubeTool-v1",
    "PushCube-v1",
    "PushT-v1",
    "RollBall-v1",
    "StackCube-v1",
    # "TwoRobotPickCube-v1",
    # "TwoRobotStackCube-v1",
]


demo_root = Path('/home/ryo/.maniskill/demos')

def get_task_demonstration_files(task_name: str):
    task_root = demo_root / task_name
    rl_traj_files = list(task_root.glob("rl/trajectory.none.*.h5"))
    mp_traj_files = list(task_root.glob("motionplanning/trajectory.h5"))
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
            replayer.replay(traj_path, count=count, visualize_existing_annotation=visualize_existing_annotation)


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


if __name__ == '__main__':
    annotate_tasks(count=2)

    # count_trajectories()

    # replayer = Replayer()
    # for traj_path in traj_paths:
    #     traj_path = str(demo_root / traj_paths[0])
    #     task = traj_path.split('/')[-3]
    #     print(f'TASK={task}, Trajectory file={traj_path}')
    #     replayer.replay(traj_path)

