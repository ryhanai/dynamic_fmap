from typing import Union
from pathlib import Path

from matplotlib.pyplot import annotate
from dynamic_fmap.benchmarks.maniskill import Replayer


tasks = [
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
    "TwoRobotPickCube-v1",
    "TwoRobotStackCube-v1",
]


demo_root = Path('/home/ryo/.maniskill/demos')

def get_task_demonstration_files(task_name: str):
    task_root = demo_root / task_name
    rl_traj_files = list(task_root.glob("rl/trajectory.none.*.h5"))
    mp_traj_files = list(task_root.glob("motionplanning/trajectory.h5"))
    return [str(p) for p in rl_traj_files] + [str(p) for p in mp_traj_files]


def annotate_all_tasks(count: Union[int, None] = None):
    replayer = Replayer()    

    for task in tasks:
        print(f"Task: {task}")
        traj_paths = get_task_demonstration_files(task)
        print(f"Number of demonstration files: {len(traj_paths)}")
        for traj_path in traj_paths:
            print(f"Processing trajectory file: {traj_path}")
            replayer.replay(traj_path, count=count)


# traj_files = [
#     'PickCube-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5',
#     'PushT-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5',
#     'PokeCube-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5',    
# ]



if __name__ == '__main__':
    annotate_all_tasks(count=2)

    # replayer = Replayer()
    # for traj_path in traj_paths:
    #     traj_path = str(demo_root / traj_paths[0])
    #     task = traj_path.split('/')[-3]
    #     print(f'TASK={task}, Trajectory file={traj_path}')
    #     replayer.replay(traj_path)

