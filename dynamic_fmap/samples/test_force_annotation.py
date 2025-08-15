from pathlib import Path
import dynamic_fmap.benchmarks.maniskill
from dynamic_fmap.benchmarks.maniskill import Replayer


# Demonstration files
demo_root = Path('/home/ryo/.maniskill/demos')
traj_files = [
#    'PickCube-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5',
    'PushT-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5',
#    'PokeCube-v1/rl/trajectory.none.pd_joint_delta_pos.physx_cuda.h5',    
]


if __name__ == '__main__':
    replayer = Replayer()

    for traj_file in traj_files:
        traj_path = str(demo_root / traj_files[0])
        env_id = traj_path.split('/')[-3]
        print(f'ENV_ID={env_id}, Trajectory file={traj_file}')

        replayer.replay(traj_path)

