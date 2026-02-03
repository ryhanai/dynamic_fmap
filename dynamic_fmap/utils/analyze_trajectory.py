import numpy as np
import h5py
from typing import List
from pathlib import Path    

from importlib.resources import files
import yaml
from dataclasses import dataclass
import tyro
import matplotlib.pyplot as plt


def get_trajectory_files():
    traj_def_file = files("dynamic_fmap.benchmarks.maniskill.configs") / 'trajectory_files.yaml'
    with open(str(traj_def_file), 'r') as f:
        traj_files = yaml.safe_load(f)['all']
    return traj_files


def analyze_demos(demo_root: Path, trajectory_files: List[Path]):
    def update_range(rng, new_vals):
        if rng is None:
            return (new_vals[0], new_vals[1])
        else:
            return (np.minimum(rng[0], new_vals[0]), np.maximum(rng[1], new_vals[1]))

    contact_range = [np.inf, -np.inf]
    force_range = [np.inf, -np.inf]         
    all_points = []
    all_forces = []

    for i, traj_file in enumerate(trajectory_files):
        traj_path = demo_root / traj_file
        print(f"{i}: {traj_path}")
        with h5py.File(traj_path) as f:
            if len(f) == 0:
                print(" no trajectory included")
                continue

            for traj in f.values():
                print(f" # of steps: {len(traj['actions'])}")
                point_forces = traj['obs/sensor_data/force_camera/point_forces']
                for cloud in point_forces:
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


def foo(task: str, traj_id=0, demo_method: str = 'motionplanning'):
    demo_file_path = Path('~').expanduser() / 'Downloads' / '250923' / task / demo_method
    if demo_method == 'motionplanning':
        demo_file_path /= 'trajectory.state+rgb+ts_force.pd_joint_pos.physx_cpu.h5'
    elif demo_method == 'rl':
        demo_file_path /= 'trajectory.state+rgb.pd_ee_delta_pose.physx_cpu.h5'    

    with h5py.File(demo_file_path) as fl:
        traj = list(fl.keys())[traj_id]
        T = fl[f'{traj}/obs/sensor_data/force_camera/point_forces'].shape[0]
        # return [fl[f'{traj}/obs/sensor_data/force_camera/point_forces'][t][:,3.6].sum(axis=0) for t in range(T)]
        pf = [fl[f'{traj}/obs/sensor_data/force_camera/point_forces'][t][:, 3:6] for t in range(T)]
        try:
            peg_states = np.asarray(fl[f'{traj}/env_states/actors/peg_0'])
        except KeyError:
            peg_states = np.asarray(fl[f'{traj}/env_states/actors/peg'])

        return np.linalg.norm(np.stack(pf), axis=-1).sum(axis=1), peg_states

        # trajs = list(fl.keys())
        # for traj in trajs:
        #     force_values = get_force_values_from_trajectory(fl, traj)
        #     print(f"Trajectory {traj} force values: min {force_values.min()}, max {force_values.max()}")    

def plot_trajectories(time_series: List[np.ndarray], y_labels: List[str] = None, title: str = ""):
    plt.figure()
    for values, y_label in zip(time_series, y_labels):
        plt.plot(values, label=y_label)

    plt.title(title)
    plt.xlabel("time step")
    plt.ylabel("value")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_force_and_peg_state(task: str, traj_id=0, demo_method: str = 'motionplanning'):
    p = foo(task, traj_id, demo_method)
    plot_trajectories(
        [p[0]*1e-3, p[1][:,0], p[1][:,1], p[1][:,2]],
        y_labels=['force', 'peg.pose.x', 'peg.pose.y', 'peg.pose.z'],
        title=demo_method,
        )


@dataclass
class Args:
    demo_root: Path  # root directory of demo data

def main(args: Args):
    demo_root = args.demo_root
    analyze_demos(demo_root, get_trajectory_files())

# if __name__ == '__main__':
#     args = tyro.cli(Args)
#     main(args)