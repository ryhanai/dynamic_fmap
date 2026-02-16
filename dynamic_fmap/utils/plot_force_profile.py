import numpy as np
import h5py
from typing import List
from pathlib import Path    

from importlib.resources import files
import yaml
from dataclasses import dataclass
import tyro
import matplotlib.pyplot as plt


def extract_trajectory(traj_path: str, task: str, traj_id=0):
    assert task == "PegInsertionSide-v1", "Only PegInsertionSide-v1 is supported"

    with h5py.File(traj_path) as fl:
        traj = list(fl.keys())[traj_id]
        T = fl[f'{traj}/obs/sensor_data/point_forces'].shape[0]
        pf = [fl[f'{traj}/obs/sensor_data/point_forces'][t][:, 3:6] for t in range(T)]
        try:
            peg_states = np.asarray(fl[f'{traj}/env_states/actors/peg_0'])
        except KeyError:
            peg_states = np.asarray(fl[f'{traj}/env_states/actors/peg'])

        return np.linalg.norm(np.stack(pf), axis=-1).sum(axis=1), peg_states

        # trajs = list(fl.keys())
        # for traj in trajs:
        #     force_values = get_force_values_from_trajectory(fl, traj)
        #     print(f"Trajectory {traj} force values: min {force_values.min()}, max {force_values.max()}")    


def do_plot(time_series: List[np.ndarray], y_labels: List[str] = None, title: str = ""):
    plt.figure()
    for values, y_label in zip(time_series, y_labels):
        plt.plot(values, label=y_label)

    plt.title(title)
    plt.xlabel("time step")
    plt.ylabel("value")
    plt.legend()
    plt.grid(True)
    plt.show()


@dataclass
class Args:
    traj_path: Path
    """trajectory file (*.h5)"""
    traj_id: int = 0
    """trajectory number to plot"""
    task: str = 'PegInsertion-v1'
    """name of the task"""
    title: str = 'motionplanning'
    """title of the graph"""


def plot_force_and_peg_state(args: Args):
    p = extract_trajectory(args.traj_path, args.task, args.traj_id)
    do_plot(
        [p[0]*1e-3, p[1][:,0], p[1][:,1], p[1][:,2]],
        y_labels=['force', 'peg.pose.x', 'peg.pose.y', 'peg.pose.z'],
        title=args.title,
        )


def main(args: Args):
    plot_force_and_peg_state(args)


if __name__ == '__main__':
    args = tyro.cli(Args)
    main(args)


# example:
#  python plot_force_profile.py --traj_path ~/Downloads/PegInsertionSide-v1/teleop/trajectory.state+rgb+ts_force.pd_ee_delta_pose.physx_cpu.h5 --task PegInsertionSide-v1 --title 'teleoperation' --traj_id 0
