import numpy as np
import h5py
from typing import List
from pathlib import Path    

from importlib.resources import files
import yaml
from dataclasses import dataclass
import tyro


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


@dataclass
class Args:
    demo_root: Path  # root directory of demo data

def main(args: Args):
    demo_root = args.demo_root
    analyze_demos(demo_root, get_trajectory_files())

if __name__ == '__main__':
    args = tyro.cli(Args)
    main(args)