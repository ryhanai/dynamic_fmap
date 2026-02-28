import argparse
from ast import parse
from typing import Annotated
import gymnasium as gym
import numpy as np
import sapien.core as sapien
from mani_skill.envs.sapien_env import BaseEnv

from mani_skill.examples.motionplanning.panda.motionplanner import \
    PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.panda.motionplanner_stick import \
    PandaStickMotionPlanningSolver
import sapien.utils.viewer
import h5py
import json
import mani_skill.trajectory.utils as trajectory_utils
from mani_skill.utils import sapien_utils
from mani_skill.utils.wrappers.record import RecordEpisode
import tyro
from dataclasses import dataclass

# --- Additional tasks and task-space force observation support
from dynamic_fmap.benchmarks.maniskill.pick_from_duplicated import PickDuplicatedEnv
from dynamic_fmap.benchmarks.maniskill.envs import *

from transforms3d.quaternions import qmult, qinverse
from transforms3d.euler import euler2quat


def pos_distance(p1, p2) -> float:
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def quat_distance_deg(q1, q2) -> float:
    q_rel = qmult(q2, qinverse(q1))
    w = np.clip(abs(q_rel[0]), -1.0, 1.0)
    return np.rad2deg(2. * np.arccos(w))


@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PickDuplicated-v1"
    obs_mode: str = "none"
    robot_uid: Annotated[str, tyro.conf.arg(aliases=["-r"])] = "panda_wristcam"
    """The robot to use. Robot setups supported for teleop in this script are panda_wristcam and panda_stick"""
    record_dir: str = "demos"
    """directory to record the demonstration data and optionally videos"""
    save_video: bool = False
    """whether to save the videos of the demonstrations after collecting them all"""
    viewer_shader: str = "rt-fast"
    """the shader to use for the viewer. 'default' is fast but lower-quality shader, 'rt' and 'rt-fast' are the ray tracing shaders"""
    video_saving_shader: str = "rt-fast"
    """the shader to use for the videos of the demonstrations. 'minimal' is the fast shader, 'rt' and 'rt-fast' are the ray tracing shaders"""

def parse_args() -> Args:
    return tyro.cli(Args)

def main(args: Args):
    output_dir = f"{args.record_dir}/{args.env_id}/teleop/"
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="none",
        robot_uids=args.robot_uid,
        enable_shadow=True,
        viewer_camera_configs=dict(shader_pack=args.viewer_shader)
    )
    env = RecordEpisode(
        env,
        output_dir=output_dir,
        trajectory_name="trajectory",
        save_video=False,
        info_on_video=False,
        source_type="teleoperation",
        source_desc="teleoperation via the click+drag system"
    )
    num_trajs = 0
    seed = 0
    env.reset(seed=seed)
    while True:
        print(f"Collecting trajectory {num_trajs+1}, seed={seed}")
        code = solve(env, debug=False, vis=True)
        if code == "quit":
            num_trajs += 1
            break
        elif code == "continue":
            seed += 1
            num_trajs += 1
            env.reset(seed=seed)
            continue
        elif code == "restart":
            env.reset(seed=seed, options=dict(save_trajectory=False))
    h5_file_path = env._h5_file.filename
    json_file_path = env._json_path
    env.close()
    del env
    print(f"Trajectories saved to {h5_file_path}")
    if args.save_video:
        print(f"Saving videos to {output_dir}")

        trajectory_data = h5py.File(h5_file_path)
        with open(json_file_path, "r") as f:
            json_data = json.load(f)
        env = gym.make(
            args.env_id,
            obs_mode=args.obs_mode,
            control_mode="pd_joint_pos",
            render_mode="rgb_array",
            reward_mode="none",
            robot_uids=args.robot_uid,
            human_render_camera_configs=dict(shader_pack=args.video_saving_shader),
        )
        env = RecordEpisode(
            env,
            output_dir=output_dir,
            trajectory_name="trajectory",
            save_video=True,
            info_on_video=False,
            save_trajectory=False,
            video_fps=30
        )
        for episode in json_data["episodes"]:
            traj_id = f"traj_{episode['episode_id']}"
            data = trajectory_data[traj_id]
            env.reset(**episode["reset_kwargs"])
            env_states_list = trajectory_utils.dict_to_list_of_dicts(data["env_states"])

            env.base_env.set_state_dict(env_states_list[0])
            for action in np.array(data["actions"]):
                env.step(action)

        trajectory_data.close()
        env.close()
        del env


class PickingController:
    def __init__(self, env: BaseEnv, debug=False, vis=False):
        self.env = env

        # self.command_seq = [  # object centered cartesian TCP trajectory
        #     ([0, 0, 0.05], ),  # approach pose
        #     ([0, 0, 0], euler2quat(0, 0, np.pi/2, axes='sxyz')),     # grasp pose
        #     'gripper-close',
        #     ([0, 0, 0.15], euler2quat(0, 0, np.pi/2, axes='sxyz')),  # lift pose
        # ]
        self.command_seq = [  # object centered cartesian TCP trajectory
            ([0, 0, 0.1], euler2quat(0, np.pi, 0, axes='sxyz')),  # approach pose
            ([0, 0, 0.05], euler2quat(0, np.pi, 0, axes='sxyz')),     # grasp pose
            'gripper-close',
            ([0, 0, 0.15], euler2quat(0, np.pi, 0, axes='sxyz')),  # lift pose
        ]

        self.planner = PandaArmMotionPlanningSolver(
            env,
            debug=debug,
            vis=vis,
            base_pose=env.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
            joint_acc_limits=0.5,
            joint_vel_limits=0.5,
        )

    def execute(self):
        for cmd in self.command_seq:
            if cmd == 'gripper-close':
                self.planner.close_gripper()
            elif cmd == 'gripepr-open':
                self.planner.open_gripper()
            else:
                p, q = cmd
                pose = self.env.target.pose * sapien.Pose(p=p, q=q)
                self.execute_pose(pose)

    def execute_pose(self, goal_pose: sapien.Pose):
        print(f"pose = {goal_pose}")
        result = self.planner.move_to_pose_with_screw(goal_pose * sapien.Pose([0, 0, 0.1]), dry_run=True)

        # print planned trajectory
        # print(result)

        if result != -1 and len(result["position"]) < 300:
            _, reward, _ ,_, info = self.planner.follow_path(result)
            print(f"Reward: {reward}, Info: {info}")
        else:
            if result == -1: print("Plan failed")
            else: print("Generated motion plan was too long. Try a closer sub-goal")


def solve(env: BaseEnv, debug=False, vis=False):
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode

    viewer = env.render_human()

    controller = PickingController(env=env, debug=debug, vis=vis)

    while True:
        env.render_human()
        execute_current_pose = False

        if viewer.window.key_press("h"):
            print("""Available commands:
            h: print this help menu
            c: stop this episode and record the trajectory and move on to a new episode
            q: quit the script and stop collecting data. Save trajectories and optionally videos.
            """)
            pass
        elif viewer.window.key_press("q"):
            return "quit"
        elif viewer.window.key_press("c"):
            return "continue"

        controller.execute()

    return args


if __name__ == "__main__":
    main(parse_args())
