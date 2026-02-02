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


# --- SpaceMouse support ---
import threading
import time
import numpy as np

try:
    import pyspacemouse
    _HAS_SPACEMOUSE = True
except Exception:
    _HAS_SPACEMOUSE = False


class SpaceMouseDevice:
    """
    Reads 6DoF motion + buttons from a SpaceMouse via pyspacemouse.
    Produces a delta pose: (dx, dy, dz, droll, dpitch, dyaw) in "user frame".
    """
    def __init__(
        self,
        translation_scale=0.005,   # meters per tick (tune)
        rotation_scale=0.02,      # rad per tick (tune)
        deadzone=0.06,            # ignore tiny noise
        hz=200,
        invert_translation=(False, False, False),
        invert_rotation=(False, False, False),
        swap_yz=False,
    ):
        self.translation_scale = float(translation_scale)
        self.rotation_scale = float(rotation_scale)
        self.deadzone = float(deadzone)
        self.hz = int(hz)

        self.invert_translation = invert_translation
        self.invert_rotation = invert_rotation
        self.swap_yz = swap_yz

        self._lock = threading.Lock()
        self._alive = False
        self._thread = None

        # latest raw
        self._t = np.zeros(3, dtype=np.float32)   # x,y,z in [-1,1] roughly
        self._r = np.zeros(3, dtype=np.float32)   # roll,pitch,yaw in [-1,1]
        self._buttons = [0, 0]  # left, right

    def start(self):
        if not _HAS_SPACEMOUSE:
            raise RuntimeError("pyspacemouse is not installed or cannot be imported.")

        ok = pyspacemouse.open()
        if not ok:
            raise RuntimeError("Failed to open SpaceMouse via pyspacemouse.open().")

        self._alive = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._alive = False
        try:
            pyspacemouse.close()
        except Exception:
            pass

    def _loop(self):
        dt = 1.0 / max(1, self.hz)
        while self._alive:
            state = pyspacemouse.read()

            if state is not None:
                # pyspacemouse returns:
                # state.x, state.y, state.z, state.roll, state.pitch, state.yaw, state.buttons
                # t = np.array([state.x, state.y, state.z], dtype=np.float32)
                # r = np.array([state.roll, state.pitch, state.yaw], dtype=np.float32)

                # StackCube-v1
                t = np.array([-state.y, state.x, -state.z], dtype=np.float32)
                r = np.array([-state.roll, -state.pitch, state.yaw], dtype=np.float32)
                # PegInsertion-v1
                t = np.array([state.x, state.y, -state.z], dtype=np.float32)
                r = np.array([-state.pitch, state.roll, state.yaw], dtype=np.float32)
                buttons = list(state.buttons) if hasattr(state, "buttons") else [0, 0]

                t = np.clip(t, -1.0, 1.0)
                r = np.clip(r, -1.0, 1.0)

                # deadzone
                t[np.abs(t) < self.deadzone] = 0.0
                r[np.abs(r) < self.deadzone] = 0.0

                if self.swap_yz:
                    t = t[[0, 2, 1]]
                    r = r[[0, 2, 1]]

                # invert axes if needed
                for i in range(3):
                    if self.invert_translation[i]:
                        t[i] *= -1.0
                    if self.invert_rotation[i]:
                        r[i] *= -1.0

                with self._lock:
                    self._t = t
                    self._r = r
                    # ensure at least 2 buttons
                    self._buttons = (buttons + [0, 0])[:2]
            time.sleep(dt)

    def get_delta(self):
        """
        Returns:
          dpos (3,) meters, drot (3,) radians, buttons [left,right]
        """
        with self._lock:
            t = self._t.copy()
            r = self._r.copy()
            b = list(self._buttons)

        dpos = t * self.translation_scale
        drot = r * self.rotation_scale
        return dpos, drot, b



@dataclass
class Args:
    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PickCube-v1"
    obs_mode: str = "none"
    robot_uid: Annotated[str, tyro.conf.arg(aliases=["-r"])] = "panda"
    """The robot to use. Robot setups supported for teleop in this script are panda and panda_stick"""
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



def solve(env: BaseEnv, debug=False, vis=False):
    assert env.unwrapped.control_mode in [
        "pd_joint_pos",
        "pd_joint_pos_vel",
    ], env.unwrapped.control_mode
    robot_has_gripper = False
    if env.unwrapped.robot_uids == "panda_stick":
        planner = PandaStickMotionPlanningSolver(
            env,
            debug=debug,
            vis=vis,
            base_pose=env.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
            joint_acc_limits=0.5,
            joint_vel_limits=0.5,
        )
    elif env.unwrapped.robot_uids == "panda" or env.unwrapped.robot_uids == "panda_wristcam":
        robot_has_gripper = True
        planner = PandaArmMotionPlanningSolver(
            env,
            debug=debug,
            vis=vis,
            base_pose=env.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
            joint_acc_limits=0.5,
            joint_vel_limits=0.5,
        )
    viewer = env.render_human()

    last_checkpoint_state = None
    gripper_open = True
    def select_panda_hand():
        viewer.select_entity(sapien_utils.get_obj_by_name(env.agent.robot.links, "panda_hand")._objs[0].entity)
    select_panda_hand()
    for plugin in viewer.plugins:
        if isinstance(plugin, sapien.utils.viewer.viewer.TransformWindow):
            transform_window = plugin


    use_spacemouse = True  # toggle as you like

    spacemouse = None
    if use_spacemouse:
        if not _HAS_SPACEMOUSE:
            print("[WARN] pyspacemouse not available. Falling back to keyboard control.")
            use_spacemouse = False
        else:
            try:
                spacemouse = SpaceMouseDevice(
                    translation_scale=0.02,  # m per tick
                    rotation_scale=0.08,     # rad per tick
                    deadzone=0.06,
                    hz=200,
                    # axis tweaks: adjust if your axes feel wrong
                    invert_translation=(False, True, False),
                    invert_rotation=(False, True, False),
                    swap_yz=False,
                ).start()
                print("[INFO] SpaceMouse connected.")
            except Exception as e:
                print(f"[WARN] SpaceMouse init failed: {e}. Falling back to keyboard.")
                spacemouse = None
                use_spacemouse = False


    while True:

        transform_window.enabled = True
        # transform_window.update_ghost_objects
        # print(transform_window.ghost_objects, transform_window._gizmo_pose)
        # planner.grasp_pose_visual.set_pose(transform_window._gizmo_pose)

        env.render_human()
        
        execute_current_pose = False
        if viewer.window.key_press("h"):
            print("""Available commands:
            h: print this help menu
            g: toggle gripper to close/open (if there is a gripper)
            u: move the panda hand up
            j: move the panda hand down
            arrow_keys: move the panda hand in the direction of the arrow keys
            n: execute command via motion planning to make the robot move to the target pose indicated by the ghost panda arm
            c: stop this episode and record the trajectory and move on to a new episode
            q: quit the script and stop collecting data. Save trajectories and optionally videos.
            """)
            pass
        # elif viewer.window.key_press("k"):
        #     print("Saving checkpoint")
        #     last_checkpoint_state = env.get_state_dict()
        # elif viewer.window.key_press("l"):
        #     if last_checkpoint_state is not None:
        #         print("Loading previous checkpoint")
        #         env.set_state_dict(last_checkpoint_state)
        #     else:
        #         print("Could not find previous checkpoint")
        elif viewer.window.key_press("q"):
            return "quit"
        elif viewer.window.key_press("c"):
            return "continue"
        # elif viewer.window.key_press("r"):
        #     viewer.select_entity(None)
        #     return "restart"
        # elif viewer.window.key_press("t"):
        #     # TODO (stao): change from position transform to rotation transform
        #     pass
        elif viewer.window.key_press("n"):
            execute_current_pose = True
        elif viewer.window.key_press("g") and robot_has_gripper:
            if gripper_open:
                gripper_open = False
                _, reward, _ ,_, info = planner.close_gripper()
            else:
                gripper_open = True
                _, reward, _ ,_, info = planner.open_gripper()
            print(f"Reward: {reward}, Info: {info}")
        elif viewer.window.key_press("u"):
            select_panda_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[0, 0, -0.01])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("j"):
            select_panda_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[0, 0, +0.01])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("down"):
            select_panda_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[+0.01, 0, 0])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("up"):
            select_panda_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[-0.01, 0, 0])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("right"):
            select_panda_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[0, -0.01, 0])).to_transformation_matrix()
            transform_window.update_ghost_objects()
        elif viewer.window.key_press("left"):
            select_panda_hand()
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * sapien.Pose(p=[0, +0.01, 0])).to_transformation_matrix()
            transform_window.update_ghost_objects()

        # handle SpaceMouse input
        dpos = np.zeros(3, dtype=np.float32)
        drot = np.zeros(3, dtype=np.float32)
        gripper_cmd = 0.0  # your existing gripper logic (open/close)

        if use_spacemouse and spacemouse is not None:
            sm_dpos, sm_drot, buttons = spacemouse.get_delta()
            dpos[:] = sm_dpos
            drot[:] = sm_drot

            # button mapping (common choice)
            # left button: close, right button: open
            # adjust signs depending on how your action space defines gripper
            if buttons[0] == 1 and buttons[1] == 0:
                _, reward, _ ,_, info = planner.close_gripper()
            elif buttons[1] == 1 and buttons[0] == 0:
                _, reward, _ ,_, info = planner.open_gripper()

            select_panda_hand()
            pose = sapien.Pose(p=dpos)
            pose.set_rpy(drot)
            transform_window.gizmo_matrix = (transform_window._gizmo_pose * pose).to_transformation_matrix()
            transform_window.update_ghost_objects()
            execute_current_pose = True
        else:
            # keep your existing keyboard control here
            pass


        if execute_current_pose:
            # z-offset of end-effector gizmo to TCP position is hardcoded for the panda robot here
            if env.unwrapped.robot_uids == "panda" or env.unwrapped.robot_uids == "panda_wristcam":
                result = planner.move_to_pose_with_screw(transform_window._gizmo_pose * sapien.Pose([0, 0, 0.1]), dry_run=True)
            elif env.unwrapped.robot_uids == "panda_stick":
                result = planner.move_to_pose_with_screw(transform_window._gizmo_pose * sapien.Pose([0, 0, 0.15]), dry_run=True)
            if result != -1 and len(result["position"]) < 150:
                _, reward, _ ,_, info = planner.follow_path(result)
                print(f"Reward: {reward}, Info: {info}")
            else:
                if result == -1: print("Plan failed")
                else: print("Generated motion plan was too long. Try a closer sub-goal")
            execute_current_pose = False

    return args


if __name__ == "__main__":
    main(parse_args())
