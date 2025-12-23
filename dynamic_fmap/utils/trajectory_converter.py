#!/usr/bin/env python
"""
Convert ManiSkill3 PegInsertionSide-v1 RL demos from pd_joint_delta_pos
to pd_ee_delta_pose using obs["qpos"] (robot joint positions).

Input:
    # ~/.maniskill/demos/PegInsertionSide-v1/rl/trajectory.h5
    ~/Downloads/250923/PenInsertionSide-v1/rl/trajectory.state+rgb.pd_joint_delta_pos.physx_cpu.h5

Output:
    ~/.maniskill/demos/PegInsertionSide-v1/rl/trajectory.state.pd_ee_delta_pose.cpu.offline.h5
"""

import os
import math
import h5py
import numpy as np
import torch
import gymnasium as gym
import mani_skill.envs  # noqa: F401  # required to register envs


# -------------------------------------------------------------
#  Utility: quaternion (wxyz) → XYZ Euler
#  (SAPIEN / ManiSkill use wxyz convention)
# -------------------------------------------------------------
def quat_to_euler_xyz(q):
    """
    q: array-like [w, x, y, z]
    return: [roll, pitch, yaw] (XYZ intrinsic Euler)
    """
    w, x, y, z = q

    # roll (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    # pitch (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    t2 = max(min(t2, 1.0), -1.0)  # clamp
    pitch = math.asin(t2)

    # yaw (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return np.array([roll, pitch, yaw], dtype=np.float32)


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


# -------------------------------------------------------------
#  HDF5 utilities
# -------------------------------------------------------------
# def extract_qpos(traj_group: h5py.Group) -> np.ndarray:
#     """
#     Try to find a dataset whose name ends with 'qpos' inside this trajectory group.
#     This is robust to different nesting layouts (obs/qpos, obs/robot_qpos, etc.).
#     """
#     candidates = []

#     def visitor(name, obj):
#         if isinstance(obj, h5py.Dataset) and name.endswith("qpos"):
#             candidates.append((name, obj))

#     traj_group.visititems(visitor)

#     if not candidates:
#         raise KeyError("No dataset ending with 'qpos' found in trajectory group")

#     # pick the first one (usually something like 'obs/robot_qpos')
#     name, ds = candidates[0]
#     print(f"  - Using qpos dataset: {name}, shape={ds.shape}")
#     return ds[...]


def extract_qpos(traj_group: h5py.Group) -> np.ndarray:
    """
    Return agent joint angles
    """
    candidates = []

    def visitor(name, obj):
        if name.endswith('obs'):
            candidates.append((name, obj['state'][:, :9]))

    traj_group.visititems(visitor)

    if not candidates:
        raise KeyError("No dataset ending with 'qpos' found in trajectory group")

    # pick the first one (usually something like 'obs/robot_qpos')
    name, ds = candidates[0]
    print(f"  - Using qpos dataset: {name}, shape={ds.shape}")
    return ds[...]


def extract_actions(traj_group: h5py.Group) -> np.ndarray:
    """
    Read original actions in this trajectory.
    """
    if "actions" not in traj_group:
        raise KeyError("No 'actions' dataset in trajectory group")
    ds = traj_group["actions"]
    print(f"  - Original actions shape: {ds.shape}")
    return ds[...]


# -------------------------------------------------------------
#  Main conversion: qpos sequence → pd_ee_delta_pose actions
# -------------------------------------------------------------
def compute_ee_delta_actions_from_observation(env, qpos_seq: np.ndarray, orig_actions: np.ndarray) -> np.ndarray:
    """
    qpos_seq: (T, dof)
    orig_actions: (T, A_orig)  (assumed pd_joint_delta_pos; last dim is gripper)

    return:
        new_actions: (T, 7)
        [dx, dy, dz, droll, dpitch, dyaw, gripper]
        scaled to [-1, 1] assuming pos/rot range 0.1 (meters/radians).
    """
    device = env.unwrapped.device
    agent = env.unwrapped.agent
    robot = agent.robot
    # Many Panda-based agents expose TCP / EE as agent.tcp or agent.ee_link
    ee_link = getattr(agent, "tcp", None)
    if ee_link is None:
        ee_link = getattr(agent, "ee_link", None)
    if ee_link is None:
        raise RuntimeError("Cannot find ee_link or tcp on agent; adjust script accordingly.")

    T, dof = qpos_seq.shape
    print(f"    - Converting sequence with T={T}, dof={dof}")

    ee_pos = []
    ee_euler = []

    # Put env on a known state
    obs, _ = env.reset(seed=0)

    for t in range(T):
        q = torch.as_tensor(qpos_seq[t], dtype=torch.float32, device=device)
        robot.set_qpos(q)
        # ensure kinematics updated
        if device.type == "cuda":
            env.unwrapped.scene._gpu_apply_all()
            env.unwrapped.scene.px.gpu_update_articulation_kinematics()
            env.unwrapped.scene._gpu_fetch_all()

        pose = ee_link.pose  # mani_skill.utils.structs.pose.Pose
        raw = pose.raw_pose[0].detach().cpu().numpy()  # (7,) [x,y,z,w,x,y,z]
        p = raw[:3]
        q_quat = raw[3:]  # (w,x,y,z)

        euler = quat_to_euler_xyz(q_quat)
        ee_pos.append(p)
        ee_euler.append(euler)

    ee_pos = np.stack(ee_pos, axis=0)      # (T, 3)
    ee_euler = np.stack(ee_euler, axis=0)  # (T, 3)

    # Δpose
    dpos = ee_pos[1:] - ee_pos[:-1]                # (T-1, 3)
    deuler = ee_euler[1:] - ee_euler[:-1]          # (T-1, 3)
    deuler = wrap_to_pi(deuler)                    # wrap angle diffs

    # 最後の1ステップはゼロにしておく
    # actionのtime stepがobservationのtime stepと同じことを期待するILを使う場合にはこれが必要
    # dpos = np.vstack([dpos, np.zeros((1, 3), dtype=np.float32)])
    # deuler = np.vstack([deuler, np.zeros((1, 3), dtype=np.float32)])

    # コントローラの典型レンジ（pos,rotともに±0.1）
    POS_RANGE = 0.1
    ROT_RANGE = 0.1
    dpos_norm = np.clip(dpos / POS_RANGE, -1.0, 1.0)
    deuler_norm = np.clip(deuler / ROT_RANGE, -1.0, 1.0)

    # グリッパ成分は元 action の最後の次元をそのままコピー
    gripper = orig_actions[:, -1:]  # (T, 1)

    new_actions = np.concatenate([dpos_norm, deuler_norm, gripper], axis=-1)
    print(f"    - New actions shape: {new_actions.shape}")
    return new_actions.astype(np.float32)


def compute_ee_delta_actions_from_observation_and_action(env, qpos_seq: np.ndarray, orig_actions: np.ndarray) -> np.ndarray:
    """
    qpos_seq: (T, dof)
    orig_actions: (T, A_orig)  (assumed pd_joint_delta_pos; last dim is gripper)

    return:
        new_actions: (T, 7)
        [dx, dy, dz, droll, dpitch, dyaw, gripper]
        scaled to [-1, 1] assuming pos/rot range 0.1 (meters/radians).

    NOTE:
      - 以前: FK(qpos_seq[t+1]) - FK(qpos_seq[t])
      - 変更後: FK(qpos_seq[t] + joint_delta[t]) - FK(qpos_seq[t])
    """
    device = env.unwrapped.device
    agent = env.unwrapped.agent
    robot = agent.robot

    ee_link = getattr(agent, "tcp", None)
    if ee_link is None:
        ee_link = getattr(agent, "ee_link", None)
    if ee_link is None:
        raise RuntimeError("Cannot find ee_link or tcp on agent; adjust script accordingly.")

    T, dof = qpos_seq.shape
    A = orig_actions.shape[1]
    if A != 8:
        raise ValueError(f"orig_actions must have 8 dims (joint_delta + gripper). Got {A}")

    joint_delta_dim = A - 1
    apply_dim = min(dof, joint_delta_dim)  # apply delta to the first apply_dim joints

    print(f"    - Converting sequence with T={T}, dof={dof}, action_dim={A} (apply_dim={apply_dim})")

    # Put env on a known state
    env.reset(seed=0)

    def _gpu_sync_if_needed():
        if device.type == "cuda":
            env.unwrapped.scene._gpu_apply_all()
            env.unwrapped.scene.px.gpu_update_articulation_kinematics()
            env.unwrapped.scene._gpu_fetch_all()

    def _fk_pose_from_qpos(qpos_1d: np.ndarray):
        q = torch.as_tensor(qpos_1d, dtype=torch.float32, device=device)
        robot.set_qpos(q)
        _gpu_sync_if_needed()
        pose = ee_link.pose  # mani_skill.utils.structs.pose.Pose
        raw = pose.raw_pose[0].detach().cpu().numpy()  # (7,) [x,y,z,w,x,y,z]
        p = raw[:3]
        q_quat = raw[3:]  # (w,x,y,z)
        euler = quat_to_euler_xyz(q_quat)
        return p, euler

    ee_pos = []
    ee_euler = []
    ee_pos_after = []
    ee_euler_after = []

    for t in range(T - 1):
        q_base = qpos_seq[t].astype(np.float32)

        # FK at qpos_seq[t]
        p0, e0 = _fk_pose_from_qpos(q_base)

        # FK at qpos_seq[t] + orig_actions[t, :-1] (applied to first apply_dim joints)
        q_after = q_base.copy()
        q_after[:apply_dim] = q_after[:apply_dim] + orig_actions[t, :apply_dim].astype(np.float32)
        p1, e1 = _fk_pose_from_qpos(q_after)

        ee_pos.append(p0)
        ee_euler.append(e0)
        ee_pos_after.append(p1)
        ee_euler_after.append(e1)

    ee_pos = np.stack(ee_pos, axis=0)                 # (T, 3)
    ee_euler = np.stack(ee_euler, axis=0)             # (T, 3)
    ee_pos_after = np.stack(ee_pos_after, axis=0)     # (T, 3)
    ee_euler_after = np.stack(ee_euler_after, axis=0) # (T, 3)

    # Δpose per-step: FK(qpos + delta) - FK(qpos)
    dpos = ee_pos_after - ee_pos                  # (T, 3)
    deuler = ee_euler_after - ee_euler            # (T, 3)
    deuler = wrap_to_pi(deuler)                   # wrap angle diffs

    # Typical controller ranges (pos,rot both ±0.1)
    POS_RANGE = 0.1
    ROT_RANGE = 0.1
    dpos_norm = np.clip(dpos / POS_RANGE, -1.0, 1.0)
    deuler_norm = np.clip(deuler / ROT_RANGE, -1.0, 1.0)

    # Keep gripper component as-is (last dim)
    gripper = orig_actions[:, -1:].astype(np.float32)  # (T, 1)

    new_actions = np.concatenate([dpos_norm, deuler_norm, gripper], axis=-1)  # (T, 7)
    print(f"    - New actions shape: {new_actions.shape}")
    return new_actions.astype(np.float32)


"""
Convert the JSON metadata of ManiSkill3 demos when you convert
PegInsertionSide-v1 RL demos from pd_joint_delta_pos -> pd_ee_delta_pose.

- 元 HDF5:  ~/.maniskill/demos/PegInsertionSide-v1/rl/trajectory.h5
- 新 HDF5:  ~/.maniskill/demos/PegInsertionSide-v1/rl/trajectory.state.pd_ee_delta_pose.cpu.offline.h5

このスクリプトは
- trajectory.h5      に対応する trajectory.json を読み取り
- control_mode を pd_ee_delta_pose に書き換えた JSON を
  trajectory.state.pd_ee_delta_pose.cpu.offline.json として保存します。
"""

import json
from copy import deepcopy


# 変換後の control_mode / obs_mode / sim_backend
TARGET_CONTROL_MODE = "pd_ee_delta_pose"
# obs_mode は元のままにしたいなら None にする
TARGET_OBS_MODE = "state+rgb"          # 例: "state" にしたければ "state"
TARGET_SIM_BACKEND = "physx_cpu"       # 例: "cpu" や "physx_cpu" にしたければ文字列


def convert_json(SRC_H5, DST_H5):
    src_json = SRC_H5.replace(".h5", ".json")
    dst_json = DST_H5.replace(".h5", ".json")

    if not os.path.exists(src_json):
        raise FileNotFoundError(f"Source JSON not found: {src_json}")

    print(f"[INFO] Loading JSON meta: {src_json}")
    with open(src_json, "r") as f:
        meta = json.load(f)

    new_meta = deepcopy(meta)

    # ---- env_info.env_kwargs の control_mode / obs_mode / sim_backend を更新 ----
    env_info = new_meta.get("env_info", {})
    env_kwargs = env_info.get("env_kwargs", {})

    # control_mode を上書き
    print(f"[INFO] env_kwargs.control_mode: {env_kwargs.get('control_mode')} -> {TARGET_CONTROL_MODE}")
    env_kwargs["control_mode"] = TARGET_CONTROL_MODE

    # obs_mode を変更したい場合（None のときはそのまま）
    if TARGET_OBS_MODE is not None:
        print(f"[INFO] env_kwargs.obs_mode: {env_kwargs.get('obs_mode')} -> {TARGET_OBS_MODE}")
        env_kwargs["obs_mode"] = TARGET_OBS_MODE

    # sim_backend を変更したい場合（None のときはそのまま）
    if TARGET_SIM_BACKEND is not None:
        print(f"[INFO] env_kwargs.sim_backend: {env_kwargs.get('sim_backend')} -> {TARGET_SIM_BACKEND}")
        env_kwargs["sim_backend"] = TARGET_SIM_BACKEND

    env_info["env_kwargs"] = env_kwargs
    new_meta["env_info"] = env_info

    # ---- 各 episode の control_mode を更新 ----
    episodes = new_meta.get("episodes", [])
    for ep in episodes:
        old_cm = ep.get("control_mode", None)
        ep["control_mode"] = TARGET_CONTROL_MODE
        print(f"[EP {ep.get('episode_id', '?')}] control_mode: {old_cm} -> {TARGET_CONTROL_MODE}")

    new_meta["episodes"] = episodes

    # 任意: source_type / source_desc を変えておくと後で分かりやすい
    src_type = new_meta.get("source_type", None)
    new_meta["source_type"] = "converted"
    new_meta["source_desc"] = (
        f"Converted from {src_json} / {src_type} to control_mode={TARGET_CONTROL_MODE} "
        f"using offline FK on qpos."
    )

    # ---- 書き出し ----
    print(f"[INFO] Saving converted JSON to: {dst_json}")
    with open(dst_json, "w") as f:
        json.dump(new_meta, f, indent=2)

    print("[DONE] JSON conversion finished.")


def do_convert(src_path: str, dst_path: str):
    # 環境：pd_ee_delta_pose / CPU
    env = gym.make(
        "PegInsertionSide-v1",
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        sim_backend="cpu",  # or "physx_cpu" depending on your install
        num_envs=1,
    )

    print(f"[INFO] Loading source HDF5: {src_path}")
    with h5py.File(src_path, "r") as f_src, h5py.File(dst_path, "w") as f_dst:
        # まず全体をコピー
        for key in f_src.keys():
            f_src.copy(key, f_dst)
        # file attributes
        for k, v in f_src.attrs.items():
            f_dst.attrs[k] = v

        print(f"[INFO] File structure copied. Now converting trajectories...")
        # 各 trajectory group に対して actions を上書き
        for traj_name, traj_group in f_dst.items():
            if not isinstance(traj_group, h5py.Group):
                continue
            print(f"[TRAJ] {traj_name}")
            try:
                qpos_seq = extract_qpos(traj_group)        # (T, dof)
                orig_actions = extract_actions(traj_group)  # (T, A_orig)
            except KeyError as e:
                print(f"  ! Skip this traj (missing key): {e}")
                continue

            new_actions = compute_ee_delta_actions_from_observation_and_action(env, qpos_seq, orig_actions)

            # 古い actions を削除して書き換え
            del traj_group["actions"]
            traj_group.create_dataset("actions", data=new_actions, compression="gzip")

            # control_mode の情報が attrs にあれば、必要に応じて書き換え
            if "control_mode" in traj_group.attrs:
                traj_group.attrs["control_mode"] = "pd_ee_delta_pose"

        print(f"[INFO] Saved converted file to: {dst_path}")

    env.close()


# -------------------------------------------------------------
#  Full file conversion
# -------------------------------------------------------------
def main():
    # src_path = os.path.expanduser("~/.maniskill/demos/PegInsertionSide-v1/rl/trajectory.h5")
    # dst_path = os.path.expanduser(
    #     "~/.maniskill/demos/PegInsertionSide-v1/rl/trajectory.state.pd_ee_delta_pose.cpu.offline.h5"
    # )
    src_path = os.path.expanduser("~/Downloads/250923/PegInsertionSide-v1/rl/trajectory.state+rgb.pd_joint_delta_pos.physx_cpu.h5")
    dst_path = os.path.expanduser(
        "~/Downloads/250923/PegInsertionSide-v1/rl/trajectory.state+rgb.pd_ee_delta_pose.physx_cpu_converted.h5"    
    )

    do_convert(src_path, dst_path)
    convert_json(src_path, dst_path)


# if __name__ == "__main__":
#     main()