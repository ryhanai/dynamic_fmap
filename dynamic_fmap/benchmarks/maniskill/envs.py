import numpy as np
import torch
from typing import Dict
# from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces

from mani_skill.envs import *
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common
from mani_skill.utils.wrappers import FrameStack


class FlattenRGBFObservationWrapper(gym.ObservationWrapper):
    """
    Flattens the rgb+force mode observations into a dictionary with three keys, "rgb", "point_forces" and "state"

    Args:
        rgb (bool): Whether to include rgb images in the observation
        state (bool): Whether to include state data in the observation
    """

    def __init__(self, env, rgb=True, force=True, state=True) -> None:
        self.base_env: BaseEnv = env.unwrapped
        super().__init__(env)
        self.include_rgb = rgb
        self.include_force = force
        self.include_state = state

        # check if rgb/depth data exists in first camera's sensor data
        first_cam = next(iter(self.base_env._init_raw_obs["sensor_data"].values()))
        if "rgb" not in first_cam:
            self.include_rgb = False
        # Currently, point_forces is ot supported by base_env. Thus, we do not check its existence here.

        new_obs = self.observation(self.base_env._init_raw_obs)
        self.base_env.update_obs_space(new_obs)

    def observation(self, observation: Dict):
        sensor_data = observation.pop("sensor_data")
        del observation["sensor_param"]
        rgb_images = []
        # depth_images = []
        for cam_data in sensor_data.values():
            if self.include_rgb:
                if "rgb" in cam_data:
                    rgb_images.append(cam_data["rgb"])

        if len(rgb_images) > 0:
            rgb_images = torch.concat(rgb_images, axis=-1)
        # flatten the rest of the data which should just be state data
        observation = common.flatten_state_dict(observation, use_torch=True, device=self.base_env.device)
        ret = dict()
        if self.include_state:
            ret["state"] = observation
        if self.include_rgb:
            ret["rgb"] = rgb_images
        if self.include_force:
            if "force_camera" in sensor_data:
                ret["point_forces"] = sensor_data["force_camera"]["point_forces"]
        return ret


class AddForceObservationWrapper(gym.Wrapper):
    """
    Adds 'point forces' to a Dict observation that already has keys:
      - 'state'
      - 'rgb'

      gym.ObservationWrapper cannot extend _init_raw_obs directly, so we extend gym.Wrapper.
    """

    def __init__(self, env: gym.Env, num_point_forces: int = 8):
        super().__init__(env)

        self._pf_key = "point_forces"
        self._num_point_forces = num_point_forces

        # if not isinstance(env.observation_space, spaces.Dict):
        #     raise TypeError("AddObsWrapper expects env.observation_space to be gym.spaces.Dict.")
        # for k in ("state", "rgb"):
        #     if k not in env.observation_space.spaces:
        #         raise KeyError(f"AddObsWrapper expects key '{k}' in env.observation_space.")

        # # Define the new space for point forces and extend observation_space
        # point_forces_space = spaces.Box(low=-np.inf, high=np.inf, shape=self._point_forces_shape, dtype=np.float32)
        # new_spaces = dict(env.observation_space.spaces)
        # if self._pf_key in new_spaces:
        #     raise KeyError("Key 'point forces' already exists in observation_space['sensor_data'].")
        # new_spaces[self._pf_key] = point_forces_space
        # self.observation_space = spaces.Dict(new_spaces)

        new_obs = self.observation(self.unwrapped._init_raw_obs)
        self.unwrapped.update_obs_space(new_obs)

        # Extend _init_raw_obs of the base env
        # This is needed so that wrappers like FrameStack that rely on _init_raw_obs work correctly.
        # base = self.env.unwrapped
        # device = getattr(base, "device", "cpu")
        # orig_init_raw_obs = getattr(base, "_init_raw_obs")
        # if isinstance(orig_init_raw_obs, dict):
        #     orig_init_raw_obs.setdefault(
        #         self._pf_key, 
        #         torch.zeros((self._num_point_forces, 9), device=device, dtype=torch.float32)
        #     )
        # else:
        #     raise AttributeError("_init_raw_obs is not dict")

    def _get_point_forces(self) -> torch.Tensor:
        dt = 1 / self.unwrapped.sim_config.sim_freq

        def get_point_forces_aux(c, flattened=True):
            if flattened:
                return [np.concatenate([p.position, p.impulse / dt, p.normal], axis=0) for p in c.points]
            else:
                return [{'position': p.position, 'force': p.impulse / dt, 'normal': p.normal} for p in c.points]

        cs = self.unwrapped.scene.get_contacts()

        if len(cs) > 0:
            A = np.concatenate([get_point_forces_aux(c, flattened=True) for c in cs])
        else:
            A = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1.]])

        # select the fixed number of point forces with largest magnitudes
        n, d = A.shape
        norms = np.linalg.norm(A[:, 3:6], axis=1)
        top_idx = np.argsort(norms)[::-1][:min(self._num_point_forces, n)]
        A_top = A[top_idx]
        if n < self._num_point_forces:
            pad = np.zeros((self._num_point_forces - n, d))  # zero-fill the rest of vectors
            A_top = np.vstack([A_top, pad])        

        pf = np.asarray(A_top, dtype=np.float32)
        return torch.unsqueeze(torch.from_numpy(pf), dim=0)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info

    def observation(self, obs):
        # obs is the underlying env's observation; add point forces.
        if not isinstance(obs, dict):
            # Some envs return OrderedDict; treat as mapping.
            raise TypeError("Underlying env must return a dict-like observation.")

        obs = dict(obs)  # avoid mutating underlying structure
        obs["point_forces"] = self._get_point_forces()
        return obs


if __name__ == "__main__":
    base_env = gym.make('PegInsertionSide-v1', obs_mode='state+rgb')
    fenv = FlattenRGBFObservationWrapper(base_env, rgb=True, force=True, state=True)
    aenv = AddForceObservationWrapper(fenv, num_point_forces=16)
    env = FrameStack(aenv, num_stack=2)
