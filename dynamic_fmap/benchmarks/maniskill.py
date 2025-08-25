from typing import Union
import numpy as np
# import transforms3d as tf
from dataclasses import dataclass
from dynamic_fmap.force_label import replay_trajectory
from mani_skill.utils.registration import register_env
from mani_skill.sensors.base_sensor import BaseSensor
from mani_skill.envs import *
from mani_skill.envs.sapien_env import BaseEnv


@dataclass
class ForceCameraConfig:
    uid: str


class ForceCamera(BaseSensor):
    """Implementation of the sensor to measure contact forces in the simulator."""

    config: ForceCameraConfig

    def __init__(
        self,
        force_camera_config: ForceCameraConfig,
        env: BaseEnv,
    ):
        super().__init__(config=force_camera_config)
        self._env = env

        # entity_uid = camera_config.entity_uid
        # if camera_config.mount is not None:
        #     self.entity = camera_config.mount
        # elif entity_uid is None:
        #     self.entity = None
        # else:
        #     if articulation is None:
        #         pass
        #     else:
        #         # if given an articulation and entity_uid (as a string), find the correct link to mount on
        #         # this is just for convenience so robot configurations can pick link to mount to by string/id
        #         self.entity = sapien_utils.get_obj_by_name(
        #             articulation.get_links(), entity_uid
        #         )
        #     if self.entity is None:
        #         raise RuntimeError(f"Mount entity ({entity_uid}) is not found")

        # intrinsic = camera_config.intrinsic
        # assert (camera_config.fov is None and intrinsic is not None) or (
        #     camera_config.fov is not None and intrinsic is None
        # )

    def capture(self):
        dt = 1 / self._env.sim_config.sim_freq

        def get_point_forces(c, flattened=True):
            if flattened:
                return [np.concatenate([p.position, p.impulse / dt, p.normal], axis=0) for p in c.points]
            else:
                return [{'position': p.position, 'force': p.impulse / dt, 'normal': p.normal} for p in c.points]

        cs = self._env.scene.get_contacts()

        if len(cs) > 0:
            self._latest_values = np.concatenate([get_point_forces(c, flattened=True) for c in cs])
            # self._latest_values = {}
            # for i, c in enumerate(cs):
            #     self._latest_values[f'contact_{i}'] = get_point_forces(c)
        else:
            # It seems that tensor with no element has issue in replay code of maniskill
            self._latest_values = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1.]])
            # self._latest_values = {}

        # It seems that tensors of different sizes cannot be saved by RecordWrapper !!
    
    def get_obs(self):
        sensor_dict = {}
        sensor_dict['point_forces'] = self._latest_values
        return sensor_dict

    def get_params(self):
        return dict()
        # return dict(
        #     extrinsic_cv=self.camera.get_extrinsic_matrix(),
        #     cam2world_gl=self.camera.get_model_matrix(),
        #     intrinsic_cv=self.camera.get_intrinsic_matrix(),
        # )


class ForceObservation:
    uid: str = 'force_camera'

    @property
    def _default_sensor_configs(self):
        return super()._default_sensor_configs


    # @property
    # def _default_sensor_configs(self):
    #     pose = sapien_utils.look_at(
    #         eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos
    #     )
    #     return [CameraConfig("force_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    def _get_obs_sensor_data(self, apply_texture_transforms: bool = True) -> dict:
        """
        Get data from all registered sensors. Auto hides any objects that are designated to be hidden

        Args:
            apply_texture_transforms (bool): Whether to apply texture transforms to the simulated sensor data to map to standard texture formats. Default is True.

        Returns:
            dict: A dictionary containing the sensor data mapping sensor name to its respective dictionary of data. The dictionary maps texture names to the data. For example the return could look like

            .. code-block:: python

                {
                    "sensor_1": {
                        "rgb": torch.Tensor,
                        "depth": torch.Tensor
                    },
                    "sensor_2": {
                        "rgb": torch.Tensor,
                        "depth": torch.Tensor
                    }
                }
        """

        if not self.uid in self._sensors:
            self._sensors[self.uid] = ForceCamera(force_camera_config=ForceCameraConfig(self.uid), env=self)

        sensor_obs = super()._get_obs_sensor_data(apply_texture_transforms=apply_texture_transforms)
        for name, sensor in self.scene.sensors.items():
            if isinstance(sensor, ForceCamera):
                sensor_obs[name] = sensor.get_obs()
                # print(f'sensor_obs[{name}] = {sensor_obs[name]}')

        return sensor_obs


# Override the already registered environments
@register_env("DrawTriangle-v1", max_episode_steps=50, override=True)
class DrawTriangleEnv(ForceObservation, DrawTriangleEnv):
    pass

@register_env("LiftPegUpright-v1", max_episode_steps=50, override=True)
class LiftPegUprightEnv(ForceObservation, LiftPegUprightEnv):
    pass

@register_env("PegInsertionSide-v1", max_episode_steps=50, override=True)
class PegInsertionSideEnv(ForceObservation, PegInsertionSideEnv):
    pass

@register_env("PickCube-v1", max_episode_steps=50, override=True)
class PickCubeEnv(ForceObservation, PickCubeEnv):
    pass

@register_env("PlugCharger-v1", max_episode_steps=50, override=True)
class PlugChargerEnv(ForceObservation, PlugChargerEnv):
    pass

@register_env("PokeCube-v1", max_episode_steps=50, override=True)
class PokeCubeEnv(ForceObservation, PokeCubeEnv):
    pass

@register_env("PullCube-v1", max_episode_steps=50, override=True)
class PullCubeEnv(ForceObservation, PullCubeEnv):
    pass

@register_env("PushCube-v1", max_episode_steps=50, override=True)
class PushCubeEnv(ForceObservation, PushCubeEnv):
    pass

@register_env("PushT-v1", max_episode_steps=50, override=True)
class PushTEnv(ForceObservation, PushTEnv):
    pass

@register_env("RollBall-v1", max_episode_steps=50, override=True)
class RollBallEnv(ForceObservation, RollBallEnv):
    pass

@register_env("StackCube-v1", max_episode_steps=50, override=True)
class StackCubeEnv(ForceObservation, StackCubeEnv):
    pass

@register_env("TwoRobotPickCube-v1", max_episode_steps=50, override=True)
class TwoRobotPickCube(ForceObservation, TwoRobotPickCube):
    pass

@register_env("TwoRobotStackCube-v1", max_episode_steps=50, override=True)
class TwoRobotStackCube(ForceObservation, TwoRobotStackCube):
    pass


class Replayer:
    def __init__(self):
        pass
        
    def replay(self, 
               traj_path: str, 
               count: Union[int, None] = None, 
               annotation=[],
               visualize_existing_annotation=False,
               ):
        self._args = replay_trajectory.Args(
            traj_path=traj_path,
            sim_backend='cpu',
            obs_mode='pointcloud', # 'rgb+depth',
            target_control_mode=None,
            verbose=False,
            save_traj=False,
            save_video=True,
            max_retry=0,
            discard_timeout=False,
            allow_failure=False,
            vis=True,
            use_env_states=True,
            use_first_env_state=False, 
            count=count,  # None by default
            reward_mode=None,
            record_rewards=False,
            shader=None, 
            video_fps=None, 
            render_mode='rgb_array', 
            num_envs=1,
            visualize_existing_annotation=False,
            )

        replay_trajectory.main(self._args)

