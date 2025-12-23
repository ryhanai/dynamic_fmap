from dataclasses import dataclass
import importlib
from wsgiref.headers import tspecials

# 1) まず observations モジュールを読み、オリジナルを取る
import mani_skill.envs.utils.observations as obs
_orig_parse_obs_mode_to_struct = obs.parse_obs_mode_to_struct


@dataclass
class ExtendedObservationModeStruct(obs.ObservationModeStruct):
    """
    A dataclass describing what observation data is being requested by the user
    """

    task_space_force: bool = False

    @property
    def use_task_space_force(self) -> bool:
        return self.task_space_force


def extended(mode):
    if isinstance(mode, str):
        flags = mode.split('+')
        if 'ts_force' in flags:
            print('ts_force flag detected!')
            orig_mode = '+'.join([f for f in flags if f != 'ts_force'])
            obs_mode_struct = _orig_parse_obs_mode_to_struct(orig_mode)
            return ExtendedObservationModeStruct(
                **obs_mode_struct.__dict__,
                task_space_force=True,
            )        

    return _orig_parse_obs_mode_to_struct(mode)


# 2) observations 側を patch（他所が module参照しているケース用）
obs.parse_obs_mode_to_struct = extended

# 3) sapien_env.py が "from ... import parse_obs_mode_to_struct" しているなら、
#    そのモジュールのローカル参照も patch
sapien_env = importlib.import_module("mani_skill.envs.sapien_env")  # 実際のパスに合わせて
sapien_env.parse_obs_mode_to_struct = extended