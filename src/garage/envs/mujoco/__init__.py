"""Garage wrappers for mujoco based gym environments."""
try:
    import mujoco_py  # noqa: F401
except Exception as e:
    raise e

from garage.envs.mujoco.half_cheetah_dir_env import HalfCheetahDirEnv
from garage.envs.mujoco.half_cheetah_vel_env import HalfCheetahVelEnv
from garage.envs.mujoco.ant_dir import AntDirEnv
from garage.envs.mujoco.humanoid_dir import HumanoidDirEnv
from garage.envs.mujoco.ant_goal import AntGoalEnv

__all__ = [
    'HalfCheetahDirEnv',
    'HalfCheetahVelEnv',
    'AntDirEnv',
    'HumanoidDirEnv',
    'AntGoalEnv'
]
