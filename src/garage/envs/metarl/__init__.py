"""Garage wrappers for metaRL mujoco based gym environments."""
try:
    import mujoco_py  # noqa: F401
except Exception as e:
    raise e

from garage.envs.metarl.half_cheetah_vel import HalfCheetahVelEnv
from garage.envs.metarl.half_cheetah_dir import HalfCheetahDirEnv

__all__ = [
    'HalfCheetahVelEnv',
    'HalfCheetahDirEnv'
]
