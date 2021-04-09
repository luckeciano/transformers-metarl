"""Garage wrappers for mujoco based gym environments."""
try:
    import mujoco_py  # noqa: F401
except Exception as e:
    raise e

from garage.rand_param_envs.walker2d_rand_params import Walker2DRandParamsEnv

__all__ = [
    'Walker2DRandParamsEnv'
]
