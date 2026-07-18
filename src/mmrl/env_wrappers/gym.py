"""Gym compatibility wrapper.

The classic Gym API is close enough to Gymnasium for most continuous-control
examples, but reset/step return shapes can differ. Add a dedicated wrapper here
when Gym support is needed.
"""

from mmrl.env_wrappers.gymnasium import GymnasiumEnvWrapper as GymEnvWrapper

