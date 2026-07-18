"""Environment wrappers for supported simulator backends."""

from mmrl.env_wrappers.base import EnvWrapper as EnvWrapper
from mmrl.env_wrappers.gym import (
    ClassicGymEnvWrapper as ClassicGymEnvWrapper,
    GymEnvWrapper as GymEnvWrapper,
)
from mmrl.env_wrappers.gymnasium import GymnasiumEnvWrapper as GymnasiumEnvWrapper
from mmrl.env_wrappers.mjlab import (
    MJLabSingleEnvWrapper as MJLabSingleEnvWrapper,
    MJLabVectorEnvWrapper as MJLabVectorEnvWrapper,
)
