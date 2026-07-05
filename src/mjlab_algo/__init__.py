"""Private MJLab algorithm extensions."""

from mjlab_algo.fastsac import (
    FastSAC,
    FastSACConfig,
    FastSACReplayBuffer,
    FastSACRunner,
    make_fastsac_config,
)
from mjlab_algo.registry import (
    load_fastsac_cfg,
    load_tdmpc2_cfg,
    register_fastsac_cfg,
    register_tdmpc2_cfg,
)
from mjlab_algo.tdmpc2 import TDMPC2, TDMPC2Config, TDMPC2Runner, make_tdmpc2_config

__all__ = [
    "FastSAC",
    "FastSACConfig",
    "FastSACReplayBuffer",
    "FastSACRunner",
    "TDMPC2",
    "TDMPC2Config",
    "TDMPC2Runner",
    "load_fastsac_cfg",
    "load_tdmpc2_cfg",
    "make_fastsac_config",
    "make_tdmpc2_config",
    "register_fastsac_cfg",
    "register_tdmpc2_cfg",
]
