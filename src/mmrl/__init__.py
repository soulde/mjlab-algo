"""Environment-agnostic reinforcement learning algorithms."""

from mmrl.fastsac import (
    FastSAC,
    FastSACConfig,
    FastSACReplayBuffer,
    FastSACRunner,
    make_fastsac_config,
)
from mmrl.registry import (
    load_fastsac_cfg,
    load_tdmpc2_cfg,
    register_fastsac_cfg,
    register_tdmpc2_cfg,
)
from mmrl.tdmpc2.config import TDMPC2Config, make_tdmpc2_config

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


def __getattr__(name: str):
    if name == "TDMPC2":
        from mmrl.tdmpc2 import TDMPC2

        return TDMPC2
    if name == "TDMPC2Runner":
        from mmrl.tdmpc2 import TDMPC2Runner

        return TDMPC2Runner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
