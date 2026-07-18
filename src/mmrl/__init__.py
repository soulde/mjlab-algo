"""Environment-agnostic reinforcement learning algorithms."""

from mmrl.config import (
    config_to_dict,
    get_config_value,
    require_config_value,
    resolve_class,
    resolve_config_class,
)
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
    "config_to_dict",
    "get_config_value",
    "load_fastsac_cfg",
    "load_tdmpc2_cfg",
    "make_fastsac_config",
    "make_tdmpc2_config",
    "require_config_value",
    "register_fastsac_cfg",
    "register_tdmpc2_cfg",
    "resolve_class",
    "resolve_config_class",
]


def __getattr__(name: str):
    if name == "TDMPC2":
        from mmrl.tdmpc2 import TDMPC2

        return TDMPC2
    if name == "TDMPC2Runner":
        from mmrl.tdmpc2 import TDMPC2Runner

        return TDMPC2Runner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
