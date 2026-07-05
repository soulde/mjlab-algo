"""Algorithm configuration registry for MJLab extension tasks."""

from copy import deepcopy

from mjlab_algo.fastsac import FastSACConfig, make_fastsac_config
from mjlab_algo.tdmpc2 import TDMPC2Config, make_tdmpc2_config

_FASTSAC_CFGS: dict[str, FastSACConfig] = {}
_TDMPC2_CFGS: dict[str, TDMPC2Config] = {}


def register_fastsac_cfg(task_id: str, cfg: FastSACConfig) -> None:
    """Register a FastSAC config for a task."""
    cfg = deepcopy(cfg)
    cfg.task = task_id
    _FASTSAC_CFGS[task_id] = cfg


def register_tdmpc2_cfg(task_id: str, cfg: TDMPC2Config) -> None:
    """Register a TD-MPC2 config for a task."""
    cfg = deepcopy(cfg)
    cfg.task = task_id
    _TDMPC2_CFGS[task_id] = cfg


def load_fastsac_cfg(task_id: str) -> FastSACConfig:
    """Load the registered FastSAC config for a task, or a task-named default."""
    if task_id not in _FASTSAC_CFGS:
        return make_fastsac_config(task=task_id)
    return deepcopy(_FASTSAC_CFGS[task_id])


def load_tdmpc2_cfg(task_id: str) -> TDMPC2Config:
    """Load the registered TD-MPC2 config for a task, or a task-named default."""
    if task_id not in _TDMPC2_CFGS:
        return make_tdmpc2_config(task=task_id)
    return deepcopy(_TDMPC2_CFGS[task_id])
