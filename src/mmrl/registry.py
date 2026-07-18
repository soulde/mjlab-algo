"""Component and task configuration registries."""

from copy import deepcopy
from collections.abc import Mapping
from typing import Any

from mmrl.config import config_to_dict, require_config_value, resolve_class
from mmrl.fastsac.config import FastSACConfig, make_fastsac_config
from mmrl.tdmpc2.config import TDMPC2Config, make_tdmpc2_config

_FASTSAC_CFGS: dict[str, FastSACConfig] = {}
_TDMPC2_CFGS: dict[str, TDMPC2Config] = {}


def build_component(
    cfg: Any,
    *,
    classes: Mapping[str, type] | None = None,
    config_arg: str | None = None,
    **overrides: Any,
) -> Any:
    """Instantiate the component selected by a config's ``class_name``.

    With ``config_arg`` set, the complete config object is passed under that
    argument name. Otherwise public config fields are expanded as keyword
    arguments. Explicit overrides always take precedence.
    """
    class_name = require_config_value(cfg, "class_name")
    if isinstance(class_name, type):
        component = class_name
    elif classes is not None and class_name in classes:
        component = classes[class_name]
    else:
        component = resolve_class(class_name)
    if config_arg is not None:
        return component(**{config_arg: cfg, **overrides})

    kwargs = config_to_dict(cfg)
    kwargs.pop("class_name", None)
    kwargs.update(overrides)
    return component(**kwargs)


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
