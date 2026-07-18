from dataclasses import dataclass

from mmrl.config import (
    config_to_dict,
    get_config_value,
    require_config_value,
    resolve_class,
    resolve_config_class,
)


@dataclass
class _ActorCfg:
    class_name: str = "builtins.dict"
    hidden_dims: tuple[int, ...] = (64, 64)


@dataclass
class _TrainCfg:
    actor: _ActorCfg


class _BaseIsaacStyleCfg:
    seed = 1
    actor = _ActorCfg()


class _IsaacStyleCfg(_BaseIsaacStyleCfg):
    max_iterations = 100


def test_config_to_dict_supports_dataclass_and_isaaclab_style_class_attrs():
    dataclass_cfg = _TrainCfg(actor=_ActorCfg(hidden_dims=(128, 128)))
    class_cfg = _IsaacStyleCfg()

    assert config_to_dict(dataclass_cfg)["actor"]["hidden_dims"] == (128, 128)
    assert config_to_dict(class_cfg)["seed"] == 1
    assert config_to_dict(class_cfg)["max_iterations"] == 100
    assert config_to_dict(class_cfg)["actor"]["class_name"] == "builtins.dict"


def test_get_and_require_config_value_support_nested_styles():
    cfg = {"runner": _IsaacStyleCfg()}

    assert get_config_value(cfg, "runner.actor.hidden_dims") == (64, 64)
    assert get_config_value(cfg, "runner.missing", default=3) == 3
    assert require_config_value(cfg, "runner.max_iterations") == 100


def test_resolve_class_supports_class_objects_and_import_paths():
    assert resolve_class(dict) is dict
    assert resolve_class("builtins.dict") is dict
    assert resolve_config_class(_ActorCfg()) is dict

