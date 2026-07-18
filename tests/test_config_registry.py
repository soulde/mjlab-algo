from mmrl.fastsac import FastSACConfig
from mmrl.registry import (
    ComponentRegistry,
    build_component,
    load_fastsac_cfg,
    load_tdmpc2_cfg,
    register_fastsac_cfg,
    register_tdmpc2_cfg,
)
from mmrl.tdmpc2 import TDMPC2Config


def test_algorithm_cfg_registry_returns_copies():
    task_id = "Test-Registry-Task"
    register_fastsac_cfg(
        task_id,
        FastSACConfig(task=task_id, total_steps=123, exp_name="fastsac_test"),
    )
    register_tdmpc2_cfg(
        task_id,
        TDMPC2Config(task=task_id, steps=456, exp_name="tdmpc2_test"),
    )

    fastsac_cfg = load_fastsac_cfg(task_id)
    tdmpc2_cfg = load_tdmpc2_cfg(task_id)
    fastsac_cfg.total_steps = 999
    tdmpc2_cfg.steps = 999

    assert load_fastsac_cfg(task_id).total_steps == 123
    assert load_fastsac_cfg(task_id).exp_name == "fastsac_test"
    assert load_tdmpc2_cfg(task_id).steps == 456
    assert load_tdmpc2_cfg(task_id).exp_name == "tdmpc2_test"


def test_algorithm_cfg_registry_falls_back_to_task_defaults():
    fastsac_cfg = load_fastsac_cfg("Unregistered-FastSAC-Task")
    tdmpc2_cfg = load_tdmpc2_cfg("Unregistered-TDMPC2-Task")

    assert fastsac_cfg.task == "Unregistered-FastSAC-Task"
    assert isinstance(fastsac_cfg, FastSACConfig)
    assert tdmpc2_cfg.task == "Unregistered-TDMPC2-Task"
    assert isinstance(tdmpc2_cfg, TDMPC2Config)


class _Component:
    def __init__(self, width, device="cpu"):
        self.width = width
        self.device = device


class _ConfiguredComponent:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env


def test_component_registry_builds_from_isaaclab_style_config():
    registry = ComponentRegistry()
    registry.register("test", _Component)

    class ComponentCfg:
        class_name = "test"
        width = 128
        device = "cuda:0"

    component = build_component(ComponentCfg(), registry=registry, device="cpu")

    assert component.width == 128
    assert component.device == "cpu"
    assert registry.available() == ("test",)


def test_component_builder_can_pass_complete_config():
    registry = ComponentRegistry()
    registry.register("configured", _ConfiguredComponent)
    cfg = {"class_name": "configured", "nested": {"value": 3}}

    component = build_component(
        cfg, registry=registry, config_arg="cfg", env="environment"
    )

    assert component.cfg is cfg
    assert component.env == "environment"
