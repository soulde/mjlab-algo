import importlib

import torch._inductor.compile_fx as compile_fx
import torch._inductor.config as inductor_config

import mjlab_algo.tdmpc2 as tdmpc2
from mjlab_algo.tdmpc2.compile import configure_tdmpc2_compile


def test_configure_tdmpc2_compile_disables_shape_padding_when_enabled():
    previous = inductor_config.shape_padding
    try:
        inductor_config.shape_padding = True

        configure_tdmpc2_compile(enabled=True)

        assert inductor_config.shape_padding is False
    finally:
        inductor_config.shape_padding = previous


def test_configure_tdmpc2_compile_disables_legacy_tf32_warning():
    previous = compile_fx._warn_tf32_disabled
    try:
        configure_tdmpc2_compile(enabled=True)

        assert compile_fx._warn_tf32_disabled is not previous
        assert compile_fx._warn_tf32_disabled() is None
    finally:
        compile_fx._warn_tf32_disabled = previous


def test_configure_tdmpc2_compile_leaves_shape_padding_when_disabled():
    previous = inductor_config.shape_padding
    try:
        inductor_config.shape_padding = True

        configure_tdmpc2_compile(enabled=False)

        assert inductor_config.shape_padding is True
    finally:
        inductor_config.shape_padding = previous


def test_tdmpc2_package_import_configures_inductor_before_agent_compile():
    previous = inductor_config.shape_padding
    try:
        inductor_config.shape_padding = True

        importlib.reload(tdmpc2)

        assert inductor_config.shape_padding is False
    finally:
        inductor_config.shape_padding = previous
