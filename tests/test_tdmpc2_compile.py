import torch._inductor.config as inductor_config

from mjlab_algo.tdmpc2.compile import configure_tdmpc2_compile


def test_configure_tdmpc2_compile_disables_shape_padding_when_enabled():
    previous = inductor_config.shape_padding
    try:
        inductor_config.shape_padding = True

        configure_tdmpc2_compile(enabled=True)

        assert inductor_config.shape_padding is False
    finally:
        inductor_config.shape_padding = previous


def test_configure_tdmpc2_compile_leaves_shape_padding_when_disabled():
    previous = inductor_config.shape_padding
    try:
        inductor_config.shape_padding = True

        configure_tdmpc2_compile(enabled=False)

        assert inductor_config.shape_padding is True
    finally:
        inductor_config.shape_padding = previous
