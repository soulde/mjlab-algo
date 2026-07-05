from mjlab_algo.tdmpc2 import Buffer, TDMPC2, TDMPC2Config, make_tdmpc2_config
from mjlab_algo.tdmpc2.vecenv_wrapper import TDMPC2VecEnvWrapper


def test_tdmpc2_public_api_imports():
    cfg = make_tdmpc2_config(model_size=1, steps=10, buffer_size=10)

    assert isinstance(cfg, TDMPC2Config)
    assert cfg.enc_dim == 256
    assert cfg.latent_dim == 128
    assert cfg.steps == 10
    assert cfg.buffer_size == 10
    assert Buffer is not None
    assert TDMPC2 is not None
    assert TDMPC2VecEnvWrapper is not None
