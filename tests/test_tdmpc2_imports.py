from mmrl.tdmpc2 import TDMPC2, TDMPC2Config, make_tdmpc2_config


def test_tdmpc2_public_api_imports():
    cfg = make_tdmpc2_config(model_size=1, steps=10, buffer_size=10)

    assert isinstance(cfg, TDMPC2Config)
    assert cfg.enc_dim == 256
    assert cfg.latent_dim == 128
    assert cfg.steps == 10
    assert cfg.buffer_size == 10
    assert TDMPC2 is not None
