from mmrl.tdmpc2 import (
    EpisodeMemoryCfg,
    TDMPC2,
    TDMPC2ModelCfg,
    TDMPC2RunnerCfg,
)
from mmrl.models import Model, WorldModel


def test_tdmpc2_public_api_imports():
    cfg = TDMPC2RunnerCfg(
        steps=10,
        model=TDMPC2ModelCfg(model_size=1),
        memory=EpisodeMemoryCfg(capacity=10),
    )

    assert isinstance(cfg, TDMPC2RunnerCfg)
    assert cfg.model.enc_dim == 256
    assert cfg.model.latent_dim == 128
    assert cfg.steps == 10
    assert cfg.memory.capacity == 10
    assert TDMPC2 is not None
    assert issubclass(WorldModel, Model)
