import torch

from mmrl.models import Model, WorldModel
from mmrl.tdmpc2 import (
    EpisodeMemoryCfg,
    TDMPC2,
    TDMPC2AlgorithmCfg,
    TDMPC2EnvSpec,
    TDMPC2ModelCfg,
    TDMPC2RunnerCfg,
)


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


def test_tdmpc2_components_use_explicit_configs():
    algorithm_cfg = TDMPC2AlgorithmCfg(
        num_bins=11,
        horizon=2,
        num_samples=8,
        num_elites=2,
        num_pi_trajs=0,
    )
    model_cfg = TDMPC2ModelCfg(
        num_enc_layers=2,
        enc_dim=16,
        mlp_dim=16,
        latent_dim=8,
        num_q=2,
        simnorm_dim=4,
    )
    env_spec = TDMPC2EnvSpec(
        obs_shape={"state": (3,)}, action_dim=2, episode_length=50
    )
    model = WorldModel(model_cfg, algorithm_cfg, env_spec)
    agent = TDMPC2(
        algorithm_cfg,
        model,
        env_spec,
        batch_size=2,
        device="cpu",
    )

    latent = model.encode(torch.zeros(1, 3))

    assert latent.shape == (1, 8)
    assert agent.model is model
