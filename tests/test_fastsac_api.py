import torch

from mjlab_algo import FastSACConfig, FastSACRunner, make_fastsac_config
from mjlab_algo.fastsac import FastSAC, FastSACReplayBuffer
from mjlab_algo.fastsac.networks import SquashedGaussianActor, TwinQNetwork


def test_fastsac_public_api_imports():
    cfg = make_fastsac_config(total_steps=32, batch_size=8)

    assert isinstance(cfg, FastSACConfig)
    assert cfg.total_steps == 32
    assert cfg.batch_size == 8
    assert FastSAC is not None
    assert FastSACRunner is not None


def test_fastsac_network_shapes():
    actor = SquashedGaussianActor(obs_dim=5, action_dim=2, hidden_dims=(16, 16))
    critic = TwinQNetwork(obs_dim=5, action_dim=2, hidden_dims=(16, 16))
    obs = torch.zeros(3, 5)

    action, log_prob = actor.sample(obs)
    q1, q2 = critic(obs, action)

    assert action.shape == (3, 2)
    assert log_prob.shape == (3, 1)
    assert q1.shape == (3, 1)
    assert q2.shape == (3, 1)
    assert torch.all(action <= 1.0)
    assert torch.all(action >= -1.0)


def test_fastsac_replay_buffer_samples_batches():
    buffer = FastSACReplayBuffer(capacity=16, obs_dim=4, action_dim=2, device="cpu")

    obs = torch.zeros(3, 4)
    action = torch.ones(3, 2)
    reward = torch.arange(3, dtype=torch.float32)
    next_obs = torch.ones(3, 4)
    done = torch.tensor([False, True, False])
    buffer.add(obs, action, reward, next_obs, done)

    batch = buffer.sample(batch_size=2)

    assert batch.obs.shape == (2, 4)
    assert batch.action.shape == (2, 2)
    assert batch.reward.shape == (2, 1)
    assert batch.next_obs.shape == (2, 4)
    assert batch.done.shape == (2, 1)


def test_fastsac_updates_from_replay_batch():
    cfg = make_fastsac_config(
        obs_dim=4,
        action_dim=2,
        hidden_dims=(16, 16),
        batch_size=4,
        buffer_size=16,
        device="cpu",
    )
    agent = FastSAC(cfg)
    buffer = FastSACReplayBuffer(capacity=16, obs_dim=4, action_dim=2, device="cpu")
    buffer.add(
        obs=torch.randn(8, 4),
        action=torch.empty(8, 2).uniform_(-1.0, 1.0),
        reward=torch.randn(8),
        next_obs=torch.randn(8, 4),
        done=torch.zeros(8, dtype=torch.bool),
    )

    metrics = agent.update(buffer.sample(cfg.batch_size))

    assert set(metrics) == {"actor_loss", "critic_loss", "alpha_loss", "alpha"}
    assert metrics["alpha"] > 0.0
