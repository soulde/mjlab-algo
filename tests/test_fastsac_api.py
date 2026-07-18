import torch

from mmrl import FastSACRunner, FastSACRunnerCfg
from mmrl.env_wrappers import EnvWrapper
from mmrl.fastsac import FastSAC, FastSACActorCfg, OffPolicyMemoryCfg
from mmrl.models import SquashedGaussianActor, TwinQNetwork
from mmrl.memories import OffPolicyReplayMemory


class _RunnerEnv(EnvWrapper):
    num_envs = 2
    obs_dim = 4
    action_dim = 2
    device = torch.device("cpu")
    unwrapped = None

    def rand_act(self):
        return torch.zeros(self.num_envs, self.action_dim)

    def reset(self):
        return torch.zeros(self.num_envs, self.obs_dim)

    def step(self, action):
        return self.reset(), torch.ones(self.num_envs), torch.ones(
            self.num_envs, dtype=torch.bool
        ), {}

    def close(self):
        pass


def test_fastsac_public_api_imports():
    cfg = FastSACRunnerCfg(
        total_steps=32, memory=OffPolicyMemoryCfg(batch_size=8)
    )

    assert isinstance(cfg, FastSACRunnerCfg)
    assert cfg.total_steps == 32
    assert cfg.memory.batch_size == 8
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
    buffer = OffPolicyReplayMemory(capacity=16, obs_dim=4, action_dim=2, device="cpu")

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
    cfg = FastSACRunnerCfg(
        actor=FastSACActorCfg(hidden_dims=(16, 16)),
        memory=OffPolicyMemoryCfg(capacity=16, batch_size=4),
        device="cpu",
    )
    agent = FastSAC(cfg, obs_dim=4, action_dim=2, device=torch.device("cpu"))
    buffer = OffPolicyReplayMemory(capacity=16, obs_dim=4, action_dim=2, device="cpu")
    buffer.add(
        obs=torch.randn(8, 4),
        action=torch.empty(8, 2).uniform_(-1.0, 1.0),
        reward=torch.randn(8),
        next_obs=torch.randn(8, 4),
        done=torch.zeros(8, dtype=torch.bool),
    )

    metrics = agent.update(buffer.sample(cfg.memory.batch_size))

    assert set(metrics) == {"actor_loss", "critic_loss", "alpha_loss", "alpha"}
    assert metrics["alpha"] > 0.0


def test_runner_builds_components_from_class_style_config(tmp_path):
    class ActorCfg:
        class_name = "SquashedGaussianActor"
        hidden_dims = (16, 16)
        log_std_min = -10.0
        log_std_max = 1.0

    class CriticCfg:
        class_name = "TwinQNetwork"
        hidden_dims = (16, 16)

    class AlgorithmCfg:
        class_name = "FastSAC"
        gamma = 0.99
        tau = 0.005
        actor_lr = 3e-4
        critic_lr = 3e-4
        alpha_lr = 3e-4
        init_alpha = 0.2
        target_entropy = None
        auto_entropy = True

    class MemoryCfg:
        class_name = "OffPolicyReplayMemory"
        capacity = 32
        batch_size = 4

    class TrainCfg:
        device = "cpu"
        total_steps = 2
        learning_starts = 100
        train_every = 1
        gradient_steps = 1
        save_interval = 0
        log_interval = 0
        actor = ActorCfg()
        critic = CriticCfg()
        algorithm = AlgorithmCfg()
        memory = MemoryCfg()

    runner = FastSACRunner(_RunnerEnv(), TrainCfg(), tmp_path)

    assert runner.agent.obs_dim == 4
    assert runner.agent.action_dim == 2
    assert runner.buffer.capacity == 32
    assert runner.get_inference_policy()(torch.zeros(2, 4)).shape == (2, 2)


def test_runner_rejects_components_outside_mmrl():
    cfg = FastSACRunnerCfg()
    cfg.actor.class_name = "environment.CustomActor"

    try:
        FastSACRunner(_RunnerEnv(), cfg, ".")
    except ValueError as error:
        assert "Unsupported actor.class_name" in str(error)
    else:
        raise AssertionError("Runner accepted an external actor implementation")
