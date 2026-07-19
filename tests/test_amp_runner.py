import torch

from mmrl import AMPRunner, AMPRunnerCfg
from mmrl.env_wrappers import EnvWrapper
from mmrl.memories import TensorAMPDataset


class _AMPEnv(EnvWrapper):
    def __init__(self):
        self._amp_obs = torch.zeros(2, 3)
        self._critic_obs = torch.zeros(2, 5)
        self.steps = 0
        self.cfg = type("Cfg", (), {"amp": object()})()

    @property
    def num_envs(self):
        return 2

    @property
    def obs_dim(self):
        return 4

    @property
    def action_dim(self):
        return 2

    @property
    def critic_obs_dim(self):
        return 5

    @property
    def device(self):
        return torch.device("cpu")

    @property
    def unwrapped(self):
        return self

    def get_amp_observations(self):
        return self._amp_obs

    def get_critic_observations(self):
        return self._critic_obs

    def select_observation_groups(self, groups):
        values = {
            "policy": torch.full((2, 4), float(self.steps)),
            "critic": self._critic_obs,
            "amp": self._amp_obs,
        }
        return torch.cat([values[group] for group in groups], dim=-1)

    def reset(self):
        self._amp_obs.zero_()
        self._critic_obs.zero_()
        return torch.zeros(2, 4)

    def step(self, action):
        self.steps += 1
        terminal = self._amp_obs + 0.5
        self._amp_obs = self._amp_obs + 1.0
        self._critic_obs.fill_(float(self.steps))
        done = torch.tensor([self.steps % 2 == 0, False])
        info = {"terminal_amp_observations": terminal[done]}
        return torch.randn(2, 4), torch.ones(2), done, info

    def rand_act(self):
        return torch.zeros(2, 2)

    def close(self):
        pass


def _cfg():
    cfg = AMPRunnerCfg(max_iterations=1, save_interval=0, log_interval=0)
    cfg.memory.num_steps_per_env = 2
    cfg.algorithm.num_learning_epochs = 1
    cfg.algorithm.num_mini_batches = 1
    cfg.algorithm.amp_replay_capacity = 16
    cfg.actor_critic.actor_hidden_dims = (8,)
    cfg.actor_critic.critic_hidden_dims = (8,)
    cfg.discriminator.hidden_dims = (8,)
    return cfg


def test_amp_runner_trains_saves_loads_and_plays(tmp_path, monkeypatch):
    expert = TensorAMPDataset(torch.randn(16, 3), torch.randn(16, 3))
    monkeypatch.setattr(
        "mmrl.amp.runner.AMPLoader.from_config",
        lambda cfg, device: expert,
    )
    runner = AMPRunner(_AMPEnv(), _cfg(), tmp_path)

    runner.learn()

    checkpoint = tmp_path / "models" / "final.pt"
    assert checkpoint.exists()
    assert runner.policy.critic.net[0].in_features == 5
    restored = AMPRunner(_AMPEnv(), _cfg(), tmp_path / "restored")
    restored.load(checkpoint)
    action = restored.get_inference_policy()(torch.zeros(2, 4))
    assert action.shape == (2, 2)


def test_amp_runner_rejects_expert_observation_mismatch(tmp_path, monkeypatch):
    expert = TensorAMPDataset(torch.randn(8, 2), torch.randn(8, 2))
    monkeypatch.setattr(
        "mmrl.amp.runner.AMPLoader.from_config",
        lambda cfg, device: expert,
    )
    try:
        AMPRunner(_AMPEnv(), _cfg(), tmp_path)
    except ValueError as error:
        assert "does not match environment" in str(error)
    else:
        raise AssertionError("AMPRunner accepted incompatible expert data")
