import torch

from mmrl.env_wrappers import EnvWrapper
from mmrl.tdmpc2 import TDMPC2Config
from mmrl.tdmpc2.runner import TDMPC2Runner


class _FakeEnv:
    def rand_act(self):
        return torch.zeros(2)


def test_to_td_moves_placeholder_and_step_tensors_to_obs_device():
    runner = object.__new__(TDMPC2Runner)
    runner.env = _FakeEnv()

    obs = torch.zeros(3)
    action = torch.zeros(2)
    reward = torch.zeros(1)
    terminated = torch.zeros(1)

    reset_td = runner._to_td(obs)
    step_td = runner._to_td(obs, action, reward, terminated)
    episode = torch.cat([reset_td, step_td])

    assert step_td["obs"].device == obs.device
    assert step_td["action"].device == obs.device
    assert step_td["reward"].device == obs.device
    assert step_td["terminated"].device == obs.device
    assert episode["reward"].shape == (2,)
    assert episode["terminated"].shape == (2,)


class _RunnerEnv(EnvWrapper):
    num_envs = 1
    obs_dim = 3
    action_dim = 2
    device = torch.device("cpu")

    class _Unwrapped:
        max_episode_length = 50

    unwrapped = _Unwrapped()

    def rand_act(self):
        return torch.zeros(2)

    def reset(self):
        return torch.zeros(3)

    def step(self, action):
        return torch.zeros(3), torch.tensor(0.0), True, {}

    def close(self):
        pass


def test_runner_builds_fixed_agent_and_memory(monkeypatch, tmp_path):
    class FakeAgent:
        def __init__(self, cfg, device):
            self.cfg = cfg
            self.device = device

    class FakeMemory:
        def __init__(self, cfg):
            self.cfg = cfg

    monkeypatch.setattr("mmrl.tdmpc2.runner.TDMPC2", FakeAgent)
    monkeypatch.setattr("mmrl.tdmpc2.runner.EpisodeMemory", FakeMemory)
    cfg = TDMPC2Config(enable_wandb=False, episode_length=0)

    runner = TDMPC2Runner(_RunnerEnv(), cfg, tmp_path)

    assert runner.cfg.obs_shape == {"state": (3,)}
    assert runner.cfg.action_dim == 2
    assert runner.cfg.episode_length == 50
    assert runner.agent.cfg is cfg
    assert runner.buffer.cfg is cfg


def test_runner_rejects_external_algorithm(monkeypatch, tmp_path):
    cfg = TDMPC2Config(class_name="environment.TDMPC2", enable_wandb=False)

    try:
        TDMPC2Runner(_RunnerEnv(), cfg, tmp_path)
    except ValueError as error:
        assert "Unsupported class_name" in str(error)
    else:
        raise AssertionError("Runner accepted an external algorithm implementation")
