import torch

from mmrl.env_wrappers import EnvWrapper
from mmrl.tdmpc2 import (
    EpisodeMemoryCfg,
    TDMPC2AlgorithmCfg,
    TDMPC2ModelCfg,
    TDMPC2RunnerCfg,
)
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
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr("mmrl.tdmpc2.runner.TDMPC2", FakeAgent)
    monkeypatch.setattr("mmrl.tdmpc2.runner.EpisodeMemory", FakeMemory)
    cfg = TDMPC2RunnerCfg(enable_wandb=False, episode_length=0)

    runner = TDMPC2Runner(_RunnerEnv(), cfg, tmp_path)

    assert runner.cfg.obs_shape == {"state": (3,)}
    assert runner.cfg.action_dim == 2
    assert runner.cfg.episode_length == 50
    assert runner.agent.cfg is runner.cfg
    assert runner.buffer.kwargs["capacity"] == cfg.memory.capacity


def test_runner_rejects_external_algorithm(monkeypatch, tmp_path):
    cfg = TDMPC2RunnerCfg(enable_wandb=False)
    cfg.algorithm.class_name = "environment.TDMPC2"

    try:
        TDMPC2Runner(_RunnerEnv(), cfg, tmp_path)
    except ValueError as error:
        assert "Unsupported algorithm.class_name" in str(error)
    else:
        raise AssertionError("Runner accepted an external algorithm implementation")


def test_runner_accepts_nested_class_style_config(monkeypatch, tmp_path):
    class TrainCfg:
        seed = 1
        device = "cpu"
        steps = 10
        seed_steps = 1
        episode_length = 50
        eval_episodes = 0
        eval_freq = 100
        log_freq = 0
        save_agent = False
        enable_wandb = False
        wandb_project = "mmrl"
        wandb_entity = None
        wandb_silent = True
        exp_name = "test"
        algorithm = TDMPC2AlgorithmCfg()
        model = TDMPC2ModelCfg(model_size=1)
        memory = EpisodeMemoryCfg(capacity=100, batch_size=2)

    class FakeAgent:
        def __init__(self, cfg, device):
            self.cfg = cfg
            self.device = device

    monkeypatch.setattr("mmrl.tdmpc2.runner.TDMPC2", FakeAgent)
    runner = TDMPC2Runner(_RunnerEnv(), TrainCfg(), tmp_path)

    assert runner.cfg.enc_dim == 256
    assert runner.cfg.buffer_size == 100
    assert runner.buffer._batch_size == 2
