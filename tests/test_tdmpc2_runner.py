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
    class FakeModel:
        def __init__(self, model_cfg, algorithm_cfg, env_spec):
            self.model_cfg = model_cfg
            self.algorithm_cfg = algorithm_cfg
            self.env_spec = env_spec

    class FakeAgent:
        def __init__(self, algorithm_cfg, model, env_spec, batch_size, device):
            self.algorithm_cfg = algorithm_cfg
            self.model = model
            self.env_spec = env_spec
            self.batch_size = batch_size
            self.device = device

    class FakeMemory:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr("mmrl.tdmpc2.runner.TDMPC2", FakeAgent)
    monkeypatch.setattr("mmrl.tdmpc2.runner.WorldModel", FakeModel)
    monkeypatch.setattr("mmrl.tdmpc2.runner.EpisodeMemory", FakeMemory)
    cfg = TDMPC2RunnerCfg(episode_length=0)

    runner = TDMPC2Runner(_RunnerEnv(), cfg, tmp_path)

    assert runner.env_spec.obs_shape == {"state": (3,)}
    assert runner.env_spec.action_dim == 2
    assert runner.env_spec.episode_length == 50
    assert runner.agent.algorithm_cfg is cfg.algorithm
    assert runner.buffer.kwargs["capacity"] == cfg.memory.capacity


def test_runner_rejects_external_algorithm(monkeypatch, tmp_path):
    cfg = TDMPC2RunnerCfg()
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
        algorithm = TDMPC2AlgorithmCfg()
        model = TDMPC2ModelCfg(model_size=1)
        memory = EpisodeMemoryCfg(capacity=100, batch_size=2)

    class FakeModel:
        def __init__(self, model_cfg, algorithm_cfg, env_spec):
            self.cfg = model_cfg

    class FakeAgent:
        def __init__(self, algorithm_cfg, model, env_spec, batch_size, device):
            self.cfg = algorithm_cfg
            self.device = device

    monkeypatch.setattr("mmrl.tdmpc2.runner.TDMPC2", FakeAgent)
    monkeypatch.setattr("mmrl.tdmpc2.runner.WorldModel", FakeModel)
    runner = TDMPC2Runner(_RunnerEnv(), TrainCfg(), tmp_path)

    assert runner.model.cfg.enc_dim == 256
    assert runner.env_spec.action_dim == 2
    assert runner.buffer._batch_size == 2


def test_inference_policy_removes_single_environment_batch_dimension():
    class FakeAgent:
        def act(self, obs, **kwargs):
            assert obs.shape == (3,)
            return torch.zeros(2)

    runner = object.__new__(TDMPC2Runner)
    runner.agent = FakeAgent()

    action = runner.get_inference_policy()(torch.zeros(1, 3), t0=True)

    assert action.shape == (2,)
