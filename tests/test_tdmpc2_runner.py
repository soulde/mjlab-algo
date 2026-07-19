import torch
from types import SimpleNamespace

from mmrl.env_wrappers import EnvWrapper
from mmrl.tdmpc2 import (
    EpisodeMemoryCfg,
    TDMPC2AlgorithmCfg,
    TDMPC2ModelCfg,
    TDMPC2RunnerCfg,
)
from mmrl.tdmpc2.runner import TDMPC2Runner
from mmrl.tdmpc2.tdmpc2 import TDMPC2


class _FakeEnv:
    action_dim = 2

    def rand_act(self):
        return torch.zeros(1, 2)


def test_agent_batches_actions_with_independent_planning_history():
    agent = TDMPC2.__new__(TDMPC2)
    torch.nn.Module.__init__(agent)
    agent.cfg = SimpleNamespace(mpc=True, horizon=2, action_dim=1)
    agent.device = torch.device("cpu")
    agent._prev_mean = torch.zeros(2, 1)

    def fake_plan(obs, t0, eval_mode, task):
        if t0:
            agent._prev_mean.zero_()
        action = agent._prev_mean[0, 0] + obs[0, 0]
        agent._prev_mean.fill_(action)
        return action.view(1)

    agent._plan_val = fake_plan

    first = agent.act(torch.tensor([[1.0], [10.0]]), t0=torch.tensor([True, True]))
    second = agent.act(torch.tensor([[2.0], [20.0]]), t0=torch.tensor([False, True]))

    assert first.tolist() == [[1.0], [10.0]]
    assert second.tolist() == [[3.0], [20.0]]


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
        return torch.zeros(1, 2)

    def reset(self):
        return torch.zeros(1, 3)

    def step(self, action):
        return (
            torch.zeros(1, 3),
            torch.tensor([0.0]),
            torch.tensor([True]),
            {"terminated": torch.tensor([False])},
        )

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


def test_inference_policy_preserves_environment_batch_dimension():
    class FakeAgent:
        def act(self, obs, **kwargs):
            assert obs.shape == (1, 3)
            return torch.zeros(1, 2)

    runner = object.__new__(TDMPC2Runner)
    runner.agent = FakeAgent()

    action = runner.get_inference_policy()(torch.zeros(1, 3), t0=True)

    assert action.shape == (1, 2)


def test_learn_stores_asynchronously_completed_vector_episodes():
    class VectorEnv:
        num_envs = 2
        obs_dim = 1
        action_dim = 1
        device = torch.device("cpu")

        def __init__(self):
            self.step_count = 0

        def reset(self):
            return torch.zeros(2, 1)

        def rand_act(self):
            return torch.zeros(2, 1)

        def step(self, action):
            self.step_count += 1
            done = {
                1: [True, False],
                2: [False, True],
                3: [True, False],
            }[self.step_count]
            return (
                torch.full((2, 1), float(self.step_count)),
                torch.ones(2),
                torch.tensor(done),
                {"terminated": torch.zeros(2, dtype=torch.bool)},
            )

    class Memory:
        def __init__(self):
            self.episodes = []

        @property
        def num_eps(self):
            return len(self.episodes)

        def add(self, episode):
            self.episodes.append(episode)
            return len(self.episodes)

    class Logger:
        def close(self):
            pass

    runner = object.__new__(TDMPC2Runner)
    runner.env = VectorEnv()
    runner.device = torch.device("cpu")
    runner.cfg = SimpleNamespace(
        steps=6,
        seed_steps=100,
        eval_freq=100,
        eval_episodes=0,
        log_freq=0,
        save_agent=False,
        algorithm=SimpleNamespace(episodic=False),
    )
    runner.train_cfg = runner.cfg
    runner.agent = SimpleNamespace(device=torch.device("cpu"))
    runner.buffer = Memory()
    runner.logger = Logger()
    runner.log_dir = None
    runner._step = 0
    runner._ep_idx = 0
    runner.seed_steps = 100
    runner._start_time = 0.0
    runner.eval = lambda: {}
    runner._log = lambda metrics, category: None

    runner.learn()

    assert runner._step == 6
    assert [episode.shape[0] for episode in runner.buffer.episodes] == [2, 3, 3]
