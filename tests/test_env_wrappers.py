from unittest.mock import patch

import numpy as np
import torch

from mmrl.env_wrappers.gymnasium import GymnasiumEnvWrapper
from mmrl.env_wrappers.isaaclab import IsaacLabEnvWrapper
from mmrl.env_wrappers.mjlab import MJLabVectorEnvWrapper


class _FakeBox:
    def __init__(self, low, high, shape):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        self.shape = shape

    def sample(self):
        return np.zeros(self.shape, dtype=np.float32)


class _FakeDictSpace:
    def __init__(self, spaces):
        self.spaces = spaces


class _FakeGymnasiumEnv:
    observation_space = _FakeBox(low=[-10.0, -10.0], high=[10.0, 10.0], shape=(2,))
    action_space = _FakeBox(low=[-2.0, 0.0], high=[2.0, 4.0], shape=(2,))

    def __init__(self):
        self.unwrapped = self
        self.last_action = None
        self.closed = False

    def reset(self):
        return np.asarray([1.0, -1.0], dtype=np.float32), {}

    def step(self, action):
        self.last_action = action
        return np.asarray([0.5, 0.25], dtype=np.float32), 1.5, False, True, {"x": 1}

    def close(self):
        self.closed = True


def test_gymnasium_wrapper_shapes_and_action_scaling():
    env = _FakeGymnasiumEnv()
    wrapped = GymnasiumEnvWrapper(env, device="cpu")

    obs = wrapped.reset()
    assert wrapped.num_envs == 1
    assert wrapped.obs_dim == 2
    assert wrapped.action_dim == 2
    assert obs.shape == (1, 2)

    next_obs, reward, done, info = wrapped.step(torch.tensor([[-1.0, 1.0]]))

    np.testing.assert_allclose(env.last_action, np.asarray([-2.0, 4.0]))
    assert next_obs.shape == (1, 2)
    assert reward.tolist() == [1.5]
    assert done.tolist() == [True]
    assert info["x"] == 1
    assert info["terminated"].tolist() == [False]
    assert info["truncated"].tolist() == [True]
    assert info["time_outs"].tolist() == [True]

    random_action = wrapped.rand_act()
    assert random_action.shape == (1, 2)
    assert torch.all(random_action >= -1.0)
    assert torch.all(random_action <= 1.0)

    wrapped.close()
    assert env.closed


def test_gymnasium_wrapper_flattens_dict_observations():
    class DictEnv(_FakeGymnasiumEnv):
        observation_space = _FakeDictSpace(
            {
                "position": _FakeBox([-1.0, -1.0], [1.0, 1.0], (2,)),
                "velocity": _FakeBox([-1.0], [1.0], (1,)),
            }
        )

        def reset(self):
            return {
                "position": np.asarray([1.0, 2.0]),
                "velocity": np.asarray([3.0]),
            }, {}

    wrapped = GymnasiumEnvWrapper(DictEnv(), device="cpu")

    assert wrapped.obs_dim == 3
    assert wrapped.reset().tolist() == [[1.0, 2.0, 3.0]]


def test_gymnasium_make_delegates_to_gymnasium():
    fake_gymnasium = type(
        "FakeGymnasium",
        (),
        {"make": staticmethod(lambda env_id, **kwargs: _FakeGymnasiumEnv())},
    )
    with patch(
        "mmrl.env_wrappers.gymnasium.import_module",
        return_value=fake_gymnasium,
    ):
        wrapped = GymnasiumEnvWrapper.make("Pendulum-v1", device="cpu")

    assert wrapped.obs_dim == 2


class _FakeGymnasiumVectorEnv(_FakeGymnasiumEnv):
    num_envs = 2
    single_observation_space = _FakeGymnasiumEnv.observation_space
    single_action_space = _FakeGymnasiumEnv.action_space
    observation_space = _FakeBox(
        low=[[-10.0, -10.0], [-10.0, -10.0]],
        high=[[10.0, 10.0], [10.0, 10.0]],
        shape=(2, 2),
    )
    action_space = _FakeBox(
        low=[[-2.0, 0.0], [-2.0, 0.0]],
        high=[[2.0, 4.0], [2.0, 4.0]],
        shape=(2, 2),
    )

    def reset(self):
        return np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), {}

    def step(self, action):
        self.last_action = action
        return (
            np.zeros((2, 2), dtype=np.float32),
            np.asarray([1.0, 2.0], dtype=np.float32),
            np.asarray([True, False]),
            np.asarray([False, True]),
            {"success": np.asarray([True, False])},
        )


def test_gymnasium_wrapper_supports_vector_environments():
    env = _FakeGymnasiumVectorEnv()
    wrapped = GymnasiumEnvWrapper(env, device="cpu")

    assert wrapped.num_envs == 2
    assert wrapped.obs_dim == 2
    assert wrapped.reset().tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert wrapped.rand_act().shape == (2, 2)

    obs, reward, done, info = wrapped.step(
        torch.tensor([[-1.0, 1.0], [1.0, -1.0]])
    )

    np.testing.assert_allclose(env.last_action, [[-2.0, 4.0], [2.0, 0.0]])
    assert obs.shape == (2, 2)
    assert reward.tolist() == [1.0, 2.0]
    assert done.tolist() == [True, True]
    assert info["terminated"].tolist() == [True, False]
    assert info["truncated"].tolist() == [False, True]
    assert info["time_outs"].tolist() == [False, True]


class _FakeMJLabEnv:
    num_envs = 2
    device = "cpu"
    observation_space = object()
    action_space = object()

    class _Unwrapped:
        num_envs = 2
        device = "cpu"
        action_manager = type("ActionManager", (), {"total_action_dim": 2})()
        max_episode_length = 100
        episode_length_buf = torch.zeros(2, dtype=torch.long)
        observation_manager = type(
            "ObservationManager",
            (),
            {"compute": lambda self: {"actor": torch.full((2, 2), 9.0)}},
        )()

        def seed(self, seed):
            return seed

    def __init__(self):
        self.unwrapped = self._Unwrapped()
        self.unwrapped.cfg = type("Cfg", (), {"is_finite_horizon": False})()
        self.last_action = None
        self.closed = False

    def reset(self):
        return {
            "actor": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "critic": np.asarray([[5.0], [6.0]], dtype=np.float32),
        }, {}

    def step(self, action):
        self.last_action = action
        return (
            {"actor": torch.zeros(2, 2), "critic": torch.ones(2, 1)},
            torch.tensor([1.0, 2.0]),
            torch.tensor([True, False]),
            torch.tensor([False, True]),
            {"success": torch.tensor([1.0, 0.0])},
        )

    def close(self):
        self.closed = True


def test_mjlab_vector_wrapper_preserves_parallel_batch():
    env = _FakeMJLabEnv()
    wrapped = MJLabVectorEnvWrapper(env, clip_actions=0.5)

    assert wrapped.num_envs == 2
    assert wrapped.obs_dim == 2
    assert wrapped.action_dim == 2
    assert wrapped.max_episode_length == 100
    assert wrapped.reset().tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert wrapped.select_observation_groups(("actor", "critic")).tolist() == [
        [1.0, 2.0, 5.0],
        [3.0, 4.0, 6.0],
    ]
    assert wrapped.rand_act().shape == (2, 2)
    assert wrapped.seed(7) == 7
    assert wrapped.get_observations().tolist() == [[9.0, 9.0], [9.0, 9.0]]

    obs, reward, done, info = wrapped.step(torch.ones(2, 2))

    assert obs.shape == (2, 2)
    assert reward.tolist() == [1.0, 2.0]
    assert done.tolist() == [True, True]
    assert info["terminated"].tolist() == [True, False]
    assert info["truncated"].tolist() == [False, True]
    assert info["time_outs"].tolist() == [False, True]
    assert env.last_action.tolist() == [[0.5, 0.5], [0.5, 0.5]]
    assert env.last_action.device == wrapped.device

    wrapped.close()
    assert env.closed


def test_mjlab_wrapper_requires_actor_observation_group():
    env = _FakeMJLabEnv()
    env.reset = lambda: ({"policy": torch.zeros(2, 2)}, {})

    with np.testing.assert_raises_regex(KeyError, "actor"):
        MJLabVectorEnvWrapper(env).reset()


def test_mjlab_finite_horizon_does_not_report_timeouts():
    env = _FakeMJLabEnv()
    env.unwrapped.cfg.is_finite_horizon = True

    _obs, _reward, _done, info = MJLabVectorEnvWrapper(env).step(
        torch.zeros(2, 2)
    )

    assert "time_outs" not in info


class _FakeIsaacLabEnv:
    num_envs = 2
    device = "cpu"
    single_action_space = _FakeBox(
        low=[-2.0, -4.0], high=[2.0, 4.0], shape=(2,)
    )

    def __init__(self):
        self.unwrapped = self
        self.last_action = None
        self.closed = False

    def reset(self):
        return {
            "policy": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "privileged": torch.tensor([[5.0], [6.0]]),
            "amp": torch.tensor([[7.0, 8.0], [9.0, 10.0]]),
        }, {}

    def step(self, action):
        self.last_action = action
        obs = {
            "policy": torch.zeros(2, 2),
            "privileged": torch.ones(2, 1),
            "amp": torch.full((2, 2), 2.0),
        }
        reward = torch.tensor([1.0, 2.0])
        terminated = torch.tensor([False, True])
        truncated = torch.tensor([True, False])
        return obs, reward, terminated, truncated, {"isaac": 1}

    def close(self):
        self.closed = True


def test_isaaclab_wrapper_vectorized_dict_observations_and_action_scaling():
    env = _FakeIsaacLabEnv()
    wrapped = IsaacLabEnvWrapper(env, device="cpu")

    obs = wrapped.reset()
    next_obs, reward, done, info = wrapped.step(
        torch.tensor([[-1.0, 1.0], [1.0, -1.0]])
    )

    assert wrapped.num_envs == 2
    assert wrapped.obs_dim == 2
    assert wrapped.amp_observation_dim == 2
    assert wrapped.action_dim == 2
    assert obs.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert wrapped.select_observation_groups(("policy", "privileged")).tolist() == [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ]
    assert wrapped.get_amp_observations().tolist() == [[2.0, 2.0], [2.0, 2.0]]
    assert env.last_action.tolist() == [[-2.0, 4.0], [2.0, -4.0]]
    assert next_obs.tolist() == [[0.0, 0.0], [0.0, 0.0]]
    assert reward.tolist() == [1.0, 2.0]
    assert done.tolist() == [True, True]
    assert info == {"isaac": 1}

    wrapped.close()
    assert env.closed


def test_wrapper_forwards_native_amp_observations():
    env = _FakeGymnasiumEnv()
    env.get_amp_observations = lambda: [1.0, 2.0, 3.0]
    wrapped = GymnasiumEnvWrapper(env, device="cpu")

    assert wrapped.amp_observation_dim == 3
    assert wrapped.get_amp_observations().tolist() == [[1.0, 2.0, 3.0]]


def test_isaaclab_wrapper_requires_configured_amp_group():
    env = _FakeIsaacLabEnv()
    wrapped = IsaacLabEnvWrapper(env, device="cpu")
    wrapped.reset()

    try:
        wrapped.select_observation_groups(("motion",))
    except KeyError as error:
        assert "motion" in str(error)
    else:
        raise AssertionError("IsaacLab wrapper accepted a missing AMP group")
