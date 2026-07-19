from unittest.mock import patch

import numpy as np
import torch

from mmrl.env_wrappers.gymnasium import GymnasiumEnvWrapper
from mmrl.env_wrappers.isaaclab import IsaacLabEnvWrapper


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
    assert info["terminated"] is False
    assert info["truncated"] is True
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
