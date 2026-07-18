import numpy as np
import torch

from mmrl.env_wrappers.gymnasium import GymnasiumEnvWrapper


class _FakeBox:
    def __init__(self, low, high, shape):
        self.low = np.asarray(low, dtype=np.float32)
        self.high = np.asarray(high, dtype=np.float32)
        self.shape = shape

    def sample(self):
        return np.zeros(self.shape, dtype=np.float32)


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
    assert info == {"x": 1}

    random_action = wrapped.rand_act()
    assert random_action.shape == (1, 2)
    assert torch.all(random_action >= -1.0)
    assert torch.all(random_action <= 1.0)

    wrapped.close()
    assert env.closed

