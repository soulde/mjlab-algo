import torch

from mjlab_algo.tdmpc2.runner import TDMPC2Runner


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
