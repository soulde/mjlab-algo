"""Classic Gym environment wrapper."""

import torch

from mmrl.env_wrappers.gymnasium import GymnasiumEnvWrapper


class GymEnvWrapper(GymnasiumEnvWrapper):
    """Adapt a current Gym continuous-control environment."""

    def reset(self) -> torch.Tensor:
        obs, _info = self.env.reset()
        return self._obs_to_tensor(obs)

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        action_np = self._scale_action(action)
        obs, reward, terminated, truncated, info = self.env.step(action_np)
        done = terminated or truncated
        info = dict(info)
        info.setdefault("terminated", bool(terminated))
        info.setdefault("truncated", bool(truncated))
        info.setdefault("time_outs", torch.tensor([truncated], dtype=torch.bool))
        return (
            self._obs_to_tensor(obs),
            torch.tensor([reward], dtype=torch.float32),
            torch.tensor([done], dtype=torch.bool),
            info,
        )
