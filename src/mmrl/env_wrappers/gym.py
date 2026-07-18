"""Classic Gym environment wrapper."""

import torch

from mmrl.env_wrappers.gymnasium import GymnasiumEnvWrapper


class GymEnvWrapper(GymnasiumEnvWrapper):
    """Adapt a single classic Gym continuous-control environment."""

    def reset(self) -> torch.Tensor:
        result = self.env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        return self._obs_to_tensor(obs)

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        action_np = self._scale_action(action)
        result = self.env.step(action_np)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        return (
            self._obs_to_tensor(obs),
            torch.tensor([reward], dtype=torch.float32),
            torch.tensor([done], dtype=torch.bool),
            info,
        )


ClassicGymEnvWrapper = GymEnvWrapper
