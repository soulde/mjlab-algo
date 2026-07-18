"""Gymnasium environment wrapper."""

from typing import Any

import numpy as np
import torch

from mmrl.env_wrappers.base import EnvWrapper


class GymnasiumEnvWrapper(EnvWrapper):
    """Adapt a single Gymnasium continuous-control environment."""

    def __init__(self, env: Any, device: str | torch.device | None = None):
        self.env = env
        self._device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self._obs_dim = int(np.prod(env.observation_space.shape))
        self._action_dim = int(np.prod(env.action_space.shape))
        self._action_low = torch.as_tensor(
            env.action_space.low, dtype=torch.float32
        ).reshape(1, -1)
        self._action_high = torch.as_tensor(
            env.action_space.high, dtype=torch.float32
        ).reshape(1, -1)

    @property
    def num_envs(self) -> int:
        return 1

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    def _obs_to_tensor(self, obs: Any) -> torch.Tensor:
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device).reshape(
            1, -1
        )

    def rand_act(self) -> torch.Tensor:
        return 2.0 * torch.rand(1, self.action_dim) - 1.0

    def _scale_action(self, action: torch.Tensor) -> np.ndarray:
        action = action.detach().cpu().to(dtype=torch.float32).reshape(1, -1)
        action = action.clamp(-1.0, 1.0)
        scaled = self._action_low + 0.5 * (action + 1.0) * (
            self._action_high - self._action_low
        )
        return scaled.numpy().reshape(self.env.action_space.shape)

    def reset(self) -> torch.Tensor:
        obs, _info = self.env.reset()
        return self._obs_to_tensor(obs)

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        action_np = self._scale_action(action)
        obs, reward, terminated, truncated, info = self.env.step(action_np)
        done = terminated or truncated
        return (
            self._obs_to_tensor(obs),
            torch.tensor([reward], dtype=torch.float32),
            torch.tensor([done], dtype=torch.bool),
            info,
        )

    def close(self) -> None:
        self.env.close()
