"""MJLab environment wrappers."""

from collections.abc import Mapping
from typing import Any

import torch

from mmrl.env_wrappers.base import EnvWrapper


class MJLabVectorEnvWrapper(EnvWrapper):
    """Adapt an MJLab vector environment to the common runner interface."""

    def __init__(self, env: Any):
        self.env = env
        self._num_envs = int(env.num_envs)
        self._action_dim = int(env.unwrapped.action_manager.total_action_dim)
        self._obs_dim: int | None = None

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def obs_dim(self) -> int:
        if self._obs_dim is None:
            obs = self.reset()
            self._obs_dim = int(obs.shape[-1])
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def device(self) -> torch.device:
        return torch.device(self.env.device)

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    def _obs_to_tensor(self, obs: Mapping[str, Any]) -> torch.Tensor:
        tensors = []
        for value in obs.values():
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=torch.float32, device=self.device)
            tensors.append(
                value.reshape(self.num_envs, -1).to(
                    device=self.device, dtype=torch.float32
                )
            )
        return torch.cat(tensors, dim=-1)

    def rand_act(self) -> torch.Tensor:
        return 2.0 * torch.rand(
            self.num_envs, self.action_dim, device=self.device
        ) - 1.0

    def reset(self) -> torch.Tensor:
        obs, _ = self.env.reset()
        return self._obs_to_tensor(obs)

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        with torch.no_grad():
            action = action.to(self.device, dtype=torch.float32)
            obs, reward, terminated, truncated, extras = self.env.step(action)
            done = terminated | truncated
            info = dict(extras)
            info.setdefault("terminated", terminated.view(-1))
            info.setdefault("truncated", truncated.view(-1))
            info.setdefault("time_outs", truncated.view(-1))
            return (
                self._obs_to_tensor(obs),
                reward.view(-1),
                done.view(-1),
                info,
            )

    def close(self) -> None:
        self.env.close()
