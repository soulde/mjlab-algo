"""MJLab environment wrappers."""

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from mmrl.env_wrappers.base import EnvWrapper


class MJLabVectorEnvWrapper(EnvWrapper):
    """Flatten MJLab vectorized observations for off-policy training."""

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
            tensors.append(value.reshape(self.num_envs, -1).to(dtype=torch.float32))
        return torch.cat(tensors, dim=-1)

    def rand_act(self) -> torch.Tensor:
        return 2.0 * torch.rand(self.num_envs, self.action_dim) - 1.0

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
            return self._obs_to_tensor(obs), reward.view(-1), done.view(-1), extras

    def close(self) -> None:
        self.env.close()


class MJLabSingleEnvWrapper(EnvWrapper):
    """Adapt a single-instance MJLab environment for sequence-based algorithms."""

    def __init__(self, env: Any):
        if env.num_envs != 1:
            raise ValueError(
                f"MJLabSingleEnvWrapper requires exactly 1 environment, got {env.num_envs}."
            )
        self.env = env
        self._action_dim = int(env.unwrapped.action_manager.total_action_dim)
        self._obs_dim: int | None = None

    @property
    def num_envs(self) -> int:
        return 1

    @property
    def obs_dim(self) -> int:
        if self._obs_dim is None:
            obs = self.reset()
            self._obs_dim = int(obs.numel())
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

    @property
    def observation_space(self):
        return self.env.single_observation_space

    @property
    def action_space(self):
        return self.env.single_action_space

    def _obs_to_tensor(self, obs_dict: Mapping[str, Any]) -> torch.Tensor:
        tensors = []
        for value in obs_dict.values():
            if isinstance(value, torch.Tensor):
                tensor = value.squeeze(0) if value.ndim > 1 else value
                tensors.append(tensor.flatten().to(dtype=torch.float32))
            elif isinstance(value, np.ndarray):
                tensor = torch.from_numpy(value).float()
                tensor = tensor.squeeze(0) if tensor.ndim > 1 else tensor
                tensors.append(tensor.flatten())
            else:
                tensors.append(torch.tensor(value, dtype=torch.float32).flatten())
        return torch.cat(tensors)

    def rand_act(self) -> torch.Tensor:
        return 2.0 * torch.rand(self.action_dim, device=self.device) - 1.0

    def reset(self) -> torch.Tensor:
        obs_dict, _ = self.env.reset()
        return self._obs_to_tensor(obs_dict)

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, bool, dict]:
        with torch.no_grad():
            action_batch = action.unsqueeze(0).to(self.device, dtype=torch.float32)
            obs_dict, reward, terminated, truncated, extras = self.env.step(
                action_batch
            )
            info: dict = defaultdict(float)
            if "log" in extras:
                info.update(extras["log"])
            info["success"] = float(extras.get("success", 0.0))
            info["terminated"] = torch.tensor(
                float(terminated.item()), dtype=torch.float32, device=self.device
            )
            done = bool((terminated | truncated).item())
            obs = self._obs_to_tensor(obs_dict)
        return obs, reward.squeeze(0).to(dtype=torch.float32), done, info

    def render(self, width: int = 384, height: int = 384) -> np.ndarray:
        return self.env.render()

    def close(self) -> None:
        self.env.close()

