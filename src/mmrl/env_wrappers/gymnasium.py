"""Gymnasium environment wrapper."""

from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any

import numpy as np
import torch

from mmrl.env_wrappers.base import EnvWrapper


class GymnasiumEnvWrapper(EnvWrapper):
    """Adapt Gymnasium scalar and vector continuous-control environments."""

    def __init__(self, env: Any, device: str | torch.device | None = None):
        self.env = env
        self._device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self._num_envs = int(getattr(env, "num_envs", 1))
        self._is_vector_env = hasattr(env, "num_envs")
        observation_space = getattr(
            env, "single_observation_space", env.observation_space
        )
        self._action_space = getattr(env, "single_action_space", env.action_space)
        self._obs_dim = _space_flat_dim(observation_space)
        self._action_dim = int(np.prod(self._action_space.shape))
        self._action_low = torch.as_tensor(
            self._action_space.low, dtype=torch.float32
        ).reshape(1, -1)
        self._action_high = torch.as_tensor(
            self._action_space.high, dtype=torch.float32
        ).reshape(1, -1)

    @classmethod
    def make(
        cls,
        env_id: str,
        device: str | torch.device | None = None,
        **kwargs: Any,
    ) -> "GymnasiumEnvWrapper":
        """Create and wrap a registered Gymnasium environment."""
        if env_id.startswith("dm_control/"):
            # Load Torch's optional native compiler modules before dm-control's
            # EGL stack. This avoids a known loader conflict in mixed installs.
            try:
                import_module("triton")
            except ImportError:
                pass
            shimmy = import_module("shimmy")
            gymnasium = import_module("gymnasium")
            gymnasium.register_envs(shimmy)
        else:
            gymnasium = import_module("gymnasium")
        return cls(gymnasium.make(env_id, **kwargs), device=device)

    @property
    def num_envs(self) -> int:
        return self._num_envs

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
        values = _flatten_observation(obs)
        return torch.cat(
            [
                torch.as_tensor(
                    value, dtype=torch.float32, device=self.device
                ).reshape(self.num_envs, -1)
                for value in values
            ],
            dim=-1,
        )

    def rand_act(self) -> torch.Tensor:
        return 2.0 * torch.rand(
            self.num_envs, self.action_dim, device=self.device
        ) - 1.0

    def _scale_action(self, action: torch.Tensor) -> np.ndarray:
        action = action.detach().cpu().to(dtype=torch.float32).reshape(
            self.num_envs, -1
        )
        action = action.clamp(-1.0, 1.0)
        scaled = self._action_low + 0.5 * (action + 1.0) * (
            self._action_high - self._action_low
        )
        shape = self._action_space.shape
        if self._is_vector_env:
            return scaled.numpy().reshape(self.num_envs, *shape)
        return scaled.numpy().reshape(shape)

    def reset(self) -> torch.Tensor:
        obs, _info = self.env.reset()
        return self._obs_to_tensor(obs)

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        action_np = self._scale_action(action)
        obs, reward, terminated, truncated, info = self.env.step(action_np)
        terminated_tensor = torch.as_tensor(
            terminated, dtype=torch.bool, device=self.device
        ).view(-1)
        truncated_tensor = torch.as_tensor(
            truncated, dtype=torch.bool, device=self.device
        ).view(-1)
        done = terminated_tensor | truncated_tensor
        info = dict(info)
        info.setdefault("terminated", terminated_tensor)
        info.setdefault("truncated", truncated_tensor)
        info.setdefault("time_outs", truncated_tensor)
        return (
            self._obs_to_tensor(obs),
            torch.as_tensor(reward, dtype=torch.float32, device=self.device).view(-1),
            done,
            info,
        )

    def close(self) -> None:
        self.env.close()

    def get_amp_observations(self) -> torch.Tensor:
        raise NotImplementedError(
            "GymnasiumEnvWrapper does not support AMP observation groups. "
            "Use an IsaacLab or MJLab environment wrapper for AMP training."
        )


def _space_flat_dim(space: Any) -> int:
    if hasattr(space, "spaces"):
        spaces = (
            space.spaces.values()
            if isinstance(space.spaces, Mapping)
            else space.spaces
        )
        return sum(_space_flat_dim(item) for item in spaces)
    if space.shape is None:
        raise TypeError(f"Observation space {space!r} has no fixed shape.")
    return int(np.prod(space.shape))


def _flatten_observation(obs: Any) -> list[Any]:
    if isinstance(obs, Mapping):
        values = []
        for item in obs.values():
            values.extend(_flatten_observation(item))
        return values
    if isinstance(obs, Sequence) and not isinstance(obs, (str, bytes, np.ndarray)):
        values = []
        for item in obs:
            values.extend(_flatten_observation(item))
        return values
    return [obs]
