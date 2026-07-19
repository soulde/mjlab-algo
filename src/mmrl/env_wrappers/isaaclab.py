"""IsaacLab environment wrappers."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from mmrl.env_wrappers.base import EnvWrapper


class IsaacLabEnvWrapper(EnvWrapper):
    """Adapt IsaacLab-style vectorized torch environments."""

    def __init__(
        self,
        env: Any,
        device: str | torch.device | None = None,
        obs_groups: Mapping[str, Sequence[str]] | None = None,
    ):
        self.env = env
        self._device = torch.device(
            device
            or getattr(env, "device", None)
            or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self._num_envs = int(getattr(env, "num_envs", 1))
        self._action_dim = self._infer_action_dim()
        self._obs_dim: int | None = None
        self._critic_obs: torch.Tensor | None = None
        self._amp_obs: torch.Tensor | None = None
        self._observations_initialized = False
        self.obs_groups = {
            "actor": ("policy",),
            "critic": ("critic",),
            "amp": ("amp",),
            **{name: tuple(groups) for name, groups in (obs_groups or {}).items()},
        }
        self._action_low, self._action_high = self._infer_action_bounds()

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def obs_dim(self) -> int:
        if self._obs_dim is None:
            self._obs_dim = int(self.reset().shape[-1])
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def unwrapped(self) -> Any:
        return getattr(self.env, "unwrapped", self.env)

    def _infer_action_dim(self) -> int:
        if hasattr(self.env, "single_action_space"):
            return int(np.prod(self.env.single_action_space.shape))
        if hasattr(self.env, "action_space"):
            shape = getattr(self.env.action_space, "shape", None)
            if shape is not None:
                return int(np.prod(shape))
        if hasattr(self.env, "num_actions"):
            return int(self.env.num_actions)
        if hasattr(self.env, "action_dim"):
            return int(self.env.action_dim)
        raise ValueError("Could not infer IsaacLab action dimension.")

    def _infer_action_bounds(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        action_space = getattr(self.env, "single_action_space", None) or getattr(
            self.env, "action_space", None
        )
        low = getattr(action_space, "low", None)
        high = getattr(action_space, "high", None)
        if low is None or high is None:
            return None, None
        return (
            torch.as_tensor(low, dtype=torch.float32, device=self.device).reshape(
                1, -1
            ),
            torch.as_tensor(high, dtype=torch.float32, device=self.device).reshape(
                1, -1
            ),
        )

    def _obs_to_tensor(self, obs: Any) -> torch.Tensor:
        if isinstance(obs, Mapping):
            tensors = [self._obs_to_tensor(value) for value in obs.values()]
            return torch.cat(tensors, dim=-1)
        if isinstance(obs, tuple) and len(obs) == 2 and isinstance(obs[1], Mapping):
            return self._obs_to_tensor(obs[0])
        if isinstance(obs, Sequence) and not isinstance(obs, (str, bytes)):
            tensors = [self._obs_to_tensor(value) for value in obs]
            return torch.cat(tensors, dim=-1)
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.reshape(self.num_envs, -1)
        else:
            tensor = tensor.reshape(self.num_envs, -1)
        return tensor

    def _select_observation_groups(
        self, observations: Mapping[str, Any], set_name: str
    ) -> torch.Tensor | None:
        values = [
            observations[group]
            for group in self.obs_groups[set_name]
            if group in observations
        ]
        if not values:
            return None
        return torch.cat([self._obs_to_tensor(value) for value in values], dim=-1)

    def _process_observations(self, observations: Any) -> torch.Tensor:
        self._observations_initialized = True
        if not isinstance(observations, Mapping):
            actor_obs = self._obs_to_tensor(observations)
            self._critic_obs = actor_obs
            self._amp_obs = None
            self._obs_dim = int(actor_obs.shape[-1])
            return actor_obs

        actor_obs = self._select_observation_groups(observations, "actor")
        if actor_obs is None:
            actor_obs = self._obs_to_tensor(observations)
        critic_obs = self._select_observation_groups(observations, "critic")
        self._critic_obs = critic_obs if critic_obs is not None else actor_obs
        self._amp_obs = self._select_observation_groups(observations, "amp")
        self._obs_dim = int(actor_obs.shape[-1])
        return actor_obs

    def get_critic_observations(self) -> torch.Tensor:
        """Return the latest critic observation set, falling back to actor."""
        if not self._observations_initialized:
            self.reset()
        assert self._critic_obs is not None
        return self._critic_obs

    @property
    def critic_obs_dim(self) -> int:
        return int(self.get_critic_observations().shape[-1])

    def get_amp_observations(self) -> torch.Tensor:
        """Return the latest IsaacLab AMP observation group."""
        if not self._observations_initialized:
            self.reset()
        if self._amp_obs is None:
            groups = ", ".join(self.obs_groups["amp"])
            raise KeyError(
                f"IsaacLab observations are missing AMP group(s): {groups}."
            )
        return self._amp_obs

    def _scale_action(self, action: torch.Tensor) -> torch.Tensor:
        action = action.to(self.device, dtype=torch.float32).reshape(
            self.num_envs, self.action_dim
        )
        action = action.clamp(-1.0, 1.0)
        if self._action_low is None or self._action_high is None:
            return action
        return self._action_low + 0.5 * (action + 1.0) * (
            self._action_high - self._action_low
        )

    def rand_act(self) -> torch.Tensor:
        return 2.0 * torch.rand(self.num_envs, self.action_dim, device=self.device) - 1.0

    def reset(self) -> torch.Tensor:
        result = self.env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        return self._process_observations(obs)

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        result = self.env.step(self._scale_action(action))
        if len(result) == 5:
            obs, reward, terminated, truncated, extras = result
            done = torch.as_tensor(terminated, device=self.device) | torch.as_tensor(
                truncated, device=self.device
            )
        elif len(result) == 4:
            obs, reward, done, extras = result
        else:
            raise ValueError(f"Unsupported IsaacLab step result length: {len(result)}")
        return (
            self._process_observations(obs),
            torch.as_tensor(reward, dtype=torch.float32, device=self.device).view(-1),
            torch.as_tensor(done, dtype=torch.bool, device=self.device).view(-1),
            dict(extras),
        )

    def close(self) -> None:
        self.env.close()
