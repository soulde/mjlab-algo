"""MJLab environment wrappers."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from mmrl.env_wrappers.base import EnvWrapper


class MJLabVectorEnvWrapper(EnvWrapper):
    """Adapt an MJLab vector environment to the common runner interface."""

    def __init__(self, env: Any, clip_actions: float | None = None):
        self.env = env
        self.clip_actions = clip_actions
        self._num_envs = int(self.unwrapped.num_envs)
        self._action_dim = int(self.unwrapped.action_manager.total_action_dim)
        self._obs_dim: int | None = None
        self._observation_groups: dict[str, torch.Tensor] = {}
        self._observations_initialized = False

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
        return torch.device(self.unwrapped.device)

    @property
    def unwrapped(self) -> Any:
        return self.env.unwrapped

    @property
    def cfg(self) -> Any:
        return self.unwrapped.cfg

    @property
    def max_episode_length(self) -> int:
        return int(self.unwrapped.max_episode_length)

    @property
    def observation_space(self) -> Any:
        return self.env.observation_space

    @property
    def action_space(self) -> Any:
        return self.env.action_space

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.unwrapped.episode_length_buf = value

    def _obs_to_tensor(self, obs: Any) -> torch.Tensor:
        if isinstance(obs, Mapping):
            return torch.cat(
                [self._obs_to_tensor(value) for value in obs.values()], dim=-1
            )
        if isinstance(obs, Sequence) and not isinstance(obs, (str, bytes)):
            tensors = [self._obs_to_tensor(value) for value in obs]
            return torch.cat(tensors, dim=-1)
        return torch.as_tensor(
            obs, device=self.device, dtype=torch.float32
        ).reshape(self.num_envs, -1)

    def _process_observations(self, observations: Mapping[str, Any]) -> torch.Tensor:
        if not isinstance(observations, Mapping):
            raise TypeError("MJLab observations must be grouped in a mapping.")
        self._observation_groups = {
            name: self._obs_to_tensor(value)
            for name, value in observations.items()
        }
        self._observations_initialized = True
        if "actor" not in self._observation_groups:
            raise KeyError("MJLab observations are missing the required 'actor' group.")
        actor_obs = self._observation_groups["actor"]
        self._obs_dim = int(actor_obs.shape[-1])
        return actor_obs

    def get_observation_groups(self) -> Mapping[str, torch.Tensor]:
        """Return observation groups from the latest reset or step."""
        if not self._observations_initialized:
            self.reset()
        return self._observation_groups

    def get_observations(self) -> torch.Tensor:
        """Compute and return the current actor observations."""
        observations = self.unwrapped.observation_manager.compute()
        return self._process_observations(observations)

    def select_observation_groups(self, groups: Sequence[str]) -> torch.Tensor:
        """Concatenate configured MJLab observation groups."""
        if not groups:
            raise ValueError("At least one MJLab observation group is required.")
        observations = self.get_observation_groups()
        missing = [name for name in groups if name not in observations]
        if missing:
            names = ", ".join(missing)
            raise KeyError(f"MJLab observations are missing group(s): {names}.")
        return torch.cat([observations[name] for name in groups], dim=-1)

    def get_amp_observations(self) -> torch.Tensor:
        return self.select_observation_groups(("amp",))

    def rand_act(self) -> torch.Tensor:
        return 2.0 * torch.rand(
            self.num_envs, self.action_dim, device=self.device
        ) - 1.0

    def reset(self) -> torch.Tensor:
        obs, _ = self.env.reset()
        return self._process_observations(obs)

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        with torch.no_grad():
            action = action.to(self.device, dtype=torch.float32)
            if self.clip_actions is not None:
                action = action.clamp(-self.clip_actions, self.clip_actions)
            obs, reward, terminated, truncated, extras = self.env.step(action)
            done = terminated | truncated
            info = dict(extras)
            info.setdefault("terminated", terminated.view(-1))
            info.setdefault("truncated", truncated.view(-1))
            if not bool(getattr(self.cfg, "is_finite_horizon", False)):
                info.setdefault("time_outs", truncated.view(-1))
            return (
                self._process_observations(obs),
                reward.view(-1),
                done.view(-1),
                info,
            )

    def close(self) -> None:
        self.env.close()

    def seed(self, seed: int = -1) -> int:
        return int(self.unwrapped.seed(seed))
