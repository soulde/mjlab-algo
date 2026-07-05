"""MJLab environment wrapper for FastSAC."""

from collections.abc import Mapping

import torch

from mjlab.envs import ManagerBasedRlEnv


class FastSACVecEnvWrapper:
    """Flatten MJLab observations for vectorized off-policy training."""

    def __init__(self, env: ManagerBasedRlEnv):
        self.env = env
        self.num_envs = env.num_envs
        self.action_dim = env.unwrapped.action_manager.total_action_dim

    @property
    def unwrapped(self) -> ManagerBasedRlEnv:
        return self.env.unwrapped

    def _obs_to_tensor(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        tensors = []
        for value in obs.values():
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, dtype=torch.float32, device=self.env.device)
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
            action = action.to(self.env.device, dtype=torch.float32)
            obs, reward, terminated, truncated, extras = self.env.step(action)
            done = terminated | truncated
            return self._obs_to_tensor(obs), reward.view(-1), done.view(-1), extras

    def close(self) -> None:
        self.env.close()
