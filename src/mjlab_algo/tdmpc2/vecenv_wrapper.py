"""VecEnv wrapper adapting ManagerBasedRlEnv for TD-MPC2.

TD-MPC2 expects a single-environment interface with flat observation
tensors (no batch dimension). This wrapper converts the batched,
dict-based observations from ManagerBasedRlEnv.
"""

from collections import defaultdict

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv


class TDMPC2VecEnvWrapper:
    """Wraps a single-instance ManagerBasedRlEnv for TD-MPC2.

    Handles batch-dimension stripping, dict-to-flat observation
    conversion, and numpy-to-torch bridging.
    """

    def __init__(self, env: ManagerBasedRlEnv):
        if env.num_envs != 1:
            raise ValueError(
                f"TD-MPC2 requires exactly 1 environment, got {env.num_envs}."
            )
        self.env = env
        self._action_dim = env.unwrapped.action_manager.total_action_dim

    @property
    def unwrapped(self) -> ManagerBasedRlEnv:
        return self.env.unwrapped

    @property
    def observation_space(self):
        return self.env.single_observation_space

    @property
    def action_space(self):
        return self.env.single_action_space

    def _obs_to_tensor(self, obs_dict: dict) -> torch.Tensor:
        """Flatten a dict observation into a single 1D tensor.

        ManagerBasedRlEnv returns dict observations like
        ``{"actor": tensor([[1, num_actor_obs]]), ...}``.
        This method strips the batch dimension and concatenates
        all observation groups.
        """
        tensors = []
        for v in obs_dict.values():
            if isinstance(v, torch.Tensor):
                # Remove batch dim (assumes batch_size=1).
                t = v.squeeze(0) if v.ndim > 1 else v
                tensors.append(t.flatten())
            elif isinstance(v, np.ndarray):
                t = torch.from_numpy(v).float()
                t = t.squeeze(0) if t.ndim > 1 else t
                tensors.append(t.flatten())
            else:
                tensors.append(torch.tensor(v, dtype=torch.float32).flatten())
        return torch.cat(tensors)

    def rand_act(self) -> torch.Tensor:
        """Sample a random action from the action space."""
        return 2 * torch.rand(self._action_dim) - 1

    def reset(self) -> torch.Tensor:
        """Reset environment and return flat observation."""
        obs_dict, _ = self.env.reset()
        return self._obs_to_tensor(obs_dict)

    def step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, bool, dict]:
        """Step environment with given action.

        Args:
            action: Action tensor of shape (action_dim,).

        Returns:
            Tuple of (obs, reward, done, info).
        """
        with torch.no_grad():
            action_batch = action.unsqueeze(0).to(self.env.device)
            obs_dict, reward, terminated, truncated, extras = self.env.step(
                action_batch
            )
            info: dict = defaultdict(float)
            if "log" in extras:
                info.update(extras["log"])
            info["success"] = float(extras.get("success", 0.0))
            info["terminated"] = torch.tensor(
                float(terminated.item()), dtype=torch.float32
            )
            done = bool((terminated | truncated).item())
            obs = self._obs_to_tensor(obs_dict)

        return (
            obs,
            reward.squeeze(0).to(dtype=torch.float32),
            done,
            info,
        )

    def render(self, width: int = 384, height: int = 384) -> np.ndarray:
        """Render the environment as an RGB array."""
        return self.env.render()

    def close(self) -> None:
        """Close the environment."""
        self.env.close()
