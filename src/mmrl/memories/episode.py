"""Episode memories for sequence-based model learning."""

import torch

from mmrl.memories.base import Memory
from mmrl.memories.storage import EpisodeListStorage


class EpisodeMemory(Memory):
    """Store full episodes and sample contiguous subsequences."""

    def __init__(
        self,
        capacity: int,
        batch_size: int,
        horizon: int,
        device: str | torch.device,
    ):
        self._capacity = int(capacity)
        self._batch_size = int(batch_size)
        self._horizon = int(horizon)
        self.device = torch.device(device)
        self.storage = EpisodeListStorage(self._capacity)

    @property
    def size(self) -> int:
        return self.storage.size

    @property
    def capacity(self):
        return self._capacity

    @property
    def num_eps(self):
        return self.storage.num_eps

    def add(self, td) -> int:
        """Add an episode to the memory."""
        return self.storage.add(td)

    def _sample_episode(self):
        """Sample a random episode that is long enough."""
        return self.storage.sample_episode(self._horizon + 1)

    def sample(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        """Sample a batch of subsequences."""
        horizon = self._horizon
        obs_list, action_list, reward_list = [], [], []
        terminated_list, task_list = [], []

        for _ in range(self._batch_size):
            ep = self._sample_episode()
            ep_len = ep.shape[0]
            max_start = ep_len - horizon - 1
            start = torch.randint(0, max_start + 1, (1,)).item()
            seq = ep[start : start + horizon + 1]

            obs_list.append(seq["obs"])
            action_list.append(seq["action"])
            reward_list.append(seq["reward"])
            terminated_list.append(seq["terminated"])
            if "task" in seq.keys():
                task_list.append(seq["task"])

        obs = torch.stack(obs_list, dim=1)
        action = torch.stack(action_list, dim=1)
        reward = torch.stack(reward_list, dim=1)
        terminated = torch.stack(terminated_list, dim=1)

        task = None
        if task_list:
            task = torch.stack(task_list, dim=1)

        obs = obs.to(self.device, non_blocking=True).contiguous()
        action = action[1:].to(self.device, non_blocking=True).contiguous()
        reward = (
            reward[1:]
            .unsqueeze(-1)
            .to(self.device, non_blocking=True)
            .contiguous()
        )
        terminated = (
            terminated[1:]
            .unsqueeze(-1)
            .to(self.device, non_blocking=True)
            .contiguous()
        )
        if task is not None:
            task = task[0].to(self.device, non_blocking=True).contiguous()

        return obs, action, reward, terminated, task
