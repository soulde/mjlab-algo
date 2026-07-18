"""Episode memories for sequence-based model learning."""

import torch

from mmrl.memories.base import Memory


class EpisodeMemory(Memory):
    """Store full episodes and sample contiguous subsequences."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._capacity = min(cfg.buffer_size, cfg.steps)
        self._batch_size = cfg.batch_size * (cfg.horizon + 1)
        self._num_eps = 0
        self._episodes: list = []
        self._num_steps = 0

    @property
    def size(self) -> int:
        return self._num_steps

    @property
    def capacity(self):
        return self._capacity

    @property
    def num_eps(self):
        return self._num_eps

    def add(self, td) -> int:
        """Add an episode to the memory."""
        while self._num_steps + td.shape[0] > self._capacity and self._num_eps > 0:
            removed = self._episodes.pop(0)
            self._num_steps -= removed.shape[0]
            self._num_eps -= 1

        self._episodes.append(td)
        self._num_steps += td.shape[0]
        self._num_eps += 1
        return self._num_eps

    def _sample_episode(self):
        """Sample a random episode that is long enough."""
        min_len = self.cfg.horizon + 1
        lengths = torch.tensor(
            [max(0, ep.shape[0] - min_len + 1) for ep in self._episodes],
            dtype=torch.float32,
        )
        probs = lengths / lengths.sum()
        idx = torch.multinomial(probs, 1).item()
        return self._episodes[idx]

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
        horizon = self.cfg.horizon
        obs_list, action_list, reward_list = [], [], []
        terminated_list, task_list = [], []

        for _ in range(self.cfg.batch_size):
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

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        obs = obs.to(device, non_blocking=True).contiguous()
        action = action[1:].to(device, non_blocking=True).contiguous()
        reward = reward[1:].unsqueeze(-1).to(device, non_blocking=True).contiguous()
        terminated = (
            terminated[1:].unsqueeze(-1).to(device, non_blocking=True).contiguous()
        )
        if task is not None:
            task = task[0].to(device, non_blocking=True).contiguous()

        return obs, action, reward, terminated, task
