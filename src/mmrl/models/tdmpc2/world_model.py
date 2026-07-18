"""TD-MPC2 implicit world model architecture.

The WorldModel contains an encoder, dynamics predictor, reward predictor,
termination predictor (optional), policy prior, and ensemble of Q-functions.
"""

from copy import deepcopy

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictParams

from mmrl.models.tdmpc2 import init, layers, math
from mmrl.models.base import Model


class WorldModel(Model):
    """TD-MPC2 implicit world model.

    Can be used for both single-task and multi-task experiments,
    and supports both state and pixel observations.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.multitask:
            self._task_emb = nn.Embedding(len(cfg.tasks), cfg.task_dim, max_norm=1)
            self.register_buffer(
                "_action_masks", torch.zeros(len(cfg.tasks), cfg.action_dim)
            )
            for i in range(len(cfg.tasks)):
                self._action_masks[i, : cfg.action_dims[i]] = 1.0
        self._encoder = layers.enc(cfg)
        self._dynamics = layers.mlp(
            cfg.latent_dim + cfg.action_dim + cfg.task_dim,
            2 * [cfg.mlp_dim],
            cfg.latent_dim,
            act=layers.SimNorm(cfg),
        )
        self._reward = layers.mlp(
            cfg.latent_dim + cfg.action_dim + cfg.task_dim,
            2 * [cfg.mlp_dim],
            max(cfg.num_bins, 1),
        )
        self._termination = (
            layers.mlp(cfg.latent_dim + cfg.task_dim, 2 * [cfg.mlp_dim], 1)
            if cfg.episodic
            else None
        )
        self._pi = layers.mlp(
            cfg.latent_dim + cfg.task_dim,
            2 * [cfg.mlp_dim],
            2 * cfg.action_dim,
        )
        self._Qs = layers.Ensemble(
            [
                layers.mlp(
                    cfg.latent_dim + cfg.action_dim + cfg.task_dim,
                    2 * [cfg.mlp_dim],
                    max(cfg.num_bins, 1),
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.num_q)
            ]
        )
        self.apply(init.weight_init)
        init.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"]])

        self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
        self.register_buffer(
            "log_std_dif",
            torch.tensor(cfg.log_std_max) - self.log_std_min,
        )
        self.init()

    def init(self):
        """Initialize target and detached Q-network copies."""
        self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True)
        self._target_Qs_params = TensorDictParams(
            self._Qs.params.data.clone(), no_convert=True
        )

        with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
            self._detach_Qs = deepcopy(self._Qs)
            self._target_Qs = deepcopy(self._Qs)

        delattr(self._detach_Qs, "params")
        self._detach_Qs.__dict__["params"] = self._detach_Qs_params
        delattr(self._target_Qs, "params")
        self._target_Qs.__dict__["params"] = self._target_Qs_params

    def __repr__(self):
        repr_str = "TD-MPC2 World Model\n"
        modules = [
            "Encoder",
            "Dynamics",
            "Reward",
            "Termination",
            "Policy prior",
            "Q-functions",
        ]
        for i, m in enumerate(
            [
                self._encoder,
                self._dynamics,
                self._reward,
                self._termination,
                self._pi,
                self._Qs,
            ]
        ):
            if m == self._termination and not self.cfg.episodic:
                continue
            repr_str += f"{modules[i]}: {m}\n"
        repr_str += "Learnable parameters: {:,}".format(self.total_params)
        return repr_str

    @property
    def total_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.init()
        return self

    def train(self, mode=True):
        """Override ``train`` to keep target Q-networks in eval mode."""
        super().train(mode)
        self._target_Qs.train(False)
        return self

    def soft_update_target_Q(self):
        """Soft-update target Q-networks using Polyak averaging."""
        self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)

    def task_emb(self, x: torch.Tensor, task) -> torch.Tensor:
        """Concatenate task embedding to input ``x``."""
        if isinstance(task, int):
            task = torch.tensor([task], device=x.device)
        emb = self._task_emb(task.long())
        if x.ndim == 3:
            emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        elif emb.shape[0] == 1:
            emb = emb.repeat(x.shape[0], 1)
        return torch.cat([x, emb], dim=-1)

    def encode(self, obs: torch.Tensor, task=None) -> torch.Tensor:
        """Encode observation into latent representation."""
        if self.cfg.multitask:
            obs = self.task_emb(obs, task)
        if self.cfg.obs == "rgb" and obs.ndim == 5:
            return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
        return self._encoder[self.cfg.obs](obs)

    def next(self, z: torch.Tensor, a: torch.Tensor, task=None) -> torch.Tensor:
        """Predict next latent state from current latent state and action."""
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        z = torch.cat([z, a], dim=-1)
        return self._dynamics(z)

    def reward(self, z: torch.Tensor, a: torch.Tensor, task=None) -> torch.Tensor:
        """Predict instantaneous (single-step) reward."""
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        z = torch.cat([z, a], dim=-1)
        return self._reward(z)

    def termination(
        self, z: torch.Tensor, task=None, unnormalized: bool = False
    ) -> torch.Tensor:
        """Predict termination signal."""
        assert task is None
        if self.cfg.multitask:
            z = self.task_emb(z, task)
        if unnormalized:
            return self._termination(z)
        return torch.sigmoid(self._termination(z))

    def pi(self, z: torch.Tensor, task=None) -> tuple[torch.Tensor, TensorDict]:
        """Sample action from the policy prior (Gaussian distribution)."""
        if self.cfg.multitask:
            z = self.task_emb(z, task)

        mean, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mean)

        if self.cfg.multitask:
            mean = mean * self._action_masks[task]
            log_std = log_std * self._action_masks[task]
            eps = eps * self._action_masks[task]
            action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
        else:
            action_dims = None

        log_prob = math.gaussian_logprob(eps, log_std)
        size = eps.shape[-1] if action_dims is None else action_dims
        scaled_log_prob = log_prob * size

        action = mean + eps * log_std.exp()
        mean, action, log_prob = math.squash(mean, action, log_prob)

        entropy_scale = scaled_log_prob / (log_prob + 1e-8)
        info = TensorDict(
            {
                "mean": mean,
                "log_std": log_std,
                "action_prob": 1.0,
                "entropy": -log_prob,
                "scaled_entropy": -log_prob * entropy_scale,
            }
        )
        return action, info

    def Q(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
        task=None,
        return_type: str = "min",
        target: bool = False,
        detach: bool = False,
    ) -> torch.Tensor:
        """Predict state-action value.

        Args:
            z: Latent state.
            a: Action.
            task: Task index (multi-task only).
            return_type: One of ``"min"``, ``"avg"``, ``"all"``.
            target: Whether to use target Q-networks.
            detach: Whether to use detached Q-networks.
        """
        assert return_type in {"min", "avg", "all"}

        if self.cfg.multitask:
            z = self.task_emb(z, task)

        z = torch.cat([z, a], dim=-1)
        if target:
            qnet = self._target_Qs
        elif detach:
            qnet = self._detach_Qs
        else:
            qnet = self._Qs
        out = qnet(z)

        if return_type == "all":
            return out

        qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
        Q_vals = math.two_hot_inv(out[qidx], self.cfg)
        if return_type == "min":
            return Q_vals.min(0).values
        return Q_vals.sum(0) / 2
