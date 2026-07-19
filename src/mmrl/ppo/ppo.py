"""Proximal Policy Optimization.

The update structure follows RSL-RL's PPO implementation under BSD-3-Clause,
adapted to mmrl's rollout memory and explicit model ownership.
"""

import torch
import torch.nn.functional as F

from mmrl.memories import OnPolicyRolloutMemory
from mmrl.ppo.actor_critic import ActorCritic


class PPO:
    """Clipped PPO with optional adaptive-KL learning-rate scheduling."""

    def __init__(
        self,
        cfg,
        policy: ActorCritic,
        device: str | torch.device,
    ):
        self.cfg = cfg
        self.policy = policy.to(device)
        self.device = torch.device(device)
        self.learning_rate = float(cfg.learning_rate)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=self.learning_rate
        )

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.policy.act(obs.to(self.device))

    def update(self, memory: OnPolicyRolloutMemory) -> dict[str, float]:
        value_loss_sum = 0.0
        surrogate_loss_sum = 0.0
        entropy_sum = 0.0
        kl_sum = 0.0
        num_updates = 0

        for _ in range(self.cfg.num_learning_epochs):
            for batch in memory.sample(self.cfg.num_mini_batches):
                advantage = batch.advantage
                if self.cfg.normalize_advantage_per_mini_batch:
                    advantage = (advantage - advantage.mean()) / advantage.std(
                        unbiased=False
                    ).clamp_min(1e-8)

                log_prob, entropy, value = self.policy.evaluate_actions(
                    batch.obs, batch.action, batch.critic_obs
                )
                log_ratio = log_prob - batch.log_prob
                ratio = log_ratio.exp()
                surrogate = -advantage * ratio
                surrogate_clipped = -advantage * ratio.clamp(
                    1.0 - self.cfg.clip_param,
                    1.0 + self.cfg.clip_param,
                )
                surrogate_loss = torch.maximum(
                    surrogate, surrogate_clipped
                ).mean()

                if self.cfg.use_clipped_value_loss:
                    value_clipped = batch.value + (value - batch.value).clamp(
                        -self.cfg.clip_param,
                        self.cfg.clip_param,
                    )
                    value_loss = torch.maximum(
                        (value - batch.ret).square(),
                        (value_clipped - batch.ret).square(),
                    ).mean()
                else:
                    value_loss = F.mse_loss(value, batch.ret)

                loss = (
                    surrogate_loss
                    + self.cfg.value_loss_coef * value_loss
                    - self.cfg.entropy_coef * entropy.mean()
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.cfg.max_grad_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                    self._adapt_learning_rate(float(approximate_kl))
                value_loss_sum += float(value_loss.detach())
                surrogate_loss_sum += float(surrogate_loss.detach())
                entropy_sum += float(entropy.mean().detach())
                kl_sum += float(approximate_kl)
                num_updates += 1

        denominator = max(num_updates, 1)
        return {
            "value_loss": value_loss_sum / denominator,
            "surrogate_loss": surrogate_loss_sum / denominator,
            "entropy": entropy_sum / denominator,
            "kl": kl_sum / denominator,
            "learning_rate": self.learning_rate,
        }

    def _adapt_learning_rate(self, kl: float) -> None:
        if self.cfg.schedule != "adaptive" or self.cfg.desired_kl <= 0.0:
            return
        if kl > 2.0 * self.cfg.desired_kl:
            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
        elif 0.0 < kl < 0.5 * self.cfg.desired_kl:
            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate
