"""Adversarial Motion Priors built on mmrl PPO."""

import torch
import torch.nn.functional as F

from mmrl.memories import (
    AMPExpertSource,
    AMPTransitionMemory,
    OnPolicyRolloutMemory,
)
from mmrl.models import AMPDiscriminator, RunningMeanStd
from mmrl.ppo import ActorCritic
from mmrl.ppo.ppo import PPO


class AMP(PPO):
    """PPO augmented with an adversarial motion discriminator."""

    def __init__(
        self,
        cfg,
        policy: ActorCritic,
        discriminator: AMPDiscriminator,
        expert_source: AMPExpertSource,
        amp_observation_dim: int,
        device: str | torch.device,
    ) -> None:
        super().__init__(cfg, policy, device)
        self.discriminator = discriminator.to(self.device)
        self.expert_source = expert_source
        self.normalizer = RunningMeanStd(amp_observation_dim).to(self.device)
        self.amp_replay = AMPTransitionMemory(
            cfg.amp_replay_capacity,
            amp_observation_dim,
            self.device,
        )
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.parameters()},
                {
                    "params": self.discriminator.trunk.parameters(),
                    "weight_decay": cfg.discriminator_weight_decay,
                },
                {
                    "params": self.discriminator.head.parameters(),
                    "weight_decay": cfg.discriminator_head_weight_decay,
                },
            ],
            lr=self.learning_rate,
        )

    def add_amp_transition(
        self, state: torch.Tensor, next_state: torch.Tensor
    ) -> None:
        self.amp_replay.add(state, next_state)

    @torch.no_grad()
    def combine_rewards(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        task_reward: torch.Tensor,
        reward_scale: float,
        task_reward_lerp: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.normalizer.normalize(state.to(self.device))
        next_state = self.normalizer.normalize(next_state.to(self.device))
        style_reward = self.discriminator.style_reward(
            state, next_state, reward_scale
        ).squeeze(-1)
        task_reward = task_reward.to(self.device).reshape_as(style_reward)
        reward = (1.0 - task_reward_lerp) * style_reward
        reward += task_reward_lerp * task_reward
        return reward, style_reward

    def update(self, memory: OnPolicyRolloutMemory) -> dict[str, float]:
        if self.amp_replay.size < memory.size // self.cfg.num_mini_batches:
            raise RuntimeError("AMP replay does not contain enough transitions.")
        totals = {
            "value_loss": 0.0,
            "surrogate_loss": 0.0,
            "entropy": 0.0,
            "kl": 0.0,
            "amp_loss": 0.0,
            "amp_gradient_penalty": 0.0,
            "amp_policy_prediction": 0.0,
            "amp_expert_prediction": 0.0,
        }
        num_updates = 0
        batch_size = memory.size // self.cfg.num_mini_batches

        for _ in range(self.cfg.num_learning_epochs):
            for batch in memory.sample(self.cfg.num_mini_batches):
                policy_amp = self.amp_replay.sample(batch_size)
                expert_amp = self.expert_source.sample(batch_size, self.device)
                policy_state_raw = policy_amp.state
                expert_state_raw = expert_amp.state
                policy_state = self.normalizer.normalize(policy_amp.state)
                policy_next_state = self.normalizer.normalize(
                    policy_amp.next_state
                )
                expert_state = self.normalizer.normalize(expert_amp.state)
                expert_next_state = self.normalizer.normalize(
                    expert_amp.next_state
                )

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
                surrogate_loss = torch.maximum(
                    -advantage * ratio,
                    -advantage
                    * ratio.clamp(
                        1.0 - self.cfg.clip_param,
                        1.0 + self.cfg.clip_param,
                    ),
                ).mean()
                if self.cfg.use_clipped_value_loss:
                    value_clipped = batch.value + (value - batch.value).clamp(
                        -self.cfg.clip_param, self.cfg.clip_param
                    )
                    value_loss = torch.maximum(
                        (value - batch.ret).square(),
                        (value_clipped - batch.ret).square(),
                    ).mean()
                else:
                    value_loss = F.mse_loss(value, batch.ret)

                policy_prediction = self.discriminator(
                    policy_state, policy_next_state
                )
                expert_prediction = self.discriminator(
                    expert_state, expert_next_state
                )
                amp_loss = 0.5 * (
                    F.mse_loss(policy_prediction, -torch.ones_like(policy_prediction))
                    + F.mse_loss(
                        expert_prediction, torch.ones_like(expert_prediction)
                    )
                )
                gradient_penalty = self.discriminator.gradient_penalty(
                    expert_state, expert_next_state
                )
                loss = (
                    surrogate_loss
                    + self.cfg.value_loss_coef * value_loss
                    - self.cfg.entropy_coef * entropy.mean()
                    + amp_loss
                    + self.cfg.gradient_penalty_coef * gradient_penalty
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
                    self.normalizer.update(policy_state_raw)
                    self.normalizer.update(expert_state_raw)

                values = {
                    "value_loss": value_loss,
                    "surrogate_loss": surrogate_loss,
                    "entropy": entropy.mean(),
                    "kl": approximate_kl,
                    "amp_loss": amp_loss,
                    "amp_gradient_penalty": gradient_penalty,
                    "amp_policy_prediction": policy_prediction.mean(),
                    "amp_expert_prediction": expert_prediction.mean(),
                }
                for name, value_metric in values.items():
                    totals[name] += float(value_metric.detach())
                num_updates += 1

        denominator = max(num_updates, 1)
        metrics = {name: value / denominator for name, value in totals.items()}
        metrics["learning_rate"] = self.learning_rate
        return metrics
