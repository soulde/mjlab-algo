"""Proximal Policy Optimization."""

from mmrl.ppo.actor_critic import ActorCritic as ActorCritic
from mmrl.ppo.config import (
    PPOActorCriticCfg as PPOActorCriticCfg,
    PPOAlgorithmCfg as PPOAlgorithmCfg,
    PPOMemoryCfg as PPOMemoryCfg,
    PPORunnerCfg as PPORunnerCfg,
)

__all__ = [
    "ActorCritic",
    "PPO",
    "PPOActorCriticCfg",
    "PPOAlgorithmCfg",
    "PPOMemoryCfg",
    "PPORunner",
    "PPORunnerCfg",
]


def __getattr__(name: str):
    if name == "PPO":
        from mmrl.ppo.ppo import PPO

        return PPO
    if name == "PPORunner":
        from mmrl.ppo.runner import PPORunner

        return PPORunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
