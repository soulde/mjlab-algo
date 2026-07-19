"""RSL-RL style Python configuration classes for AMP."""

from dataclasses import dataclass, field

from mmrl.logging import LoggerCfg
from mmrl.ppo.config import PPOActorCriticCfg, PPOAlgorithmCfg


@dataclass
class AMPAlgorithmCfg(PPOAlgorithmCfg):
    class_name: str = "AMP"
    amp_replay_capacity: int = 1_000_000
    gradient_penalty_coef: float = 10.0
    discriminator_weight_decay: float = 1e-3
    discriminator_head_weight_decay: float = 1e-1


@dataclass
class AMPDiscriminatorCfg:
    class_name: str = "AMPDiscriminator"
    hidden_dims: tuple[int, ...] = (1024, 512)
    reward_scale: float = 0.2
    task_reward_lerp: float = 0.8


@dataclass
class AMPMemoryCfg:
    class_name: str = "OnPolicyRolloutMemory"
    num_steps_per_env: int = 24


@dataclass
class AMPRunnerCfg:
    seed: int = 1
    device: str | None = None
    max_iterations: int = 1_500
    save_interval: int = 50
    log_interval: int = 1
    obs_groups: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "actor": ("policy",),
            "critic": ("critic",),
            "amp": ("amp",),
        }
    )
    actor_critic: PPOActorCriticCfg = field(default_factory=PPOActorCriticCfg)
    algorithm: AMPAlgorithmCfg = field(default_factory=AMPAlgorithmCfg)
    discriminator: AMPDiscriminatorCfg = field(
        default_factory=AMPDiscriminatorCfg
    )
    memory: AMPMemoryCfg = field(default_factory=AMPMemoryCfg)
    logger: LoggerCfg = field(default_factory=LoggerCfg)
