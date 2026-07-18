"""Python configuration classes for FastSAC."""

from dataclasses import dataclass, field


@dataclass
class FastSACActorCfg:
    class_name: str = "SquashedGaussianActor"
    hidden_dims: tuple[int, ...] = (256, 256)
    log_std_min: float = -20.0
    log_std_max: float = 2.0


@dataclass
class FastSACCriticCfg:
    class_name: str = "TwinQNetwork"
    hidden_dims: tuple[int, ...] = (256, 256)


@dataclass
class FastSACAlgorithmCfg:
    class_name: str = "FastSAC"
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    init_alpha: float = 0.2
    target_entropy: float | None = None
    auto_entropy: bool = True


@dataclass
class OffPolicyMemoryCfg:
    class_name: str = "OffPolicyReplayMemory"
    capacity: int = 1_000_000
    batch_size: int = 256


@dataclass
class FastSACRunnerCfg:
    """RSL-RL style runner configuration owned by environment packages."""

    seed: int = 1
    total_steps: int = 1_000_000
    learning_starts: int = 5_000
    train_every: int = 1
    gradient_steps: int = 1
    save_interval: int = 100_000
    log_interval: int = 1_000
    device: str | None = None
    algorithm: FastSACAlgorithmCfg = field(default_factory=FastSACAlgorithmCfg)
    actor: FastSACActorCfg = field(default_factory=FastSACActorCfg)
    critic: FastSACCriticCfg = field(default_factory=FastSACCriticCfg)
    memory: OffPolicyMemoryCfg = field(default_factory=OffPolicyMemoryCfg)
