"""Configuration for FastSAC."""

from dataclasses import dataclass, field


@dataclass
class FastSACConfig:
    """Single-task Soft Actor-Critic configuration for MJLab environments."""

    task: str = ""
    seed: int = 1
    total_steps: int = 1_000_000
    num_envs: int = 1
    batch_size: int = 256
    buffer_size: int = 1_000_000
    learning_starts: int = 5_000
    train_every: int = 1
    gradient_steps: int = 1
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    init_alpha: float = 0.2
    target_entropy: float | None = None
    auto_entropy: bool = True
    hidden_dims: tuple[int, ...] = (256, 256)
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    log_root: str = "logs/fastsac"
    exp_name: str = "default"
    save_agent: bool = True
    save_interval: int = 100_000
    log_interval: int = 1_000
    device: str | None = None

    obs_dim: int = 0
    action_dim: int = 0
    episode_length: int = 0
    metrics: dict[str, float] = field(default_factory=dict)


def make_fastsac_config(**overrides) -> FastSACConfig:
    """Create a FastSAC config with keyword overrides."""
    return FastSACConfig(**overrides)
