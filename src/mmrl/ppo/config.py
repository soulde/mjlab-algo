"""Python configuration classes for PPO."""

from dataclasses import dataclass, field


@dataclass
class PPOActorCriticCfg:
    class_name: str = "ActorCritic"
    actor_hidden_dims: tuple[int, ...] = (256, 256, 256)
    critic_hidden_dims: tuple[int, ...] = (256, 256, 256)
    activation: str = "elu"
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"


@dataclass
class PPOAlgorithmCfg:
    class_name: str = "PPO"
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    clip_param: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.0
    learning_rate: float = 1e-3
    max_grad_norm: float = 1.0
    use_clipped_value_loss: bool = True
    schedule: str = "adaptive"
    desired_kl: float = 0.01
    normalize_advantage_per_mini_batch: bool = False


@dataclass
class PPOMemoryCfg:
    class_name: str = "OnPolicyRolloutMemory"
    num_steps_per_env: int = 24


@dataclass
class PPORunnerCfg:
    seed: int = 1
    device: str | None = None
    max_iterations: int = 1_500
    save_interval: int = 50
    log_interval: int = 1
    actor_critic: PPOActorCriticCfg = field(default_factory=PPOActorCriticCfg)
    algorithm: PPOAlgorithmCfg = field(default_factory=PPOAlgorithmCfg)
    memory: PPOMemoryCfg = field(default_factory=PPOMemoryCfg)
