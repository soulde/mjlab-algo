"""RSL-RL style Python configuration classes for TD-MPC2."""

from dataclasses import dataclass, field


MODEL_SIZE: dict[int, dict[str, int]] = {
    1: {
        "enc_dim": 256,
        "mlp_dim": 384,
        "latent_dim": 128,
        "num_enc_layers": 2,
        "num_q": 2,
    },
    5: {
        "enc_dim": 256,
        "mlp_dim": 512,
        "latent_dim": 512,
        "num_enc_layers": 2,
    },
    19: {
        "enc_dim": 1024,
        "mlp_dim": 1024,
        "latent_dim": 768,
        "num_enc_layers": 3,
    },
    48: {
        "enc_dim": 1792,
        "mlp_dim": 1792,
        "latent_dim": 768,
        "num_enc_layers": 4,
    },
    317: {
        "enc_dim": 4096,
        "mlp_dim": 4096,
        "latent_dim": 1376,
        "num_enc_layers": 5,
        "num_q": 8,
    },
}


@dataclass
class TDMPC2AlgorithmCfg:
    class_name: str = "TDMPC2"
    reward_coef: float = 0.1
    value_coef: float = 0.1
    termination_coef: float = 1.0
    consistency_coef: float = 20.0
    rho: float = 0.5
    lr: float = 3e-4
    enc_lr_scale: float = 0.3
    grad_clip_norm: float = 20.0
    tau: float = 0.01
    discount_denom: float = 5.0
    discount_min: float = 0.95
    discount_max: float = 0.995
    mpc: bool = True
    iterations: int = 6
    num_samples: int = 512
    num_elites: int = 64
    num_pi_trajs: int = 24
    horizon: int = 3
    min_std: float = 0.05
    max_std: float = 2.0
    temperature: float = 0.5
    entropy_coef: float = 1e-4
    num_bins: int = 101
    vmin: float = -10.0
    vmax: float = 10.0
    episodic: bool = False


@dataclass
class TDMPC2ModelCfg:
    class_name: str = "WorldModel"
    obs: str = "state"
    model_size: int | None = None
    num_enc_layers: int = 2
    enc_dim: int = 256
    num_channels: int = 32
    mlp_dim: int = 512
    latent_dim: int = 512
    task_dim: int = 96
    num_q: int = 5
    dropout: float = 0.01
    simnorm_dim: int = 8
    log_std_min: float = -10.0
    log_std_max: float = 2.0
    multitask: bool = False
    tasks: list[str] = field(default_factory=list)
    obs_shapes: list = field(default_factory=list)
    action_dims: list[int] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.model_size is None:
            return
        if self.model_size not in MODEL_SIZE:
            raise ValueError(
                f"Invalid model_size {self.model_size}; "
                f"expected one of {tuple(MODEL_SIZE)}."
            )
        for name, value in MODEL_SIZE[self.model_size].items():
            setattr(self, name, value)


@dataclass
class EpisodeMemoryCfg:
    class_name: str = "EpisodeMemory"
    capacity: int = 1_000_000
    batch_size: int = 256


@dataclass(frozen=True)
class TDMPC2EnvSpec:
    obs_shape: dict[str, tuple[int, ...]]
    action_dim: int
    episode_length: int


@dataclass
class TDMPC2RunnerCfg:
    """Top-level config passed by an environment package to TDMPC2Runner."""

    seed: int = 1
    device: str | None = None
    steps: int = 10_000_000
    seed_steps: int = 0
    episode_length: int = 0
    eval_episodes: int = 10
    eval_freq: int = 50_000
    log_freq: int = 1_000
    save_agent: bool = True
    enable_wandb: bool = False
    wandb_project: str = "mmrl"
    wandb_entity: str | None = None
    wandb_silent: bool = False
    exp_name: str = "default"
    algorithm: TDMPC2AlgorithmCfg = field(default_factory=TDMPC2AlgorithmCfg)
    model: TDMPC2ModelCfg = field(default_factory=TDMPC2ModelCfg)
    memory: EpisodeMemoryCfg = field(default_factory=EpisodeMemoryCfg)
