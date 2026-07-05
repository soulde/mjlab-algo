"""Configuration for TD-MPC2.

Dataclass-based config replacing Hydra/OmegaConf, consistent with
mjlab's ``rl/config.py`` pattern.
"""

from dataclasses import dataclass, field
from typing import Literal

# Model size presets (parameters in millions)
MODEL_SIZE: dict = {
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

# Task sets for multi-task training (kept for compatibility)
TASK_SET: dict = {
    "mt30": [],
    "mt80": [],
}


def _apply_model_size(cfg_dict: dict, model_size: int | None) -> dict:
    """Apply model size preset to config dict, overriding dimension fields."""
    if model_size is not None:
        if model_size not in MODEL_SIZE:
            raise ValueError(
                f"Invalid model_size {model_size}. Must be one of {list(MODEL_SIZE.keys())}."
            )
        for k, v in MODEL_SIZE[model_size].items():
            cfg_dict[k] = v
    return cfg_dict


@dataclass
class TDMPC2Config:
    """Configuration for TD-MPC2 training and architecture.

    All algorithm hyperparameters, planning settings, architecture
    dimensions, and logging options live here. Equivalent to the
    original ``config.yaml``.
    """

    # ── Environment ────────────────────────────────────────────────
    task: str = "dog-run"
    """Task name (e.g., ``Mjlab-Cartpole-Balance``)."""
    obs: Literal["state", "rgb"] = "state"
    """Observation type."""
    episodic: bool = False
    """Whether the task has early termination signals."""

    # ── Evaluation ─────────────────────────────────────────────────
    eval_episodes: int = 10
    """Number of episodes per evaluation."""
    eval_freq: int = 50000
    """Evaluation frequency in environment steps."""

    # ── Training ───────────────────────────────────────────────────
    steps: int = 10_000_000
    """Total environment steps."""
    batch_size: int = 256
    """Batch size (number of subsequences per update)."""
    reward_coef: float = 0.1
    """Weight for reward prediction loss."""
    value_coef: float = 0.1
    """Weight for value (Q) prediction loss."""
    termination_coef: float = 1.0
    """Weight for termination prediction loss."""
    consistency_coef: float = 20.0
    """Weight for latent consistency loss."""
    rho: float = 0.5
    """Discount factor for loss weighting over the horizon."""
    lr: float = 3e-4
    """Base learning rate."""
    enc_lr_scale: float = 0.3
    """Encoder learning rate multiplier (relative to ``lr``)."""
    grad_clip_norm: float = 20.0
    """Maximum gradient norm for clipping."""
    tau: float = 0.01
    """Polyak averaging coefficient for target Q-networks."""
    discount_denom: float = 5.0
    """Denominator for discount factor heuristic."""
    discount_min: float = 0.95
    """Minimum discount factor."""
    discount_max: float = 0.995
    """Maximum discount factor."""
    buffer_size: int = 1_000_000
    """Replay buffer capacity in steps."""
    exp_name: str = "default"
    """Experiment name (used in log directory)."""

    # ── Planning (MPPI) ────────────────────────────────────────────
    mpc: bool = True
    """Whether to use MPPI planning. If False, uses policy prior."""
    iterations: int = 6
    """Number of MPPI optimization iterations."""
    num_samples: int = 512
    """Total number of action sequence samples."""
    num_elites: int = 64
    """Number of elite samples kept per iteration."""
    num_pi_trajs: int = 24
    """Number of policy-prior trajectories to inject."""
    horizon: int = 3
    """Planning horizon (steps)."""
    min_std: float = 0.05
    """Minimum action standard deviation in MPPI."""
    max_std: float = 2.0
    """Maximum action standard deviation in MPPI."""
    temperature: float = 0.5
    """MPPI temperature for score weighting."""

    # ── Actor ──────────────────────────────────────────────────────
    log_std_min: float = -10.0
    """Minimum log-standard-deviation for the policy."""
    log_std_max: float = 2.0
    """Maximum log-standard-deviation for the policy."""
    entropy_coef: float = 1e-4
    """Entropy regularization coefficient."""

    # ── Critic (distributional) ────────────────────────────────────
    num_bins: int = 101
    """Number of bins for distributional value prediction."""
    vmin: float = -10.0
    """Minimum value for two-hot discretization."""
    vmax: float = 10.0
    """Maximum value for two-hot discretization."""

    # ── Architecture ───────────────────────────────────────────────
    model_size: int | None = None
    """Preset model size: 1, 5, 19, 48, or 317 (M params)."""
    num_enc_layers: int = 2
    """Number of encoder layers."""
    enc_dim: int = 256
    """Encoder hidden dimension."""
    num_channels: int = 32
    """Number of channels in conv encoder (RGB only)."""
    mlp_dim: int = 512
    """MLP hidden dimension."""
    latent_dim: int = 512
    """Latent state dimension."""
    task_dim: int = 96
    """Task embedding dimension (multi-task only)."""
    num_q: int = 5
    """Number of Q-functions in the ensemble."""
    dropout: float = 0.01
    """Dropout rate in Q-function MLPs."""
    simnorm_dim: int = 8
    """SimNorm grouping dimension."""

    # ── Logging ────────────────────────────────────────────────────
    wandb_project: str = "mjlab"
    """W&B project name."""
    wandb_entity: str | None = None
    """W&B entity (team/user)."""
    wandb_silent: bool = False
    """Suppress W&B output."""
    enable_wandb: bool = True
    """Whether to log to W&B."""
    save_csv: bool = True
    """Whether to save evaluation metrics to CSV."""
    save_video: bool = True
    """Whether to record evaluation videos."""
    save_agent: bool = True
    """Whether to save model checkpoints."""
    log_root: str = "logs/tdmpc2"
    """Root directory for logs/checkpoints."""

    # ── Misc ───────────────────────────────────────────────────────
    compile: bool = True
    """Whether to use torch.compile for update and planning."""
    seed: int = 1
    """Random seed."""

    # ── Convenience (populated at runtime) ─────────────────────────
    work_dir: str = ""
    """Working directory for this run (set automatically)."""
    task_title: str = ""
    """Human-readable task title."""
    multitask: bool = False
    """Whether this is a multi-task run (set automatically)."""
    tasks: list[str] = field(default_factory=list)
    """List of task names (multi-task only)."""
    obs_shape: dict = field(default_factory=dict)
    """Dictionary of observation shapes."""
    action_dim: int = 0
    """Action dimension."""
    episode_length: int = 0
    """Max episode length."""
    obs_shapes: list = field(default_factory=list)
    """List of observation shapes (multi-task only)."""
    action_dims: list[int] = field(default_factory=list)
    """List of action dimensions (multi-task only)."""
    episode_lengths: list[int] = field(default_factory=list)
    """List of episode lengths (multi-task only)."""
    seed_steps: int = 0
    """Number of random seed steps (set automatically)."""
    bin_size: float = 0.0
    """Bin size for discrete regression (set automatically)."""


def make_tdmpc2_config(
    model_size: int | None = None,
    **overrides,
) -> TDMPC2Config:
    """Create a TDMPC2Config with model size preset and overrides.

    Args:
        model_size: Optional preset size (1, 5, 19, 48, 317).
        **overrides: Any config fields to override.

    Returns:
        Configured TDMPC2Config instance.
    """
    cfg_dict = {}
    _apply_model_size(cfg_dict, model_size)
    cfg_dict.update(overrides)
    return TDMPC2Config(**cfg_dict)
