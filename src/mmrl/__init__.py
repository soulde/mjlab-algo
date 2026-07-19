"""Environment-agnostic reinforcement learning algorithms."""

from mmrl.config import (
    config_to_dict,
    get_config_value,
    require_config_value,
)
from mmrl.fastsac import (
    FastSAC,
    FastSACActorCfg,
    FastSACAlgorithmCfg,
    FastSACCriticCfg,
    FastSACRunner,
    FastSACRunnerCfg,
    OffPolicyMemoryCfg,
)
from mmrl.logging import LoggerCfg, MetricLogger
from mmrl.ppo import (
    ActorCritic,
    PPO,
    PPOActorCriticCfg,
    PPOAlgorithmCfg,
    PPOMemoryCfg,
    PPORunner,
    PPORunnerCfg,
)
from mmrl.tdmpc2.config import (
    EpisodeMemoryCfg,
    TDMPC2AlgorithmCfg,
    TDMPC2EnvSpec,
    TDMPC2ModelCfg,
    TDMPC2RunnerCfg,
)

__all__ = [
    "FastSAC",
    "FastSACActorCfg",
    "FastSACAlgorithmCfg",
    "FastSACCriticCfg",
    "FastSACRunner",
    "FastSACRunnerCfg",
    "OffPolicyMemoryCfg",
    "LoggerCfg",
    "MetricLogger",
    "ActorCritic",
    "PPO",
    "PPOActorCriticCfg",
    "PPOAlgorithmCfg",
    "PPOMemoryCfg",
    "PPORunner",
    "PPORunnerCfg",
    "TDMPC2",
    "EpisodeMemoryCfg",
    "TDMPC2AlgorithmCfg",
    "TDMPC2EnvSpec",
    "TDMPC2ModelCfg",
    "TDMPC2Runner",
    "TDMPC2RunnerCfg",
    "config_to_dict",
    "get_config_value",
    "require_config_value",
]


def __getattr__(name: str):
    if name == "TDMPC2":
        from mmrl.tdmpc2 import TDMPC2

        return TDMPC2
    if name == "TDMPC2Runner":
        from mmrl.tdmpc2 import TDMPC2Runner

        return TDMPC2Runner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
