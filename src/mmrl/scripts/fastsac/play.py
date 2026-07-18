"""Evaluation script for FastSAC."""

import sys
from dataclasses import dataclass
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mmrl.fastsac import FastSAC, make_fastsac_config
from mmrl.fastsac.vecenv_wrapper import FastSACVecEnvWrapper
from mmrl.scripts._cli import maybe_print_top_level_help


@dataclass(frozen=True)
class PlayArgs:
    task_id: str = ""
    checkpoint: str | None = None
    agent: Literal["zero", "random", "trained"] = "trained"
    device: str | None = None
    viewer: Literal["native", "viser"] = "viser"


def main():
    maybe_print_top_level_help("fastsac-play")

    import mjlab.tasks  # noqa: F401

    all_tasks = list_tasks()
    if not all_tasks:
        print("No tasks registered.")
        sys.exit(1)

    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
    )
    args = tyro.cli(
        PlayArgs,
        args=remaining_args,
        default=PlayArgs(task_id=chosen_task),
        prog=sys.argv[0] + f" {chosen_task}",
    )
    if args.agent == "trained" and args.checkpoint is None:
        raise ValueError("--checkpoint is required when agent=trained")

    configure_torch_backends()
    env_cfg = load_env_cfg(args.task_id)
    env_cfg.scene.num_envs = 1
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped_env = FastSACVecEnvWrapper(env)
    obs = wrapped_env.reset()

    if args.agent == "trained":
        cfg = make_fastsac_config(
            task=args.task_id,
            device=device,
            obs_dim=obs.shape[-1],
            action_dim=wrapped_env.action_dim,
            episode_length=wrapped_env.unwrapped.max_episode_length,
        )
        policy = FastSAC(cfg)
        policy.load(args.checkpoint)
    else:
        policy = None

    class _PolicyWrapper:
        def __call__(self, obs_dict):
            flat_obs = wrapped_env._obs_to_tensor(obs_dict)
            if args.agent == "zero":
                return torch.zeros(1, wrapped_env.action_dim)
            if args.agent == "random":
                return wrapped_env.rand_act()
            assert policy is not None
            return policy.act(flat_obs, eval_mode=True)

    try:
        if args.viewer == "native":
            NativeMujocoViewer(env, _PolicyWrapper()).run()
        else:
            ViserPlayViewer(env, _PolicyWrapper()).run()
    finally:
        wrapped_env.close()
