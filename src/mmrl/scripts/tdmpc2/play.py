"""Play/evaluation script for TD-MPC2.

Loads a trained checkpoint and runs the policy in the environment
with an optional viewer.
"""

import os
import sys
from dataclasses import dataclass
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mmrl.tdmpc2 import TDMPC2, make_tdmpc2_config
from mmrl.tdmpc2.vecenv_wrapper import TDMPC2VecEnvWrapper
from mmrl.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class PlayArgs:
    """CLI arguments for TD-MPC2 play/evaluation."""

    task_id: str = ""
    """Task ID (e.g., ``Mjlab-Cartpole-Balance``)."""
    checkpoint: str | None = None
    """Path to checkpoint file (.pt)."""
    model_size: int | None = 5
    """Model size preset: 1, 5, 19, 48, or 317."""
    agent: Literal["zero", "random", "trained"] = "trained"
    """Agent type: zero actions, random actions, or trained policy."""
    device: str | None = None
    """Device override."""
    viewer: Literal["auto", "native", "viser"] = "auto"
    """Viewer backend."""
    video: bool = False
    """Record a video of the episode."""
    video_length: int = 500
    """Maximum video length in steps."""
    eval_episodes: int = 1
    """Number of evaluation episodes."""


def main():
    maybe_print_top_level_help("tdmpc2-play")

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

    configure_torch_backends()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load environment config.
    env_cfg = load_env_cfg(args.task_id)
    env_cfg.scene.num_envs = 1

    render_mode = "rgb_array" if args.video else None
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
    env = TDMPC2VecEnvWrapper(env)

    # Determine if we're using a dummy agent.
    DUMMY_MODE = args.agent in {"zero", "random"}

    if not DUMMY_MODE and args.checkpoint is None:
        raise ValueError("--checkpoint is required when agent=trained")

    if DUMMY_MODE:
        if args.agent == "zero":

            class PolicyZero:
                def __call__(self, obs, t0=False, eval_mode=False):
                    return torch.zeros(env._action_dim)

            policy = PolicyZero()
        else:

            class PolicyRandom:
                def __call__(self, obs, t0=False, eval_mode=False):
                    return 2 * torch.rand(env._action_dim) - 1

            policy = PolicyRandom()
    else:
        # Build config and load agent.
        cfg = make_tdmpc2_config(
            model_size=args.model_size,
            task=args.task_id,
            mpc=True,
        )
        cfg.obs_shape = {"state": env.observation_space.shape}
        cfg.action_dim = int(env._action_dim)
        cfg.episode_length = env.unwrapped.max_episode_length
        cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
        cfg.multitask = False
        cfg.task_dim = 0

        agent = TDMPC2(cfg)
        agent.load(args.checkpoint)
        policy = agent

    # Handle viewer selection.
    if args.viewer == "auto":
        has_display = bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        resolved_viewer = "native" if has_display else "viser"
    else:
        resolved_viewer = args.viewer

    # Wrap the policy to match viewer interface.
    class _PolicyWrapper:
        def __init__(self, policy, env_wrapper):
            self._policy = policy
            self._env = env_wrapper
            self._t = 0

        def __call__(self, obs):
            # Viewer passes batched obs; strip and flatten for TD-MPC2.
            if isinstance(obs, dict):
                tensors = []
                for v in obs.values():
                    t = v.squeeze(0) if v.ndim > 1 else v
                    tensors.append(t.flatten())
                obs_flat = torch.cat(tensors).to(self._env.env.device)
            else:
                obs_flat = obs.squeeze(0).flatten().to(self._env.env.device)
            action = self._policy(obs_flat, t0=self._t == 0, eval_mode=True)
            self._t += 1
            return action.unsqueeze(0)  # Add batch dim back for viewer

    wrapped_policy = _PolicyWrapper(policy, env)

    if resolved_viewer == "native":
        NativeMujocoViewer(env.env, wrapped_policy).run()
    elif resolved_viewer == "viser":
        ViserPlayViewer(env.env, wrapped_policy).run()
    else:
        raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

    env.close()


if __name__ == "__main__":
    main()
