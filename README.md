# mmrl

Environment-agnostic reinforcement learning algorithms for MJLab, IsaacLab,
Gym, and Gymnasium style environments.

`mmrl` is a core algorithm package. Install it into the same Python
environment as your simulator or environment framework, then write thin
environment-specific `train.py` and `play.py` entry points that compose its
wrappers, memories, agents, and runners.

Environment packages own task-specific algorithm config classes. They pass a
config instance and a wrapped environment into an `mmrl` runner; they do not
register tasks, algorithms, or components in this package.

## Algorithms

- AMP
  - PPO with an LSGAN motion discriminator, policy transition replay,
    expert-motion sampling, feature normalization, gradient penalty, and
    configurable task/style reward blending.
- PPO
  - Native continuous-action implementation with clipped objectives, GAE,
    adaptive-KL scheduling, and vectorized rollout collection.
- TD-MPC2
  - Model-based RL implementation migrated from the original `tdmpc2` branch.
- FastSAC
  - Lightweight Soft Actor-Critic implementation for flat continuous-control
    tasks.

## Requirements

- Python version compatible with your target environment framework.
- A working MJLab, IsaacLab, Gym, or Gymnasium environment.
- `uv` for local editable installation and command execution.

This package intentionally does not vendor environment frameworks; install the
target environment package separately and then install this algorithm package
into the same environment.

## Install

From a workspace that contains both your environment package and this
repository:

```sh
uv pip install --python .venv/bin/python -e ./mmrl --no-build-isolation
```

From GitHub:

```sh
uv pip install --python .venv/bin/python \
  git+https://github.com/soulde/mmrl.git
```

Install the optional Gymnasium dm-control stack for dog-run:

```sh
uv pip install --python .venv/bin/python -e './mmrl[dm-control]' \
  --no-build-isolation
```

## Entry Points

`mmrl` intentionally provides no training CLI. MJLab, IsaacLab, Gym, and
Gymnasium packages construct their own environment and own their `train.py` and
`play.py`. See `examples/gymnasium/` for the integration template.

## Python API

Top-level imports:

```python
from mmrl import FastSAC, FastSACRunner, FastSACRunnerCfg
from mmrl import AMP, AMPRunner, AMPRunnerCfg
from mmrl import TDMPC2, TDMPC2Runner, TDMPC2RunnerCfg
from mmrl import PPO, PPORunner, PPORunnerCfg
```

Algorithm-specific imports:

```python
from mmrl.fastsac import FastSAC
from mmrl.memories import OffPolicyReplayMemory
from mmrl.tdmpc2 import TDMPC2, TDMPC2ModelCfg, TDMPC2RunnerCfg
```

Environment and memory imports:

```python
from mmrl.env_wrappers import GymnasiumEnvWrapper
from mmrl.env_wrappers import MJLabSingleEnvWrapper, MJLabVectorEnvWrapper
from mmrl.memories import EpisodeMemory, OffPolicyReplayMemory
from mmrl.amp import AMPLoader
```

### AMP Integration

AMP is implemented as a PPO specialization and does not depend on RSL-RL.
`AMPRunner` constructs its expert motion loader directly from `env.cfg.amp`:

```python
runner = AMPRunner(env, AMPRunnerCfg(), log_dir)
runner.learn()
```

The environment-owned AMP config provides the motion and robot layout:

```python
class AMPCfg:
    dt = 0.02
    motion_files = ("motions/walk.pkl",)
    motion_weights = {"walk.pkl": 1.0}
    joint_names = (...,)
    anchor_base = "base"
    anchor_links = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
    urdf_path = "robots/go2/go2.urdf"
    preload_transitions = True
    num_preload_transitions = 1_000_000
```

The loader validates GMR pickle motions, derives velocities, computes
anchor-relative link positions from the URDF, builds the shared AMP feature
layout, applies motion weights, interpolates frames, and optionally preloads
transitions. The reference-style names `amp_motion_files`,
`amp_motion_weights`, `amp_anchor_base`, `amp_anchor_links`, and
`amp_num_preload_transitions` are also accepted inside `env.cfg.amp`.

For wrappers without observation-group support, the wrapped environment must
implement:

```python
def get_amp_observations() -> torch.Tensor:
    # Shape: (num_envs, amp_observation_dim)
    ...
```

If `step()` automatically resets completed environments, its info dictionary
must include `terminal_amp_observations`, containing either one row per
environment or one row per completed environment. This preserves the real
terminal transition instead of pairing the previous state with a reset state.

### IsaacLab Observation Groups

IsaacLab environments should return separate observation groups for policy,
privileged critic input, and AMP motion features. The wrapper preserves those
groups, while the runner config maps them to algorithm observation sets:

```python
env = IsaacLabEnvWrapper(env)
cfg = AMPRunnerCfg(
    obs_groups={
        "actor": ("policy",),
        "critic": ("policy", "privileged"),
        "amp": ("amp",),
    },
)
```

`reset()` and `step()` keep the full group mapping cached in the wrapper and
return the conventional `policy` group through the common environment API.
PPO and AMP runners select and concatenate their configured sets from the
latest cache. Missing critic groups fall back to actor observations; a missing
configured actor or AMP group raises an error. This prevents privileged or AMP
features from leaking into the deployed actor policy.

## Logging

All runners always print a compact training block to the terminal. It includes
progress, throughput, collection and learning time, losses, recent episode
reward and length, elapsed time, ETA, and algorithm-specific values such as
entropy, KL, alpha, or replay size.

TensorBoard and Weights & Biases use the same scalar metrics and global step.
They are disabled by default and configured through the nested Python runner
config:

```python
from mmrl import LoggerCfg, PPORunnerCfg

cfg = PPORunnerCfg(
    logger=LoggerCfg(
        backends=("tensorboard", "wandb"),
        color=True,
        wandb_project="dog-run",
        wandb_group="ppo",
        run_name="seed-1",
    )
)
```

Set `color=False` for white/plain terminal output, for example when redirecting
training logs to a file.

TensorBoard event files and checkpoints are written below the runner's
`log_dir`. Start TensorBoard with `tensorboard --logdir <log-root>`. W&B uses
the same directory for its local run files. TensorBoard and W&B are installed
as core dependencies, but remain inactive until listed in `logger.backends`.

## Package Layout

```text
src/mmrl/
  env_wrappers/
    base.py
    gym.py
    gymnasium.py
    isaaclab.py
    mjlab.py
  fastsac/
    config.py
    fastsac.py
    runner.py
  amp/
    amp.py
    config.py
    runner.py
  memories/
    amp.py
    base.py
    episode.py
    off_policy.py
    on_policy.py
    storage.py
  models/
    amp.py
    actors.py
    base.py
    critics.py
    mlp.py
    tdmpc2/
      init.py
      layers.py
      math.py
      world_model.py
    world_models.py
  ppo/
    actor_critic.py
    config.py
    ppo.py
    runner.py
  runners/
    base.py
    model_based.py
    off_policy.py
    on_policy.py
  tdmpc2/
    config.py
    runner.py
    tdmpc2.py
```

Examples:

```text
examples/gymnasium/
  fastsac_train.py
  fastsac_play.py
  ppo_train.py
  ppo_play.py
  tdmpc2_train.py
  tdmpc2_play.py
```

### Gymnasium Dog Run

The examples support the Shimmy environment ID `dm_control/dog-run-v0`:

```sh
MUJOCO_GL=egl python examples/gymnasium/ppo_train.py \
  dm_control/dog-run-v0
MUJOCO_GL=egl python examples/gymnasium/fastsac_train.py \
  dm_control/dog-run-v0
MUJOCO_GL=egl python examples/gymnasium/tdmpc2_train.py \
  dm_control/dog-run-v0
```

Each corresponding `*_play.py` accepts its checkpoint path. Use
`render_mode="human"` on a machine with a display; use an EGL-backed render
mode for headless playback or recording.

## Notes

- TD-MPC2 uses single-environment training.
- TD-MPC2 runs without `torch.compile` by default to avoid PyTorch 2.9
  Inductor/TF32 compatibility issues.
- FastSAC supports vectorized MJLab environments through its wrapper, but is
  intentionally single-process and lightweight.
- PPO is implemented inside `mmrl` and has no runtime dependency on RSL-RL.
  Its design was adapted from RSL-RL under BSD-3-Clause; the upstream license
  is retained in `licenses/rsl_rl-BSD-3-Clause.txt`.
- AMP extends the native PPO implementation. Its LSGAN reward, policy replay,
  expert sampling, and terminal-state handling follow the reference structure
  used by the RSL-RL fork in `amp_go2-main`, adapted to mmrl ownership
  boundaries.
- Policy runners consume flattened observations supplied by environment
  wrappers; AMP observations remain a separate environment-defined tensor.

## Config Compatibility

`mmrl` should follow the same integration shape as RSL-RL: environment packages
own their `train.py`/`play.py`, construct the environment, prepare a nested
runner config, then pass both into an `mmrl` runner.

New runner/model code should accept `train_cfg` objects that are organized like:

```text
runner:
  ...
  algorithm:
    class_name: ...
    ...
  actor:
    class_name: ...
    ...
  critic:
    class_name: ...
    ...
  memory:
    class_name: ...
    ...
  logger:
    backends: ...
    ...
```

Environment packages pass Python config classes or instances into
`mmrl` runners and models. New code should not assume that config objects are
dataclasses only. It should support:

- dataclass instances
- dictionaries
- plain Python objects with attributes
- IsaacLab-style class configs with inherited class attributes

YAML configuration is intentionally outside the framework boundary. Component
selection remains Python-owned and limited to implementations provided by
`mmrl`; environment packages may select and configure them, but cannot inject
new algorithm, model, memory, or runner classes into the library.

Prefer shared config access helpers over direct `cfg.__dict__` reads when
implementing reusable runners, models, wrappers, and memory factories.
Use `class_name` only to select implementations explicitly supported by the
corresponding `mmrl` runner.

## Memory Roadmap

Current memory storage is split by sampling pattern:

- `TensorRingStorage` backs off-policy replay. It preallocates fixed-shape
  tensors and uses ring writes for stable high-frequency transition storage.
- `EpisodeListStorage` backs TD-MPC2 style episode replay. It keeps variable
  length episodes intact and evicts by total timestep capacity.
- `TensorRolloutStorage` backs on-policy rollouts with preallocated
  `(num_steps, num_envs, ...)` tensors, GAE returns, and shuffled mini-batches.
- `TensorRingStorage` also backs AMP policy transition replay. AMP replay only
  trains the discriminator; PPO policy updates remain on-policy.

Planned upgrades:

- Add optional pinned-memory and device-resident storage modes for faster CPU to
  GPU transfer.
- Add prioritized off-policy replay without changing the `OffPolicyReplayMemory`
  public batch API.
- Add n-step return support for off-policy memory.
- Add episode-level metadata and sequence masks for model-based algorithms that
  need padded batched episode storage.
- Add serialization hooks for saving and restoring memory state with checkpoints.

## Development Checks

Run these from the repository root:

```sh
uv run ruff check src tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests
```

To refresh the editable installation after changing package metadata:

```sh
uv pip install --python .venv/bin/python -e ./mmrl --no-build-isolation
```
