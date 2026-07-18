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
```

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
  memories/
    base.py
    episode.py
    off_policy.py
    on_policy.py
  models/
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
- Both algorithms flatten MJLab dict observations for their policy inputs.

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
