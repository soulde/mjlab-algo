# mmrl

Environment-agnostic reinforcement learning algorithms for MJLab, IsaacLab,
Gym, and Gymnasium style environments.

`mmrl` is a core algorithm package. Install it into the same Python
environment as your simulator or environment framework, then write thin
environment-specific `train.py` and `play.py` entry points that compose its
wrappers, memories, agents, and runners.

Algorithm defaults can be registered by task extension packages. For example,
`mjlabplusplus` registers DR02 FastSAC and TD-MPC2 defaults from its
`tasks/velocity/dr02/rl_cfg.py`, so the training command only needs the task ID.

## Algorithms

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

## Command Line Usage

TD-MPC2:

```sh
uv run tdmpc2-train Mjlab-Cartpole-Balance
uv run tdmpc2-train Mjlab-Velocity-Flat-DR02
uv run tdmpc2-play Mjlab-Cartpole-Balance --agent zero
```

FastSAC:

```sh
uv run fastsac-train Mjlab-Cartpole-Balance
uv run fastsac-train Mjlab-Velocity-Flat-DR02
uv run fastsac-play Mjlab-Cartpole-Balance --agent zero
```

When a task has registered algorithm defaults, command-line flags are temporary
overrides on top of that task config:

```sh
uv run fastsac-train Mjlab-Velocity-Flat-DR02 --total-steps 10000
uv run tdmpc2-train Mjlab-Velocity-Flat-DR02 --steps 10000
```

Run a short FastSAC smoke test:

```sh
uv run fastsac-train Mjlab-Cartpole-Balance \
  --total-steps 4 \
  --learning-starts 1 \
  --batch-size 2 \
  --buffer-size 16 \
  --num-envs 1 \
  --device cpu \
  --log-interval 2 \
  --no-save-agent
```

## Python API

Top-level imports:

```python
from mmrl import FastSAC, FastSACConfig, FastSACRunner
from mmrl import TDMPC2, TDMPC2Config, TDMPC2Runner
```

Algorithm-specific imports:

```python
from mmrl.fastsac import FastSAC, FastSACReplayBuffer
from mmrl.fastsac import FastSACConfig, make_fastsac_config
from mmrl.registry import load_fastsac_cfg, load_tdmpc2_cfg
from mmrl.tdmpc2 import TDMPC2, TDMPC2Config, make_tdmpc2_config
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
    buffer.py
    config.py
    fastsac.py
    networks.py
    runner.py
    vecenv_wrapper.py
  memories/
    base.py
    episode.py
    off_policy.py
    on_policy.py
  models/
    actors.py
    base.py
    critics.py
    world_models.py
  runners/
    base.py
    model_based.py
    off_policy.py
    on_policy.py
  tdmpc2/
    buffer.py
    config.py
    runner.py
    tdmpc2.py
    vecenv_wrapper.py
    world_model.py
  registry.py
  scripts/
    fastsac/
    tdmpc2/
```

Examples:

```text
examples/gymnasium/
  fastsac_train.py
  fastsac_play.py
```

## Notes

- TD-MPC2 uses single-environment training.
- TD-MPC2 runs without `torch.compile` by default to avoid PyTorch 2.9
  Inductor/TF32 compatibility issues.
- FastSAC supports vectorized MJLab environments through its wrapper, but is
  intentionally single-process and lightweight.
- Both algorithms flatten MJLab dict observations for their policy inputs.
- Legacy imports such as `mmrl.fastsac.vecenv_wrapper` and
  `mmrl.tdmpc2.buffer` are kept as compatibility shims while the common
  namespaces mature.

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

Environment packages may pass IsaacLab-style config classes or instances into
`mmrl` runners and models. New code should not assume that config objects are
dataclasses only. It should support:

- dataclass instances
- dictionaries
- plain Python objects with attributes
- IsaacLab-style class configs with inherited class attributes

Prefer shared config access helpers over direct `cfg.__dict__` reads when
implementing reusable runners, models, wrappers, and memory factories.
Prefer `class_name` based construction hooks for algorithm/model/memory
components so environment packages can swap implementations without editing
`mmrl` internals.

## Memory Roadmap

Current memory storage is split by sampling pattern:

- `TensorRingStorage` backs off-policy replay. It preallocates fixed-shape
  tensors and uses ring writes for stable high-frequency transition storage.
- `EpisodeListStorage` backs TD-MPC2 style episode replay. It keeps variable
  length episodes intact and evicts by total timestep capacity.
- `TensorListStorage` backs the initial on-policy rollout memory. It is a
  short-lived append/clear store for future PPO/A2C style runners.

Planned upgrades:

- Add a preallocated rollout tensor storage shaped like
  `(rollout_steps, num_envs, ...)` for mature on-policy algorithms.
- Add optional pinned-memory and device-resident storage modes for faster CPU to
  GPU transfer.
- Add prioritized off-policy replay without changing the `OffPolicyReplayMemory`
  public batch API.
- Add n-step return support for off-policy memory.
- Add episode-level metadata and sequence masks for model-based algorithms that
  need padded batched episode storage.
- Add serialization hooks for saving and restoring memory state with checkpoints.

## Development Checks

Run these from the parent `mjlab` workspace:

```sh
uv run ruff check mmrl/src mmrl/tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest mmrl/tests
```

To refresh console scripts after changing `pyproject.toml` or `setup.py`:

```sh
uv pip install --python .venv/bin/python -e ./mmrl --no-build-isolation
```
