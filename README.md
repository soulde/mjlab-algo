# mjlab-algo

Additional reinforcement learning algorithms for
[MJLab](https://github.com/mujocolab/mjlab).

`mjlab-algo` is an extension package. Install it into the same Python
environment as `mjlab`, and it provides additional algorithm APIs and command
line entry points without modifying the upstream `mjlab` repository.

Algorithm defaults can be registered by task extension packages. For example,
`mjlabplusplus` registers DR02 FastSAC and TD-MPC2 defaults from its
`tasks/velocity/dr02/rl_cfg.py`, so the training command only needs the task ID.

## Algorithms

- TD-MPC2
  - Model-based RL implementation migrated from the original `tdmpc2` branch.
  - Commands: `tdmpc2-train`, `tdmpc2-play`
- FastSAC
  - Lightweight Soft Actor-Critic implementation for flat continuous-control
    MJLab tasks.
  - Commands: `fastsac-train`, `fastsac-play`

## Requirements

- Python version compatible with your `mjlab` checkout.
- A working `mjlab` environment.
- `uv` for local editable installation and command execution.

This package intentionally does not vendor `mjlab`; install `mjlab` separately
and then install this algorithm package into the same environment.

## Install

From a workspace that contains both `mjlab` and this repository:

```sh
uv pip install --python .venv/bin/python -e ./mjlab-algo --no-build-isolation
```

From GitHub:

```sh
uv pip install --python .venv/bin/python \
  git+https://github.com/soulde/mjlab-algo.git
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
from mjlab_algo import FastSAC, FastSACConfig, FastSACRunner
from mjlab_algo import TDMPC2, TDMPC2Config, TDMPC2Runner
```

Algorithm-specific imports:

```python
from mjlab_algo.fastsac import FastSAC, FastSACReplayBuffer
from mjlab_algo.fastsac import FastSACConfig, make_fastsac_config
from mjlab_algo.registry import load_fastsac_cfg, load_tdmpc2_cfg
from mjlab_algo.tdmpc2 import TDMPC2, TDMPC2Config, make_tdmpc2_config
```

## Package Layout

```text
src/mjlab_algo/
  fastsac/
    buffer.py
    config.py
    fastsac.py
    networks.py
    runner.py
    vecenv_wrapper.py
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

## Notes

- TD-MPC2 uses single-environment training.
- TD-MPC2 runs without `torch.compile` by default to avoid PyTorch 2.9
  Inductor/TF32 compatibility issues.
- FastSAC supports vectorized MJLab environments through its wrapper, but is
  intentionally single-process and lightweight.
- Both algorithms flatten MJLab dict observations for their policy inputs.

## Development Checks

Run these from the parent `mjlab` workspace:

```sh
uv run ruff check mjlab-algo/src mjlab-algo/tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest mjlab-algo/tests
```

To refresh console scripts after changing `pyproject.toml` or `setup.py`:

```sh
uv pip install --python .venv/bin/python -e ./mjlab-algo --no-build-isolation
```
