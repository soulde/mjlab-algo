# mjlab-algo

Private MJLab extension package for additional reinforcement learning algorithms.

This package currently provides the TD-MPC2 implementation migrated from the
`tdmpc2` branch of `mjlab`, plus a lightweight SAC implementation for flat
continuous-control MJLab tasks.

## Install

Install `mjlab` first, then install this package in the same environment:

```sh
uv pip install --python .venv/bin/python -e ./mjlab-algo --no-build-isolation
```

## Commands

```sh
uv run tdmpc2-train Mjlab-Cartpole-Balance
uv run tdmpc2-play Mjlab-Cartpole-Balance --agent zero
uv run fastsac-train Mjlab-Cartpole-Balance
uv run fastsac-play Mjlab-Cartpole-Balance --agent zero
```

## Python API

```python
from mjlab_algo import FastSACConfig, FastSACRunner, TDMPC2Config, TDMPC2Runner
from mjlab_algo.fastsac import FastSAC, FastSACReplayBuffer
from mjlab_algo.tdmpc2 import TDMPC2
```
