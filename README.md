# mjlab-algo

Private MJLab extension package for additional reinforcement learning algorithms.

This package currently provides the TD-MPC2 implementation migrated from the
`tdmpc2` branch of `mjlab`.

## Install

Install `mjlab` first, then install this package in the same environment:

```sh
uv pip install --python .venv/bin/python -e ./mjlab-algo --no-build-isolation
```

## Commands

```sh
uv run tdmpc2-train Mjlab-Cartpole-Balance
uv run tdmpc2-play Mjlab-Cartpole-Balance --agent zero
```
