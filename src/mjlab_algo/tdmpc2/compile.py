"""Torch compile compatibility helpers for TD-MPC2."""

from __future__ import annotations


def configure_tdmpc2_compile(enabled: bool) -> None:
    """Configure torch inductor before compiling TD-MPC2 functions.

    PyTorch 2.9's inductor shape-padding pass benchmarks padded matmul variants
    through a path that reads the legacy ``cuda.matmul.allow_tf32`` property.
    MJLab configures TF32 through the new ``fp32_precision`` API on PyTorch 2.9+,
    and mixing those APIs raises during ``torch.compile``. Disabling shape
    padding keeps compile enabled while avoiding that legacy TF32 read.

    Inductor also calls a private TF32 performance-warning helper during codegen
    that reads the same legacy property. Replacing that warning helper with a
    no-op avoids the PyTorch 2.9 API conflict without changing compiled math.
    """
    if not enabled:
        return
    try:
        import torch._inductor.config as inductor_config
    except Exception:
        return
    if hasattr(inductor_config, "shape_padding"):
        inductor_config.shape_padding = False

    try:
        import torch._inductor.compile_fx as compile_fx
    except Exception:
        return
    if hasattr(compile_fx, "_warn_tf32_disabled"):
        compile_fx._warn_tf32_disabled = lambda: None
