"""Configuration helpers for dict, dataclass, and class-style configs."""

from dataclasses import asdict, is_dataclass
from importlib import import_module
from typing import Any


def config_to_dict(cfg: Any) -> dict[str, Any]:
    """Convert common config styles into a plain dictionary."""
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return {key: _convert_value(value) for key, value in cfg.items()}
    if is_dataclass(cfg):
        return {key: _convert_value(value) for key, value in asdict(cfg).items()}

    values: dict[str, Any] = {}
    for cls in reversed(type(cfg).mro()):
        if cls is object:
            continue
        values.update(_public_config_items(cls.__dict__))
    values.update(_public_config_items(getattr(cfg, "__dict__", {})))
    return {key: _convert_value(value) for key, value in values.items()}


def get_config_value(cfg: Any, path: str, default: Any = None) -> Any:
    """Read a dotted config path from dicts, dataclasses, or objects."""
    current = cfg
    for part in path.split("."):
        if current is None:
            return default
        if isinstance(current, dict):
            if part not in current:
                return default
            current = current[part]
        else:
            if not hasattr(current, part):
                return default
            current = getattr(current, part)
    return current


def require_config_value(cfg: Any, path: str) -> Any:
    """Read a dotted config path and raise if it is missing."""
    missing = object()
    value = get_config_value(cfg, path, missing)
    if value is missing:
        raise KeyError(f"Missing required config value: {path}")
    return value


def resolve_class(class_name: str | type) -> type:
    """Resolve a class object from a class or import path string."""
    if isinstance(class_name, type):
        return class_name
    if not isinstance(class_name, str):
        raise TypeError(f"class_name must be a class or string, got {type(class_name)!r}")
    module_name, separator, attr_name = class_name.rpartition(".")
    if not separator:
        raise ValueError(
            f"Class name {class_name!r} must be a fully qualified import path."
        )
    module = import_module(module_name)
    resolved = getattr(module, attr_name)
    if not isinstance(resolved, type):
        raise TypeError(f"Resolved object {class_name!r} is not a class.")
    return resolved


def resolve_config_class(cfg: Any, path: str = "class_name") -> type:
    """Resolve a class from a config object's ``class_name`` field."""
    return resolve_class(require_config_value(cfg, path))


def _public_config_items(mapping) -> dict[str, Any]:
    result = {}
    for key, value in mapping.items():
        if key.startswith("_"):
            continue
        if isinstance(value, (staticmethod, classmethod, property)):
            continue
        if callable(value):
            continue
        result[key] = value
    return result


def _convert_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _convert_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_convert_value(item) for item in value)
    if is_dataclass(value) or hasattr(value, "__dict__"):
        return config_to_dict(value)
    return value

