"""Factory and Registry pattern for CDADE components.

Usage
-----
Register a class::

    @register_detector("iforest")
    class IForestDetector(BaseDetector):
        ...

Instantiate by name::

    cls = get_detector("iforest")
    detector = cls(cfg)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

_REGISTRIES: dict[str, dict[str, type]] = {}


def _get_registry(kind: str) -> dict[str, type]:
    if kind not in _REGISTRIES:
        _REGISTRIES[kind] = {}
    return _REGISTRIES[kind]


def register(kind: str, name: str) -> Callable[[type[T]], type[T]]:
    """Generic decorator that registers *cls* under *name* in registry *kind*.

    Args:
        kind: Registry namespace (e.g. "detector", "reconciler", "selector").
        name: Lookup key.

    Returns:
        The decorated class unchanged.
    """

    def decorator(cls: type[T]) -> type[T]:
        registry = _get_registry(kind)
        if name in registry:
            raise KeyError(f"'{name}' already registered in '{kind}' registry.")
        registry[name] = cls
        return cls

    return decorator


def get(kind: str, name: str) -> type:
    """Return the class registered under *name* in registry *kind*.

    Args:
        kind: Registry namespace.
        name: Lookup key.

    Returns:
        The registered class.

    Raises:
        KeyError: If *name* is not found in *kind*.
    """
    registry = _get_registry(kind)
    if name not in registry:
        available = sorted(registry.keys())
        raise KeyError(f"'{name}' not found in '{kind}' registry. Available: {available}")
    return registry[name]


def list_registered(kind: str) -> list[str]:
    """Return all registered names for *kind*."""
    return sorted(_get_registry(kind).keys())


def list_detectors() -> list[str]:
    """Return all registered detector names."""
    return list_registered("detector")


# Convenience aliases for each component family


def register_detector(name: str) -> Callable[[type[T]], type[T]]:
    """Register a detector class."""
    return register("detector", name)


def get_detector(name: str) -> type:
    """Retrieve a detector class by name."""
    return get("detector", name)


def register_reconciler(name: str) -> Callable[[type[T]], type[T]]:
    """Register a reconciler class."""
    return register("reconciler", name)


def get_reconciler(name: str) -> type:
    """Retrieve a reconciler class by name."""
    return get("reconciler", name)


def register_selector(name: str) -> Callable[[type[T]], type[T]]:
    """Register a selector class."""
    return register("selector", name)


def get_selector(name: str) -> type:
    """Retrieve a selector class by name."""
    return get("selector", name)


def make(kind: str, name: str, *args: Any, **kwargs: Any) -> Any:
    """Instantiate the class registered under *name* in *kind*.

    Args:
        kind: Registry namespace.
        name: Lookup key.
        *args: Positional arguments forwarded to the constructor.
        **kwargs: Keyword arguments forwarded to the constructor.

    Returns:
        A new instance of the registered class.
    """
    cls = get(kind, name)
    return cls(*args, **kwargs)


# Baseline detector registry


def register_baseline_detector(name: str) -> Callable[[type[T]], type[T]]:
    """Register a baseline detector class.

    Args:
        name: Registry name for the baseline detector.

    Returns:
        The decorated class unchanged.
    """
    return register("baseline_detector", name)


def get_baseline_detector(name: str) -> type:
    """Retrieve a baseline detector class by name.

    Args:
        name: Lookup key.

    Returns:
        The registered class.
    """
    return get("baseline_detector", name)


def list_baseline_detectors() -> list[str]:
    """Return all registered baseline detector names."""
    return sorted(list_registered("baseline_detector"))
