"""Loading and access helpers for ``configs/classifier.yaml``.

Kept separate so ``train.py`` and ``inference.py`` share one definition of where
settings live and how defaults are applied, without either importing the other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

DEFAULT_CONFIG_PATH = Path("configs/classifier.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Read and parse the classifier YAML config.

    Args:
        path: Path to the config file, relative to the working directory or absolute.

    Returns:
        The parsed configuration as a nested dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the file does not parse to a mapping or omits ``classes``.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(config).__name__}: {config_path}")
    if not config.get("classes"):
        raise ValueError(f"Config is missing a non-empty 'classes' list: {config_path}")
    return config


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve the ``training.device`` setting to a concrete torch device.

    Args:
        requested: ``"auto"`` picks CUDA when available, otherwise CPU. Any other
            value is passed through to :class:`torch.device`.

    Returns:
        The device to run on.
    """
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)
