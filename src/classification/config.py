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
    """Resolve the configured device.

    Auto-selection order:
        CUDA -> Apple MPS -> CPU
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")

        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")

        return torch.device("cpu")

    return torch.device(requested)