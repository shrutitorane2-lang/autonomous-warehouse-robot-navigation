"""
Configuration loader utility.
Reads config/config.yaml and provides typed access to all parameters.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


# Locate the project root (two levels up from src/utils/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"


class Config:
    """
    Lightweight wrapper around a YAML configuration dictionary.

    Supports attribute-style access for nested dicts, e.g.::

        cfg = Config.load()
        lr = cfg.q_learning.learning_rate
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Return the raw dictionary representation."""
        return self._data

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """
        Load configuration from a YAML file.

        Parameters
        ----------
        path : str or Path, optional
            Explicit path to config.yaml. Defaults to config/config.yaml
            in the project root.

        Returns
        -------
        Config
            Parsed configuration object.
        """
        config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                "Expected location: config/config.yaml"
            )
        with open(config_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return cls(raw)

    def override(self, **kwargs: Any) -> "Config":
        """
        Create a shallow copy of the config with top-level keys overridden.
        Useful for quick hyperparameter sweeps.

        Parameters
        ----------
        **kwargs : Any
            Key-value pairs to override in the top-level config dictionary.

        Returns
        -------
        Config
            New Config instance with overrides applied.
        """
        import copy
        updated = copy.deepcopy(self._data)
        updated.update(kwargs)
        return Config(updated)


def load_config(path: str | Path | None = None) -> Config:
    """Convenience function — equivalent to Config.load(path)."""
    return Config.load(path)
