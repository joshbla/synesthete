import yaml
from pathlib import Path
import torch
import os
from typing import Any

def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base (dicts only)."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _flatten_keypaths(d: dict[str, Any], prefix: str = "") -> set[str]:
    """Return dotted keypaths for nested dict keys."""
    out: set[str] = set()
    for k, v in d.items():
        p = f"{prefix}.{k}" if prefix else str(k)
        out.add(p)
        if isinstance(v, dict):
            out |= _flatten_keypaths(v, prefix=p)
    return out


def load_config(
    config_path: str | None = None,
    *,
    override_path: str | None = None,
    warn_unknown_override_keys: bool = True,
) -> dict[str, Any]:
    """
    Load config, optionally applying an override file.

    Sources (in order):
    - config_path argument (or env SYNESTHETE_CONFIG, else config/default.yaml)
    - override_path argument (or env SYNESTHETE_CONFIG_OVERRIDE, else none)

    Rationale:
    - lets you run a fast smoke pipeline via a small override YAML without
      copying the whole config and risking drift.
    """
    if config_path is None:
        config_path = os.environ.get("SYNESTHETE_CONFIG", "config/default.yaml")
    if override_path is None:
        override_path = os.environ.get("SYNESTHETE_CONFIG_OVERRIDE")

    with open(config_path, "r") as f:
        base = yaml.safe_load(f) or {}
    if not isinstance(base, dict):
        raise ValueError(f"Base config must be a mapping, got {type(base)} from {config_path}")

    if override_path:
        with open(override_path, "r") as f:
            override = yaml.safe_load(f) or {}
        if not isinstance(override, dict):
            raise ValueError(f"Override config must be a mapping, got {type(override)} from {override_path}")

        if warn_unknown_override_keys:
            base_paths = _flatten_keypaths(base)
            override_paths = _flatten_keypaths(override)
            unknown = sorted(p for p in override_paths if p not in base_paths)
            if unknown:
                print(
                    "[Config] Warning: override contains unknown keys (possible typos / drift):\n"
                    + "\n".join(f"  - {p}" for p in unknown),
                    flush=True,
                )

        return _deep_merge(base, override)

    return base

def get_device():
    """
    Returns the best available device.
    Prioritizes MPS (Apple Silicon) > CUDA > CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
