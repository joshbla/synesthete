"""
Validate config override keypaths against the base config.

Usage:
  uv run python scripts/validate_config_overrides.py config/default.yaml config/smoke.yaml
"""

import sys
from pathlib import Path

# Ensure we can import from repo root when run directly
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import load_config, _flatten_keypaths  # type: ignore


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: uv run python scripts/validate_config_overrides.py <base.yaml> <override.yaml>")
        return 2

    base_path = sys.argv[1]
    override_path = sys.argv[2]

    # Load both separately so we can compare keypaths.
    base = load_config(base_path)
    override = load_config(override_path)

    base_paths = _flatten_keypaths(base)
    override_paths = _flatten_keypaths(override)
    unknown = sorted(p for p in override_paths if p not in base_paths)

    if unknown:
        print("Unknown override keys (possible typos / drift):")
        for p in unknown:
            print(f"- {p}")
        return 1

    print("OK: all override keys exist in base config.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

