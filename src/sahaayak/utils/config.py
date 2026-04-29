"""YAML configuration loader.

The default config lives in `config/default.yaml`. A per-user calibration
profile may live in `config/calibration_profile.yaml` (gitignored) and is
*shallow-merged* on top.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from sahaayak.utils.logger import get_logger

logger = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "default.yaml"
USER_PROFILE_PATH = _REPO_ROOT / "config" / "calibration_profile.yaml"


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `overlay` into `base`. Lists are replaced, not concatenated."""
    out = deepcopy(base)
    for key, value in overlay.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_config(
    default_path: Path | None = None,
    user_path: Path | None = None,
) -> dict[str, Any]:
    """Load default config and merge any user calibration profile on top.

    Args:
        default_path: Path to the default YAML. Defaults to ``config/default.yaml``.
        user_path: Optional per-user override path.

    Returns:
        A merged configuration dict.

    Raises:
        FileNotFoundError: If the default config file is missing.
    """
    default_path = default_path or DEFAULT_CONFIG_PATH
    user_path = user_path or USER_PROFILE_PATH

    if not default_path.exists():
        raise FileNotFoundError(f"Default config not found at {default_path}")

    with default_path.open("r", encoding="utf-8") as fh:
        config: dict[str, Any] = yaml.safe_load(fh) or {}

    if user_path.exists():
        try:
            with user_path.open("r", encoding="utf-8") as fh:
                user_overlay = yaml.safe_load(fh) or {}
            config = _deep_merge(config, user_overlay)
            logger.info("Merged user profile from %s", user_path)
        except yaml.YAMLError as exc:
            logger.warning("Ignoring malformed user profile %s: %s", user_path, exc)

    return config
