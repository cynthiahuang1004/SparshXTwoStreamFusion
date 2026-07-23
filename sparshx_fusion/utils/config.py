from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def update_epochs(cfg: dict[str, Any], epochs: int | None) -> dict[str, Any]:
    if epochs is not None:
        cfg["train"]["epochs"] = epochs
    return cfg

