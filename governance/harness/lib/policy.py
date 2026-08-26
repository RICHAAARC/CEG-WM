"""Load the minimal JSON policies used by the detachable harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_policy(root: str | Path, name: str) -> dict[str, Any]:
    path = Path(root) / "governance" / "policies" / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))
