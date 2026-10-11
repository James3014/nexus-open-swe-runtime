#!/usr/bin/env python3
"""Print the pinned Nexus-new consumer revision.

The single source of truth is ``expected_identity`` of the
``nexus-new-consumer`` material in ``.nexus-core/config.toml`` (nexus-certify
requires that value to be a literal and rejects unknown top-level keys, so the
pin cannot live elsewhere).  CI, the Core verifier commands and the
cross-repo test all read it through this module.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[2] / ".nexus-core" / "config.toml"
MATERIAL_ID = "nexus-new-consumer"
_PREFIX = "git-commit:"
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


def read_consumer_pin(config_path: Path = CONFIG_PATH) -> str:
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    for material in config.get("materials", []):
        if material.get("id") == MATERIAL_ID:
            identity = material.get("expected_identity", "")
            sha = identity[len(_PREFIX) :] if identity.startswith(_PREFIX) else ""
            if not _HEX40.match(sha):
                raise ValueError(f"{MATERIAL_ID}.expected_identity is not git-commit:<40-hex>")
            return sha
    raise ValueError(f"material {MATERIAL_ID!r} not found in {config_path}")


if __name__ == "__main__":
    try:
        print(read_consumer_pin())
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"consumer_pin: {exc}", file=sys.stderr)
        sys.exit(1)
