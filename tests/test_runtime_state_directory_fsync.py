"""Regression coverage for durable operation/session state publication."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from nexus_open_swe_runtime import cli


def test_atomic_json_fsyncs_file_and_parent_directory(tmp_path: Path, monkeypatch) -> None:
    observed: list[str] = []
    original_fsync = os.fsync
    original_fstat = os.fstat

    def recording_fsync(fd: int) -> None:
        mode = original_fstat(fd).st_mode
        observed.append("directory" if stat.S_ISDIR(mode) else "file")
        original_fsync(fd)

    monkeypatch.setattr(cli.os, "fsync", recording_fsync)

    target = tmp_path / "state" / "operations" / "op.json"
    cli._atomic_json(target, {"status": "STARTED"})

    assert target.exists()
    assert "file" in observed
    assert "directory" in observed
