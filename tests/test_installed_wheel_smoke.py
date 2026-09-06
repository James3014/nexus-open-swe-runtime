"""Test that built standalone nexus-open-swe-runtime wheel installs and executes cleanly."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_installed_wheel_smoke_in_isolated_venv(tmp_path: Path):
    repo_root = Path(__file__).parent.parent.resolve()
    dist_dir = repo_root / "dist"
    wheels = sorted(dist_dir.glob("nexus_open_swe_runtime-*.whl"))
    if not wheels:
        try:
            subprocess.run(["uv", "build"], cwd=str(repo_root), check=True)
            wheels = sorted(dist_dir.glob("nexus_open_swe_runtime-*.whl"))
        except Exception:
            pass
    if not wheels:
        pytest.skip("No nexus-open-swe-runtime wheel found in dist/ and uv build not available.")

    target_wheel = wheels[-1]

    # Create isolated venv
    venv_dir = tmp_path / "smoke_venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    venv_pip = venv_dir / "bin" / "pip"
    venv_runtime_bin = venv_dir / "bin" / "nexus-open-swe-runtime"

    # Install the wheel
    subprocess.run([str(venv_pip), "install", "--no-cache-dir", str(target_wheel)], check=True)

    # 1. Verify CLI executable exists
    assert venv_runtime_bin.is_file()

    # 2. Invoke --identity outside repo checkout
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()

    res_identity = subprocess.run(
        [str(venv_runtime_bin), "--identity"],
        cwd=str(outside_dir),
        capture_output=True,
        text=True,
        check=True,
    )
    identity_data = json.loads(res_identity.stdout)
    assert identity_data["status"] == "IDENTIFIED"
    assert identity_data["kind"] == "identity"
    assert identity_data["distribution_name"] == "nexus-open-swe-runtime"
    assert "site-packages" in identity_data["artifact_identity"]["module_file"]
    assert str(repo_root) not in identity_data["artifact_identity"]["module_file"]

    # 3. Invoke via stdin JSON protocol
    stdin_req = json.dumps({"schema": "nexus.open_swe_runtime.request.v1", "operation": "identity"})
    res_stdin = subprocess.run(
        [str(venv_runtime_bin)],
        input=stdin_req,
        cwd=str(outside_dir),
        capture_output=True,
        text=True,
        check=True,
    )
    stdin_data = json.loads(res_stdin.stdout)
    assert stdin_data["status"] == "IDENTIFIED"
    assert stdin_data["distribution_name"] == "nexus-open-swe-runtime"

    # 4. Invoke with invalid schema (fail-closed)
    bad_req = json.dumps({"schema": "nexus.open_swe_runtime.request.v999", "operation": "identity"})
    res_bad = subprocess.run(
        [str(venv_runtime_bin)],
        input=bad_req,
        cwd=str(outside_dir),
        capture_output=True,
        text=True,
    )
    bad_data = json.loads(res_bad.stdout)
    assert bad_data["status"] == "OPEN_SWE_RUNTIME_PROTOCOL_FAILED"
    assert bad_data["process_started"] is False
    assert bad_data["retry_safe"] is False
