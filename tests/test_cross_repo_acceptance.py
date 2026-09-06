"""Cross-repo acceptance test verifying legacy Nexus-new client compatibility
against the standalone nexus-open-swe-runtime executable.

Proven contracts:
1. Legacy client builds accepted v1 request
2. Standalone runtime accepts it via stdin/stdout subprocess
3. Result parses through legacy client parsing logic
4. Operation identity and result schemas are strictly preserved
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def test_cross_repo_legacy_client_compatibility_acceptance():
    repo_root = Path(__file__).parent.parent.resolve()
    nexus_new_candidates = [
        Path("/Users/jameschen/workspace/nexus-new"),
        Path("/Users/jameschen/Workspace/nexus-cutover-worktree"),
        Path("/Users/jameschen/Workspace/Nexus-new"),
    ]
    nexus_new_root = None
    for candidate in nexus_new_candidates:
        if (candidate / "nexus" / "services" / "open_swe_external_intelligence.py").is_file():
            nexus_new_root = candidate
            break

    if nexus_new_root is None:
        pytest.skip("Nexus-new checkout not available for live cross-repo import")

    # Read Git SHAs for exact physical receipt
    res_sha_nexus = subprocess.run(
        ["git", "-C", str(nexus_new_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    nexus_new_sha = res_sha_nexus.stdout.strip()
    assert len(nexus_new_sha) == 40

    res_sha_runtime = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    runtime_sha = res_sha_runtime.stdout.strip()
    assert len(runtime_sha) == 40

    # 1. Prepare standalone executable
    dist_dir = repo_root / "dist"
    wheels = sorted(dist_dir.glob("nexus_open_swe_runtime-*.whl"))
    if not wheels:
        res_build = subprocess.run(["uv", "build"], cwd=str(repo_root), capture_output=True, text=True)
        if res_build.returncode != 0:
            pytest.fail(f"Failed to build runtime wheel: {res_build.stderr}")
        wheels = sorted(dist_dir.glob("nexus_open_swe_runtime-*.whl"))

    with tempfile.TemporaryDirectory() as tmp_venv:
        venv_path = Path(tmp_venv) / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
        venv_pip = venv_path / "bin" / "pip"
        venv_runtime = venv_path / "bin" / "nexus-open-swe-runtime"
        subprocess.run([str(venv_pip), "install", "--no-cache-dir", str(wheels[-1])], check=True)

        # 2. Dynamic execution script using Nexus-new client code against standalone runtime
        outside_test = Path(tmp_venv) / "outside_run"
        outside_test.mkdir()
        runner_script = outside_test / "verify_client.py"
        runner_script.write_text(f"""\
import sys
import json
from pathlib import Path

sys.path.insert(0, "{nexus_new_root}")
from nexus.services.open_swe_external_intelligence import (
    OpenSWEExternalIntelligenceTransport,
    PROTOCOL_REQUEST_SCHEMA,
    PROTOCOL_RESULT_SCHEMA,
    _runtime_call,
)

# Test 1: Invoke identity directly via subprocess protocol
op_id = "a" * 64
req_payload = {{
    "schema": PROTOCOL_REQUEST_SCHEMA,
    "operation": "identity",
    "operation_id": op_id,
}}
raw_result, stderr, process_started, error = _runtime_call(
    "{venv_runtime}",
    req_payload,
    provider_id="google_genai",
    timeout=30.0,
)
assert error == "", f"Runtime call returned error: {{error}}, stderr: {{stderr}}"
assert process_started is True
assert isinstance(raw_result, dict)
assert raw_result["schema"] == PROTOCOL_RESULT_SCHEMA
assert raw_result["status"] == "IDENTIFIED"
assert raw_result["kind"] == "identity"

# Test 2: Invoke semantic reconcile via legacy client method
dummy_repo = Path("{tmp_venv}") / "dummy_repo"
dummy_repo.mkdir(exist_ok=True)
state_root = Path("{tmp_venv}") / "state_root"

transport = OpenSWEExternalIntelligenceTransport(
    repository_root=dummy_repo,
    model_provider="google_genai",
    model_id="gemini-test",
    executable="{venv_runtime}",
    runtime_state_root=state_root,
)

reconcile_res = transport.reconcile("test prompt")
assert reconcile_res.status == "OPEN_SWE_OUTCOME_UNKNOWN"
assert reconcile_res.outcome_unknown is True
assert reconcile_res.retry_safe is False

print("CROSS_REPO_ACCEPTANCE_OK")
""")

        # Execute using Python with access to Nexus-new dependencies
        res_run = subprocess.run(
            [sys.executable, str(runner_script)],
            cwd=str(outside_test),
            capture_output=True,
            text=True,
        )
        assert res_run.returncode == 0, f"Cross-repo acceptance failed:\\nSTDOUT: {res_run.stdout}\\nSTDERR: {res_run.stderr}"
        assert "CROSS_REPO_ACCEPTANCE_OK" in res_run.stdout
