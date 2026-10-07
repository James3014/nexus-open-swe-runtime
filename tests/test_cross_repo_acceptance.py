"""Portable cross-repository compatibility contract.

The test is local opt-in (set ``NEXUS_NEW_CONSUMER_ROOT``) and CI-required
(the workflow checks out the pinned consumer and sets
``NEXUS_NEW_CONSUMER_REQUIRED=1``).  It never invokes a model provider: the
consumer performs runtime identity and semantic-reconcile read-back only.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

EXPECTED_CONSUMER_REVISION = "2999d62e0fb463646a12a659eefeee0aaf3367ea"
CONSUMER_ROOT_ENV = "NEXUS_NEW_CONSUMER_ROOT"
CONSUMER_REQUIRED_ENV = "NEXUS_NEW_CONSUMER_REQUIRED"
CONSUMER_CLIENT = Path("nexus/services/open_swe_external_intelligence.py")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _consumer_is_required() -> bool:
    return os.environ.get(CONSUMER_REQUIRED_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _require_consumer_checkout(
    root: Path, *, expected_revision: str = EXPECTED_CONSUMER_REVISION
) -> Path:
    if not root.is_dir() or not (root / CONSUMER_CLIENT).is_file():
        raise AssertionError("CROSS_REPO_REQUIRED_CONSUMER_MISSING")

    status = subprocess.run(
        ["git", "-C", str(root), "status", "--short"],
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode != 0:
        raise AssertionError("CROSS_REPO_REQUIRED_CONSUMER_MISSING")
    if status.stdout.strip():
        raise AssertionError("CROSS_REPO_CONSUMER_DIRTY")

    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    observed = revision.stdout.strip()
    if revision.returncode != 0 or HEX40.fullmatch(observed) is None:
        raise AssertionError("CROSS_REPO_CONSUMER_REVISION_MISMATCH")
    if observed != expected_revision:
        raise AssertionError(
            f"CROSS_REPO_CONSUMER_REVISION_MISMATCH: expected {expected_revision}, observed {observed}"
        )
    return root


def _configured_consumer_checkout() -> Path:
    configured = os.environ.get(CONSUMER_ROOT_ENV, "").strip()
    if not configured:
        if _consumer_is_required():
            raise AssertionError("CROSS_REPO_REQUIRED_CONSUMER_MISSING")
        pytest.skip(
            "cross-repo contract is local opt-in; set NEXUS_NEW_CONSUMER_ROOT to enable"
        )
    return _require_consumer_checkout(Path(configured).expanduser().resolve())


def _build_fresh_wheel(repo_root: Path, build_root: Path) -> Path:
    output_dir = build_root / "dist"
    cache_dir = build_root / "uv-cache"
    output_dir.mkdir()
    build_env = {
        **os.environ,
        "UV_CACHE_DIR": str(cache_dir),
        "PIP_CACHE_DIR": str(cache_dir / "pip"),
    }
    uv_result = subprocess.run(
        ["uv", "build", "--out-dir", str(output_dir)],
        cwd=str(repo_root),
        env=build_env,
        capture_output=True,
        text=True,
        check=False,
    )
    if uv_result.returncode != 0:
        # Some macOS hosts cannot initialize uv's system-configuration
        # resolver inside a restricted test process.  Keep the build fresh
        # and source-bound with the standard PEP 517 frontend as a fallback.
        pip_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-cache-dir",
                "--no-build-isolation",
                "--wheel-dir",
                str(output_dir),
                str(repo_root),
            ],
            cwd=str(repo_root),
            env=build_env,
            capture_output=True,
            text=True,
            check=False,
        )
        if pip_result.returncode != 0:
            raise AssertionError(
                "CROSS_REPO_RUNTIME_WHEEL_BUILD_FAILED: "
                f"uv={uv_result.stderr}; pip={pip_result.stderr}"
            )
    wheels = sorted(output_dir.glob("nexus_open_swe_runtime-*.whl"))
    if len(wheels) != 1:
        raise AssertionError("CROSS_REPO_RUNTIME_WHEEL_MISSING_OR_AMBIGUOUS")
    return wheels[0]


def _venv_executable(venv_root: Path, name: str) -> Path:
    directory = "Scripts" if os.name == "nt" else "bin"
    return venv_root / directory / name


def _install_fresh_wheel(wheel: Path, build_root: Path) -> Path:
    venv_root = build_root / "venv"
    result = subprocess.run(
        [sys.executable, "-m", "venv", str(venv_root)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(f"CROSS_REPO_RUNTIME_VENV_FAILED: {result.stderr}")
    pip = _venv_executable(venv_root, "pip")
    result = subprocess.run(
        [str(pip), "install", "--disable-pip-version-check", "--no-cache-dir", str(wheel)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(f"CROSS_REPO_RUNTIME_WHEEL_INSTALL_FAILED: {result.stderr}")
    executable = _venv_executable(venv_root, "nexus-open-swe-runtime")
    if not executable.is_file():
        raise AssertionError("CROSS_REPO_RUNTIME_EXECUTABLE_MISSING")
    return executable


def _runtime_identity(executable: Path, outside_root: Path) -> dict[str, object]:
    result = subprocess.run(
        [str(executable), "--identity"],
        cwd=str(outside_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(f"CROSS_REPO_RUNTIME_IDENTITY_FAILED: {result.stderr}")
    try:
        identity = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError("CROSS_REPO_RUNTIME_IDENTITY_INVALID") from exc
    if not isinstance(identity, dict):
        raise AssertionError("CROSS_REPO_RUNTIME_IDENTITY_INVALID")
    return identity


def _import_consumer(consumer_root: Path):
    original_path = sys.path.copy()
    try:
        sys.path.insert(0, str(consumer_root))
        module = importlib.import_module("nexus.services.open_swe_external_intelligence")
        _assert_consumer_module_origin(module, consumer_root)
        return module
    except Exception as exc:
        if isinstance(exc, AssertionError):
            raise
        raise AssertionError(f"CROSS_REPO_CONSUMER_IMPORT_FAILED: {exc}") from exc
    finally:
        sys.path[:] = original_path


def _assert_consumer_module_origin(module: object, consumer_root: Path) -> None:
    expected = (consumer_root / CONSUMER_CLIENT).resolve()
    actual_value = getattr(module, "__file__", None)
    actual = Path(str(actual_value)).resolve() if actual_value else None
    if actual != expected:
        raise AssertionError(
            "CROSS_REPO_CONSUMER_IMPORT_ORIGIN_MISMATCH: "
            f"expected {expected}, observed {actual}"
        )


def test_cross_repo_consumer_revision_mismatch_is_explicit(tmp_path: Path):
    consumer_root = tmp_path / "consumer"
    client = consumer_root / CONSUMER_CLIENT
    client.parent.mkdir(parents=True)
    client.write_text("# contract fixture\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(consumer_root)], check=True)
    subprocess.run(["git", "-C", str(consumer_root), "add", str(CONSUMER_CLIENT)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(consumer_root),
            "-c",
            "user.name=contract-test",
            "-c",
            "user.email=contract-test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )

    with pytest.raises(AssertionError, match="CROSS_REPO_CONSUMER_REVISION_MISMATCH"):
        _require_consumer_checkout(consumer_root)


def test_cross_repo_required_consumer_missing_is_explicit(tmp_path: Path):
    with pytest.raises(AssertionError, match="CROSS_REPO_REQUIRED_CONSUMER_MISSING"):
        _require_consumer_checkout(tmp_path / "missing")


def test_cross_repo_consumer_wrong_import_origin_is_explicit(tmp_path: Path):
    expected_root = tmp_path / "expected"
    foreign_module = tmp_path / "foreign" / "nexus" / "services" / "open_swe_external_intelligence.py"
    (expected_root / CONSUMER_CLIENT).parent.mkdir(parents=True)
    foreign_module.parent.mkdir(parents=True)
    (expected_root / CONSUMER_CLIENT).write_text("# expected\n", encoding="utf-8")
    foreign_module.write_text("# foreign\n", encoding="utf-8")

    class ForeignModule:
        __file__ = str(foreign_module)

    with pytest.raises(AssertionError, match="CROSS_REPO_CONSUMER_IMPORT_ORIGIN_MISMATCH"):
        _assert_consumer_module_origin(ForeignModule(), expected_root)


@pytest.fixture(scope="module")
def _fresh_runtime_artifact() -> tuple[Path, str, Path, Path, dict[str, object], Path]:
    consumer_root = _configured_consumer_checkout()
    repo_root = Path(__file__).parent.parent.resolve()
    temp_parent = "/private/tmp" if Path("/private/tmp").is_dir() else None
    with tempfile.TemporaryDirectory(
        prefix="nexus-open-swe-cross-repo-", dir=temp_parent
    ) as raw_tmp:
        build_root = Path(raw_tmp)
        wheel = _build_fresh_wheel(repo_root, build_root)
        executable = _install_fresh_wheel(wheel, build_root)
        outside_root = build_root / "outside"
        outside_root.mkdir()
        identity = _runtime_identity(executable, outside_root)
        assert identity["schema"] == "nexus.open_swe_runtime.result.v1"
        assert identity["status"] == "IDENTIFIED"
        assert identity["kind"] == "identity"
        assert identity["distribution_name"] == "nexus-open-swe-runtime"
        artifact = identity.get("artifact_identity")
        assert isinstance(artifact, dict)
        module_file = Path(str(artifact.get("module_file", ""))).resolve()
        module_hash = str(artifact.get("module_sha256", "")).lower()
        assert module_file.is_file()
        assert HEX64.fullmatch(module_hash)
        assert module_hash == hashlib.sha256(module_file.read_bytes()).hexdigest()
        assert module_hash == hashlib.sha256(
            (repo_root / "nexus_open_swe_runtime" / "cli.py").read_bytes()
        ).hexdigest()
        assert module_file.is_relative_to(build_root / "venv")
        assert not module_file.is_relative_to(repo_root)
        yield executable, module_hash, build_root, outside_root, identity, consumer_root


def test_cross_repo_legacy_client_compatibility_acceptance(_fresh_runtime_artifact):
    executable, module_hash, build_root, outside_root, identity, consumer_root = _fresh_runtime_artifact

    consumer = _import_consumer(consumer_root)
    transport = consumer.OpenSWEExternalIntelligenceTransport(
        repository_root=outside_root,
        model_provider="google_genai",
        model_id="cross-repo-contract",
        executable=str(executable),
        expected_artifact_sha256=module_hash,
        runtime_state_root=build_root / "runtime-state",
        timeout=30.0,
    )
    result = transport.reconcile("cross-repo compatibility contract")

    assert identity["distribution_name"] == "nexus-open-swe-runtime"
    assert result.status == "OPEN_SWE_OUTCOME_UNKNOWN"
    assert result.outcome_unknown is True
    assert result.retry_safe is False


def test_cross_repo_runtime_artifact_binding_requires_hash(tmp_path: Path):
    consumer_root = _configured_consumer_checkout()
    consumer = _import_consumer(consumer_root)
    with pytest.raises(
        consumer.OpenSWEExternalIntelligenceError,
        match="OPEN_SWE_EXPECTED_ARTIFACT_HASH_REQUIRED",
    ):
        consumer.OpenSWEExternalIntelligenceTransport(
            repository_root=tmp_path,
            model_provider="google_genai",
            model_id="cross-repo-contract",
            executable=str(tmp_path / "runtime"),
            expected_artifact_sha256="",
        )

def test_cross_repo_runtime_artifact_binding_rejects_wrong_hash(_fresh_runtime_artifact):
    executable, _module_hash, build_root, outside_root, _identity, consumer_root = _fresh_runtime_artifact
    consumer = _import_consumer(consumer_root)
    transport = consumer.OpenSWEExternalIntelligenceTransport(
        repository_root=outside_root,
        model_provider="google_genai",
        model_id="cross-repo-contract",
        executable=str(executable),
        expected_artifact_sha256="0" * 64,
        runtime_state_root=build_root / "wrong-hash-state",
        timeout=30.0,
    )
    result = transport.reconcile("cross-repo wrong artifact hash")
    assert result.status == "OPEN_SWE_RUNTIME_IDENTITY_FAILED"
    assert result.outcome_unknown is True
    assert result.retry_safe is False
