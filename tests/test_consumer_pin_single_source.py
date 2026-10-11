"""Guard: the Nexus-new consumer pin has exactly one literal source.

The single source is ``expected_identity`` in ``.nexus-core/config.toml``
(nexus-certify requires a literal there and rejects unknown top-level keys).
Everything else must derive the revision from it.  Every tracked file other
than config.toml is scanned, with no exemptions.
"""

from __future__ import annotations

import importlib.util
import subprocess
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / ".nexus-core" / "config.toml"
HELPER = REPO_ROOT / "scripts" / "ci" / "consumer_pin.py"


def _helper():
    spec = importlib.util.spec_from_file_location("consumer_pin", HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
        capture_output=True,
        check=True,
    ).stdout
    return [REPO_ROOT / n.decode() for n in out.split(b"\0") if n]


def test_pin_literal_appears_only_in_config_toml() -> None:
    sha = _helper().read_consumer_pin()
    offenders = []
    for path in _tracked_files():
        if path.resolve() == CONFIG.resolve() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if sha in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == [], f"literal consumer pin outside single source: {offenders}"
    assert CONFIG.read_text(encoding="utf-8").count(sha) == 1


def test_config_commands_derive_pin_from_helper() -> None:
    config = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    commands = [config["materials"][0]["observe_command"]]
    commands += [
        v["command"] for v in config["verifiers"] if v["id"] in {"cross-repo-acceptance", "full-tests"}
    ]
    assert len(commands) == 3
    for command in commands:
        assert "scripts/ci/consumer_pin.py" in command[-1]


def test_ci_workflow_reads_pin_instead_of_embedding_it() -> None:
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "scripts/ci/consumer_pin.py" in ci
    assert "origin \"$consumer_sha\"" in ci
    assert '= "$consumer_sha"' in ci


def test_changing_single_source_changes_all_consumers(tmp_path: Path) -> None:
    helper = _helper()
    new_sha = "0123456789abcdef0123456789abcdef01234567"
    old_sha = helper.read_consumer_pin()
    cfg = tmp_path / "config.toml"
    cfg.write_text(CONFIG.read_text(encoding="utf-8").replace(old_sha, new_sha), encoding="utf-8")
    assert helper.read_consumer_pin(cfg) == new_sha
    done = subprocess.run(
        ["python3", str(HELPER)], capture_output=True, text=True, check=True
    )
    assert done.stdout.strip() == old_sha
