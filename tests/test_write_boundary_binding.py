from __future__ import annotations

from pathlib import Path

import pytest

from nexus_open_swe_runtime import cli
from nexus_open_swe_runtime.tool_projection import ValidatedToolProjection


def _projection(write_paths: object) -> ValidatedToolProjection:
    return ValidatedToolProjection(
        authorization_hash="a" * 64,
        projection_hash="b" * 64,
        operation_id="op",
        attempt_id="attempt",
        provider="provider",
        backend_id="nexus-open-swe-runtime",
        selected_tools=("edit_file",),
        selected_effects={"filesystem": {"write_paths": write_paths}},
    )


def test_allowed_paths_within_authorized_write_paths_pass() -> None:
    cli._assert_projected_write_paths(_projection(["src"]), ("src/a.py",))
    cli._assert_projected_write_paths(None, ("anything.py",))


@pytest.mark.parametrize(
    ("write_paths", "allowed"),
    [
        (["a.py"], ("b.py",)),
        (["a.py"], ("a.py", "b.py")),
        (["src/a.py"], ("src",)),
        ([], ("a.py",)),
        (None, ("a.py",)),
    ],
)
def test_allowed_paths_wider_than_authorization_fail_closed(
    write_paths: object, allowed: tuple[str, ...]
) -> None:
    with pytest.raises(cli.RuntimeErrorBounded, match="OPEN_SWE_TOOL_PROJECTION_EFFECTS_WIDENED"):
        cli._assert_projected_write_paths(_projection(write_paths), allowed)


def test_scoped_backend_rejects_symlinked_parent(tmp_path: Path) -> None:
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / "src").symlink_to(tmp_path / ".github" / "workflows", target_is_directory=True)
    backend = cli.ScopedRepairBackend(object(), tmp_path, ("src/a.py",))
    with pytest.raises(PermissionError, match="OPEN_SWE_MUTATION_PATH_FORBIDDEN"):
        backend._authorize("src/a.py")


def test_scoped_backend_rejects_symlinked_leaf_inside_workspace(tmp_path: Path) -> None:
    (tmp_path / "real.py").write_text("x\n")
    (tmp_path / "a.py").symlink_to(tmp_path / "real.py")
    backend = cli.ScopedRepairBackend(object(), tmp_path, ("a.py",))
    with pytest.raises(PermissionError, match="OPEN_SWE_MUTATION_PATH_FORBIDDEN"):
        backend._authorize("a.py")


def test_scoped_backend_allows_plain_authorized_path(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    cli.ScopedRepairBackend(object(), tmp_path, ("src/a.py",))._authorize("src/a.py")


def test_regular_file_ancestor_is_a_violation_not_a_crash(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x\n")
    assert cli._workspace_path_has_symlink(tmp_path, "a.py/b.py") is True
    backend = cli.ScopedRepairBackend(object(), tmp_path, ("a.py/b.py",))
    with pytest.raises(PermissionError, match="OPEN_SWE_MUTATION_PATH_FORBIDDEN"):
        backend._authorize("a.py/b.py")
