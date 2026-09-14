from __future__ import annotations

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "nexus_open_swe_runtime" / "core_binding.py"
SPEC = importlib.util.spec_from_file_location("open_swe_core_binding_test_target", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
core_binding = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core_binding)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _contract(*, allowed_paths: list[str], deletion_policy: str = "FORBID") -> dict:
    value = {
        "contract_id": "open-swe-core-binding-test",
        "requirements_hash": "sha256:" + "1" * 64,
        "required_verifier_ids": ["unit"],
        "allowed_paths": allowed_paths,
        "deletion_policy": deletion_policy,
    }
    return value


def _binding(repo: Path, operation_id: str, *, allowed_paths: list[str], deletion_policy: str = "FORBID") -> dict:
    head = _git(repo, "rev-parse", "HEAD")
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    contract = _contract(allowed_paths=allowed_paths, deletion_policy=deletion_policy)
    base = {
        "schema": core_binding.CORE_BINDING_SCHEMA,
        "binding_id": "binding-open-swe-unit",
        "operation_id": operation_id,
        "attempt_id": "attempt-open-swe-unit",
        "repository": {
            "canonical_id": "James3014/example",
            "origin": "https://github.com/James3014/example.git",
            "source_revision": f"git-commit:{head}",
            "source_tree": f"git-tree:{tree}",
            "workspace_identity": "sha256:" + "2" * 64,
            "workspace_mode": "target",
        },
        "integration_authority": {
            "execution_lane": "DIRECT_DELEGATED",
            "authority_ref": "James3014/example#1",
            "authority_hash": "sha256:" + "3" * 64,
        },
        "capability_discovery": {
            "required": True,
            "receipt_hash": "sha256:" + "4" * 64,
            "index_revision": "git-commit:" + "5" * 40,
        },
        "core": {
            "protocol_version": core_binding.CORE_PROTOCOL_VERSION,
            "acceptance_contract": contract,
            "acceptance_contract_hash": core_binding.acceptance_contract_hash(contract),
        },
        "freshness": {
            "created_at": "2026-09-14T00:00:00Z",
            "valid_until": None,
            "revalidate_before_first_effect": True,
        },
    }
    return {**base, "binding_hash": core_binding.repository_mutation_binding_hash(base)}


class CoreBindingProtocolTests(unittest.TestCase):
    def test_unicode_acceptance_contract_matches_core_candidate_vector(self) -> None:
        value = {
            "contract_id": "ac-雪-1",
            "requirements_hash": "sha256:" + "c" * 64,
            "required_verifier_ids": ["lint", "測試"],
            "allowed_paths": ["src/雪.py"],
            "deletion_policy": "ALLOW",
        }
        self.assertEqual(
            core_binding.acceptance_contract_hash(value),
            "sha256:21face1a76514a320f54660053e4c1e9e51f54ab31f346a599db88a2107174d4",
        )

    def test_binding_rejects_source_tree_substitution(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            _git(repo, "init", "--initial-branch=main")
            _git(repo, "config", "user.email", "test@example.com")
            _git(repo, "config", "user.name", "Test User")
            (repo / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
            _git(repo, "add", "a.py")
            _git(repo, "commit", "-m", "base")
            binding = _binding(repo, "operation-1", allowed_paths=["a.py"])
            head = _git(repo, "rev-parse", "HEAD")
            tree = _git(repo, "rev-parse", "HEAD^{tree}")
            parsed = core_binding.validate_worker_core_binding(
                binding,
                operation_id="operation-1",
                workspace=repo,
                envelope_repository="James3014/example",
                expected_base_sha=head,
                observed_source_tree=tree,
                allowed_paths=("a.py",),
            )
            self.assertEqual(parsed["binding_hash"], binding["binding_hash"])
            with self.assertRaisesRegex(ValueError, "source_tree_binding"):
                core_binding.validate_worker_core_binding(
                    binding,
                    operation_id="operation-1",
                    workspace=repo,
                    envelope_repository="James3014/example",
                    expected_base_sha=head,
                    observed_source_tree="f" * 40,
                    allowed_paths=("a.py",),
                )

    def test_binding_rejects_origin_drift_and_timestamps_require_timezone(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            _git(repo, "init", "--initial-branch=main")
            _git(repo, "config", "user.email", "test@example.com")
            _git(repo, "config", "user.name", "Test User")
            (repo / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
            _git(repo, "add", "a.py")
            _git(repo, "commit", "-m", "base")
            binding = _binding(repo, "operation-origin", allowed_paths=["a.py"])
            head = _git(repo, "rev-parse", "HEAD")
            tree = _git(repo, "rev-parse", "HEAD^{tree}")

            drifted = {**binding, "repository": {**binding["repository"], "origin": "https://github.com/James3014/other.git"}}
            drifted_base = {key: value for key, value in drifted.items() if key != "binding_hash"}
            drifted["binding_hash"] = core_binding.repository_mutation_binding_hash(drifted_base)
            with self.assertRaisesRegex(ValueError, "origin_binding"):
                core_binding.validate_worker_core_binding(
                    drifted,
                    operation_id="operation-origin",
                    workspace=repo,
                    envelope_repository="James3014/example",
                    expected_base_sha=head,
                    observed_source_tree=tree,
                    allowed_paths=("a.py",),
                )

            naive = {**binding, "freshness": {**binding["freshness"], "created_at": "2026-09-14T00:00:00"}}
            naive_base = {key: value for key, value in naive.items() if key != "binding_hash"}
            naive["binding_hash"] = core_binding.repository_mutation_binding_hash(naive_base)
            with self.assertRaisesRegex(ValueError, "created_at"):
                core_binding.parse_core_binding(naive)

    def test_allowed_path_order_is_set_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            _git(repo, "init", "--initial-branch=main")
            _git(repo, "config", "user.email", "test@example.com")
            _git(repo, "config", "user.name", "Test User")
            (repo / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
            _git(repo, "add", "a.py")
            _git(repo, "commit", "-m", "base")
            binding = _binding(repo, "operation-order", allowed_paths=["a.py", "new.py"])
            parsed = core_binding.validate_worker_core_binding(
                binding,
                operation_id="operation-order",
                workspace=repo,
                envelope_repository="James3014/example",
                expected_base_sha=_git(repo, "rev-parse", "HEAD"),
                observed_source_tree=_git(repo, "rev-parse", "HEAD^{tree}"),
                allowed_paths=("new.py", "a.py"),
            )
            self.assertEqual(parsed["binding_hash"], binding["binding_hash"])

    def test_physical_changeset_materializes_uncommitted_tree(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            _git(repo, "init", "--initial-branch=main")
            _git(repo, "config", "user.email", "test@example.com")
            _git(repo, "config", "user.name", "Test User")
            (repo / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
            _git(repo, "add", "a.py")
            _git(repo, "commit", "-m", "base")
            binding = _binding(repo, "operation-2", allowed_paths=["a.py", "new.py"])
            source_head = _git(repo, "rev-parse", "HEAD")
            source_tree = _git(repo, "rev-parse", "HEAD^{tree}")

            (repo / "a.py").write_text("VALUE = 2\n", encoding="utf-8")
            (repo / "new.py").write_text("ADDED = True\n", encoding="utf-8")
            result = core_binding.physical_changeset(repo, binding)

            self.assertEqual(result["source_revision"], f"git-commit:{source_head}")
            self.assertEqual(result["source_tree"], f"git-tree:{source_tree}")
            self.assertTrue(result["target_revision"].startswith("git-tree:"))
            self.assertEqual(result["changed_paths"], ["a.py", "new.py"])
            self.assertEqual(result["deleted_paths"], [])
            self.assertEqual(result["scope_escape_paths"], [])
            self.assertFalse(result["deletion_violation"])
            self.assertEqual(
                [row["change_type"] for row in result["change_manifest"]["entries"]],
                ["MODIFY", "ADD"],
            )
            self.assertTrue(result["diff_hash"].startswith("sha256:"))
            self.assertEqual(_git(repo, "rev-parse", "HEAD"), source_head)

    def test_zero_change_is_not_a_physical_changeset(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = Path(raw)
            _git(repo, "init", "--initial-branch=main")
            _git(repo, "config", "user.email", "test@example.com")
            _git(repo, "config", "user.name", "Test User")
            (repo / "a.py").write_text("VALUE = 1\n", encoding="utf-8")
            _git(repo, "add", "a.py")
            _git(repo, "commit", "-m", "base")
            binding = _binding(repo, "operation-3", allowed_paths=["a.py"])
            with self.assertRaisesRegex(ValueError, "changeset_empty"):
                core_binding.physical_changeset(repo, binding)


if __name__ == "__main__":
    unittest.main()
