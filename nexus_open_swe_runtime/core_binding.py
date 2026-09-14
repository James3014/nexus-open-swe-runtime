"""Transport-level validation for the frozen Nexus repository mutation binding.

This module validates and carries the integration projection used by Open SWE v2.
It does not perform Core verification/certification or gain Nexus authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

CORE_BINDING_SCHEMA = "nexus.repository_mutation_binding.v1"
CORE_PROTOCOL_VERSION = "0.1.0-experimental"
CORE_PROTOCOL_CANDIDATE = (
    "James3014/nexus-core#29@aabf2d4d00be4a3a357646be97b4d784d486ccb4"
)
CORE_CHANGE_MANIFEST_SCHEMA = "nexus.core.git-change-manifest.v1-experimental"

_SHA = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT = re.compile(r"^git-commit:([0-9a-f]{40})$")
_TREE = re.compile(r"^git-tree:([0-9a-f]{40})$")


def _canonical(value: Any) -> str:
    # Mirrors nexus-core Candidate #29 public canonical JSON semantics.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _exact(value: Mapping[str, Any], keys: set[str], name: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{name}_keys")


def _text(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > 512
    ):
        raise ValueError(name)
    return value


def _hash_text(value: Any, name: str) -> str:
    text = _text(value, name)
    if not _SHA.fullmatch(text):
        raise ValueError(name)
    return text


def _path(value: Any, name: str) -> str:
    text = _text(value, name)
    if text.startswith("/") or "\\" in text:
        raise ValueError(name)
    parts = text.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(name)
    return text


def _timestamp(value: Any, name: str) -> str:
    text = _text(value, name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(name) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(name)
    return text


def _string_list(value: Any, name: str, *, paths: bool = False) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(name)
    mapper = _path if paths else _text
    result = [mapper(item, name) for item in value]
    if len(set(result)) != len(result):
        raise ValueError(name)
    return result


def acceptance_contract_hash(contract: Mapping[str, Any]) -> str:
    return _hash([
        contract["contract_id"],
        contract["requirements_hash"],
        sorted(contract["required_verifier_ids"]),
        sorted(contract["allowed_paths"]),
        contract["deletion_policy"],
    ])


def parse_core_binding(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("binding_shape")
    binding = dict(value)
    _exact(binding, {
        "schema", "binding_id", "operation_id", "attempt_id", "repository",
        "integration_authority", "capability_discovery", "core", "freshness",
        "binding_hash",
    }, "binding")
    if binding["schema"] != CORE_BINDING_SCHEMA:
        raise ValueError("binding_schema")

    repository = binding["repository"]
    authority = binding["integration_authority"]
    discovery = binding["capability_discovery"]
    core = binding["core"]
    freshness = binding["freshness"]
    if not all(isinstance(item, Mapping) for item in (repository, authority, discovery, core, freshness)):
        raise ValueError("binding_nested_shape")
    repository = dict(repository)
    authority = dict(authority)
    discovery = dict(discovery)
    core = dict(core)
    freshness = dict(freshness)
    _exact(repository, {"canonical_id", "origin", "source_revision", "source_tree", "workspace_identity", "workspace_mode"}, "repository")
    _exact(authority, {"execution_lane", "authority_ref", "authority_hash"}, "authority")
    _exact(discovery, {"required", "receipt_hash", "index_revision"}, "discovery")
    _exact(core, {"protocol_version", "acceptance_contract", "acceptance_contract_hash"}, "core")
    _exact(freshness, {"created_at", "valid_until", "revalidate_before_first_effect"}, "freshness")

    contract = core["acceptance_contract"]
    if not isinstance(contract, Mapping):
        raise ValueError("contract_shape")
    contract = dict(contract)
    _exact(contract, {"contract_id", "requirements_hash", "required_verifier_ids", "allowed_paths", "deletion_policy"}, "contract")
    contract = {
        "contract_id": _text(contract["contract_id"], "contract_id"),
        "requirements_hash": _hash_text(contract["requirements_hash"], "requirements_hash"),
        "required_verifier_ids": _string_list(contract["required_verifier_ids"], "required_verifier_ids"),
        "allowed_paths": _string_list(contract["allowed_paths"], "allowed_paths", paths=True),
        "deletion_policy": contract["deletion_policy"],
    }
    if contract["deletion_policy"] not in {"FORBID", "ALLOW"}:
        raise ValueError("deletion_policy")
    contract_hash = _hash_text(core["acceptance_contract_hash"], "acceptance_contract_hash")
    if contract_hash != acceptance_contract_hash(contract):
        raise ValueError("acceptance_contract_hash")

    source_revision = _text(repository["source_revision"], "source_revision")
    source_tree = _text(repository["source_tree"], "source_tree")
    index_revision = _text(discovery["index_revision"], "index_revision")
    if not _COMMIT.fullmatch(source_revision) or not _TREE.fullmatch(source_tree) or not _COMMIT.fullmatch(index_revision):
        raise ValueError("revision_identity")
    if repository["workspace_mode"] not in {"checkout", "managed_worktree", "target"}:
        raise ValueError("workspace_mode")
    if authority["execution_lane"] not in {"DIRECT_CANONICAL", "DIRECT_DELEGATED", "GOVERNED"}:
        raise ValueError("execution_lane")
    if discovery["required"] is not True or freshness["revalidate_before_first_effect"] is not True:
        raise ValueError("required_binding_flags")
    if core["protocol_version"] != CORE_PROTOCOL_VERSION:
        raise ValueError("core_protocol_version")

    normalized = {
        "schema": CORE_BINDING_SCHEMA,
        "binding_id": _text(binding["binding_id"], "binding_id"),
        "operation_id": _text(binding["operation_id"], "operation_id"),
        "attempt_id": _text(binding["attempt_id"], "attempt_id"),
        "repository": {
            "canonical_id": _text(repository["canonical_id"], "canonical_id"),
            "origin": _text(repository["origin"], "origin"),
            "source_revision": source_revision,
            "source_tree": source_tree,
            "workspace_identity": _hash_text(repository["workspace_identity"], "workspace_identity"),
            "workspace_mode": repository["workspace_mode"],
        },
        "integration_authority": {
            "execution_lane": authority["execution_lane"],
            "authority_ref": _text(authority["authority_ref"], "authority_ref"),
            "authority_hash": _hash_text(authority["authority_hash"], "authority_hash"),
        },
        "capability_discovery": {
            "required": True,
            "receipt_hash": _hash_text(discovery["receipt_hash"], "receipt_hash"),
            "index_revision": index_revision,
        },
        "core": {
            "protocol_version": CORE_PROTOCOL_VERSION,
            "acceptance_contract": contract,
            "acceptance_contract_hash": contract_hash,
        },
        "freshness": {
            "created_at": _timestamp(freshness["created_at"], "created_at"),
            "valid_until": None if freshness["valid_until"] is None else _timestamp(freshness["valid_until"], "valid_until"),
            "revalidate_before_first_effect": True,
        },
    }
    supplied = _hash_text(binding["binding_hash"], "binding_hash")
    if supplied != _hash(normalized):
        raise ValueError("binding_hash")
    return {**normalized, "binding_hash": supplied}


def _canonical_repo(value: str) -> str:
    text = value.strip().removesuffix(".git")
    for prefix in ("https://github.com/", "http://github.com/", "git@github.com:"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return text.strip("/")


def validate_worker_core_binding(
    value: Any,
    *,
    operation_id: str,
    workspace: Path,
    envelope_repository: str,
    expected_base_sha: str,
    observed_source_tree: str,
    allowed_paths: tuple[str, ...],
) -> dict[str, Any]:
    binding = parse_core_binding(value)
    freshness = binding["freshness"]
    now = datetime.now(timezone.utc)
    created_at = datetime.fromisoformat(str(freshness["created_at"]).replace("Z", "+00:00"))
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    if created_at > now.replace(microsecond=0) and (created_at - now).total_seconds() > 60:
        raise ValueError("binding_not_yet_valid")
    valid_until = freshness["valid_until"]
    if valid_until is not None:
        expires = datetime.fromisoformat(str(valid_until).replace("Z", "+00:00"))
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        if expires <= now:
            raise ValueError("binding_expired")
    if binding["operation_id"] != operation_id:
        raise ValueError("operation_binding")
    repository = binding["repository"]
    canonical_repository = _canonical_repo(repository["canonical_id"])
    if canonical_repository != _canonical_repo(envelope_repository):
        raise ValueError("repository_binding")
    if canonical_repository != _canonical_repo(repository["origin"]):
        raise ValueError("origin_binding")
    if repository["source_revision"] != f"git-commit:{expected_base_sha}":
        raise ValueError("base_binding")
    if repository["source_tree"] != f"git-tree:{observed_source_tree}":
        raise ValueError("source_tree_binding")
    contract = binding["core"]["acceptance_contract"]
    if set(contract["allowed_paths"]) != set(allowed_paths):
        raise ValueError("allowed_paths_binding")
    return binding


def repository_mutation_binding_hash(binding_without_hash: Mapping[str, Any]) -> str:
    return _hash(dict(binding_without_hash))


def binding_transport_hash(binding: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(dict(binding)).encode("utf-8")).hexdigest()


def _git(workspace: Path, *args: str, env: Mapping[str, str] | None = None) -> str:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.check_output(
        ["git", "-C", str(workspace), *args],
        text=True,
        env=merged,
    ).rstrip("\n")


def _tree_entry(workspace: Path, tree: str, path: str) -> dict[str, str] | None:
    raw = subprocess.check_output(
        ["git", "-C", str(workspace), "ls-tree", "-z", tree, "--", path]
    )
    if not raw:
        return None
    text = raw.decode("utf-8").rstrip("\x00")
    match = re.match(r"^([0-7]{6})\s+\S+\s+([0-9a-f]{40})\t", text)
    if not match:
        raise ValueError("tree_entry")
    return {"mode": match.group(1), "oid": match.group(2)}


def physical_changeset(workspace: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize exact current bytes as a deterministic Git tree for downstream Core verify."""
    source_tree_ref = str(binding["repository"]["source_tree"])
    source_tree = _TREE.fullmatch(source_tree_ref)
    if source_tree is None:
        raise ValueError("source_tree")
    source_tree_oid = source_tree.group(1)
    current_head = _git(workspace, "rev-parse", "HEAD")
    status = _git(workspace, "status", "--porcelain=v1", "--untracked-files=all")
    dirty = bool(status.strip())
    if dirty:
        with tempfile.TemporaryDirectory(prefix="open-swe-core-index-") as tmp:
            index_path = str(Path(tmp) / "index")
            env = {"GIT_INDEX_FILE": index_path}
            _git(workspace, "read-tree", "HEAD", env=env)
            _git(workspace, "add", "-A", "--", ".", env=env)
            target_tree = _git(workspace, "write-tree", env=env)
    else:
        target_tree = _git(workspace, "rev-parse", "HEAD^{tree}")

    raw = subprocess.check_output([
        "git", "-C", str(workspace), "diff-tree", "--no-commit-id", "--name-status",
        "-r", "-z", "--no-renames", source_tree_oid, target_tree,
    ])
    fields = [field.decode("utf-8") for field in raw.split(b"\x00") if field]
    if len(fields) % 2:
        raise ValueError("diff_tree")
    entries: list[dict[str, Any]] = []
    for index in range(0, len(fields), 2):
        status_code, path = fields[index], fields[index + 1]
        before = _tree_entry(workspace, source_tree_oid, path)
        after = _tree_entry(workspace, target_tree, path)
        change_type = "ADD" if status_code.startswith("A") else "DELETE" if status_code.startswith("D") else "MODIFY"
        entries.append({
            "path": path,
            "change_type": change_type,
            "before_oid": before["oid"] if before else None,
            "after_oid": after["oid"] if after else None,
            "before_mode": before["mode"] if before else None,
            "after_mode": after["mode"] if after else None,
        })
    entries.sort(key=lambda item: item["path"])
    if not entries:
        raise ValueError("changeset_empty")
    source_ref = f"git-tree:{source_tree_oid}"
    target_ref = f"git-tree:{target_tree}"
    manifest = {"source_tree": source_ref, "target_tree": target_ref, "entries": entries}
    diff_hash = _hash([
        CORE_CHANGE_MANIFEST_SCHEMA,
        source_ref,
        target_ref,
        [[entry[key] for key in (
            "path", "change_type", "before_oid", "after_oid", "before_mode", "after_mode"
        )] for entry in entries],
    ])
    changed = [entry["path"] for entry in entries]
    deleted = [entry["path"] for entry in entries if entry["change_type"] == "DELETE"]
    allowed = set(binding["core"]["acceptance_contract"]["allowed_paths"])
    scope_escape = [path for path in changed if path not in allowed]
    deletion_violation = (
        binding["core"]["acceptance_contract"]["deletion_policy"] == "FORBID" and bool(deleted)
    )
    return {
        "source_revision": binding["repository"]["source_revision"],
        "source_tree": source_ref,
        "target_revision": f"git-tree:{target_tree}" if dirty else f"git-commit:{current_head}",
        "target_tree": target_ref,
        "changed_paths": changed,
        "deleted_paths": deleted,
        "scope_escape_paths": scope_escape,
        "deletion_violation": deletion_violation,
        "diff_hash": diff_hash,
        "change_manifest": manifest,
    }
