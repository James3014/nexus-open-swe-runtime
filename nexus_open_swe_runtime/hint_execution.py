"""Additive retrieval-hint execution evidence (#68).

Execution-side seam that accepts externally prepared, revision-bound additive
retrieval hints, preserves worker search/read recovery, and emits bounded
derived evidence about whether hints reduced exploration. It never computes a
retrieval policy, never removes authorized tools, and never treats Top-K as
complete.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from langchain_core.callbacks import BaseCallbackHandler

HINT_EXECUTION_SCHEMA = "nexus.open_swe_runtime.hint_execution_evidence.v1"
HINT_EXECUTION_CLAIM_CEILING = "OPEN_SWE_RETRIEVAL_HINT_EXECUTION_EVIDENCE_ONLY"
SEARCH_TOOLS = ("search", "grep", "read", "read_file", "ls", "glob", "find")


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _hash(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def validate_hint_input(hints):
    blockers = []
    if hints is None:
        return ["hint_unavailable_canonical_execution"]
    if not isinstance(hints, Mapping):
        return ["hint_input_not_mapping"]
    for key in ("repository", "revision", "query_evidence_hash"):
        if not isinstance(hints.get(key), str) or not hints[key].strip():
            blockers.append(f"hint_missing_{key}")
    candidates = hints.get("candidates", [])
    if candidates is not None and not isinstance(candidates, (list, tuple)):
        blockers.append("hint_candidates_not_list")
    elif any(not isinstance(e if isinstance(e, str) else e.get("candidate_ref") if isinstance(e, Mapping) else None, str) for e in (candidates or [])):
        blockers.append("hint_candidate_invalid")
    return sorted(set(blockers))


def bind_execution_hints(*, expected_repository, expected_revision, hint_input, authorized_tools):
    expected_repository = _text(expected_repository, "expected_repository")
    expected_revision = _text(expected_revision, "expected_revision")
    authorized = tuple(sorted({str(t).strip() for t in (authorized_tools or []) if str(t).strip()}))
    blockers = validate_hint_input(hint_input)
    evidence = dict(hint_input) if isinstance(hint_input, Mapping) else {}
    if not blockers:
        if str(evidence.get("repository") or "").strip() != expected_repository:
            blockers.append("hint_foreign_repository")
        actual_revision = str(evidence.get("revision") or "").strip().removeprefix("git-commit:")
        expected_revision = expected_revision.removeprefix("git-commit:")
        if actual_revision != expected_revision:
            blockers.append("hint_stale_revision")
    if blockers:
        return {"bound": False, "blockers": sorted(set(blockers)), "hints": [], "authorized_tools": list(authorized), "hint_identity": {}}
    refs = []
    for entry in (evidence.get("candidates") or []):
        ref = entry if isinstance(entry, str) else (entry.get("candidate_ref") if isinstance(entry, Mapping) else "")
        if isinstance(ref, str) and ref.strip():
            refs.append(ref.strip())
    identity = {"repository": str(evidence.get("repository")), "revision": str(evidence.get("revision")), "query_evidence_hash": str(evidence.get("query_evidence_hash")), "retriever_policy": str(evidence.get("retriever_policy") or "")}
    identity["binding_hash"] = _hash(identity)
    identity["candidate_hash"] = _hash(refs)
    return {"bound": True, "blockers": [], "hints": refs, "authorized_tools": list(authorized), "hint_identity": identity}


def build_hint_execution_receipt(*, bound, hint_identity=None, initial_hints=None, hinted_reads=(), discovered_outside=(), tool_calls=None, recovery_events=(), widened=False, context_failures=()):
    hinted = [str(r).strip() for r in (hinted_reads or []) if str(r).strip()]
    discovered = [str(r).strip() for r in (discovered_outside or []) if str(r).strip()]
    calls = []
    for entry in (tool_calls or []):
        if isinstance(entry, Mapping) and str(entry.get("tool") or "").strip():
            count = entry.get("count")
            if type(count) is not int or count < 0:
                raise ValueError("tool count must be an observed non-negative integer")
            calls.append({"tool": str(entry["tool"]).strip(), "count": count})
    search_calls = None if tool_calls is None else sum(c["count"] for c in calls if c["tool"] in SEARCH_TOOLS)
    initial = [str(r).strip() for r in (hinted if initial_hints is None else initial_hints) if str(r).strip()]
    receipt = {"schema": HINT_EXECUTION_SCHEMA, "bound": bool(bound), "hint_identity": dict(hint_identity or {}), "hinted_total": len(set(initial)) if bound else 0, "hinted_read": sorted(set(hinted)), "discovered_outside_hints": sorted(set(discovered)), "tool_calls": calls, "search_tool_calls": search_calls, "widened_beyond_hints": bool(widened or discovered), "recovery_events": [str(e) for e in (recovery_events or [])], "context_failures": [str(e) for e in (context_failures or [])], "claim_ceiling": HINT_EXECUTION_CLAIM_CEILING}
    receipt["missing"] = ["tool_calls", "search_tool_calls"] if tool_calls is None else []
    receipt["receipt_hash"] = _hash({k: v for k, v in receipt.items() if k != "receipt_hash"})
    return receipt


def validate_hint_execution_receipt(receipt):
    blockers = []
    if not isinstance(receipt, Mapping):
        return ["receipt_not_mapping"]
    if receipt.get("schema") != HINT_EXECUTION_SCHEMA:
        blockers.append("invalid_receipt_schema")
    if receipt.get("claim_ceiling") != HINT_EXECUTION_CLAIM_CEILING:
        blockers.append("invalid_claim_ceiling")
    expected = dict(receipt)
    digest = expected.pop("receipt_hash", "")
    if _hash(expected) != digest:
        blockers.append("receipt_hash_mismatch")
    return sorted(set(blockers))



def _candidate_paths(refs):
    paths = set()
    for raw in refs or ():
        value = str(raw).strip().lstrip("/")
        if not value:
            continue
        paths.add(value)
        if ":" in value:
            prefix = value.split(":", 1)[0].strip().lstrip("/")
            if prefix:
                paths.add(prefix)
    return paths


def _result_paths(value):
    values = value if isinstance(value, (list, tuple)) else None
    if values is None and isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, (list, tuple)):
                values = parsed
    if values is None:
        values = [value]
    result = []
    for item in values:
        if not isinstance(item, str):
            continue
        text = item.strip().lstrip("/")
        if text and not text.endswith("/") and PurePosixPath(text).name:
            result.append(text)
    return result


def add_recovery_observation(receipt, observation):
    """Append replay observations without discarding the original execution evidence."""
    if not isinstance(receipt, Mapping) or validate_hint_execution_receipt(receipt):
        return receipt
    if not isinstance(observation, Mapping):
        return receipt
    counts = Counter()
    for source in (receipt.get("tool_calls") or [], observation.get("tool_calls") or []):
        for entry in source:
            if isinstance(entry, Mapping) and isinstance(entry.get("tool"), str) and type(entry.get("count")) is int:
                counts[entry["tool"]] += entry["count"]
    merged = dict(receipt)
    merged["hinted_read"] = sorted(set(receipt.get("hinted_read") or []) | set(observation.get("hinted_read") or []))
    merged["discovered_outside_hints"] = sorted(set(receipt.get("discovered_outside_hints") or []) | set(observation.get("discovered_outside_hints") or []))
    merged["tool_calls"] = [{"tool": tool, "count": count} for tool, count in sorted(counts.items())]
    merged["search_tool_calls"] = sum(entry["count"] for entry in merged["tool_calls"] if entry["tool"] in SEARCH_TOOLS)
    merged["widened_beyond_hints"] = bool(merged["discovered_outside_hints"] or receipt.get("widened_beyond_hints") or observation.get("widened_beyond_hints"))
    merged["recovery_events"] = list(receipt.get("recovery_events") or []) + ["worker_reconcile_replay"] + list(observation.get("recovery_events") or [])
    merged["context_failures"] = list(receipt.get("context_failures") or []) + list(observation.get("context_failures") or [])
    merged["missing"] = sorted(set(receipt.get("missing") or []) & set(observation.get("missing") or []))
    merged["receipt_hash"] = _hash({k: v for k, v in merged.items() if k != "receipt_hash"})
    return merged


class HintExecutionObserver(BaseCallbackHandler):
    """Observe execution without granting or narrowing tool authority."""

    def __init__(self, bound, root=None):
        super().__init__()
        self.bound = dict(bound or {})
        self.root = Path(root).resolve() if root is not None else None
        self._candidates = _candidate_paths(self.bound.get("hints", ()))
        self._calls = Counter()
        self._hinted_reads = set()
        self._discovered = set()
        self._failures = []
        self._recovery = []
        self._pending_reads = {}

    @staticmethod
    def _tool_name(serialized):
        if isinstance(serialized, Mapping):
            return str(serialized.get("name") or serialized.get("id") or "").strip()
        return ""

    @staticmethod
    def _input_paths(tool_input):
        if isinstance(tool_input, str):
            text = tool_input.strip()
            if text.startswith("{"):
                try:
                    parsed = json.loads(text)
                except (TypeError, ValueError):
                    parsed = None
                if isinstance(parsed, Mapping):
                    return HintExecutionObserver._input_paths(parsed)
            return _result_paths(tool_input)
        if not isinstance(tool_input, Mapping):
            return []
        values = []
        for key in ("file_path", "path", "pattern"):
            if key in tool_input:
                values.extend(_result_paths(tool_input.get(key)))
        return values

    @staticmethod
    def _success(output):
        status = getattr(output, "status", None)
        if status is not None:
            return str(status).lower() == "success"
        if isinstance(output, Mapping) and output.get("status") is not None:
            return str(output.get("status")).lower() == "success"
        text = str(getattr(output, "content", output) or "")
        return not bool(__import__("re").match(r"^\s*(?:tool\s+)?(?:error|failed|failure)\s*:", text, __import__("re").I))

    def _existing_files(self, values):
        for path in _result_paths(values):
            if self.root is None:
                yield path
                continue
            candidate = (self.root / path).resolve()
            if candidate.is_relative_to(self.root) and candidate.is_file():
                yield path

    def on_tool_start(self, serialized, input_str, **kwargs):
        name = self._tool_name(serialized)
        if name:
            self._calls[name] += 1
        if name in {"read", "read_file"}:
            run_id = str(kwargs.get("run_id") or "")
            self._pending_reads[run_id] = self._input_paths(input_str)

    def on_tool_end(self, output, **kwargs):
        name = str(kwargs.get("name") or "")
        run_id = str(kwargs.get("run_id") or "")
        if run_id in self._pending_reads:
            paths = self._pending_reads.pop(run_id)
            if self._success(output):
                for path in paths:
                    if path in self._candidates:
                        self._hinted_reads.add(path)
                    elif self.bound.get("bound"):
                        self._discovered.add(path)
        if name in {"ls", "glob", "grep", "find", "search"} and self.bound.get("bound") and self._success(output):
            for path in self._existing_files(getattr(output, "content", output)):
                if path not in self._candidates:
                    self._discovered.add(path)

    def on_tool_error(self, error, **kwargs):
        run_id = str(kwargs.get("run_id") or "")
        self._pending_reads.pop(run_id, None)
        self._failures.append(type(error).__name__)

    def inspect_output(self, output):
        if not self.bound.get("bound"):
            return
        messages = output.get("messages", []) if isinstance(output, Mapping) else []
        for message in messages:
            name = getattr(message, "name", "")
            content = getattr(message, "content", None)
            if name in {"ls", "glob", "grep", "find", "search"} and self._success(message):
                for path in self._existing_files(content):
                    if path not in self._candidates:
                        self._discovered.add(path)

    def add_recovery_observation(self, event):
        text = str(event).strip()
        if text:
            self._recovery.append(text)

    def receipt(self):
        return build_hint_execution_receipt(
            bound=bool(self.bound.get("bound")),
            hint_identity=self.bound.get("hint_identity") or {},
            initial_hints=self.bound.get("hints") or (),
            hinted_reads=sorted(self._hinted_reads),
            discovered_outside=sorted(self._discovered),
            tool_calls=[{"tool": tool, "count": count} for tool, count in sorted(self._calls.items())],
            recovery_events=self._recovery,
            widened=bool(self._discovered),
            context_failures=self._failures,
        )
