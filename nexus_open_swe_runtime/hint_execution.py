"""Additive retrieval-hint execution evidence (#68).

Execution-side seam that accepts externally prepared, revision-bound additive
retrieval hints, preserves worker search/read recovery, and emits bounded
derived evidence about whether hints reduced exploration. It never computes a
retrieval policy, never removes authorized tools, and never treats Top-K as
complete.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

HINT_EXECUTION_SCHEMA = "nexus.open_swe_runtime.hint_execution_evidence.v1"
HINT_EXECUTION_CLAIM_CEILING = "OPEN_SWE_RETRIEVAL_HINT_EXECUTION_EVIDENCE_ONLY"
SEARCH_TOOLS = ("search", "grep", "read", "glob", "find")


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
        if str(evidence.get("revision") or "").strip() != expected_revision:
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


def build_hint_execution_receipt(*, bound, hint_identity=None, hinted_reads=(), discovered_outside=(), tool_calls=None, recovery_events=(), widened=False, context_failures=()):
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
    receipt = {"schema": HINT_EXECUTION_SCHEMA, "bound": bool(bound), "hint_identity": dict(hint_identity or {}), "hinted_total": len(set(hinted)) if bound else 0, "hinted_read": sorted(set(hinted)), "discovered_outside_hints": sorted(set(discovered)), "tool_calls": calls, "search_tool_calls": search_calls, "widened_beyond_hints": bool(widened or discovered), "recovery_events": [str(e) for e in (recovery_events or [])], "context_failures": [str(e) for e in (context_failures or [])], "claim_ceiling": HINT_EXECUTION_CLAIM_CEILING}
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
