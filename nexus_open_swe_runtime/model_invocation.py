from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

"""Model-invocation execution evidence.

A stable, hash-bound receipt per physical provider/model attempt.  Deliberately
carries no routing, selection, correctness, or acceptance authority: it is
``DERIVED_EXECUTION_EVIDENCE_ONLY`` evidence for one external model transport
attempt.  Missing telemetry is recorded as ``NOT_MEASURED`` and is never coerced
to a numeric value.
"""

MODEL_INVOCATION_RECEIPT_SCHEMA = "nexus.open_swe_runtime.model_invocation_receipt.v1"
MODEL_INVOCATION_AUTHORITY_KIND = "DERIVED_EXECUTION_EVIDENCE_ONLY"
MODEL_INVOCATION_BACKEND_ID = "opencli-chatgpt-web"
NOT_MEASURED = "NOT_MEASURED"

TRANSPORT_OUTCOME_OBSERVED_OK = "OBSERVED_OK"
TRANSPORT_OUTCOME_PROVIDER_STARTUP_FAILURE = "PROVIDER_STARTUP_FAILURE"
TRANSPORT_OUTCOME_PROVIDER_TIMEOUT = "PROVIDER_TIMEOUT"
TRANSPORT_OUTCOME_TRANSPORT_FAILURE = "TRANSPORT_FAILURE"
TRANSPORT_OUTCOME_INVALID_RESPONSE = "INVALID_RESPONSE"
TRANSPORT_OUTCOME_OUTCOME_UNKNOWN = "OUTCOME_UNKNOWN"

MODEL_INVOCATION_BINDING_KEYS = (
    "operation_id",
    "provider_id",
    "model_id",
    "worker_identity_sha256",
    "transport_config_sha256",
    "runtime_identity_sha256",
    "core_binding_hash",
    "acceptance_contract_hash",
    "effect_authorization_hash",
    "tool_projection_hash",
    "workspace",
    "checkpoint_namespace",
)

MODEL_INVOCATION_RECEIPT_FIELDS = (
    "schema",
    "authority_kind",
    "operation_id",
    "provider_id",
    "model_id",
    "provider_model_revision",
    "backend_id",
    "backend_identity",
    "worker_identity_sha256",
    "core_binding_hash",
    "acceptance_contract_hash",
    "effect_authorization_hash",
    "tool_projection_hash",
    "transport_config_sha256",
    "runtime_identity_sha256",
    "workspace",
    "checkpoint_namespace",
    "turn_id",
    "attempt_ordinal",
    "new_conversation",
    "prior_conversation_id_sha256",
    "conversation_id_sha256",
    "started_at",
    "started_monotonic",
    "finished_monotonic",
    "setup_latency_seconds",
    "active_latency_seconds",
    "readback_latency_seconds",
    "total_latency_seconds",
    "input_tokens",
    "output_tokens",
    "cached_tokens",
    "request_bytes",
    "response_bytes",
    "request_sha256",
    "response_sha256",
    "content_hash",
    "transport_outcome",
    "provider_status",
    "error_class",
    "timeout_class",
    "response_valid",
    "external_effect_outcome_known",
    "blind_replay_prohibited",
    "recovered_after_timeout",
    "resume_required",
    "invocation_id",
    "receipt_sha256",
)

_TIME_CONVERSION_CODES = frozenset({
    "OPENCLI_WEB_TIMEOUT",
    "OPENCLI_WEB_TIMEOUT_RECONCILE_UNKNOWN",
    "OPENCLI_WEB_RESUME_REQUIRED",
    "OPENCLI_WEB_REPAIR_RESUME_EXHAUSTED",
})


class ModelInvocationError(ValueError):
    """Bounded model-invocation evidence failure."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: Any) -> str:
    if isinstance(value, str):
        return hashlib.sha256(value.encode("utf-8")).hexdigest()
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _fsync_replace(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_symlink() or path.parent.is_symlink():
        raise ModelInvocationError("MODEL_INVOCATION_STATE_SYMLINK")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(dict(value), stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def invocation_binding_from_identity(identity: Any) -> dict[str, str]:
    """Normalize a recovery identity or mapping into the 12-key binding."""
    binding: dict[str, str] = {}
    for key in MODEL_INVOCATION_BINDING_KEYS:
        if isinstance(identity, Mapping):
            value = identity.get(key)
        else:
            value = getattr(identity, key, None)
        if value is None:
            value = ""
        if isinstance(value, (list, tuple)):
            value = "|".join(str(item) for item in value)
        else:
            value = str(value)
        binding[key] = value
    return binding


def classify_transport_outcome(error_code: str) -> str:
    """Map a bounded transport error code to the invocation outcome taxonomy."""
    invalid_response = frozenset({
        "OPENCLI_WEB_RESPONSE_INVALID",
        "OPENCLI_WEB_CONVERSATION_ID_INVALID",
        "OPENCLI_WEB_CONVERSATION_ID_MISMATCH",
        "OPENCLI_WEB_RECONCILE_INVALID",
        "OPENCLI_WEB_TURN_IDENTITY_UNKNOWN",
        "OPENCLI_WEB_RECONCILE_INCOMPLETE",
    })
    if error_code in {"OPENCLI_NOT_FOUND", "OPENCLI_WEB_PROCESS_FAILURE", "OPENCLI_WEB_BUSY"}:
        return TRANSPORT_OUTCOME_TRANSPORT_FAILURE
    if error_code == "OPENCLI_WEB_HARD_BLOCK":
        return TRANSPORT_OUTCOME_PROVIDER_STARTUP_FAILURE
    if error_code == "OPENCLI_WEB_TIMEOUT":
        return TRANSPORT_OUTCOME_PROVIDER_TIMEOUT
    if error_code in {
        "OPENCLI_WEB_TIMEOUT_RECONCILE_UNKNOWN",
        "OPENCLI_WEB_RESUME_REQUIRED",
        "OPENCLI_WEB_REPAIR_RESUME_EXHAUSTED",
    }:
        return TRANSPORT_OUTCOME_OUTCOME_UNKNOWN
    if error_code in invalid_response:
        return TRANSPORT_OUTCOME_INVALID_RESPONSE
    return TRANSPORT_OUTCOME_OUTCOME_UNKNOWN


def provider_status_for_error(error_code: str) -> str:
    """The provider was reachable and answered when its response was rejected."""
    if classify_transport_outcome(error_code) == TRANSPORT_OUTCOME_INVALID_RESPONSE:
        return "ok"
    return "failed"


def external_outcome_known_for_error(error_code: str) -> bool:
    """Known-effect codes: the transport reports a definitive provider state."""
    known_effect_codes = frozenset({
        "OPENCLI_NOT_FOUND",
        "OPENCLI_WEB_BUSY",
        "OPENCLI_WEB_HARD_BLOCK",
    })
    return (
        classify_transport_outcome(error_code) == TRANSPORT_OUTCOME_INVALID_RESPONSE
        or error_code in known_effect_codes
    )


def _required_text(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value:
        raise ModelInvocationError(code)
    return value


def build_model_invocation_receipt(
    binding: Mapping[str, Any],
    attempt: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one self-hashed receipt for a physical transport attempt."""
    operation_id = _required_text(
        binding.get("operation_id"), "MODEL_INVOCATION_RECEIPT_FIELDS_INVALID"
    )
    provider_id = _required_text(
        binding.get("provider_id"), "MODEL_INVOCATION_RECEIPT_FIELDS_INVALID"
    )
    model_id = _required_text(
        binding.get("model_id"), "MODEL_INVOCATION_RECEIPT_FIELDS_INVALID"
    )
    backend_id = _required_text(
        attempt.get("backend_id"), "MODEL_INVOCATION_RECEIPT_FIELDS_INVALID"
    )
    conversation_id_sha256 = attempt.get("conversation_id_sha256", NOT_MEASURED)
    response_sha256 = attempt.get("response_sha256", NOT_MEASURED)
    subject = {
        "request_sha256": attempt.get("request_sha256", ""),
        "response_sha256": response_sha256,
        "conversation_id_sha256": conversation_id_sha256,
        "attempt_ordinal": attempt.get("attempt_ordinal", 0),
    }
    content_hash = _sha256(_canonical_json(subject))
    invocation_identity = _sha256(_canonical_json({
        "operation_id": operation_id,
        "provider_id": provider_id,
        "model_id": model_id,
        "backend_id": backend_id,
        "turn_id": attempt.get("turn_id", ""),
        "attempt_ordinal": attempt.get("attempt_ordinal", 0),
        "request_sha256": attempt.get("request_sha256", ""),
        "new_conversation": attempt.get("new_conversation", False),
    }))
    invocation_id = f"inv_{invocation_identity}"
    backend_identity = dict(attempt.get("backend_identity") or {})
    backend_identity["configured_identity"] = {
        "provider_id": provider_id,
        "model_id": model_id,
    }
    backend_identity["observed_execution_identity"] = {
        "provider_id": attempt.get("observed_provider_id", NOT_MEASURED),
        "model_id": attempt.get("observed_model_id", NOT_MEASURED),
        "model_revision": attempt.get(
            "observed_model_revision",
            attempt.get("provider_model_revision", NOT_MEASURED),
        ),
        "observation_source": attempt.get("identity_observation_source", NOT_MEASURED),
    }
    receipt: dict[str, Any] = {
        "schema": MODEL_INVOCATION_RECEIPT_SCHEMA,
        "authority_kind": MODEL_INVOCATION_AUTHORITY_KIND,
        "operation_id": operation_id,
        "provider_id": provider_id,
        "model_id": model_id,
        "provider_model_revision": attempt.get("provider_model_revision", NOT_MEASURED),
        "backend_id": backend_id,
        "backend_identity": backend_identity,
        "worker_identity_sha256": str(binding.get("worker_identity_sha256") or ""),
        "core_binding_hash": str(binding.get("core_binding_hash") or ""),
        "acceptance_contract_hash": str(binding.get("acceptance_contract_hash") or ""),
        "effect_authorization_hash": str(binding.get("effect_authorization_hash") or ""),
        "tool_projection_hash": str(binding.get("tool_projection_hash") or ""),
        "transport_config_sha256": str(binding.get("transport_config_sha256") or ""),
        "runtime_identity_sha256": str(binding.get("runtime_identity_sha256") or ""),
        "workspace": str(binding.get("workspace") or ""),
        "checkpoint_namespace": str(binding.get("checkpoint_namespace") or ""),
        "turn_id": attempt.get("turn_id", ""),
        "attempt_ordinal": attempt.get("attempt_ordinal", 1),
        "new_conversation": bool(attempt.get("new_conversation")),
        "prior_conversation_id_sha256": attempt.get(
            "prior_conversation_id_sha256", NOT_MEASURED
        ),
        "conversation_id_sha256": conversation_id_sha256,
        "started_at": attempt.get("started_at", ""),
        "started_monotonic": attempt.get("started_monotonic", NOT_MEASURED),
        "finished_monotonic": attempt.get("finished_monotonic", NOT_MEASURED),
        "setup_latency_seconds": attempt.get("setup_latency_seconds", NOT_MEASURED),
        "active_latency_seconds": attempt.get("active_latency_seconds", NOT_MEASURED),
        "readback_latency_seconds": attempt.get("readback_latency_seconds", NOT_MEASURED),
        "total_latency_seconds": attempt.get("total_latency_seconds", NOT_MEASURED),
        "input_tokens": attempt.get("input_tokens", NOT_MEASURED),
        "output_tokens": attempt.get("output_tokens", NOT_MEASURED),
        "cached_tokens": attempt.get("cached_tokens", NOT_MEASURED),
        "request_bytes": attempt.get("request_bytes", NOT_MEASURED),
        "response_bytes": attempt.get("response_bytes", NOT_MEASURED),
        "request_sha256": attempt.get("request_sha256", ""),
        "response_sha256": response_sha256,
        "content_hash": content_hash,
        "transport_outcome": attempt.get(
            "transport_outcome", TRANSPORT_OUTCOME_OUTCOME_UNKNOWN
        ),
        "provider_status": attempt.get("provider_status", NOT_MEASURED),
        "error_class": attempt.get("error_class", NOT_MEASURED),
        "timeout_class": attempt.get("timeout_class", NOT_MEASURED),
        "response_valid": attempt.get("response_valid", NOT_MEASURED),
        "external_effect_outcome_known": bool(
            attempt.get("external_effect_outcome_known")
        ),
        "blind_replay_prohibited": bool(attempt.get("blind_replay_prohibited")),
        "recovered_after_timeout": bool(attempt.get("recovered_after_timeout")),
        "resume_required": bool(attempt.get("resume_required")),
        "invocation_id": invocation_id,
    }
    receipt["receipt_sha256"] = _sha256(
        _canonical_json({key: value for key, value in receipt.items() if key != "receipt_sha256"})
    )
    return receipt


def _validated_receipt(
    receipt: Any,
    binding: Mapping[str, str] | None,
) -> dict[str, Any]:
    if not isinstance(receipt, Mapping):
        raise ModelInvocationError("MODEL_INVOCATION_RECEIPT_FIELDS_INVALID")
    if set(receipt) != set(MODEL_INVOCATION_RECEIPT_FIELDS):
        raise ModelInvocationError("MODEL_INVOCATION_RECEIPT_FIELDS_INVALID")
    if receipt.get("schema") != MODEL_INVOCATION_RECEIPT_SCHEMA:
        raise ModelInvocationError("MODEL_INVOCATION_RECEIPT_FIELDS_INVALID")
    for key in ("operation_id", "provider_id", "model_id", "backend_id", "invocation_id"):
        if not isinstance(receipt.get(key), str) or not receipt[key]:
            raise ModelInvocationError("MODEL_INVOCATION_RECEIPT_FIELDS_INVALID")
    ordered = {key: receipt[key] for key in MODEL_INVOCATION_RECEIPT_FIELDS}
    hashed = {key: value for key, value in ordered.items() if key != "receipt_sha256"}
    computed = _sha256(_canonical_json(hashed))
    if ordered.get("receipt_sha256") != computed:
        raise ModelInvocationError("MODEL_INVOCATION_RECEIPT_HASH_INVALID")
    if binding is not None:
        for key in MODEL_INVOCATION_BINDING_KEYS:
            if binding.get(key, "") != ordered.get(key, ""):
                raise ModelInvocationError("MODEL_INVOCATION_IDENTITY_MISMATCH")
    return ordered


class DurableInvocationJournal:
    """Fail-closed per-invocation evidence store rooted under ``recovery/``."""

    def __init__(self, state_root: str | Path, identity: Any) -> None:
        self.identity = identity
        self.binding = invocation_binding_from_identity(identity)
        self.journal_root = Path(state_root).expanduser() / "recovery" / "invocations"

    def _ensure_root(self) -> None:
        if self.journal_root.is_symlink() or self.journal_root.parent.is_symlink():
            raise ModelInvocationError("MODEL_INVOCATION_STATE_SYMLINK")
        self.journal_root.mkdir(mode=0o700, parents=True, exist_ok=True)

    def record(self, receipt: Mapping[str, Any]) -> None:
        receipt = _validated_receipt(receipt, self.binding)
        self._ensure_root()
        _fsync_replace(
            self.journal_root / f"{receipt['invocation_id']}.json",
            receipt,
        )

    def result_invocations(self, *, operation_id: str | None = None) -> list[dict[str, Any]]:
        if not self.journal_root.is_dir():
            return []
        if self.journal_root.is_symlink() or self.journal_root.parent.is_symlink():
            raise ModelInvocationError("MODEL_INVOCATION_STATE_SYMLINK")
        receipts: list[dict[str, Any]] = []
        for path in sorted(self.journal_root.glob("inv_*.json")):
            if path.is_symlink():
                raise ModelInvocationError("MODEL_INVOCATION_STATE_SYMLINK")
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ModelInvocationError("MODEL_INVOCATION_RECORD_CORRUPT") from exc
            try:
                receipt = _validated_receipt(raw, self.binding)
            except ModelInvocationError as exc:
                if str(exc) == "MODEL_INVOCATION_IDENTITY_MISMATCH":
                    raise
                raise ModelInvocationError("MODEL_INVOCATION_RECORD_CORRUPT") from exc
            if operation_id is not None and receipt["operation_id"] != operation_id:
                continue
            receipts.append(receipt)
        return receipts


__all__ = [
    "MODEL_INVOCATION_AUTHORITY_KIND",
    "MODEL_INVOCATION_BACKEND_ID",
    "MODEL_INVOCATION_RECEIPT_SCHEMA",
    "NOT_MEASURED",
    "TRANSPORT_OUTCOME_OBSERVED_OK",
    "TRANSPORT_OUTCOME_PROVIDER_STARTUP_FAILURE",
    "TRANSPORT_OUTCOME_PROVIDER_TIMEOUT",
    "TRANSPORT_OUTCOME_TRANSPORT_FAILURE",
    "TRANSPORT_OUTCOME_INVALID_RESPONSE",
    "TRANSPORT_OUTCOME_OUTCOME_UNKNOWN",
    "DurableInvocationJournal",
    "ModelInvocationError",
    "build_model_invocation_receipt",
    "classify_transport_outcome",
    "external_outcome_known_for_error",
    "invocation_binding_from_identity",
    "provider_status_for_error",
]