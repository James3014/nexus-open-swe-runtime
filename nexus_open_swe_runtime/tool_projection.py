"""Consume Runtime-owned effect authorization and tool projection contracts.

This module never selects or widens capability authority. It validates the exact
Wave 1 transport contracts, narrows them to the tools physically exposed by this
execution backend, and emits derived exposure evidence.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

EFFECT_AUTHORIZATION_SCHEMA = "nexus.runtime.effect_authorization.v1"
TOOL_PROJECTION_SCHEMA = "nexus.runtime.tool_projection_manifest.v1"
EXPOSURE_RECEIPT_SCHEMA = "nexus.open_swe_runtime.execution_exposure_receipt.v1"
TOOL_PROJECTION_BACKEND_ID = "nexus-open-swe-runtime"

_EFFECT_AUTHORIZATION_FIELDS = frozenset(
    {
        "schema",
        "authority_id",
        "authority_ref",
        "operation_id",
        "attempt_id",
        "repository",
        "source_revision",
        "base_revision",
        "workspace_id",
        "target_id",
        "expires_at",
        "effects",
        "authorization_hash",
    }
)
_TOOL_PROJECTION_FIELDS = frozenset(
    {
        "schema",
        "authorization_hash",
        "operation_id",
        "attempt_id",
        "provider",
        "backend_id",
        "selected_tools",
        "selected_effects",
        "authority_kind",
        "projection_hash",
    }
)


class ToolProjectionError(ValueError):
    """Raised when a supplied projection cannot be safely consumed."""


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ToolProjectionError("OPEN_SWE_TOOL_PROJECTION_NONCANONICAL") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _required_text(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolProjectionError(code)
    return value.strip()


def _normalize_json_value(value: Any, code: str) -> Any:
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = _required_text(raw_key, code)
            normalized[key] = _normalize_json_value(raw_value, code)
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_json_value(item, code) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        _canonical_json(value)
        return value
    raise ToolProjectionError(code)


def _normalize_effects(value: Any, code: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ToolProjectionError(code)
    normalized = _normalize_json_value(value, code)
    if not isinstance(normalized, dict):
        raise ToolProjectionError(code)
    _canonical_json(normalized)
    return normalized


def _is_subset(selected: Any, ceiling: Any) -> bool:
    if isinstance(selected, Mapping):
        if not isinstance(ceiling, Mapping):
            return False
        return all(
            key in ceiling and _is_subset(selected_value, ceiling[key])
            for key, selected_value in selected.items()
        )
    if isinstance(selected, list):
        if not isinstance(ceiling, list):
            return False
        ceiling_values = {_canonical_json(item) for item in ceiling}
        return all(_canonical_json(item) in ceiling_values for item in selected)
    return selected == ceiling


def _parse_expiry(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ToolProjectionError("OPEN_SWE_EFFECT_AUTHORIZATION_EXPIRY_INVALID") from exc
    if parsed.tzinfo is None:
        raise ToolProjectionError("OPEN_SWE_EFFECT_AUTHORIZATION_EXPIRY_INVALID")
    return parsed.astimezone(UTC)


@dataclass(frozen=True)
class ValidatedToolProjection:
    authorization_hash: str
    projection_hash: str
    operation_id: str
    attempt_id: str
    provider: str
    backend_id: str
    selected_tools: tuple[str, ...]
    selected_effects: Mapping[str, Any]


def validate_tool_projection(
    request: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> ValidatedToolProjection | None:
    raw_authorization = request.get("effect_authorization")
    raw_projection = request.get("tool_projection_manifest")
    if raw_authorization is None and raw_projection is None:
        return None
    if not isinstance(raw_authorization, Mapping) or not isinstance(raw_projection, Mapping):
        raise ToolProjectionError("OPEN_SWE_TOOL_PROJECTION_PAIR_REQUIRED")
    if set(raw_authorization) != _EFFECT_AUTHORIZATION_FIELDS:
        raise ToolProjectionError("OPEN_SWE_EFFECT_AUTHORIZATION_FIELDS_INVALID")
    if raw_authorization.get("schema") != EFFECT_AUTHORIZATION_SCHEMA:
        raise ToolProjectionError("OPEN_SWE_EFFECT_AUTHORIZATION_SCHEMA_INVALID")

    authorization_hash = _required_text(
        raw_authorization.get("authorization_hash"),
        "OPEN_SWE_EFFECT_AUTHORIZATION_HASH_INVALID",
    )
    authorization_material = dict(raw_authorization)
    authorization_material.pop("authorization_hash", None)
    if _sha256(authorization_material) != authorization_hash:
        raise ToolProjectionError("OPEN_SWE_EFFECT_AUTHORIZATION_HASH_INVALID")

    operation_id = _required_text(
        raw_authorization.get("operation_id"),
        "OPEN_SWE_EFFECT_AUTHORIZATION_IDENTITY_INVALID",
    )
    attempt_id = _required_text(
        raw_authorization.get("attempt_id"),
        "OPEN_SWE_EFFECT_AUTHORIZATION_IDENTITY_INVALID",
    )
    if operation_id != request.get("operation_id"):
        raise ToolProjectionError("OPEN_SWE_EFFECT_AUTHORIZATION_OPERATION_MISMATCH")
    if attempt_id != request.get("attempt_id"):
        raise ToolProjectionError("OPEN_SWE_EFFECT_AUTHORIZATION_ATTEMPT_MISMATCH")

    expires_at = raw_authorization.get("expires_at")
    if expires_at is not None:
        expires_text = _required_text(
            expires_at,
            "OPEN_SWE_EFFECT_AUTHORIZATION_EXPIRY_INVALID",
        )
        current = now or datetime.now(UTC)
        if current.tzinfo is None:
            current = current.replace(tzinfo=UTC)
        if current.astimezone(UTC) >= _parse_expiry(expires_text):
            raise ToolProjectionError("OPEN_SWE_EFFECT_AUTHORIZATION_EXPIRED")

    authorized_effects = _normalize_effects(
        raw_authorization.get("effects"),
        "OPEN_SWE_EFFECT_AUTHORIZATION_EFFECTS_INVALID",
    )

    if set(raw_projection) != _TOOL_PROJECTION_FIELDS:
        raise ToolProjectionError("OPEN_SWE_TOOL_PROJECTION_FIELDS_INVALID")
    if raw_projection.get("schema") != TOOL_PROJECTION_SCHEMA:
        raise ToolProjectionError("OPEN_SWE_TOOL_PROJECTION_SCHEMA_INVALID")
    if raw_projection.get("authority_kind") != "DERIVED_PROJECTION_ONLY":
        raise ToolProjectionError("OPEN_SWE_TOOL_PROJECTION_AUTHORITY_INVALID")
    if raw_projection.get("authorization_hash") != authorization_hash:
        raise ToolProjectionError("OPEN_SWE_TOOL_PROJECTION_AUTHORIZATION_MISMATCH")
    if raw_projection.get("operation_id") != operation_id:
        raise ToolProjectionError("OPEN_SWE_TOOL_PROJECTION_OPERATION_MISMATCH")
    if raw_projection.get("attempt_id") != attempt_id:
        raise ToolProjectionError("OPEN_SWE_TOOL_PROJECTION_ATTEMPT_MISMATCH")

    provider = _required_text(
        raw_projection.get("provider"),
        "OPEN_SWE_TOOL_PROJECTION_PROVIDER_INVALID",
    )
    if provider != request.get("provider_id"):
        raise ToolProjectionError("OPEN_SWE_TOOL_PROJECTION_PROVIDER_MISMATCH")
    backend_id = _required_text(
        raw_projection.get("backend_id"),
        "OPEN_SWE_TOOL_PROJECTION_BACKEND_INVALID",
    )
    if backend_id != TOOL_PROJECTION_BACKEND_ID:
        raise ToolProjectionError("OPEN_SWE_TOOL_PROJECTION_BACKEND_MISMATCH")

    raw_tools = raw_projection.get("selected_tools")
    if not isinstance(raw_tools, list) or not raw_tools:
        raise ToolProjectionError("OPEN_SWE_TOOL_PROJECTION_TOOLS_INVALID")
    selected_tools = tuple(
        _required_text(tool, "OPEN_SWE_TOOL_PROJECTION_TOOLS_INVALID") for tool in raw_tools
    )
    if tuple(sorted(set(selected_tools))) != selected_tools:
        raise ToolProjectionError("OPEN_SWE_TOOL_PROJECTION_TOOLS_INVALID")

    selected_effects = _normalize_effects(
        raw_projection.get("selected_effects"),
        "OPEN_SWE_TOOL_PROJECTION_EFFECTS_INVALID",
    )
    if not _is_subset(selected_effects, authorized_effects):
        raise ToolProjectionError("OPEN_SWE_TOOL_PROJECTION_EFFECTS_WIDENED")

    projection_hash = _required_text(
        raw_projection.get("projection_hash"),
        "OPEN_SWE_TOOL_PROJECTION_HASH_INVALID",
    )
    projection_material = dict(raw_projection)
    projection_material.pop("projection_hash", None)
    if _sha256(projection_material) != projection_hash:
        raise ToolProjectionError("OPEN_SWE_TOOL_PROJECTION_HASH_INVALID")

    return ValidatedToolProjection(
        authorization_hash=authorization_hash,
        projection_hash=projection_hash,
        operation_id=operation_id,
        attempt_id=attempt_id,
        provider=provider,
        backend_id=backend_id,
        selected_tools=selected_tools,
        selected_effects=copy.deepcopy(selected_effects),
    )


def require_supported_projection_tools(
    projection: ValidatedToolProjection,
    supported_tools: set[str] | frozenset[str],
) -> None:
    unsupported = set(projection.selected_tools) - set(supported_tools)
    if unsupported:
        raise ToolProjectionError("OPEN_SWE_TOOL_PROJECTION_UNSUPPORTED_TOOL")


def assert_exposed_tools(
    projection: ValidatedToolProjection,
    actual_tools: Sequence[str],
) -> tuple[str, ...]:
    normalized = tuple(sorted({_required_text(tool, "OPEN_SWE_TOOL_EXPOSURE_INVALID") for tool in actual_tools}))
    if not set(normalized).issubset(projection.selected_tools):
        raise ToolProjectionError("OPEN_SWE_TOOL_EXPOSURE_WIDENED")
    return normalized


def build_exposure_receipt(
    projection: ValidatedToolProjection,
    phase_surfaces: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    normalized_phases: dict[str, list[str]] = {}
    union: set[str] = set()
    for raw_phase, raw_tools in sorted(phase_surfaces.items()):
        phase = _required_text(raw_phase, "OPEN_SWE_TOOL_EXPOSURE_INVALID")
        tools = list(assert_exposed_tools(projection, raw_tools))
        normalized_phases[phase] = tools
        union.update(tools)
    material = {
        "schema": EXPOSURE_RECEIPT_SCHEMA,
        "operation_id": projection.operation_id,
        "attempt_id": projection.attempt_id,
        "provider": projection.provider,
        "backend_id": projection.backend_id,
        "authorization_hash": projection.authorization_hash,
        "projection_hash": projection.projection_hash,
        "actual_exposed_tools": sorted(union),
        "phase_tool_surfaces": normalized_phases,
        "authority_kind": "DERIVED_EXPOSURE_EVIDENCE_ONLY",
    }
    return {**material, "exposure_hash": _sha256(material)}
