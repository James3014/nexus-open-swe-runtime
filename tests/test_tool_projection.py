from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import pytest

from nexus_open_swe_runtime.tool_projection import (
    EXPOSURE_RECEIPT_SCHEMA,
    TOOL_PROJECTION_BACKEND_ID,
    ToolProjectionError,
    assert_exposed_tools,
    build_exposure_receipt,
    require_supported_projection_tools,
    validate_tool_projection,
)


def _canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _hash(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def projected_request(
    *,
    operation_id="a" * 64,
    attempt_id="attempt-1",
    provider="google_genai",
    tools=("read_file", "record_worker_result", "write_file"),
    selected_effects=None,
    authorized_effects=None,
):
    authorization_material = {
        "schema": "nexus.runtime.effect_authorization.v1",
        "authority_id": "owner-grant-wave2",
        "authority_ref": "wave2",
        "operation_id": operation_id,
        "attempt_id": attempt_id,
        "repository": "James3014/nexus-open-swe-runtime",
        "source_revision": "1" * 40,
        "base_revision": "1" * 40,
        "workspace_id": "workspace-wave2",
        "target_id": "open-swe-runtime",
        "expires_at": None,
        "effects": authorized_effects
        or {
            "filesystem": {"write_paths": ["a.py"]},
            "process": {"commands": []},
            "network": {"hosts": []},
            "git": {"operations": []},
        },
    }
    authorization = {
        **authorization_material,
        "authorization_hash": _hash(authorization_material),
    }
    projection_material = {
        "schema": "nexus.runtime.tool_projection_manifest.v1",
        "authorization_hash": authorization["authorization_hash"],
        "operation_id": operation_id,
        "attempt_id": attempt_id,
        "provider": provider,
        "backend_id": TOOL_PROJECTION_BACKEND_ID,
        "selected_tools": sorted(tools),
        "selected_effects": selected_effects
        or {"filesystem": {"write_paths": ["a.py"]}},
        "authority_kind": "DERIVED_PROJECTION_ONLY",
    }
    projection = {
        **projection_material,
        "projection_hash": _hash(projection_material),
    }
    return {
        "operation_id": operation_id,
        "attempt_id": attempt_id,
        "provider_id": provider,
        "effect_authorization": authorization,
        "tool_projection_manifest": projection,
    }


def test_wave1_projection_contract_is_consumed_without_becoming_authority():
    request = projected_request()
    projection = validate_tool_projection(request)

    assert projection is not None
    assert projection.backend_id == TOOL_PROJECTION_BACKEND_ID
    assert projection.selected_tools == (
        "read_file",
        "record_worker_result",
        "write_file",
    )
    assert projection.authorization_hash == request["effect_authorization"]["authorization_hash"]
    assert projection.projection_hash == request["tool_projection_manifest"]["projection_hash"]


@pytest.mark.parametrize(
    ("mutator", "error"),
    [
        (
            lambda request: request["effect_authorization"]["effects"]["filesystem"][
                "write_paths"
            ].append("escape.py"),
            "OPEN_SWE_EFFECT_AUTHORIZATION_HASH_INVALID",
        ),
        (
            lambda request: request["tool_projection_manifest"].__setitem__(
                "selected_tools",
                ["edit_file", "read_file", "record_worker_result", "write_file"],
            ),
            "OPEN_SWE_TOOL_PROJECTION_HASH_INVALID",
        ),
        (
            lambda request: request["tool_projection_manifest"].__setitem__(
                "provider", "other-provider"
            ),
            "OPEN_SWE_TOOL_PROJECTION_PROVIDER_MISMATCH",
        ),
        (
            lambda request: request["tool_projection_manifest"].__setitem__(
                "backend_id", "other-runtime"
            ),
            "OPEN_SWE_TOOL_PROJECTION_BACKEND_MISMATCH",
        ),
    ],
)
def test_tampered_or_substituted_projection_fails_closed(mutator, error):
    request = projected_request()
    mutator(request)
    if error.endswith("PROVIDER_MISMATCH") or error.endswith("BACKEND_MISMATCH"):
        material = dict(request["tool_projection_manifest"])
        material.pop("projection_hash")
        request["tool_projection_manifest"]["projection_hash"] = _hash(material)

    with pytest.raises(ToolProjectionError, match=error):
        validate_tool_projection(request)


def test_projection_effect_widening_fails_closed_even_with_valid_hash():
    request = projected_request(
        selected_effects={"filesystem": {"write_paths": ["escape.py"]}}
    )

    with pytest.raises(
        ToolProjectionError,
        match="OPEN_SWE_TOOL_PROJECTION_EFFECTS_WIDENED",
    ):
        validate_tool_projection(request)


def test_projection_expiry_is_revalidated_by_open_swe_consumer():
    request = projected_request()
    authorization = request["effect_authorization"]
    authorization["expires_at"] = "2026-09-17T00:00:00+00:00"
    material = dict(authorization)
    material.pop("authorization_hash")
    authorization["authorization_hash"] = _hash(material)
    projection = request["tool_projection_manifest"]
    projection["authorization_hash"] = authorization["authorization_hash"]
    projection_material = dict(projection)
    projection_material.pop("projection_hash")
    projection["projection_hash"] = _hash(projection_material)

    with pytest.raises(ToolProjectionError, match="OPEN_SWE_EFFECT_AUTHORIZATION_EXPIRED"):
        validate_tool_projection(
            request,
            now=datetime(2026, 9, 18, tzinfo=UTC),
        )


def test_unsupported_tool_name_cannot_be_translated_into_open_swe_capability():
    projection = validate_tool_projection(
        projected_request(tools=("read_file", "record_worker_result", "shell"))
    )
    assert projection is not None

    with pytest.raises(
        ToolProjectionError,
        match="OPEN_SWE_TOOL_PROJECTION_UNSUPPORTED_TOOL",
    ):
        require_supported_projection_tools(
            projection,
            {"read_file", "record_worker_result", "write_file"},
        )


def test_actual_exposure_must_be_subset_of_projection():
    projection = validate_tool_projection(projected_request())
    assert projection is not None

    with pytest.raises(ToolProjectionError, match="OPEN_SWE_TOOL_EXPOSURE_WIDENED"):
        assert_exposed_tools(
            projection,
            ("read_file", "record_worker_result", "write_file", "edit_file"),
        )


def test_exposure_receipt_is_deterministic_and_derived_only():
    projection = validate_tool_projection(
        projected_request(
            tools=(
                "read_file",
                "record_diagnosis",
                "record_worker_result",
                "write_file",
            )
        )
    )
    assert projection is not None

    first = build_exposure_receipt(
        projection,
        {
            "repair": ("read_file", "record_worker_result", "write_file"),
            "diagnosis": ("read_file", "record_diagnosis"),
        },
    )
    second = build_exposure_receipt(
        projection,
        {
            "diagnosis": ("record_diagnosis", "read_file"),
            "repair": ("write_file", "read_file", "record_worker_result"),
        },
    )

    assert first == second
    assert first["schema"] == EXPOSURE_RECEIPT_SCHEMA
    assert first["authority_kind"] == "DERIVED_EXPOSURE_EVIDENCE_ONLY"
    assert first["actual_exposed_tools"] == [
        "read_file",
        "record_diagnosis",
        "record_worker_result",
        "write_file",
    ]
    assert len(first["exposure_hash"]) == 64
