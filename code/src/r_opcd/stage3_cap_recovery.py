"""Fail-closed helpers for non-destructive Stage 3 trajectory cap recovery."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


STABLE_GENERATION_FIELDS = (
    "source_id",
    "source_row_sha256",
    "model_revision",
    "student_adapter",
    "prompt_view",
    "prompt_token_ids_sha256",
    "prompt_rendered_sha256",
    "prompt_attention_mask_sha256",
)


def capped_source_ids(records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return capped source IDs without treating non-EOS stops as semantic labels."""

    return [
        str(record["source_id"])
        for record in records
        if record.get("stop_reason") == "max_new_tokens"
    ]


def select_cap_recovery_rows(
    source_rows: Sequence[Mapping[str, Any]],
    original_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select exactly the capped source rows while retaining source order."""

    source_ids = [str(row["id"]) for row in source_rows]
    generation_ids = [str(row["source_id"]) for row in original_records]
    if source_ids != generation_ids:
        raise ValueError("source and original generation order/IDs differ")
    capped = capped_source_ids(original_records)
    if len(capped) != len(set(capped)):
        raise ValueError("capped source IDs are not unique")
    capped_set = set(capped)
    selected = [dict(row) for row in source_rows if str(row["id"]) in capped_set]
    if len(selected) != len(capped):
        raise ValueError("could not recover every capped source row")
    return selected


def merge_cap_recovery(
    original_records: Sequence[Mapping[str, Any]],
    recovered_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Replace only capped rows after exact prompt and greedy-prefix validation."""

    capped = capped_source_ids(original_records)
    recovered_by_id = {str(row["source_id"]): row for row in recovered_records}
    if len(recovered_by_id) != len(recovered_records):
        raise ValueError("recovered source IDs are not unique")
    if set(recovered_by_id) != set(capped):
        raise ValueError("recovered IDs differ from the original capped ID set")

    merged: list[dict[str, Any]] = []
    for original in original_records:
        source_id = str(original["source_id"])
        if source_id not in recovered_by_id:
            merged.append(dict(original))
            continue
        recovered = recovered_by_id[source_id]
        for field in STABLE_GENERATION_FIELDS:
            if original.get(field) != recovered.get(field):
                raise ValueError(f"{source_id}: stable field changed: {field}")
        for field in ("prompt_token_ids", "prompt_attention_mask"):
            if original.get(field) != recovered.get(field):
                raise ValueError(f"{source_id}: prompt material changed: {field}")
        original_tokens = [int(value) for value in original["response_token_ids"]]
        recovered_tokens = [int(value) for value in recovered["response_token_ids"]]
        if recovered_tokens[: len(original_tokens)] != original_tokens:
            raise ValueError(f"{source_id}: recovered greedy token prefix changed")
        if int(recovered["max_new_tokens"]) <= int(original["max_new_tokens"]):
            raise ValueError(f"{source_id}: recovery cap did not increase")
        replacement = dict(recovered)
        replacement["schema"] = "r-opcd-stage3-student-rollout-cap-recovered-v1"
        replacement["cap_recovery"] = {
            "replaced_original_capped_record": True,
            "original_max_new_tokens": int(original["max_new_tokens"]),
            "recovery_max_new_tokens": int(recovered["max_new_tokens"]),
            "original_response_token_ids_sha256": original["response_token_ids_sha256"],
            "original_response_tokens": len(original_tokens),
            "exact_original_token_prefix_preserved": True,
        }
        merged.append(replacement)
    return merged
