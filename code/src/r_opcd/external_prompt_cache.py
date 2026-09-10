"""Frozen Qwen3-8B prompt-cache records for external injection cases.

The logical case remains the semantic authority.  This module materializes a
tokenizer-bound cache that shares one ordinary prompt between Student and T-
and stores the latest exact-span attention-quarantine T+ prompt separately.
Reference answers are intentionally emitted through a different record type.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Mapping

from r_opcd.attack_pilot import ATTENTION_QUARANTINE_CONTRACT_VERSION
from r_opcd.external_injection_data import canonical_sha256, validate_logical_case
from r_opcd.frontier_collection import (
    PromptTokenization,
    tokenize_student_prompt,
    tokenize_teacher_plus_prompt,
)


PROMPT_CACHE_SCHEMA = "r-opcd-external-prompt-cache-qwen3-8b-v1"
REFERENCE_SCHEMA = "r-opcd-external-reference-sidecar-v1"
POSITION_IDS_POLICY = "absolute_arange_unpadded"
PREFIX_POLICY = "append_identical_response_token_ids_with_attention_one"
FORBIDDEN_PROMPT_CACHE_FIELDS = frozenset(
    {
        "target_answer",
        "target_answers",
        "reference_response",
        "risk_label",
        "risk_weight",
        "sanitized_context",
        "attack_str",
        "verifier_label",
        "scorer_prompt",
    }
)


def build_prompt_cache_record(
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    max_prompt_tokens: int = 1536,
) -> dict[str, Any]:
    """Compile one logical case into a tokenizer-bound Student/T-/T+ cache."""

    validate_logical_case(row)
    if max_prompt_tokens < 1:
        raise ValueError("max_prompt_tokens must be positive")
    ordinary = tokenize_student_prompt(tokenizer, row)
    privileged = tokenize_teacher_plus_prompt(tokenizer, row)
    if ordinary.prompt_tokens > max_prompt_tokens:
        raise ValueError(
            f"T- prompt for {row['id']} has {ordinary.prompt_tokens} tokens, "
            f"above cap {max_prompt_tokens}"
        )
    if privileged.prompt_tokens > max_prompt_tokens:
        raise ValueError(
            f"T+ prompt for {row['id']} has {privileged.prompt_tokens} tokens, "
            f"above cap {max_prompt_tokens}"
        )
    quarantined = privileged.quarantined_token_indices
    if not quarantined:
        raise ValueError("T+ cache must contain a non-empty q1 token span")
    expected_indices = tuple(range(quarantined[0], quarantined[-1] + 1))
    if quarantined != expected_indices:
        raise ValueError("q1 token indices must form one contiguous interval")

    all_one_plus_mask = (1,) * privileged.prompt_tokens
    record = {
        "schema": PROMPT_CACHE_SCHEMA,
        "logical_case_id": row["id"],
        "logical_case_sha256": canonical_sha256(dict(row)),
        "split": row["split"],
        "source_dataset": row["source_dataset"],
        "task_name": row["task_name"],
        "task_type": row["task_type"],
        "attack_name": row["attack_name"],
        "attack_category": row["attack_category"],
        "sampling_group": {
            "base_task_id": row["base_task_id"],
            "target_source_digest": row["target_source_digest"],
            "injection_source_digest": row["injection_source_digest"],
        },
        "position_ids_policy": POSITION_IDS_POLICY,
        "response_prefix_policy": PREFIX_POLICY,
        "student_teacher_minus": {
            "view": "ordinary_attacked_prompt",
            "student_equals_teacher_minus": True,
            "input_ids": list(ordinary.input_ids),
            "input_ids_sha256": canonical_sha256(list(ordinary.input_ids)),
            "attention_mask": list(ordinary.attention_mask),
            "attention_mask_sha256": ordinary.attention_mask_sha256,
            "prompt_tokens": ordinary.prompt_tokens,
            "rendered_sha256": ordinary.rendered_sha256,
        },
        "teacher_plus": {
            "view": "attention_quarantine_guard_oracle",
            "contract_version": ATTENTION_QUARANTINE_CONTRACT_VERSION,
            "input_ids": list(privileged.input_ids),
            "input_ids_sha256": canonical_sha256(list(privileged.input_ids)),
            "attention_mask": list(privileged.attention_mask),
            "attention_mask_sha256": privileged.attention_mask_sha256,
            "prompt_tokens": privileged.prompt_tokens,
            "rendered_sha256": privileged.rendered_sha256,
            "q1_sha256": row["malicious_span_sha256"],
            "quarantined_token_indices": list(quarantined),
            "quarantined_token_start": quarantined[0],
            "quarantined_token_end_exclusive": quarantined[-1] + 1,
            "quarantined_tokens": len(quarantined),
        },
        "teacher_plus_all_one_control": {
            "input_ids_source": "teacher_plus.input_ids",
            "attention_mask_policy": "all_ones",
            "attention_mask_sha256": sha256(bytes(all_one_plus_mask)).hexdigest(),
        },
    }
    validate_prompt_cache_record(record, max_prompt_tokens=max_prompt_tokens)
    return record


def build_reference_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """Emit supervision into a sidecar that is never a Teacher prompt input."""

    validate_logical_case(row)
    record = {
        "schema": REFERENCE_SCHEMA,
        "logical_case_id": row["id"],
        "logical_case_sha256": canonical_sha256(dict(row)),
        "split": row["split"],
        "task_name": row["task_name"],
        "task_type": row["task_type"],
        "target_answer": row["target_answer"],
        "target_answers": row["target_answers"],
    }
    record["reference_sha256"] = canonical_sha256(record)
    return record


def validate_prompt_cache_record(
    record: Mapping[str, Any], *, max_prompt_tokens: int = 1536
) -> None:
    """Fail closed if a cache can leak supervision or cannot reconstruct masks."""

    required = {
        "schema",
        "logical_case_id",
        "logical_case_sha256",
        "split",
        "source_dataset",
        "task_name",
        "task_type",
        "attack_name",
        "attack_category",
        "sampling_group",
        "position_ids_policy",
        "response_prefix_policy",
        "student_teacher_minus",
        "teacher_plus",
        "teacher_plus_all_one_control",
    }
    missing = sorted(required - set(record))
    if missing:
        raise ValueError(f"prompt-cache record missing fields: {missing}")
    if record["schema"] != PROMPT_CACHE_SCHEMA:
        raise ValueError("prompt-cache schema changed")
    if record["split"] not in {"train", "dev"}:
        raise ValueError("prompt-cache split must be train or dev")
    if record["position_ids_policy"] != POSITION_IDS_POLICY:
        raise ValueError("position-ID policy changed")
    if record["response_prefix_policy"] != PREFIX_POLICY:
        raise ValueError("response-prefix policy changed")
    forbidden = _recursive_keys(record) & FORBIDDEN_PROMPT_CACHE_FIELDS
    if forbidden:
        raise ValueError(
            "prompt cache contains forbidden supervision fields: "
            + ", ".join(sorted(forbidden))
        )

    ordinary = record["student_teacher_minus"]
    plus = record["teacher_plus"]
    if not isinstance(ordinary, Mapping) or not isinstance(plus, Mapping):
        raise ValueError("cached prompt views must be mappings")
    if ordinary.get("student_equals_teacher_minus") is not True:
        raise ValueError("Student and T- must share one prompt cache")
    if plus.get("contract_version") != ATTENTION_QUARANTINE_CONTRACT_VERSION:
        raise ValueError("T+ contract version changed")
    _validate_cached_view(ordinary, name="Student/T-", max_prompt_tokens=max_prompt_tokens)
    _validate_cached_view(plus, name="T+", max_prompt_tokens=max_prompt_tokens)
    if any(int(value) != 1 for value in ordinary["attention_mask"]):
        raise ValueError("Student/T- prompt mask must be all ones")

    plus_mask = tuple(int(value) for value in plus["attention_mask"])
    indices = tuple(int(value) for value in plus["quarantined_token_indices"])
    if not indices or indices != tuple(sorted(set(indices))):
        raise ValueError("T+ q1 indices must be non-empty, sorted, and unique")
    if indices != tuple(range(indices[0], indices[-1] + 1)):
        raise ValueError("T+ q1 indices must be contiguous")
    if indices[0] < 0 or indices[-1] >= len(plus_mask):
        raise ValueError("T+ q1 token index is out of range")
    if tuple(index for index, value in enumerate(plus_mask) if value == 0) != indices:
        raise ValueError("T+ zero mask positions no longer equal q1 token indices")
    if int(plus["quarantined_token_start"]) != indices[0]:
        raise ValueError("T+ q1 start changed")
    if int(plus["quarantined_token_end_exclusive"]) != indices[-1] + 1:
        raise ValueError("T+ q1 end changed")
    if int(plus["quarantined_tokens"]) != len(indices):
        raise ValueError("T+ q1 token count changed")

    control = record["teacher_plus_all_one_control"]
    if not isinstance(control, Mapping):
        raise ValueError("T+ all-one control must be a mapping")
    expected_control_hash = sha256(bytes([1] * len(plus_mask))).hexdigest()
    if control.get("input_ids_source") != "teacher_plus.input_ids":
        raise ValueError("T+ control must reuse identical T+ input IDs")
    if control.get("attention_mask_policy") != "all_ones":
        raise ValueError("T+ control mask policy changed")
    if control.get("attention_mask_sha256") != expected_control_hash:
        raise ValueError("T+ all-one control mask hash changed")


def prompt_tokenizations_from_cache_record(
    record: Mapping[str, Any], *, max_prompt_tokens: int = 1536
) -> dict[str, PromptTokenization]:
    """Recover runtime prompt objects without re-tokenizing text."""

    validate_prompt_cache_record(record, max_prompt_tokens=max_prompt_tokens)
    ordinary = _prompt_tokenization(record["student_teacher_minus"])
    plus = _prompt_tokenization(record["teacher_plus"])
    return {
        "student": ordinary,
        "teacher_minus": ordinary,
        "teacher_plus": plus,
    }


def teacher_plus_all_one_control_from_cache_record(
    record: Mapping[str, Any], *, max_prompt_tokens: int = 1536
) -> PromptTokenization:
    """Build the matched serialization control with q1 visible to attention."""

    validate_prompt_cache_record(record, max_prompt_tokens=max_prompt_tokens)
    plus = record["teacher_plus"]
    input_ids = tuple(int(value) for value in plus["input_ids"])
    attention_mask = (1,) * len(input_ids)
    return PromptTokenization(
        input_ids=input_ids,
        attention_mask=attention_mask,
        rendered_sha256=str(plus["rendered_sha256"]),
        attention_mask_sha256=sha256(bytes(attention_mask)).hexdigest(),
    )


def _validate_cached_view(
    view: Mapping[str, Any], *, name: str, max_prompt_tokens: int
) -> None:
    required = {
        "input_ids",
        "input_ids_sha256",
        "attention_mask",
        "attention_mask_sha256",
        "prompt_tokens",
        "rendered_sha256",
    }
    missing = sorted(required - set(view))
    if missing:
        raise ValueError(f"{name} cached view missing fields: {missing}")
    input_ids = [int(value) for value in view["input_ids"]]
    attention_mask = [int(value) for value in view["attention_mask"]]
    if not input_ids or len(input_ids) != len(attention_mask):
        raise ValueError(f"{name} input IDs and attention mask are not aligned")
    if any(value not in (0, 1) for value in attention_mask):
        raise ValueError(f"{name} attention mask must be binary")
    if not any(attention_mask):
        raise ValueError(f"{name} attention mask hides the complete prompt")
    if len(input_ids) > max_prompt_tokens:
        raise ValueError(f"{name} prompt exceeds the frozen token cap")
    if int(view["prompt_tokens"]) != len(input_ids):
        raise ValueError(f"{name} prompt-token count changed")
    if view["input_ids_sha256"] != canonical_sha256(input_ids):
        raise ValueError(f"{name} input-ID hash changed")
    if view["attention_mask_sha256"] != sha256(bytes(attention_mask)).hexdigest():
        raise ValueError(f"{name} attention-mask hash changed")


def _prompt_tokenization(view: Mapping[str, Any]) -> PromptTokenization:
    return PromptTokenization(
        input_ids=tuple(int(value) for value in view["input_ids"]),
        attention_mask=tuple(int(value) for value in view["attention_mask"]),
        rendered_sha256=str(view["rendered_sha256"]),
        attention_mask_sha256=str(view["attention_mask_sha256"]),
        quarantined_token_indices=tuple(
            int(value) for value in view.get("quarantined_token_indices", ())
        ),
    )


def _recursive_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(_recursive_keys(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            keys.update(_recursive_keys(item))
    return keys
