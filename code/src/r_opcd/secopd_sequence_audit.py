"""CPU-only sequence-length audit for fixed-rollout SecOPD training data.

The audit deliberately receives the clean utility batch builder as a callback.
The production CLI passes the exact ``utility_batch`` function used by the
training runtime, while the three attacked views use the shared recovery-batch
builder directly.  No model forward is needed to reproduce the four input
sequence widths seen by training.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256
import json
from math import ceil
from statistics import fmean
from typing import Any

from r_opcd.prompt_views import TeacherForcingBatch
from r_opcd.tokenizer_adapter import (
    AlignedRecoveryBatches,
    build_attention_quarantine_recovery_batches,
)


UtilityBatchBuilder = Callable[[Any, Mapping[str, Any]], TeacherForcingBatch]
RecoveryBatchBuilder = Callable[..., AlignedRecoveryBatches]


def callable_path(function: Callable[..., Any]) -> str:
    """Return a stable-enough provenance label for a rendering callback."""

    module = getattr(function, "__module__", type(function).__module__)
    name = getattr(function, "__qualname__", type(function).__qualname__)
    return f"{module}.{name}"


def canonical_row_sha256(row: Mapping[str, Any]) -> str:
    """Hash one decoded JSON row without depending on file compression."""

    payload = json.dumps(
        row,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def nearest_rank(values: Sequence[int], probability: float) -> int:
    """Return a deterministic nearest-rank percentile for non-empty integers."""

    if not values:
        raise ValueError("percentile input must be non-empty")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("percentile probability must lie in [0, 1]")
    ordered = sorted(int(value) for value in values)
    if probability == 0.0:
        return ordered[0]
    return ordered[min(len(ordered) - 1, ceil(probability * len(ordered)) - 1)]


def summarize_lengths(values: Sequence[int], *, cap: int) -> dict[str, Any]:
    """Summarize one route and expose every cap-relevant count."""

    if not values:
        raise ValueError("length summary input must be non-empty")
    normalized = [int(value) for value in values]
    if any(value < 1 for value in normalized):
        raise ValueError("all sequence lengths must be positive")
    return {
        "count": len(normalized),
        "min": min(normalized),
        "mean": fmean(normalized),
        "p50_nearest_rank": nearest_rank(normalized, 0.50),
        "p90_nearest_rank": nearest_rank(normalized, 0.90),
        "p95_nearest_rank": nearest_rank(normalized, 0.95),
        "p99_nearest_rank": nearest_rank(normalized, 0.99),
        "max": max(normalized),
        "at_cap": sum(value == cap for value in normalized),
        "over_cap": sum(value > cap for value in normalized),
        "minimum_headroom": cap - max(normalized),
    }


def _require_nonempty_string(row: Mapping[str, Any], key: str, *, context: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return value


def _response_ids(row: Mapping[str, Any], *, index: int) -> list[int]:
    trajectory = row.get("student_trajectory")
    if not isinstance(trajectory, Mapping):
        raise ValueError(f"attack row {index} lacks student_trajectory mapping")
    raw_ids = trajectory.get("response_content_token_ids")
    if (
        not isinstance(raw_ids, Sequence)
        or isinstance(raw_ids, (str, bytes))
        or not raw_ids
    ):
        raise ValueError(
            f"attack row {index} response_content_token_ids must be non-empty"
        )
    ids = [int(token_id) for token_id in raw_ids]
    if any(token_id < 0 for token_id in ids):
        raise ValueError(f"attack row {index} contains a negative response token")
    return ids


def _single_sequence_width(batch: TeacherForcingBatch, *, expected_view: str) -> int:
    if batch.view_name != expected_view:
        raise ValueError(
            f"batch view mismatch: expected {expected_view}, observed {batch.view_name}"
        )
    if batch.input_ids.ndim != 2 or batch.input_ids.shape[0] != 1:
        raise ValueError(f"{expected_view} audit batch must contain exactly one row")
    width = int(batch.input_ids.shape[1])
    if width != int(batch.prompt_width + batch.responses.shape[1]):
        raise ValueError(f"{expected_view} prompt/response width invariant failed")
    return width


def audit_training_sequence_lengths(
    *,
    tokenizer: Any,
    attack_rows: Sequence[Mapping[str, Any]],
    utility_rows: Sequence[Mapping[str, Any]],
    max_sequence_tokens: int,
    enable_thinking: bool,
    utility_batch_builder: UtilityBatchBuilder,
    recovery_batch_builder: RecoveryBatchBuilder = (
        build_attention_quarantine_recovery_batches
    ),
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Audit all four training routes and return rows, report, and gate selection.

    Each attacked example is rendered independently.  Consequently the tensor
    width is exactly ``prompt + that real Student response`` and cannot be
    inflated by padding from a neighboring example.
    """

    if max_sequence_tokens < 1:
        raise ValueError("max_sequence_tokens must be positive")
    if not attack_rows or not utility_rows:
        raise ValueError("attack and utility training rows must both be non-empty")
    if len(attack_rows) != len(utility_rows):
        raise ValueError("attack and utility training row counts must match")

    route_values: dict[str, list[int]] = {
        "student": [],
        "teacher_minus": [],
        "teacher_plus": [],
        "clean_utility": [],
    }
    per_row: list[dict[str, Any]] = []

    for index, (attack, utility) in enumerate(
        zip(attack_rows, utility_rows, strict=True)
    ):
        if not isinstance(attack, Mapping) or not isinstance(utility, Mapping):
            raise ValueError(f"training row {index} must be a mapping")
        source_id = _require_nonempty_string(
            attack, "source_id", context=f"attack[{index}]"
        )
        base_task_id = _require_nonempty_string(
            attack, "base_task_id", context=f"attack[{index}]"
        )
        attack_source = attack.get("attack_source")
        if not isinstance(attack_source, Mapping):
            raise ValueError(f"attack row {index} lacks attack_source mapping")
        response_ids = _response_ids(attack, index=index)
        batches = recovery_batch_builder(
            tokenizer,
            [attack_source],
            [response_ids],
            template_kwargs={"enable_thinking": bool(enable_thinking)},
        )
        attack_lengths = {
            "student": _single_sequence_width(
                batches.student, expected_view="student"
            ),
            "teacher_minus": _single_sequence_width(
                batches.teacher_minus, expected_view="teacher_minus"
            ),
            "teacher_plus": _single_sequence_width(
                batches.teacher_plus, expected_view="teacher_plus"
            ),
        }
        if attack_lengths["student"] != attack_lengths["teacher_minus"]:
            raise ValueError(f"Student/T- sequence widths diverged at attack row {index}")

        utility_id = _require_nonempty_string(
            utility, "id", context=f"utility[{index}]"
        )
        clean_batch = utility_batch_builder(tokenizer, utility)
        clean_length = _single_sequence_width(
            clean_batch, expected_view="clean_utility"
        )

        for route, length in attack_lengths.items():
            route_values[route].append(length)
        route_values["clean_utility"].append(clean_length)
        max_attack_length = max(attack_lengths.values())
        governing_views = sorted(
            route
            for route, length in attack_lengths.items()
            if length == max_attack_length
        )
        per_row.append(
            {
                "schema": "r-opcd-secopd-training-sequence-length-row-v1",
                "index_zero_based": index,
                "index_one_based": index + 1,
                "attack": {
                    "source_id": source_id,
                    "base_task_id": base_task_id,
                    "row_sha256": canonical_row_sha256(attack),
                    "response_tokens": len(response_ids),
                    "route_total_tokens": attack_lengths,
                    "maximum_total_tokens": max_attack_length,
                    "governing_views": governing_views,
                    "within_cap": max_attack_length <= max_sequence_tokens,
                },
                "clean_utility": {
                    "id": utility_id,
                    "base_task_id": str(utility.get("base_task_id", "")),
                    "row_sha256": canonical_row_sha256(utility),
                    "total_tokens": clean_length,
                    "within_cap": clean_length <= max_sequence_tokens,
                },
            }
        )

    route_summaries = {
        route: summarize_lengths(values, cap=max_sequence_tokens)
        for route, values in route_values.items()
    }
    longest_by_route: dict[str, dict[str, Any]] = {}
    for route, values in route_values.items():
        longest_index = max(range(len(values)), key=values.__getitem__)
        row = per_row[longest_index]
        identity = (
            {
                "source_id": row["attack"]["source_id"],
                "base_task_id": row["attack"]["base_task_id"],
                "row_sha256": row["attack"]["row_sha256"],
            }
            if route != "clean_utility"
            else {
                "id": row["clean_utility"]["id"],
                "base_task_id": row["clean_utility"]["base_task_id"],
                "row_sha256": row["clean_utility"]["row_sha256"],
            }
        )
        longest_by_route[route] = {
            "index_zero_based": longest_index,
            "index_one_based": longest_index + 1,
            "total_tokens": values[longest_index],
            **identity,
        }

    attack_maxima = [row["attack"]["maximum_total_tokens"] for row in per_row]
    longest_attack_index = max(
        range(len(attack_maxima)), key=attack_maxima.__getitem__
    )
    longest_clean_index = max(
        range(len(route_values["clean_utility"])),
        key=route_values["clean_utility"].__getitem__,
    )
    selected_attack = per_row[longest_attack_index]["attack"]
    selected_clean = per_row[longest_clean_index]["clean_utility"]
    selection = {
        "schema": "r-opcd-secopd-longest-example-selection-v1",
        "selection_rule": (
            "first file-order row attaining max total tokens; attack maximum is "
            "taken across Student, T-, and T+"
        ),
        "max_sequence_tokens": max_sequence_tokens,
        "attack": {
            "index_zero_based": longest_attack_index,
            "index_one_based": longest_attack_index + 1,
            **selected_attack,
        },
        "clean_utility": {
            "index_zero_based": longest_clean_index,
            "index_one_based": longest_clean_index + 1,
            **selected_clean,
        },
        "usage_note": (
            "The two indices identify independent worst-case records in package "
            "file order; do not assume the training scheduler pairs them."
        ),
    }

    route_violation_count = sum(
        summary["over_cap"] for summary in route_summaries.values()
    )
    attack_record_violations = sum(
        not row["attack"]["within_cap"] for row in per_row
    )
    utility_record_violations = sum(
        not row["clean_utility"]["within_cap"] for row in per_row
    )
    status = "pass" if route_violation_count == 0 else "fail"
    report = {
        "schema": "r-opcd-secopd-training-sequence-length-audit-v1",
        "status": status,
        "decision": (
            "all_training_sequences_within_runtime_cap"
            if status == "pass"
            else "training_blocked_sequence_cap_exceeded"
        ),
        "max_sequence_tokens": max_sequence_tokens,
        "attack_records": len(attack_rows),
        "utility_records": len(utility_rows),
        "route_summaries": route_summaries,
        "violations": {
            "route_instances_over_cap": route_violation_count,
            "attack_records_over_cap": attack_record_violations,
            "utility_records_over_cap": utility_record_violations,
        },
        "longest_by_route": longest_by_route,
        "longest_attack_index_zero_based": longest_attack_index,
        "longest_attack_index_one_based": longest_attack_index + 1,
        "rendering_contract": {
            "attack_builder": callable_path(recovery_batch_builder),
            "clean_builder": callable_path(utility_batch_builder),
            "attack_response": "stored real Student response_content_token_ids",
            "enable_thinking": bool(enable_thinking),
            "model_forward_calls": 0,
            "percentile_method": "nearest_rank",
        },
        "evidence_boundary": (
            "CPU tokenizer/rendering preflight only; no model forward, loss, "
            "optimizer, OPD target, or T+ policy mutation"
        ),
    }
    return per_row, report, selection


def _audit_attack_partition(
    *,
    tokenizer: Any,
    attack_rows: Sequence[Mapping[str, Any]],
    partition: str,
    max_sequence_tokens: int,
    enable_thinking: bool,
    recovery_batch_builder: RecoveryBatchBuilder,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Audit an attack-only partition such as reliability calibration."""

    if not attack_rows:
        raise ValueError(f"{partition} attack rows must be non-empty")
    route_values: dict[str, list[int]] = {
        "student": [],
        "teacher_minus": [],
        "teacher_plus": [],
    }
    rows: list[dict[str, Any]] = []
    for index, attack in enumerate(attack_rows):
        if not isinstance(attack, Mapping):
            raise ValueError(f"{partition} attack row {index} must be a mapping")
        source_id = _require_nonempty_string(
            attack, "source_id", context=f"{partition}_attack[{index}]"
        )
        base_task_id = _require_nonempty_string(
            attack, "base_task_id", context=f"{partition}_attack[{index}]"
        )
        attack_source = attack.get("attack_source")
        if not isinstance(attack_source, Mapping):
            raise ValueError(
                f"{partition} attack row {index} lacks attack_source mapping"
            )
        response_ids = _response_ids(attack, index=index)
        batches = recovery_batch_builder(
            tokenizer,
            [attack_source],
            [response_ids],
            template_kwargs={"enable_thinking": bool(enable_thinking)},
        )
        lengths = {
            "student": _single_sequence_width(
                batches.student, expected_view="student"
            ),
            "teacher_minus": _single_sequence_width(
                batches.teacher_minus, expected_view="teacher_minus"
            ),
            "teacher_plus": _single_sequence_width(
                batches.teacher_plus, expected_view="teacher_plus"
            ),
        }
        if lengths["student"] != lengths["teacher_minus"]:
            raise ValueError(
                f"Student/T- widths diverged at {partition} attack row {index}"
            )
        for route, length in lengths.items():
            route_values[route].append(length)
        maximum = max(lengths.values())
        rows.append(
            {
                "schema": "r-opcd-secopd-training-sequence-length-row-v1",
                "partition": partition,
                "index_zero_based": index,
                "index_one_based": index + 1,
                "attack": {
                    "source_id": source_id,
                    "base_task_id": base_task_id,
                    "row_sha256": canonical_row_sha256(attack),
                    "response_tokens": len(response_ids),
                    "route_total_tokens": lengths,
                    "maximum_total_tokens": maximum,
                    "governing_views": sorted(
                        route for route, length in lengths.items() if length == maximum
                    ),
                    "within_cap": maximum <= max_sequence_tokens,
                },
            }
        )

    summaries = {
        route: summarize_lengths(values, cap=max_sequence_tokens)
        for route, values in route_values.items()
    }
    longest: dict[str, dict[str, Any]] = {}
    for route, values in route_values.items():
        index = max(range(len(values)), key=values.__getitem__)
        attack = rows[index]["attack"]
        longest[route] = {
            "index_zero_based": index,
            "index_one_based": index + 1,
            "total_tokens": values[index],
            "source_id": attack["source_id"],
            "base_task_id": attack["base_task_id"],
            "row_sha256": attack["row_sha256"],
        }
    return rows, {
        "attack_records": len(rows),
        "route_summaries": summaries,
        "longest_by_route": longest,
        "route_instances_over_cap": sum(
            summary["over_cap"] for summary in summaries.values()
        ),
        "attack_records_over_cap": sum(
            not row["attack"]["within_cap"] for row in rows
        ),
    }


def _collect_violation_instances(
    rows: Sequence[Mapping[str, Any]], *, cap: int
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for row in rows:
        partition = str(row["partition"])
        index_zero = int(row["index_zero_based"])
        index_one = int(row["index_one_based"])
        attack = row.get("attack")
        if isinstance(attack, Mapping):
            for route, raw_length in attack["route_total_tokens"].items():
                length = int(raw_length)
                if length > cap:
                    violations.append(
                        {
                            "partition": partition,
                            "data_role": "attack",
                            "route": str(route),
                            "index_zero_based": index_zero,
                            "index_one_based": index_one,
                            "record_id": str(attack["source_id"]),
                            "base_task_id": str(attack["base_task_id"]),
                            "total_tokens": length,
                            "cap": cap,
                            "excess_tokens": length - cap,
                        }
                    )
        utility = row.get("clean_utility")
        if isinstance(utility, Mapping):
            length = int(utility["total_tokens"])
            if length > cap:
                violations.append(
                    {
                        "partition": partition,
                        "data_role": "clean_utility",
                        "route": "clean_utility",
                        "index_zero_based": index_zero,
                        "index_one_based": index_one,
                        "record_id": str(utility["id"]),
                        "base_task_id": str(utility["base_task_id"]),
                        "total_tokens": length,
                        "cap": cap,
                        "excess_tokens": length - cap,
                    }
                )
    return violations


def audit_reliability_package_sequence_lengths(
    *,
    tokenizer: Any,
    train_attack_rows: Sequence[Mapping[str, Any]],
    calibration_attack_rows: Sequence[Mapping[str, Any]],
    train_utility_rows: Sequence[Mapping[str, Any]],
    max_sequence_tokens: int,
    enable_thinking: bool,
    utility_batch_builder: UtilityBatchBuilder,
    recovery_batch_builder: RecoveryBatchBuilder = (
        build_attention_quarantine_recovery_batches
    ),
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Audit every sequence used by reliability calibration or training."""

    train_rows, train_report, selection = audit_training_sequence_lengths(
        tokenizer=tokenizer,
        attack_rows=train_attack_rows,
        utility_rows=train_utility_rows,
        max_sequence_tokens=max_sequence_tokens,
        enable_thinking=enable_thinking,
        utility_batch_builder=utility_batch_builder,
        recovery_batch_builder=recovery_batch_builder,
    )
    for row in train_rows:
        row["partition"] = "train"
    calibration_rows, calibration_report = _audit_attack_partition(
        tokenizer=tokenizer,
        attack_rows=calibration_attack_rows,
        partition="calibration",
        max_sequence_tokens=max_sequence_tokens,
        enable_thinking=enable_thinking,
        recovery_batch_builder=recovery_batch_builder,
    )
    all_rows = [*train_rows, *calibration_rows]
    violations = _collect_violation_instances(all_rows, cap=max_sequence_tokens)

    candidates: list[dict[str, Any]] = []
    for row in all_rows:
        for route, raw_length in row["attack"]["route_total_tokens"].items():
            candidates.append(
                {
                    "partition": row["partition"],
                    "data_role": "attack",
                    "route": route,
                    "index_zero_based": row["index_zero_based"],
                    "index_one_based": row["index_one_based"],
                    "record_id": row["attack"]["source_id"],
                    "base_task_id": row["attack"]["base_task_id"],
                    "total_tokens": int(raw_length),
                }
            )
        utility = row.get("clean_utility")
        if isinstance(utility, Mapping):
            candidates.append(
                {
                    "partition": row["partition"],
                    "data_role": "clean_utility",
                    "route": "clean_utility",
                    "index_zero_based": row["index_zero_based"],
                    "index_one_based": row["index_one_based"],
                    "record_id": utility["id"],
                    "base_task_id": utility["base_task_id"],
                    "total_tokens": int(utility["total_tokens"]),
                }
            )
    global_longest = max(candidates, key=lambda candidate: candidate["total_tokens"])
    train_partition = {
        "attack_records": len(train_attack_rows),
        "utility_records": len(train_utility_rows),
        "route_summaries": train_report["route_summaries"],
        "longest_by_route": train_report["longest_by_route"],
        **train_report["violations"],
    }
    status = "pass" if not violations else "fail"
    report = dict(train_report)
    report.update(
        {
            "schema": "r-opcd-secopd-reliability-sequence-length-audit-v2",
            "status": status,
            "decision": (
                "all_calibration_and_training_sequences_within_runtime_cap"
                if status == "pass"
                else "calibration_and_training_blocked_sequence_cap_exceeded"
            ),
            "calibration_attack_records": len(calibration_attack_rows),
            "partitions": {
                "train": train_partition,
                "calibration": calibration_report,
            },
            "global_longest": global_longest,
            "violations": {
                "route_instances_over_cap": len(violations),
                "attack_records_over_cap": len(
                    {
                        (item["partition"], item["record_id"])
                        for item in violations
                        if item["data_role"] == "attack"
                    }
                ),
                "utility_records_over_cap": len(
                    {
                        item["record_id"]
                        for item in violations
                        if item["data_role"] == "clean_utility"
                    }
                ),
            },
            "violation_record_ids": sorted(
                {str(item["record_id"]) for item in violations}
            ),
            "violation_instances": violations,
        }
    )
    selection["partition"] = "train"
    selection["scope"] = "one_step_gate_train_only"
    return all_rows, report, selection
