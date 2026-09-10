"""Deterministic Stage 3F attack/utility pilot-mixture construction."""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from typing import Any, Mapping, Sequence

from r_opcd.attack_pilot import POSITIONS, build_base_messages
from r_opcd.external_injection_data import canonical_sha256


MIXTURE_SCHEMA = "r-opcd-stage3f-pilot-mixture-v1"
ATTACK_CANDIDATE_SCHEMA = "r-opcd-stage3f-attack-candidate-v1"
UTILITY_CONTROL_SCHEMA = "r-opcd-stage3f-utility-control-v1"
MIXTURE_NAMESPACE = "stage3f_pilot_mixture_v1"
MIXTURE_SEED = 20260902

BIPIA_FAMILY = "bipia_four_task"
STRUQ_FAMILY = "struq"
OPEN_PI_FAMILY = "open_prompt_injection"
SOURCE_FAMILIES = (BIPIA_FAMILY, STRUQ_FAMILY, OPEN_PI_FAMILY)

ATTACK_CANDIDATE_ANNOTATION_FIELDS = {
    "mixture_schema_version",
    "mixture_namespace",
    "mixture_seed",
    "mixture_branch",
    "mixture_source_family",
    "mixture_selection_cell",
    "attack_gate_status",
    "recoverability_c_status",
    "mixture_record_sha256",
}

BIPIA_SOURCE_DATASET = "bipia_grouped_official_train"
STRUQ_SOURCE_DATASET = "struq_alpaca_cleaned_adaptation"
OPEN_PI_SOURCE_DATASET = "open_prompt_injection_sst2_sms_adaptation"

ATTACK_QUOTAS = {
    "train": {
        BIPIA_FAMILY: 480,
        STRUQ_FAMILY: 480,
        OPEN_PI_FAMILY: 480,
    },
    "dev": {
        BIPIA_FAMILY: 120,
        STRUQ_FAMILY: 120,
        OPEN_PI_FAMILY: 120,
    },
}
UTILITY_QUOTAS = {
    "train": {
        BIPIA_FAMILY: 120,
        STRUQ_FAMILY: 120,
        OPEN_PI_FAMILY: 120,
    },
    "dev": {
        BIPIA_FAMILY: 32,
        STRUQ_FAMILY: 32,
        OPEN_PI_FAMILY: 32,
    },
}


def stable_order(
    rows: Sequence[Mapping[str, Any]], *, purpose: str, seed: int = MIXTURE_SEED
) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            sha256(f"{seed}|{purpose}|{row['id']}".encode("utf-8")).hexdigest(),
            str(row["id"]),
        ),
    )


def select_bipia_attack_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    per_task_attack_name: int,
    seed: int = MIXTURE_SEED,
) -> list[dict[str, Any]]:
    """Select every task/attack-name cell with position balance when possible."""

    attacked = [
        dict(row)
        for row in rows
        if row.get("split") == split
        and row.get("risk_label") == "context_injection"
    ]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in attacked:
        grouped[(str(row["task_name"]), str(row["attack_name"]))].append(row)
    if not grouped:
        raise ValueError(f"no BIPIA attack cells for {split}")

    selected: list[dict[str, Any]] = []
    for (task, attack_name), members in sorted(grouped.items()):
        if len(members) < per_task_attack_name:
            raise ValueError(
                f"BIPIA cell {task}/{attack_name} has {len(members)}, "
                f"needs {per_task_attack_name}"
            )
        chosen: list[dict[str, Any]] = []
        if per_task_attack_name >= len(POSITIONS):
            for position in POSITIONS:
                position_rows = [
                    row for row in members if row.get("attack_position") == position
                ]
                if not position_rows:
                    raise ValueError(
                        f"BIPIA cell {task}/{attack_name} lacks position {position}"
                    )
                chosen.append(
                    stable_order(
                        position_rows,
                        purpose=f"bipia:{split}:{task}:{attack_name}:{position}",
                        seed=seed,
                    )[0]
                )
        remaining = [row for row in members if row["id"] not in {x["id"] for x in chosen}]
        chosen.extend(
            stable_order(
                remaining,
                purpose=f"bipia:{split}:{task}:{attack_name}:remainder",
                seed=seed,
            )[: per_task_attack_name - len(chosen)]
        )
        if len(chosen) != per_task_attack_name:
            raise ValueError(f"BIPIA cell {task}/{attack_name} selection underflow")
        selected.extend(
            annotate_attack_candidate(
                row,
                source_family=BIPIA_FAMILY,
                selection_cell=f"{task}::{attack_name}",
                seed=seed,
            )
            for row in chosen
        )
    return sorted(selected, key=lambda row: str(row["id"]))


def select_external_attack_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    source_dataset: str,
    source_family: str,
    group_fields: Sequence[str],
    per_cell: int,
    seed: int = MIXTURE_SEED,
) -> list[dict[str, Any]]:
    eligible = [
        dict(row)
        for row in rows
        if row.get("split") == split
        and row.get("source_dataset") == source_dataset
    ]
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        grouped[tuple(str(row[field]) for field in group_fields)].append(row)
    if not grouped:
        raise ValueError(f"no {source_family} attack cells for {split}")
    selected: list[dict[str, Any]] = []
    for cell, members in sorted(grouped.items()):
        if len(members) < per_cell:
            raise ValueError(
                f"{source_family} cell {'::'.join(cell)} has {len(members)}, "
                f"needs {per_cell}"
            )
        purpose = f"{source_family}:{split}:{'::'.join(cell)}"
        selected.extend(
            annotate_attack_candidate(
                row,
                source_family=source_family,
                selection_cell="::".join(cell),
                seed=seed,
            )
            for row in stable_order(members, purpose=purpose, seed=seed)[:per_cell]
        )
    return sorted(selected, key=lambda row: str(row["id"]))


def annotate_attack_candidate(
    row: Mapping[str, Any],
    *,
    source_family: str,
    selection_cell: str,
    seed: int = MIXTURE_SEED,
) -> dict[str, Any]:
    result = dict(row)
    result.update(
        {
            "mixture_schema_version": ATTACK_CANDIDATE_SCHEMA,
            "mixture_namespace": MIXTURE_NAMESPACE,
            "mixture_seed": seed,
            "mixture_branch": "attack_candidate",
            "mixture_source_family": source_family,
            "mixture_selection_cell": selection_cell,
            "attack_gate_status": "pending_current_student_A_U_verification",
            "recoverability_c_status": "pending_verified_recovery_frontier",
        }
    )
    result["mixture_record_sha256"] = canonical_sha256(result)
    return result


def original_source_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only Stage 3F selection annotations from a source-backed row."""

    return {
        key: value
        for key, value in row.items()
        if key not in ATTACK_CANDIDATE_ANNOTATION_FIELDS
    }


def select_bipia_utility_controls(
    source_rows: Sequence[Mapping[str, Any]],
    static_rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    per_task: int,
    seed: int = MIXTURE_SEED,
) -> list[dict[str, Any]]:
    """Use independent benign rows and exclude source replicas."""

    static_by_source = {str(row["source_id"]): dict(row) for row in static_rows}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        if (
            row.get("split") == split
            and row.get("risk_label") == "benign"
            and not row.get("source_replica_of")
        ):
            grouped[str(row["task_name"])].append(dict(row))
    if not grouped:
        raise ValueError(f"no canonical BIPIA benign rows for {split}")

    controls: list[dict[str, Any]] = []
    for task, members in sorted(grouped.items()):
        if len(members) < per_task:
            raise ValueError(
                f"BIPIA benign task {task} has {len(members)}, needs {per_task}"
            )
        chosen = stable_order(
            members, purpose=f"bipia:{split}:{task}:utility", seed=seed
        )[:per_task]
        for row in chosen:
            source_id = str(row["id"])
            if source_id not in static_by_source:
                raise ValueError(f"BIPIA benign row lacks static SFT join: {source_id}")
            static = static_by_source[source_id]
            controls.append(
                build_utility_control(
                    source_row=row,
                    source_family=BIPIA_FAMILY,
                    instruction=str(static["instruction"]),
                    response=str(static["response"]),
                    role="independent_benign",
                    matched_attack_candidate_ids=(),
                    seed=seed,
                )
            )
    return sorted(controls, key=lambda row: str(row["id"]))


def select_external_utility_controls(
    attack_candidates: Sequence[Mapping[str, Any]],
    *,
    source_family: str,
    per_task: Mapping[str, int],
    seed: int = MIXTURE_SEED,
) -> list[dict[str, Any]]:
    """Build clean counterparts only from base tasks selected for attack use."""

    grouped_by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in attack_candidates:
        if row.get("mixture_source_family") == source_family:
            grouped_by_base[str(row["base_task_id"])].append(dict(row))
    representatives: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for members in grouped_by_base.values():
        representative = sorted(members, key=lambda row: str(row["id"]))[0]
        representatives[str(representative["task_name"])].append(representative)

    controls: list[dict[str, Any]] = []
    for task, quota in sorted(per_task.items()):
        candidates = representatives.get(task, [])
        if len(candidates) < quota:
            raise ValueError(
                f"{source_family} utility task {task} has {len(candidates)} "
                f"selected bases, needs {quota}"
            )
        chosen = stable_order(
            candidates,
            purpose=f"{source_family}:{task}:matched_utility",
            seed=seed,
        )[:quota]
        for row in chosen:
            source_row = original_source_row(row)
            clean_row = dict(source_row)
            clean_row["contaminated_context"] = row["clean_context"]
            messages = build_base_messages(clean_row)
            if len(messages) != 1 or messages[0]["role"] != "user":
                raise ValueError("clean utility prompt must be one ordinary user message")
            matched_ids = sorted(
                str(member["id"])
                for member in grouped_by_base[str(row["base_task_id"])]
            )
            controls.append(
                build_utility_control(
                    source_row=source_row,
                    source_family=source_family,
                    instruction=str(messages[0]["content"]),
                    response=str(row["target_answer"]),
                    role="matched_clean_counterpart",
                    matched_attack_candidate_ids=matched_ids,
                    seed=seed,
                )
            )
    return sorted(controls, key=lambda row: str(row["id"]))


def build_utility_control(
    *,
    source_row: Mapping[str, Any],
    source_family: str,
    instruction: str,
    response: str,
    role: str,
    matched_attack_candidate_ids: Sequence[str],
    seed: int = MIXTURE_SEED,
) -> dict[str, Any]:
    if not instruction.strip() or not response.strip():
        raise ValueError("utility instruction and response must be non-empty")
    source_id = str(source_row["id"])
    control_digest = canonical_sha256(
        [MIXTURE_NAMESPACE, source_id, source_family, role]
    )
    result = {
        "schema_version": UTILITY_CONTROL_SCHEMA,
        "id": (
            f"r-opcd::{MIXTURE_NAMESPACE}::{source_row['split']}::utility::"
            f"{control_digest}"
        ),
        "split": source_row["split"],
        "mixture_namespace": MIXTURE_NAMESPACE,
        "mixture_seed": seed,
        "mixture_branch": "benign_utility_control",
        "mixture_source_family": source_family,
        "control_role": role,
        "source_id": source_id,
        "source_row_sha256": canonical_sha256(dict(source_row)),
        "base_task_id": source_row["base_task_id"],
        "base_task_digest": source_row["base_task_digest"],
        "task_name": source_row["task_name"],
        "task_type": source_row["task_type"],
        "unified_task": source_row["unified_task"],
        "user_query": source_row["user_query"],
        "clean_context": source_row["clean_context"],
        "instruction": instruction,
        "response": response,
        "target_answer": source_row["target_answer"],
        "target_answers": source_row["target_answers"],
        "has_q1": False,
        "attack_gate": 0.0,
        "recoverability_c": 0.0,
        "teacher_pair_required": False,
        "training_objective": "ordinary_utility_sft_only",
        "matched_attack_candidate_ids": list(matched_attack_candidate_ids),
    }
    validate_utility_control(result)
    result["mixture_record_sha256"] = canonical_sha256(result)
    return result


def validate_utility_control(row: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "id",
        "split",
        "mixture_branch",
        "mixture_source_family",
        "source_id",
        "base_task_id",
        "instruction",
        "response",
        "has_q1",
        "attack_gate",
        "recoverability_c",
        "teacher_pair_required",
        "training_objective",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"utility control missing fields: {missing}")
    if row["schema_version"] != UTILITY_CONTROL_SCHEMA:
        raise ValueError("utility-control schema changed")
    if row["mixture_branch"] != "benign_utility_control":
        raise ValueError("utility-control branch changed")
    if row["has_q1"] is not False:
        raise ValueError("utility control must not fabricate q1")
    if float(row["attack_gate"]) != 0.0 or float(row["recoverability_c"]) != 0.0:
        raise ValueError("utility control must have attack_gate=0 and c=0")
    if row["teacher_pair_required"] is not False:
        raise ValueError("utility control must not require T-/T+ correction")
    forbidden = {
        "malicious_span",
        "malicious_span_char_start",
        "malicious_span_char_end",
        "quarantined_token_indices",
    }
    if forbidden & set(row):
        raise ValueError("utility control contains q1-only fields")
    if not str(row["instruction"]).strip() or not str(row["response"]).strip():
        raise ValueError("utility-control text must be non-empty")


def audit_pilot_mixture(
    attack_rows: Sequence[Mapping[str, Any]],
    utility_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not attack_rows or not utility_rows:
        raise ValueError("pilot mixture requires both attack and utility rows")
    for row in utility_rows:
        validate_utility_control(row)
    attack_ids = [str(row["id"]) for row in attack_rows]
    utility_ids = [str(row["id"]) for row in utility_rows]
    if len(attack_ids) != len(set(attack_ids)):
        raise ValueError("attack-candidate IDs are not unique")
    if len(utility_ids) != len(set(utility_ids)):
        raise ValueError("utility-control IDs are not unique")
    if set(attack_ids) & set(utility_ids):
        raise ValueError("attack and utility IDs overlap")

    split_report: dict[str, Any] = {}
    for split in ("train", "dev"):
        attacks = [row for row in attack_rows if row["split"] == split]
        controls = [row for row in utility_rows if row["split"] == split]
        observed_attacks = Counter(str(row["mixture_source_family"]) for row in attacks)
        observed_controls = Counter(str(row["mixture_source_family"]) for row in controls)
        if dict(observed_attacks) != ATTACK_QUOTAS[split]:
            raise ValueError(
                f"{split} attack quotas changed: {dict(observed_attacks)}"
            )
        if dict(observed_controls) != UTILITY_QUOTAS[split]:
            raise ValueError(
                f"{split} utility quotas changed: {dict(observed_controls)}"
            )
        split_report[split] = {
            "attack_candidates": len(attacks),
            "utility_controls": len(controls),
            "attack_sources": dict(sorted(observed_attacks.items())),
            "utility_sources": dict(sorted(observed_controls.items())),
            "attack_tasks": dict(
                sorted(Counter(str(row["task_name"]) for row in attacks).items())
            ),
            "utility_tasks": dict(
                sorted(Counter(str(row["task_name"]) for row in controls).items())
            ),
            "attack_positions": dict(
                sorted(
                    Counter(str(row["attack_position"]) for row in attacks).items()
                )
            ),
        }

    train_attack_bases = {
        str(row["base_task_id"]) for row in attack_rows if row["split"] == "train"
    }
    dev_attack_bases = {
        str(row["base_task_id"]) for row in attack_rows if row["split"] == "dev"
    }
    train_utility_bases = {
        str(row["base_task_id"]) for row in utility_rows if row["split"] == "train"
    }
    dev_utility_bases = {
        str(row["base_task_id"]) for row in utility_rows if row["split"] == "dev"
    }
    if (train_attack_bases | train_utility_bases) & (
        dev_attack_bases | dev_utility_bases
    ):
        raise ValueError("train/dev mixture base-task leakage")
    return {
        "schema": "r-opcd-stage3f-pilot-mixture-audit-v1",
        "status": "pass",
        "rows": len(attack_rows) + len(utility_rows),
        "attack_candidates": len(attack_rows),
        "utility_controls": len(utility_rows),
        "splits": split_report,
        "train_dev_id_overlap": 0,
        "train_dev_base_task_overlap": 0,
        "utility_q1_rows": 0,
        "utility_nonzero_c_rows": 0,
    }
