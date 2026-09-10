"""Deterministic split-safe materialization for Stage 3 training sources."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
import json
from typing import Any, Mapping, Sequence


SOURCE_NAMESPACE = "bipia_official_train"
SOURCE_SCHEMA = "r-opcd-stage3-grouped-source-v1"
STATIC_SFT_SCHEMA = "r-opcd-stage3-grouped-static-sft-v1"
SPLIT_SEED = 20260902


@dataclass(frozen=True)
class GroupedSplit:
    split_by_base_digest: Mapping[str, str]
    diagnostics: Mapping[str, Any]


def base_task_digest(row: Mapping[str, Any]) -> str:
    """Hash the task, query, and clean context that define one split group."""

    payload = json.dumps(
        [row.get("task_name"), row.get("user_query"), row.get("clean_context")],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def base_task_id(
    row: Mapping[str, Any], *, source_namespace: str = SOURCE_NAMESPACE
) -> str:
    return f"r-opcd::{source_namespace}::base::{base_task_digest(row)}"


def namespaced_sample_id(
    raw_id: str,
    split: str,
    *,
    source_namespace: str = SOURCE_NAMESPACE,
) -> str:
    if split not in {"train", "dev"}:
        raise ValueError(f"unsupported derived split: {split}")
    if not raw_id.strip():
        raise ValueError("raw sample ID must be non-empty")
    return f"r-opcd::{source_namespace}::{split}::{raw_id}"


def choose_grouped_train_dev_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = SPLIT_SEED,
    dev_denominator: int = 5,
    attack_dev_groups_per_task: int = 2,
) -> GroupedSplit:
    """Choose a base-context grouped split with balanced attacked dev groups.

    The current source has ten attacked base contexts per task.  Selecting two
    makes the attacked development fraction exactly one fifth. Candidate pairs
    first maximize attack-name coverage, then minimize category/position and
    attack-name allocation error, with a seeded hash used only as a final tie
    breaker. Benign-only groups fill the remaining one-fifth group quota.
    """

    if not rows:
        raise ValueError("cannot split an empty source")
    if dev_denominator < 2:
        raise ValueError("dev_denominator must be at least two")
    if attack_dev_groups_per_task < 1:
        raise ValueError("attack_dev_groups_per_task must be positive")

    rows_by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        task = str(row.get("task_name", "")).strip()
        if not task:
            raise ValueError("every row must have a non-empty task_name")
        rows_by_task[task].append(row)

    assignments: dict[str, str] = {}
    diagnostics: dict[str, Any] = {}
    for task in sorted(rows_by_task):
        task_rows = rows_by_task[task]
        all_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        attack_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in task_rows:
            digest = base_task_digest(row)
            all_groups[digest].append(row)
            if row.get("risk_label") == "context_injection":
                attack_groups[digest].append(row)

        if len(attack_groups) < attack_dev_groups_per_task:
            raise ValueError(
                f"{task} has only {len(attack_groups)} attacked groups, cannot "
                f"select {attack_dev_groups_per_task}"
            )
        attacked_rows = [
            row
            for group_rows in attack_groups.values()
            for row in group_rows
        ]
        full_names = Counter(str(row["attack_name"]) for row in attacked_rows)
        full_categories = Counter(
            str(row["attack_category"]) for row in attacked_rows
        )
        full_positions = Counter(
            str(row["attack_position"]) for row in attacked_rows
        )

        candidates: list[tuple[tuple[Any, ...], tuple[str, ...]]] = []
        for selected in combinations(
            sorted(attack_groups), attack_dev_groups_per_task
        ):
            selected_rows = [
                row for digest in selected for row in attack_groups[digest]
            ]
            observed_names = Counter(
                str(row["attack_name"]) for row in selected_rows
            )
            observed_categories = Counter(
                str(row["attack_category"]) for row in selected_rows
            )
            observed_positions = Counter(
                str(row["attack_position"]) for row in selected_rows
            )
            missing_names = len(set(full_names) - set(observed_names))
            category_position_error = sum(
                (
                    dev_denominator * observed_categories[name]
                    - full_categories[name]
                )
                ** 2
                for name in full_categories
            ) + sum(
                (
                    dev_denominator * observed_positions[name]
                    - full_positions[name]
                )
                ** 2
                for name in full_positions
            )
            name_error = sum(
                (dev_denominator * observed_names[name] - full_names[name]) ** 2
                for name in full_names
            )
            tie_break = sha256(
                f"{seed}|{task}|{'|'.join(selected)}".encode("utf-8")
            ).hexdigest()
            candidates.append(
                (
                    (
                        missing_names,
                        category_position_error,
                        name_error,
                        tie_break,
                    ),
                    selected,
                )
            )
        candidates.sort()
        best_score, selected_attack_groups = candidates[0]
        if best_score[0] != 0:
            raise ValueError(
                f"{task} cannot cover every attack name with "
                f"{attack_dev_groups_per_task} development groups"
            )

        dev_groups = set(selected_attack_groups)
        target_dev_groups = (
            len(all_groups) + dev_denominator - 1
        ) // dev_denominator
        benign_only_groups = [
            digest
            for digest in all_groups
            if digest not in attack_groups and digest not in dev_groups
        ]
        benign_only_groups.sort(
            key=lambda digest: (
                sha256(
                    f"{seed}|{task}|benign|{digest}".encode("utf-8")
                ).hexdigest(),
                digest,
            )
        )
        for digest in benign_only_groups:
            if len(dev_groups) >= target_dev_groups:
                break
            dev_groups.add(digest)
        if len(dev_groups) != target_dev_groups:
            raise ValueError(
                f"{task} selected {len(dev_groups)} dev groups, expected "
                f"{target_dev_groups}"
            )

        for digest in all_groups:
            derived_split = "dev" if digest in dev_groups else "train"
            if digest in assignments and assignments[digest] != derived_split:
                raise RuntimeError("one base digest received conflicting splits")
            assignments[digest] = derived_split

        dev_rows = [
            row for row in task_rows if base_task_digest(row) in dev_groups
        ]
        dev_attack_rows = [
            row
            for row in dev_rows
            if row.get("risk_label") == "context_injection"
        ]
        diagnostics[task] = {
            "all_base_groups": len(all_groups),
            "attack_base_groups": len(attack_groups),
            "dev_base_groups": len(dev_groups),
            "selected_attack_base_digests": list(selected_attack_groups),
            "selection_score": {
                "missing_attack_names": best_score[0],
                "category_position_error": best_score[1],
                "attack_name_error": best_score[2],
            },
            "dev_rows": len(dev_rows),
            "dev_attack_rows": len(dev_attack_rows),
            "dev_benign_rows": len(dev_rows) - len(dev_attack_rows),
            "dev_attack_names": dict(
                sorted(
                    Counter(
                        str(row["attack_name"]) for row in dev_attack_rows
                    ).items()
                )
            ),
            "dev_attack_categories": dict(
                sorted(
                    Counter(
                        str(row["attack_category"]) for row in dev_attack_rows
                    ).items()
                )
            ),
            "dev_attack_positions": dict(
                sorted(
                    Counter(
                        str(row["attack_position"]) for row in dev_attack_rows
                    ).items()
                )
            ),
        }

    observed_digests = {base_task_digest(row) for row in rows}
    if set(assignments) != observed_digests:
        raise RuntimeError("not every source base group received a split")
    return GroupedSplit(
        split_by_base_digest=assignments,
        diagnostics=diagnostics,
    )


def materialize_grouped_source_rows(
    rows: Sequence[Mapping[str, Any]],
    grouped_split: GroupedSplit,
    *,
    seed: int = SPLIT_SEED,
    source_namespace: str = SOURCE_NAMESPACE,
) -> list[dict[str, Any]]:
    """Copy rows into split-safe IDs while preserving raw provenance."""

    raw_rows = [dict(row) for row in rows]
    raw_ids = [str(row.get("id", "")) for row in raw_rows]
    if len(raw_ids) != len(set(raw_ids)) or any(not raw_id for raw_id in raw_ids):
        raise ValueError("raw source IDs must be unique and non-empty")
    split_by_raw_id = {
        str(row["id"]): grouped_split.split_by_base_digest[base_task_digest(row)]
        for row in raw_rows
    }
    raw_id_set = set(raw_ids)

    materialized: list[dict[str, Any]] = []
    for row in raw_rows:
        raw_id = str(row["id"])
        digest = base_task_digest(row)
        derived_split = grouped_split.split_by_base_digest[digest]
        source_split = str(row.get("split", ""))
        output = dict(row)
        output.update(
            {
                "schema_version": SOURCE_SCHEMA,
                "id": namespaced_sample_id(
                    raw_id,
                    derived_split,
                    source_namespace=source_namespace,
                ),
                "raw_id": raw_id,
                "source_split": source_split,
                "split": derived_split,
                "split_namespace": source_namespace,
                "split_seed": seed,
                "base_task_id": (
                    f"r-opcd::{source_namespace}::base::{digest}"
                ),
                "base_task_digest": digest,
            }
        )
        raw_replica = row.get("source_replica_of")
        if raw_replica:
            raw_replica = str(raw_replica)
            if raw_replica not in raw_id_set:
                raise ValueError(f"replica target is absent: {raw_replica}")
            if split_by_raw_id[raw_replica] != derived_split:
                raise ValueError("benign replica and source were split apart")
            output["raw_source_replica_of"] = raw_replica
            output["source_replica_of"] = namespaced_sample_id(
                raw_replica,
                derived_split,
                source_namespace=source_namespace,
            )
        materialized.append(output)

    ids = [row["id"] for row in materialized]
    if len(ids) != len(set(ids)):
        raise RuntimeError("materialized IDs are not unique")
    return materialized


def materialize_grouped_static_sft_rows(
    static_rows: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = SPLIT_SEED,
    source_namespace: str = SOURCE_NAMESPACE,
) -> list[dict[str, Any]]:
    """Apply the exact source split and namespace to the static-SFT control."""

    source_by_raw_id = {str(row["raw_id"]): row for row in source_rows}
    if len(source_by_raw_id) != len(source_rows):
        raise ValueError("materialized source raw IDs are not unique")
    outputs: list[dict[str, Any]] = []
    for row in static_rows:
        raw_source_id = str(row.get("source_id", ""))
        if raw_source_id not in source_by_raw_id:
            raise ValueError(f"static-SFT source is absent: {raw_source_id}")
        source = source_by_raw_id[raw_source_id]
        output = dict(row)
        output.update(
            {
                "schema_version": STATIC_SFT_SCHEMA,
                "source_id": source["id"],
                "raw_source_id": raw_source_id,
                "split": source["split"],
                "split_namespace": source_namespace,
                "split_seed": seed,
                "base_task_id": source["base_task_id"],
                "base_task_digest": source["base_task_digest"],
            }
        )
        outputs.append(output)
    if len(outputs) != len(source_rows):
        raise ValueError("static-SFT and source row counts differ")
    return outputs
