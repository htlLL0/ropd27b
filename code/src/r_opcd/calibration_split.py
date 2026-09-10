"""Outcome-independent grouped split for reliability calibration."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import math
from typing import Any, Mapping, Sequence


ASSIGNMENT_SCHEMA = "r-opcd-stage3n-reliability-split-assignment-v1"


def selection_digest(*, seed: int, selection_salt: str, base_task_id: str) -> str:
    if not selection_salt or not base_task_id:
        raise ValueError("selection_salt and base_task_id must be non-empty")
    return sha256(
        f"{seed}|{selection_salt}|{base_task_id}".encode("utf-8")
    ).hexdigest()


def build_group_table(
    candidate_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse attack-positive trajectories into indivisible base-task groups."""

    groups: dict[str, dict[str, Any]] = {}
    seen_sources: set[str] = set()
    for row in candidate_rows:
        source_id = str(row["source_id"])
        if source_id in seen_sources:
            raise ValueError(f"duplicate source_id: {source_id}")
        seen_sources.add(source_id)
        base_task_id = str(row["base_task_id"])
        family = str(row["mixture_source_family"])
        group = groups.setdefault(
            base_task_id,
            {
                "base_task_id": base_task_id,
                "mixture_source_family": family,
                "source_ids": [],
                "attack_categories": set(),
            },
        )
        if group["mixture_source_family"] != family:
            raise ValueError(f"base task crosses source families: {base_task_id}")
        group["source_ids"].append(source_id)
        group["attack_categories"].add(str(row["attack_category"]))

    return [
        {
            "base_task_id": str(group["base_task_id"]),
            "mixture_source_family": str(group["mixture_source_family"]),
            "source_ids": sorted(group["source_ids"]),
            "source_records": len(group["source_ids"]),
            "attack_categories": sorted(group["attack_categories"]),
        }
        for group in sorted(groups.values(), key=lambda value: value["base_task_id"])
    ]


def assign_grouped_calibration(
    groups: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    namespace: str,
    selection_salt: str,
    calibration_fraction: float,
) -> list[dict[str, Any]]:
    """Select whole base tasks by hash rank within each source family."""

    if not namespace:
        raise ValueError("namespace must be non-empty")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must lie strictly between zero and one")
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for group in groups:
        base_task_id = str(group["base_task_id"])
        if base_task_id in seen:
            raise ValueError(f"duplicate base_task_id: {base_task_id}")
        seen.add(base_task_id)
        by_family[str(group["mixture_source_family"])].append(group)

    assignments: list[dict[str, Any]] = []
    for family, members in sorted(by_family.items()):
        if len(members) < 2:
            raise ValueError(f"source family has fewer than two groups: {family}")
        calibration_groups = math.floor(
            len(members) * calibration_fraction + 0.5
        )
        calibration_groups = min(
            len(members) - 1, max(1, calibration_groups)
        )
        ranked = sorted(
            members,
            key=lambda row: (
                selection_digest(
                    seed=seed,
                    selection_salt=selection_salt,
                    base_task_id=str(row["base_task_id"]),
                ),
                str(row["base_task_id"]),
            ),
        )
        for rank, group in enumerate(ranked):
            base_task_id = str(group["base_task_id"])
            assignments.append(
                {
                    "schema": ASSIGNMENT_SCHEMA,
                    "namespace": namespace,
                    "seed": seed,
                    "selection_salt": selection_salt,
                    "base_task_id": base_task_id,
                    "mixture_source_family": family,
                    "selection_digest": selection_digest(
                        seed=seed,
                        selection_salt=selection_salt,
                        base_task_id=base_task_id,
                    ),
                    "family_group_count": len(members),
                    "family_calibration_groups": calibration_groups,
                    "family_rank": rank,
                    "selection_split": (
                        "calibration" if rank < calibration_groups else "train"
                    ),
                    "source_ids": list(group["source_ids"]),
                    "source_records": int(group["source_records"]),
                    "attack_categories": list(group["attack_categories"]),
                }
            )
    return sorted(assignments, key=lambda row: str(row["base_task_id"]))
