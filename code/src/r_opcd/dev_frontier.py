"""Deterministic construction helpers for the held-out Stage 3 dev frontier."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from typing import Any, Iterable, Mapping, Sequence


MAIN_FAMILIES = ("open_prompt_injection", "struq")
EXCLUDED_FAMILY = "bipia_four_task"


def stable_key(seed: int, namespace: str, row_id: str) -> str:
    return sha256(f"{seed}|{namespace}|{row_id}".encode("utf-8")).hexdigest()


def assert_group_disjoint(
    train_rows: Sequence[Mapping[str, Any]],
    dev_rows: Sequence[Mapping[str, Any]],
    *,
    fields: Sequence[str] = ("id", "raw_id", "base_task_id", "base_task_digest"),
) -> dict[str, int]:
    """Fail closed unless every declared identity level is train/dev disjoint."""

    overlaps: dict[str, int] = {}
    for field in fields:
        train_values = [str(row[field]) for row in train_rows]
        dev_values = [str(row[field]) for row in dev_rows]
        if field in {"id", "raw_id"}:
            if len(set(train_values)) != len(train_values):
                raise ValueError(f"train {field} values are not unique")
            if len(set(dev_values)) != len(dev_values):
                raise ValueError(f"dev {field} values are not unique")
        overlap = set(train_values) & set(dev_values)
        overlaps[field] = len(overlap)
        if overlap:
            raise ValueError(f"train/dev {field} overlap: {len(overlap)}")
    return overlaps


def partition_dev_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate automatically judgeable main dev from preserved BIPIA sidecar."""

    main: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        family = str(row["mixture_source_family"])
        if family in MAIN_FAMILIES:
            row["dev_frontier_role"] = "automatic_main"
            main.append(row)
        elif family == EXCLUDED_FAMILY:
            row["dev_frontier_role"] = "preserved_ood_sidecar"
            row["dev_frontier_exclusion_reason"] = (
                "no_qualified_bipia_task_success_automatic_route"
            )
            excluded.append(row)
        else:
            raise ValueError(f"unknown Stage 3 source family: {family}")
    return main, excluded


def choose_stratified_smoke(
    rows: Sequence[Mapping[str, Any]], *, seed: int, per_stratum: int = 3
) -> list[dict[str, Any]]:
    """Choose three rows per family and attack category by default."""

    strata: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        family = str(row["mixture_source_family"])
        if family in MAIN_FAMILIES:
            stratum = f"{family}::category={row['attack_category']}"
        else:
            raise ValueError(f"smoke row is not in the automatic main: {family}")
        strata[stratum].append(row)
    selected: list[dict[str, Any]] = []
    for stratum in sorted(strata):
        candidates = sorted(
            strata[stratum],
            key=lambda row: stable_key(seed, f"smoke::{stratum}", str(row["id"])),
        )
        if len(candidates) < per_stratum:
            raise ValueError(f"too few rows for smoke stratum {stratum}")
        selected.extend(dict(row) for row in candidates[:per_stratum])
    return selected


def balanced_shards(
    rows: Sequence[Mapping[str, Any]],
    *,
    count: int,
    max_new_tokens_by_task: Mapping[str, int],
) -> tuple[list[list[dict[str, Any]]], list[int]]:
    """Allocate longest estimated Student generations first, deterministically."""

    if count < 1:
        raise ValueError("shard count must be positive")
    weighted = [
        (
            int(max_new_tokens_by_task[str(row["unified_task"])]),
            str(row["id"]),
            row,
        )
        for row in rows
    ]
    shards: list[list[dict[str, Any]]] = [[] for _ in range(count)]
    loads = [0] * count
    for work, _, row in sorted(weighted, key=lambda item: (-item[0], item[1])):
        index = min(range(count), key=lambda value: (loads[value], value))
        shards[index].append(dict(row))
        loads[index] += work
    for shard in shards:
        shard.sort(key=lambda row: str(row["id"]))
    return shards, loads


def flatten(items: Iterable[Iterable[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    return [dict(row) for group in items for row in group]
