"""Freeze an outcome-independent, quota-balanced subset of existing rollouts.

Selection reads source metadata and membership only.  Generation contents are
validated for integrity, but neither their text nor stop status is a selection
input.  Interrupted prefixes retain their original, explicitly weaker evidence.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from r_opcd.frontier_collection import canonical_sha256
from r_opcd.interrupted_rollout_finalizer import _validate_completed_run
from r_opcd.interrupted_rollout_recovery import (
    PREPARE_SCHEMA, file_sha256, read_json, require, validate_student_generation,
)
from r_opcd.secopd_data import audit_transfer_package, build_matched_utility_rows


DATA_SCHEMA = "r-opcd-secopd-existing-subset-v1"
ROLLOUT_SCHEMA = "r-opcd-secopd-existing-subset-rollouts-v1"
SALT = "secopd-existing6000-v1-20260905"
QUOTAS = {"straightforward_prepend": 2700, "straightforward_append": 2700, "completion": 600}
BOUNDARY = (
    "Selection is quota-balanced within the frozen completed-by-cutoff availability "
    "pool, not an unconditional sample of all 9600 candidates. Source-length and "
    "scheduling availability bias is not removed by hashing. No response text, "
    "EOS/cap status, A/U, or attack success determines selection."
)


def production_spec(root: Path) -> dict[str, Any]:
    root = root.resolve()
    def pinned(relative: str, digest: str, rows: int | None = None) -> dict[str, Any]:
        value: dict[str, Any] = {"path": str(root / relative), "sha256": digest}
        if rows is not None:
            value["rows"] = rows
        return value
    completion_hashes = (
        "9cc67bf9593c22fe7a34db886c580728d2075b50623efa3c96f7306911ba426e",
        "14876a4d61f9de2863543d6d2a436aa78fcc5b28a90a79ec343fb17cf17e5f5f",
        "507702f707b224e17667e6b912f49ed3960803891b61c4118ea8e4bd83379a70",
        "e93eb944c36063b929978ceef2fa9edf949f6377471f2a7618c6bc3416ca32d7",
        "9b5c7909dff8b1e1998903523340dc6e1f7ebb4e948a8da3b354398fd3012543",
    )
    return {
        "schema": "r-opcd-secopd-existing-subset-spec-v1",
        "source": pinned("data/secopd_alpaca_transfer_9600_v2_5way/pilot_train/attack_candidates.jsonl", "5af1c49c5e18d83836f190cf57b85a73533ec08b7e1e2807c1e6e780df61a21a", 9600),
        "utility": pinned("data/secopd_alpaca_transfer_9600_v2_5way/pilot_train/utility_controls.jsonl", "dc8c22b21e9b7ab6955c579bcefe51bb08ef2bfc8ac0619694a6f03dc8857c28", 9600),
        "dev_source": pinned("data/secopd_alpaca_transfer_v1/pilot_dev/attack_candidates.jsonl", "43c27531718fa6980271e8f1033b5127ecd3342b3d40a97dfbc6e2494a893716", 120),
        "dev_utility": pinned("data/secopd_alpaca_transfer_v1/pilot_dev/utility_controls.jsonl", "075d9b8a477af1da7ffec830d6d75ad77c88f40498f781f109bb352538a21ebf", 120),
        "recovery_manifest": pinned("artifacts/secopd_scale9600_v2_interrupted_recovery_10way_20260905/manifest.json", "1dfebfd2fe4717a3080c9ea63b2a1a28ae9be81d69dbec455698c01cdfd7a602"),
        "completed_remainders": [dict(pinned(f"artifacts/secopd_scale9600_v2_remainder_rollout_shard{i:02d}_20260905/completion.json", digest), index=i) for i, digest in enumerate(completion_hashes)],
        "expected_partial_counts": [824, 828, 784, 817, 813],
        "expected_completed_counts": [554, 554, 554, 554, 553],
        "expected_pool_rows": 6835,
        "expected_pool_cells": {"straightforward_prepend": 3104, "straightforward_append": 3050, "completion": 681},
        "quotas": dict(QUOTAS), "salt": SALT, "expected_count": 6000,
    }


def _read_lines(path: Path) -> tuple[list[dict[str, Any]], list[bytes]]:
    """Preserve each JSON record's bytes; normalize only line delimiters on output."""
    raw = path.read_bytes()
    require(bool(raw), f"empty JSONL: {path}")
    lines = raw.splitlines()
    rows = []
    for index, line in enumerate(lines):
        require(bool(line.strip()), f"blank JSONL row: {path}:{index + 1}")
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError) as error:
            raise RuntimeError(f"invalid JSONL: {path}:{index + 1}") from error
        require(isinstance(row, dict), f"non-object JSONL: {path}:{index + 1}")
        rows.append(row)
    return rows, lines


def _ids(rows: Sequence[Mapping[str, Any]], field: str) -> list[str]:
    values = [row.get(field) for row in rows]
    require(all(isinstance(v, str) and v for v in values), f"empty or invalid {field}")
    require(len(values) == len(set(values)), f"duplicate {field}")
    return values


def _checked_file(spec: Mapping[str, Any], evidence: dict[str, str]) -> Path:
    require(isinstance(spec.get("path"), str), "missing pinned file path")
    path = Path(spec["path"]).resolve()
    require(path.is_file(), f"missing pinned file: {path}")
    actual = file_sha256(path)
    require(actual == spec.get("sha256"), f"pinned file SHA mismatch: {path}")
    evidence[str(path)] = actual
    return path


def _checked_rows(spec: Mapping[str, Any], evidence: dict[str, str], field: str) -> tuple[list[dict[str, Any]], list[bytes]]:
    path = _checked_file(spec, evidence)
    rows, lines = _read_lines(path)
    ids = _ids(rows, field)
    require(len(rows) == spec.get("rows"), f"pinned row count mismatch: {path}")
    if "source_ids_sha256" in spec:
        require(canonical_sha256(ids) == spec["source_ids_sha256"], f"pinned IDs mismatch: {path}")
    return rows, lines


def choose_source_ids(sources: Sequence[Mapping[str, Any]], available_ids: Sequence[str], *, quotas: Mapping[str, int], salt: str) -> list[str]:
    """The entire selection function: source ID/cell + availability membership."""
    source_ids = _ids(sources, "id")
    require(len(available_ids) == len(set(available_ids)), "duplicate available source ID")
    available = set(available_ids)
    require(available <= set(source_ids), "unknown available source ID")
    require(bool(salt) and "\0" not in salt, "selection salt must be nonempty and NUL-free")
    require(bool(quotas) and all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in quotas.values()), "quotas must be positive integers")
    cells: dict[str, list[str]] = {cell: [] for cell in quotas}
    for row in sources:
        if row["id"] in available:
            cell = row.get("secopd_attack_plan_cell")
            require(cell in cells, f"unrecognized selection cell: {cell}")
            cells[cell].append(row["id"])
    selected: set[str] = set()
    for cell, count in quotas.items():
        require(len(cells[cell]) >= count, f"insufficient availability for cell {cell}")
        ranking = sorted(cells[cell], key=lambda source_id: (sha256((salt + "\0" + source_id).encode("utf-8")).hexdigest(), source_id))
        selected.update(ranking[:count])
    require(len(selected) == sum(quotas.values()), "selection count mismatch")
    return [source_id for source_id in source_ids if source_id in selected]


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def build_existing_subset(*, spec: Mapping[str, Any], output_dir: Path, rollout_output_dir: Path, dry_run: bool = False) -> dict[str, Any]:
    output_dir, rollout_output_dir = output_dir.resolve(), rollout_output_dir.resolve()
    require(output_dir != rollout_output_dir and output_dir not in rollout_output_dir.parents and rollout_output_dir not in output_dir.parents, "output directories must be distinct and non-nested")
    for path in (output_dir, rollout_output_dir):
        if path.exists():
            raise FileExistsError(f"refusing existing output: {path}")
    require(spec.get("schema") == "r-opcd-secopd-existing-subset-spec-v1", "wrong subset specification schema")
    evidence: dict[str, str] = {}
    project_root = Path(__file__).resolve().parents[2]
    producer_code = [
        {"path": str(path), "sha256": file_sha256(path)}
        for path in (
            Path(__file__).resolve(), project_root / "tools/build_secopd_existing_subset.py",
            project_root / "src/r_opcd/interrupted_rollout_finalizer.py",
            project_root / "src/r_opcd/interrupted_rollout_recovery.py",
            project_root / "src/r_opcd/secopd_data.py",
            project_root / "src/r_opcd/training_mixture.py",
            project_root / "src/r_opcd/external_injection_data.py",
        )
    ]
    for entry in producer_code:
        _checked_file(entry, evidence)
    sources, source_lines = _checked_rows(spec["source"], evidence, "id")
    source_ids = _ids(sources, "id")
    source_by_id = dict(zip(source_ids, sources, strict=True))
    require(len({canonical_sha256(row) for row in sources}) == len(sources), "duplicate source row hashes")
    require(all(row.get("split") == "train" for row in sources), "nontrain row in parent source")
    recovery_path = _checked_file(spec["recovery_manifest"], evidence)
    recovery = read_json(recovery_path)
    require(recovery.get("status") == "pass" and recovery.get("schema") == PREPARE_SCHEMA, "recovery manifest did not pass")
    require(recovery["authoritative_source"]["sha256"] == spec["source"]["sha256"], "recovery source SHA mismatch")
    require(recovery["authoritative_source"]["rows"] == len(sources) and recovery["authoritative_source"]["source_ids_sha256"] == canonical_sha256(source_ids), "recovery source count/IDs mismatch")
    config_path = _checked_file(recovery["config"], evidence)
    config = read_json(config_path)
    code = recovery["asserted_interrupted_run_code_bundle"]
    code_hash = canonical_sha256(code)
    require(code_hash == recovery["asserted_interrupted_run_code_bundle_sha256"], "asserted code bundle SHA mismatch")
    for entry in code.values():
        _checked_file(entry, evidence)
    policy = {"checkpoint": config.get("checkpoint"), "student_adapter": None, "prompt": config.get("prompt"), "generation": config.get("generation")}
    policy_hash = canonical_sha256(policy)
    require(policy == recovery["model_policy_bundle"] and policy_hash == recovery["model_policy_bundle_sha256"], "model policy bundle mismatch")
    generations: dict[str, dict[str, Any]] = {}
    generation_lines: dict[str, bytes] = {}
    partials, completed = [], []
    original_ids: list[str] = []
    expected_remainder: set[str] = set()
    original_specs = recovery["original_shards"]
    require([item["partial"]["rows"] for item in original_specs] == spec["expected_partial_counts"], "original partial count contract mismatch")
    def retain(records: Sequence[dict[str, Any]], lines: Sequence[bytes]) -> None:
        for record, line in zip(records, lines, strict=True):
            source_id = record["source_id"]
            require(source_id not in generations, f"duplicate available source ID: {source_id}")
            require(source_id in source_by_id, f"unknown available source ID: {source_id}")
            validate_student_generation(record, source_by_id[source_id], config, context=f"availability {source_id}")
            generations[source_id], generation_lines[source_id] = record, line
    for item in original_specs:
        shard, _ = _checked_rows(item["source"], evidence, "id")
        shard_ids = _ids(shard, "id")
        for row in shard:
            require(row["id"] in source_by_id and row == source_by_id[row["id"]], "original shard row differs from source")
        original_ids.extend(shard_ids)
        records, lines = _checked_rows(item["partial"], evidence, "source_id")
        require(_ids(records, "source_id") == shard_ids[:len(records)], "partial is not a continuous original prefix")
        expected_remainder.update(shard_ids[len(records):])
        retain(records, lines)
        partials.append({**item["partial"], "status": "authenticated_interrupted_prefix_without_completion", "completion_synthesized": False})
    require(len(original_ids) == len(set(original_ids)) == len(sources) and set(original_ids) == set(source_ids), "original shards do not partition source")
    remainder, _ = _checked_rows(recovery["remainder"], evidence, "id")
    remainder_ids = _ids(remainder, "id")
    require(set(remainder_ids) == expected_remainder and set(remainder_ids).isdisjoint(generations), "prepared remainder/partial partition mismatch")
    expected_samples, concatenated = {}, []
    for shard_spec in recovery["remainder_sharding"]["shards"]:
        shard, _ = _checked_rows(shard_spec, evidence, "id")
        for row in shard:
            require(row["id"] in source_by_id and row == source_by_id[row["id"]], "prepared remainder row differs from source")
        require(shard_spec["sha256"] not in expected_samples, "duplicate prepared sample hash")
        expected_samples[shard_spec["sha256"]] = shard_spec
        concatenated.extend(_ids(shard, "id"))
    require(concatenated == remainder_ids, "prepared remainder shards do not reconstruct remainder")
    used_samples: set[str] = set()
    remainder_specs = recovery["remainder_sharding"]["shards"]
    for entry in spec["completed_remainders"]:
        completion_path = _checked_file(entry, evidence)
        run_dir = completion_path.parent
        records, proof, sample_hash = _validate_completed_run(
            run_dir, source_by_id=source_by_id, expected_samples=expected_samples, config=config,
            expected_config_sha256=recovery["config"]["sha256"], expected_code_sha256=code_hash,
            expected_policy_sha256=policy_hash,
        )
        require(sample_hash == remainder_specs[entry["index"]]["sha256"], "completed remainder index mismatch")
        require(sample_hash not in used_samples, "duplicate completed remainder sample")
        used_samples.add(sample_hash)
        actual_records, lines = _read_lines(run_dir / "generations.jsonl")
        require(actual_records == records, "completed generation changed during validation")
        evidence[str(run_dir / "generations.jsonl")] = proof["generations_sha256"]
        evidence[str(run_dir / "collection_summary.json")] = proof["collection_summary_sha256"]
        retain(records, lines)
        completed.append(proof)
    require([item["records"] for item in completed] == spec["expected_completed_counts"], "completed count contract mismatch")
    require(len(generations) == spec["expected_pool_rows"], "availability pool count mismatch")
    pool_ids = [source_id for source_id in source_ids if source_id in generations]
    pool_cells = dict(Counter(source_by_id[source_id]["secopd_attack_plan_cell"] for source_id in pool_ids))
    require(pool_cells == spec["expected_pool_cells"], "availability pool cell count mismatch")
    _ids(list(generations.values()), "trajectory_id")
    selected_ids = choose_source_ids(sources, pool_ids, quotas=spec["quotas"], salt=spec["salt"])
    require(len(selected_ids) == spec["expected_count"], "selected total differs from declared expected count")
    selected = set(selected_ids)
    selected_sources = [source_by_id[source_id] for source_id in selected_ids]
    utilities, utility_lines = _checked_rows(spec["utility"], evidence, "id")
    utility_source_ids = _ids(utilities, "source_id")
    require(set(utility_source_ids) == set(source_ids), "parent utility coverage mismatch")
    chosen_utility = [row for row in utilities if row["source_id"] in selected]
    expected_utility = build_matched_utility_rows(selected_sources, pairing_seed=selected_sources[0]["mixture_seed"])
    require(sorted(chosen_utility, key=lambda row: row["id"]) == expected_utility, "matched clean utility differs from converter contract")
    require(len({row["base_task_id"] for row in selected_sources}) == len(selected_sources), "selected clean tasks repeat")
    dev, _ = _checked_rows(spec["dev_source"], evidence, "id")
    dev_utility, _ = _checked_rows(spec["dev_utility"], evidence, "id")
    require(all(row.get("split") == "dev" for row in dev), "nondev row in frozen dev")
    audit = audit_transfer_package({"train": selected_sources, "dev": dev}, {"train": chosen_utility, "dev": dev_utility})
    require(all(value == 0 for value in audit["train_dev_overlap"].values()), "train/dev overlap")
    for path, digest in evidence.items():
        require(file_sha256(Path(path)) == digest, f"input changed during validation: {path}")
    source_bytes = b"".join(line + b"\n" for row, line in zip(sources, source_lines, strict=True) if row["id"] in selected)
    utility_bytes = b"".join(line + b"\n" for row, line in zip(utilities, utility_lines, strict=True) if row["source_id"] in selected)
    generation_bytes = b"".join(generation_lines[source_id] + b"\n" for source_id in selected_ids)
    # Stop status is inspected for downstream recovery ONLY AFTER ID selection.
    cap_ids = {source_id for source_id in selected_ids if generations[source_id]["stop_reason"] == "max_new_tokens"}
    cap_bytes = b"".join(line + b"\n" for row, line in zip(sources, source_lines, strict=True) if row["id"] in cap_ids)
    files: dict[Path, bytes] = {
        output_dir / "pilot_train/attack_candidates.jsonl": source_bytes,
        output_dir / "pilot_train/utility_controls.jsonl": utility_bytes,
        output_dir / "pilot_dev/attack_candidates.jsonl": Path(spec["dev_source"]["path"]).read_bytes(),
        output_dir / "pilot_dev/utility_controls.jsonl": Path(spec["dev_utility"]["path"]).read_bytes(),
        rollout_output_dir / "generations.jsonl": generation_bytes,
        rollout_output_dir / "cap_rows.jsonl": cap_bytes,
    }
    def output_spec(path: Path, rows: int) -> dict[str, Any]:
        return {"path": str(path), "sha256": sha256(files[path]).hexdigest(), "rows": rows}
    source_out = output_spec(output_dir / "pilot_train/attack_candidates.jsonl", len(selected_ids))
    source_out["source_ids_sha256"] = canonical_sha256(selected_ids)
    selection = {
        "salt": spec["salt"], "method": "sha256(salt + NUL + source_id) within cell",
        "quotas": dict(spec["quotas"]), "decision_fields": ["source_id", "secopd_attack_plan_cell"],
        "membership_input": "frozen_availability_source_ids", "semantic_labels_read": False,
        "response_content_used_for_selection": False, "stop_status_used_for_selection": False,
        "output_order": "parent authoritative source order",
        "source_ids_sha256_definition": "frontier_collection.canonical_sha256(list_of_source_ids)",
    }
    pool = {"rows": len(pool_ids), "cells": pool_cells, "source_ids_sha256": canonical_sha256(pool_ids), "completed_remainder_indices": [entry["index"] for entry in spec["completed_remainders"]], "evidence_boundary": BOUNDARY}
    provenance = {
        "schema": "r-opcd-secopd-existing-subset-provenance-v1", "status": "pass",
        "recovery_manifest": dict(spec["recovery_manifest"]), "partials": partials,
        "completed_remainder_runs": completed, "original_completion_synthesized": False,
        "code_evidence_boundary": recovery["code_evidence_boundary"],
        "config": dict(recovery["config"]), "config_sha256": recovery["config"]["sha256"],
        "code_bundle": code, "code_bundle_sha256": code_hash,
        "model_policy_bundle": policy, "model_policy_bundle_sha256": policy_hash,
        "verified_inputs": [{"path": path, "sha256": digest} for path, digest in sorted(evidence.items())],
        "selection_spec": dict(spec), "selection_spec_sha256": canonical_sha256(spec),
        "producer_code": producer_code,
        "availability_pool": pool, "evidence_boundary": BOUNDARY,
    }
    provenance_path = rollout_output_dir / "interrupted_run_provenance.json"
    files[provenance_path] = _json_bytes(provenance)
    provenance_ref = {"path": str(provenance_path), "sha256": sha256(files[provenance_path]).hexdigest()}
    manifest = {
        "schema": DATA_SCHEMA, "status": "pass", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_count": len(selected_ids), "selection": selection, "parent_source": dict(spec["source"]),
        "parent_utility_controls": dict(spec["utility"]), "authoritative_source": source_out,
        "utility_controls": output_spec(output_dir / "pilot_train/utility_controls.jsonl", len(chosen_utility)),
        "frozen_dev": {"attacks": output_spec(output_dir / "pilot_dev/attack_candidates.jsonl", len(dev)), "utilities": output_spec(output_dir / "pilot_dev/utility_controls.jsonl", len(dev_utility)), "original_attacks": dict(spec["dev_source"]), "original_utilities": dict(spec["dev_utility"]), "byte_exact_copies": True},
        "availability_pool": pool, "audit": audit, "interrupted_run_provenance": provenance_ref,
        "semantic_labels": "not_collected", "official_sep_test": "not_read_or_modified",
        "evidence_boundary": BOUNDARY,
    }
    manifest_path = output_dir / "manifest.json"
    files[manifest_path] = _json_bytes(manifest)
    initial = {
        "schema": ROLLOUT_SCHEMA, "status": "pass", "authoritative_source": source_out,
        "generations": output_spec(rollout_output_dir / "generations.jsonl", len(selected_ids)),
        "cap_rows": output_spec(rollout_output_dir / "cap_rows.jsonl", len(cap_ids)),
        "selection_manifest": {"path": str(manifest_path), "sha256": sha256(files[manifest_path]).hexdigest()},
        "interrupted_run_provenance": provenance_ref,
        "coverage": {"total_rows": len(selected_ids), "exact_disjoint_coverage": True, "authoritative_order_preserved": True, "unique_source_ids": True, "unique_trajectory_ids": True},
        "config": dict(recovery["config"]), "config_sha256": recovery["config"]["sha256"],
        "code_bundle": code, "code_bundle_sha256": code_hash,
        "model_policy_bundle": policy, "model_policy_bundle_sha256": policy_hash,
        "semantic_labels": "not_collected", "training_eligibility": "blocked_until_cap_recovery_and_independent_A_label",
        "evidence_boundary": BOUNDARY,
    }
    files[rollout_output_dir / "manifest.json"] = _json_bytes(initial)
    for path, digest in evidence.items():
        require(file_sha256(Path(path)) == digest, f"input changed before publication: {path}")
    if not dry_run:
        # Validation is complete before the first filesystem mutation. Exclusive
        # creates retain any interrupted build for inspection; never overwrite.
        output_dir.mkdir(parents=True, exist_ok=False)
        rollout_output_dir.mkdir(parents=True, exist_ok=False)
        for path, content in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(content)
    return {"status": "pass", "dry_run": dry_run, "dataset_manifest": manifest, "initial_manifest": initial, "selected_cap_rows": len(cap_ids), "selected_cells": dict(Counter(row["secopd_attack_plan_cell"] for row in selected_sources)), "output_file_hashes": {str(path): sha256(content).hexdigest() for path, content in files.items()}}
