"""Final assembly for authenticated interrupted SecOPD Student rollouts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from r_opcd.frontier_collection import canonical_sha256
from r_opcd.interrupted_rollout_recovery import (
    PREPARE_SCHEMA,
    file_sha256,
    read_json,
    read_jsonl,
    require,
    validate_student_generation,
    write_jsonl,
)


FINAL_SCHEMA = "r-opcd-interrupted-student-rollout-finalization-v1"
PROVENANCE_SCHEMA = "r-opcd-interrupted-student-rollout-provenance-v1"
RUN_COMPLETION_SCHEMA = "r-opcd-stage3-frontier-collection-completion-v1"
ASSEMBLY_SCHEMA = "r-opcd-secopd-rollout-assembly-v1"


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _declared_file(root: Path, raw_path: Any, *, label: str) -> Path:
    require(isinstance(raw_path, str) and raw_path, f"{root}: missing {label} path")
    declared = Path(raw_path).expanduser()
    candidates = [declared] if declared.is_absolute() else [root / declared]
    copied = root / declared.name
    if copied not in candidates:
        candidates.append(copied)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(f"{root}: cannot resolve declared {label} file {raw_path!r}")


def _validated_code_bundle(root: Path, value: Any) -> str:
    require(isinstance(value, Mapping) and value, f"{root}: code bundle is empty")
    for name, entry in value.items():
        require(isinstance(name, str) and name, f"{root}: invalid code entry name")
        require(isinstance(entry, Mapping), f"{root}: malformed code entry {name}")
        require(
            isinstance(entry.get("path"), str) and entry.get("path"),
            f"{root}: code entry {name} lacks path",
        )
        require(_is_sha256(entry.get("sha256")), f"{root}: code entry {name} lacks SHA-256")
    return canonical_sha256(value)


def _expand_remainder_inputs(
    inputs: Sequence[Path],
    *,
    expected_remainder: Mapping[str, Any],
    expected_config_sha256: str,
    expected_code_sha256: str,
    expected_policy_sha256: str,
) -> tuple[list[Path], list[dict[str, Any]]]:
    require(bool(inputs), "at least one completed remainder input is required")
    run_dirs: list[Path] = []
    assembly_evidence: list[dict[str, Any]] = []
    for raw_root in inputs:
        root = raw_root.expanduser().resolve()
        require(root.is_dir(), f"remainder input is not a directory: {root}")
        if (root / "completion.json").is_file():
            run_dirs.append(root)
            continue
        manifest_path = root / "manifest.json"
        provenance_path = root / "completion_provenance.json"
        generations_path = root / "generations.jsonl"
        require(
            manifest_path.is_file()
            and provenance_path.is_file()
            and generations_path.is_file(),
            f"not a completed run or assembled output: {root}",
        )
        manifest = read_json(manifest_path)
        provenance = read_json(provenance_path)
        require(
            manifest.get("schema") == ASSEMBLY_SCHEMA and manifest.get("status") == "pass",
            f"{root}: invalid assembled manifest",
        )
        require(provenance.get("status") == "pass", f"{root}: invalid assembled provenance")
        require(
            manifest.get("source", {}).get("sha256") == expected_remainder.get("sha256"),
            f"{root}: assembled source SHA drift",
        )
        require(
            manifest.get("source", {}).get("rows") == expected_remainder.get("rows"),
            f"{root}: assembled source row-count drift",
        )
        require(
            manifest.get("generations", {}).get("sha256") == file_sha256(generations_path),
            f"{root}: assembled generations hash mismatch",
        )
        require(
            manifest.get("completion_provenance", {}).get("sha256")
            == file_sha256(provenance_path),
            f"{root}: assembled provenance hash mismatch",
        )
        require(manifest.get("config_sha256") == expected_config_sha256, f"{root}: assembled config drift")
        require(manifest.get("code_bundle_sha256") == expected_code_sha256, f"{root}: assembled code drift")
        require(manifest.get("policy_bundle_sha256") == expected_policy_sha256, f"{root}: assembled policy drift")
        shards = provenance.get("shards")
        require(isinstance(shards, list) and shards, f"{root}: assembled provenance has no shards")
        referenced_runs: list[str] = []
        for entry in shards:
            require(isinstance(entry, Mapping), f"{root}: malformed assembled shard provenance")
            run_dir = Path(str(entry.get("run_dir", ""))).expanduser().resolve()
            completion_path = run_dir / "completion.json"
            require(completion_path.is_file(), f"{root}: referenced completion missing: {run_dir}")
            require(
                entry.get("completion_sha256") == file_sha256(completion_path),
                f"{root}: referenced completion hash mismatch: {run_dir}",
            )
            run_dirs.append(run_dir)
            referenced_runs.append(str(run_dir))
        assembly_evidence.append(
            {
                "path": str(root),
                "manifest_sha256": file_sha256(manifest_path),
                "completion_provenance_sha256": file_sha256(provenance_path),
                "generations_sha256": file_sha256(generations_path),
                "run_dirs": referenced_runs,
            }
        )
    require(len(run_dirs) == len(set(run_dirs)), "completed run dirs repeat across inputs")
    return run_dirs, assembly_evidence


def _validate_completed_run(
    root: Path,
    *,
    source_by_id: Mapping[str, Mapping[str, Any]],
    expected_samples: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
    expected_config_sha256: str,
    expected_code_sha256: str,
    expected_policy_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    completion_path = root / "completion.json"
    summary_path = root / "collection_summary.json"
    generations_path = root / "generations.jsonl"
    require(completion_path.is_file(), f"{root}: missing completion.json")
    require(summary_path.is_file(), f"{root}: missing collection_summary.json")
    require(generations_path.is_file(), f"{root}: missing generations.jsonl")
    completion = read_json(completion_path)
    require(completion.get("schema") == RUN_COMPLETION_SCHEMA, f"{root}: wrong completion schema")
    require(
        completion.get("status") == "pass" and completion.get("mode") == "student",
        f"{root}: run did not complete as Student",
    )
    require(
        completion.get("generations_sha256") == file_sha256(generations_path),
        f"{root}: generations hash mismatch",
    )
    require(
        completion.get("collection_summary_sha256") == file_sha256(summary_path),
        f"{root}: summary hash mismatch",
    )
    config_path = _declared_file(root, completion.get("config"), label="config")
    sample_path = _declared_file(root, completion.get("sample"), label="sample")
    config_hash = file_sha256(config_path)
    sample_hash = file_sha256(sample_path)
    require(
        completion.get("config_sha256") == config_hash == expected_config_sha256,
        f"{root}: config hash drift",
    )
    require(completion.get("sample_sha256") == sample_hash, f"{root}: sample hash mismatch")
    require(sample_hash in expected_samples, f"{root}: sample is not a prepared remainder shard")
    expected_sample = expected_samples[sample_hash]
    sample = read_jsonl(sample_path)
    sample_ids = [str(row.get("id", "")) for row in sample]
    require(len(sample) == expected_sample.get("rows"), f"{root}: sample row-count drift")
    require(
        canonical_sha256(sample_ids) == expected_sample.get("source_ids_sha256"),
        f"{root}: sample ID drift",
    )
    require(all(sample_ids) and len(sample_ids) == len(set(sample_ids)), f"{root}: sample IDs invalid")
    for row, source_id in zip(sample, sample_ids, strict=True):
        require(source_id in source_by_id, f"{root}: unknown source ID {source_id}")
        require(
            canonical_sha256(row) == canonical_sha256(source_by_id[source_id]),
            f"{root}: sample row differs from authoritative source: {source_id}",
        )

    generations = read_jsonl(generations_path)
    generation_ids = [str(record.get("source_id", "")) for record in generations]
    require(generation_ids == sample_ids, f"{root}: generation order/coverage differs from sample")
    require(len(generation_ids) == len(set(generation_ids)), f"{root}: generation IDs repeat")
    for row_number, (record, source) in enumerate(zip(generations, sample, strict=True), start=1):
        validate_student_generation(record, source, config, context=f"{root} row {row_number}")

    summary = read_json(summary_path)
    require(completion.get("records") == len(generations), f"{root}: completion record count mismatch")
    require(summary.get("source_rows") == len(sample), f"{root}: summary source count mismatch")
    require(summary.get("trajectories") == len(generations), f"{root}: summary trajectory count mismatch")
    require(summary.get("do_sample") is False, f"{root}: summary is not greedy")
    require(
        summary.get("decode_seeds") == [None] and summary.get("decode_seed_count") == 1,
        f"{root}: summary decode contract drift",
    )
    require(completion.get("checkpoint") == config.get("checkpoint"), f"{root}: checkpoint drift")
    require(completion.get("student_adapter") is None, f"{root}: adapter is forbidden")
    require(completion.get("prompt") == config.get("prompt"), f"{root}: prompt drift")
    require(completion.get("generation") == config.get("generation"), f"{root}: generation drift")
    policy = {
        "checkpoint": completion.get("checkpoint"),
        "student_adapter": completion.get("student_adapter"),
        "prompt": completion.get("prompt"),
        "generation": completion.get("generation"),
    }
    require(canonical_sha256(policy) == expected_policy_sha256, f"{root}: policy hash drift")
    code_hash = _validated_code_bundle(root, completion.get("code"))
    require(code_hash == expected_code_sha256, f"{root}: code bundle drift")
    runtime = completion.get("runtime")
    require(isinstance(runtime, Mapping), f"{root}: runtime evidence missing")
    require(
        runtime.get("observed_gpu_name") == config.get("runtime", {}).get("expected_gpu_name"),
        f"{root}: observed GPU drift",
    )
    evidence = {
        "run_dir": str(root),
        "completion": str(completion_path),
        "completion_sha256": file_sha256(completion_path),
        "sample": str(sample_path),
        "sample_sha256": sample_hash,
        "sample_rows": len(sample),
        "generations": str(generations_path),
        "generations_sha256": file_sha256(generations_path),
        "records": len(generations),
        "collection_summary": str(summary_path),
        "collection_summary_sha256": file_sha256(summary_path),
        "code_bundle_sha256": code_hash,
        "model_policy_bundle_sha256": canonical_sha256(policy),
        "git_head_before_run": completion.get("git_head_before_run"),
        "cuda_visible_devices": runtime.get("cuda_visible_devices"),
    }
    return generations, evidence, sample_hash


def finalize_interrupted_recovery(
    *,
    recovery_manifest_path: Path,
    source_path: Path,
    partial_paths: Sequence[Path],
    remainder_inputs: Sequence[Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Merge authenticated partial prefixes with completed remainder runs."""

    recovery_manifest_path = recovery_manifest_path.expanduser().resolve()
    source_path = source_path.expanduser().resolve()
    partial_paths = [path.expanduser().resolve() for path in partial_paths]
    output_dir = output_dir.expanduser().resolve()
    require(recovery_manifest_path.is_file(), "recovery manifest is missing")
    require(source_path.is_file(), "authoritative source is missing")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    recovery = read_json(recovery_manifest_path)
    require(
        recovery.get("schema") == PREPARE_SCHEMA and recovery.get("status") == "pass",
        "recovery preparation did not pass",
    )
    source_spec = recovery.get("authoritative_source", {})
    require(
        source_spec.get("sha256") == file_sha256(source_path),
        "authoritative source hash differs from recovery manifest",
    )
    sources = read_jsonl(source_path)
    source_ids = [str(row.get("id", "")) for row in sources]
    require(
        len(sources) == source_spec.get("rows")
        and canonical_sha256(source_ids) == source_spec.get("source_ids_sha256"),
        "authoritative source rows/IDs drifted",
    )
    source_by_id = dict(zip(source_ids, sources, strict=True))

    config_spec = recovery.get("config", {})
    config_path = Path(str(config_spec.get("path", ""))).expanduser().resolve()
    require(
        config_path.is_file() and file_sha256(config_path) == config_spec.get("sha256"),
        "frozen recovery config is missing or changed",
    )
    config = read_json(config_path)
    original_specs = recovery.get("original_shards")
    require(isinstance(original_specs, list) and original_specs, "manifest lacks original shards")
    require(len(partial_paths) == len(original_specs), "partial file count differs from manifest")
    partial_records: dict[str, dict[str, Any]] = {}
    partial_evidence: list[dict[str, Any]] = []
    for index, (path, spec) in enumerate(zip(partial_paths, original_specs, strict=True)):
        partial_spec = spec.get("partial", {})
        declared_path = Path(str(partial_spec.get("path", ""))).expanduser().resolve()
        require(path == declared_path, f"partial {index} path differs from manifest")
        require(path.is_file(), f"partial {index} is missing")
        before_hash = file_sha256(path)
        rows = read_jsonl(path)
        after_hash = file_sha256(path)
        require(before_hash == after_hash == partial_spec.get("sha256"), f"partial {index} hash changed")
        require(len(rows) == partial_spec.get("rows"), f"partial {index} row count changed")
        ids = [str(row.get("source_id", "")) for row in rows]
        require(
            canonical_sha256(ids) == partial_spec.get("source_ids_sha256"),
            f"partial {index} ID sequence changed",
        )
        for row_number, record in enumerate(rows, start=1):
            source_id = str(record.get("source_id", ""))
            require(source_id in source_by_id, f"partial {index} has unknown source ID")
            require(source_id not in partial_records, f"partial ID repeats: {source_id}")
            validate_student_generation(
                record,
                source_by_id[source_id],
                config,
                context=f"partial {index} row {row_number}",
            )
            partial_records[source_id] = record
        partial_evidence.append(
            {
                "index": index,
                "path": str(path),
                "sha256": before_hash,
                "rows": len(rows),
                "source_ids_sha256": canonical_sha256(ids),
                "status": "authenticated_interrupted_prefix_without_completion",
                "completion_synthesized": False,
            }
        )

    remainder_spec = recovery.get("remainder", {})
    remainder_path = Path(str(remainder_spec.get("path", ""))).expanduser().resolve()
    require(
        remainder_path.is_file() and file_sha256(remainder_path) == remainder_spec.get("sha256"),
        "prepared remainder source is missing or changed",
    )
    remainder_rows = read_jsonl(remainder_path)
    remainder_ids = [str(row.get("id", "")) for row in remainder_rows]
    require(
        len(remainder_rows) == remainder_spec.get("rows")
        and canonical_sha256(remainder_ids) == remainder_spec.get("source_ids_sha256"),
        "prepared remainder rows/IDs drifted",
    )
    require(set(partial_records).isdisjoint(remainder_ids), "partial/remainder IDs overlap")
    require(
        set(partial_records) | set(remainder_ids) == set(source_ids),
        "partial plus remainder do not exactly cover source",
    )

    shard_specs = recovery.get("remainder_sharding", {}).get("shards")
    require(isinstance(shard_specs, list) and shard_specs, "manifest lacks remainder shards")
    expected_samples: dict[str, Mapping[str, Any]] = {}
    concatenated_ids: list[str] = []
    for spec in shard_specs:
        path = Path(str(spec.get("path", ""))).expanduser().resolve()
        require(
            path.is_file() and file_sha256(path) == spec.get("sha256"),
            f"prepared shard missing or changed: {path}",
        )
        rows = read_jsonl(path)
        ids = [str(row.get("id", "")) for row in rows]
        require(
            len(rows) == spec.get("rows")
            and canonical_sha256(ids) == spec.get("source_ids_sha256"),
            f"prepared shard rows/IDs drifted: {path}",
        )
        require(str(spec["sha256"]) not in expected_samples, "prepared shard hashes repeat")
        expected_samples[str(spec["sha256"])] = spec
        concatenated_ids.extend(ids)
    require(concatenated_ids == remainder_ids, "prepared shards do not reconstruct remainder")

    expected_config_sha256 = str(config_spec.get("sha256"))
    expected_code_sha256 = str(recovery.get("asserted_interrupted_run_code_bundle_sha256"))
    expected_policy_sha256 = str(recovery.get("model_policy_bundle_sha256"))
    run_dirs, assembly_evidence = _expand_remainder_inputs(
        remainder_inputs,
        expected_remainder=remainder_spec,
        expected_config_sha256=expected_config_sha256,
        expected_code_sha256=expected_code_sha256,
        expected_policy_sha256=expected_policy_sha256,
    )
    completed_records: dict[str, dict[str, Any]] = {}
    completed_evidence: list[dict[str, Any]] = []
    used_samples: set[str] = set()
    for root in run_dirs:
        records, evidence, sample_hash = _validate_completed_run(
            root,
            source_by_id=source_by_id,
            expected_samples=expected_samples,
            config=config,
            expected_config_sha256=expected_config_sha256,
            expected_code_sha256=expected_code_sha256,
            expected_policy_sha256=expected_policy_sha256,
        )
        require(sample_hash not in used_samples, "prepared shard completed more than once")
        used_samples.add(sample_hash)
        for record in records:
            source_id = str(record["source_id"])
            require(source_id not in completed_records, f"completed ID repeats: {source_id}")
            completed_records[source_id] = record
        completed_evidence.append(evidence)
    require(used_samples == set(expected_samples), "completed runs do not cover every prepared shard")
    require(set(completed_records) == set(remainder_ids), "completed records do not cover remainder")
    require(set(partial_records).isdisjoint(completed_records), "partial/completed records overlap")

    all_records = {**partial_records, **completed_records}
    require(set(all_records) == set(source_ids), "final records do not exactly cover source")
    merged = [all_records[source_id] for source_id in source_ids]
    trajectory_ids = [str(record.get("trajectory_id", "")) for record in merged]
    require(
        all(trajectory_ids) and len(trajectory_ids) == len(set(trajectory_ids)),
        "final trajectory IDs are empty or repeat",
    )
    cap_ids = {
        str(record["source_id"])
        for record in merged
        if record.get("stop_reason") == "max_new_tokens"
    }
    cap_rows = [source_by_id[source_id] for source_id in source_ids if source_id in cap_ids]

    output_dir.mkdir(parents=True)
    generations_out = output_dir / "generations.jsonl"
    cap_rows_out = output_dir / "cap_rows.jsonl"
    provenance_out = output_dir / "interrupted_run_provenance.json"
    write_jsonl(generations_out, merged)
    write_jsonl(cap_rows_out, cap_rows)
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "status": "pass",
        "recovery_manifest": {
            "path": str(recovery_manifest_path),
            "sha256": file_sha256(recovery_manifest_path),
        },
        "interrupted_run_status": "incomplete_without_completion_or_failure_record",
        "original_completion_synthesized": False,
        "partials": partial_evidence,
        "completed_remainder_runs": completed_evidence,
        "accepted_assembled_inputs": assembly_evidence,
        "config_sha256": expected_config_sha256,
        "code_bundle_sha256": expected_code_sha256,
        "model_policy_bundle_sha256": expected_policy_sha256,
        "evidence_boundary": (
            "Interrupted prefixes are authenticated record-by-record but remain "
            "non-completed runs; only remainder runs carry normal completion evidence."
        ),
    }
    provenance_out.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": FINAL_SCHEMA,
        "status": "pass",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authoritative_source": {
            "path": str(source_path),
            "sha256": file_sha256(source_path),
            "rows": len(sources),
            "source_ids_sha256": canonical_sha256(source_ids),
        },
        "coverage": {
            "partial_rows": len(partial_records),
            "completed_remainder_rows": len(completed_records),
            "total_rows": len(merged),
            "exact_disjoint_coverage": True,
            "authoritative_order_preserved": True,
            "unique_source_ids": True,
            "unique_trajectory_ids": True,
        },
        "generations": {
            "path": str(generations_out),
            "sha256": file_sha256(generations_out),
            "rows": len(merged),
        },
        "cap_rows": {
            "path": str(cap_rows_out),
            "sha256": file_sha256(cap_rows_out),
            "rows": len(cap_rows),
            "semantics": "authoritative source rows whose rollout stopped at max_new_tokens",
        },
        "interrupted_run_provenance": {
            "path": str(provenance_out),
            "sha256": file_sha256(provenance_out),
        },
        "config_sha256": expected_config_sha256,
        "code_bundle_sha256": expected_code_sha256,
        "model_policy_bundle_sha256": expected_policy_sha256,
        "semantic_labels": "not_collected",
        "training_eligibility": "blocked_until_cap_recovery_and_independent_A_label",
        "evidence_boundary": (
            "lossless interrupted-run recovery and provenance integrity only; "
            "no original completion was fabricated and no semantic outcome is inferred"
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
