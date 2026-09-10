"""Fail-closed recovery for interrupted deterministic Student rollouts.

An interrupted runner has no completion record, so its generated prefix is
never promoted to a successful run.  This module only authenticates and
freezes that prefix, materializes the not-yet-generated source rows, and later
combines it with separately completed remainder runs under explicit provenance.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from r_opcd.frontier_collection import canonical_sha256


PREPARE_SCHEMA = "r-opcd-interrupted-student-rollout-recovery-v1"
STUDENT_SCHEMA = "r-opcd-stage3-student-rollout-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return dict(value)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    require(bool(raw), f"JSONL is empty: {path}")
    require(raw.endswith(b"\n"), f"JSONL lacks terminal newline: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        require(bool(line.strip()), f"blank JSONL row at {path}:{line_number}")
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"invalid UTF-8 JSON at {path}:{line_number}") from error
        require(isinstance(value, dict), f"expected JSON object at {path}:{line_number}")
        rows.append(dict(value))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _hex_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_int_tokens(value: Any, *, label: str, allow_empty: bool = False) -> list[int]:
    require(isinstance(value, list), f"{label} is not a list")
    require(allow_empty or bool(value), f"{label} is empty")
    require(
        all(isinstance(token, int) and not isinstance(token, bool) and token >= 0 for token in value),
        f"{label} contains an invalid token ID",
    )
    return list(value)


def validate_student_generation(
    record: Mapping[str, Any],
    source: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    context: str,
) -> None:
    """Validate a deterministic base-model Student record against source/config."""

    source_id = str(source.get("id", ""))
    require(record.get("schema") == STUDENT_SCHEMA, f"{context}: wrong schema")
    require(record.get("source_id") == source_id, f"{context}: source ID/order drift")
    require(
        record.get("source_row_sha256") == canonical_sha256(source),
        f"{context}: source row hash mismatch",
    )
    require(
        record.get("trajectory_id") == f"{source_id}::student_on_policy",
        f"{context}: unexpected trajectory ID",
    )
    for field in (
        "raw_id", "base_task_id", "split", "unified_task", "task_name", "task_type",
        "attack_name", "attack_category", "attack_position", "mixture_schema_version",
        "mixture_namespace", "mixture_seed", "mixture_branch", "mixture_source_family",
        "mixture_selection_cell", "mixture_record_sha256", "attack_gate_status",
        "recoverability_c_status",
    ):
        if field in source:
            require(record.get(field) == source.get(field), f"{context}: propagated {field} drift")

    checkpoint = config.get("checkpoint", {})
    generation = config.get("generation", {})
    prompt = config.get("prompt", {})
    require(record.get("policy_role") == "current_student_on_policy", f"{context}: wrong policy role")
    require(record.get("model_revision") == checkpoint.get("snapshot_revision"), f"{context}: model revision drift")
    require(record.get("student_adapter") is None, f"{context}: adapter is forbidden")
    require(generation.get("do_sample") is False, "recovery only supports frozen greedy config")
    require(record.get("do_sample") is False, f"{context}: rollout is not greedy")
    require(record.get("decode_seed_index") == 0, f"{context}: unexpected decode seed index")
    require(record.get("decode_seed") is None, f"{context}: decode seed must be null")
    require(record.get("effective_sampling_seed") is None, f"{context}: effective seed must be null")
    require(float(record.get("temperature")) == float(generation.get("temperature", 0.0)), f"{context}: temperature drift")
    require(float(record.get("top_p")) == float(generation.get("top_p", 1.0)), f"{context}: top_p drift")
    require(int(record.get("top_k")) == int(generation.get("top_k", 0)), f"{context}: top_k drift")
    task = str(source.get("unified_task", ""))
    expected_cap = int(generation.get("max_new_tokens_by_task", {}).get(task, -1))
    require(expected_cap > 0, f"{context}: config has no generation cap for task {task!r}")
    require(record.get("max_new_tokens") == expected_cap, f"{context}: generation cap drift")
    require(record.get("prompt_view") == prompt.get("student_view"), f"{context}: prompt view drift")
    require(record.get("prompt_consumed_fields") == ["user_query", "contaminated_context"], f"{context}: prompt fields drift")

    prompt_ids = _validate_int_tokens(record.get("prompt_token_ids"), label=f"{context}: prompt IDs")
    mask = record.get("prompt_attention_mask")
    require(isinstance(mask, list) and len(mask) == len(prompt_ids), f"{context}: prompt/mask length mismatch")
    require(all(value in (1, True) for value in mask), f"{context}: Student prompt mask is not all ones")
    require(record.get("prompt_tokens") == len(prompt_ids), f"{context}: prompt token count mismatch")
    require(record.get("prompt_token_ids_sha256") == canonical_sha256(prompt_ids), f"{context}: prompt token hash mismatch")
    require(
        record.get("prompt_attention_mask_sha256") == sha256(bytes(mask)).hexdigest(),
        f"{context}: prompt mask hash mismatch",
    )
    require(_hex_sha256(record.get("prompt_rendered_sha256")), f"{context}: invalid rendered prompt hash")
    require(record.get("attention_quarantined_token_indices") == [], f"{context}: Student prompt unexpectedly quarantined")
    require(record.get("attention_quarantined_tokens") == 0, f"{context}: Student quarantine count drift")

    response_ids = _validate_int_tokens(record.get("response_token_ids"), label=f"{context}: response IDs")
    content_ids = _validate_int_tokens(record.get("response_content_token_ids"), label=f"{context}: response content IDs", allow_empty=True)
    require(len(content_ids) <= len(response_ids), f"{context}: response content exceeds raw response")
    require(response_ids[: len(content_ids)] == content_ids, f"{context}: response content is not raw-response prefix")
    require(record.get("response_token_ids_sha256") == canonical_sha256(response_ids), f"{context}: response token hash mismatch")
    require(record.get("generated_tokens_including_terminal") == len(response_ids), f"{context}: generated token count mismatch")
    require(record.get("response_content_tokens") == len(content_ids), f"{context}: response content count mismatch")
    stop_reason = record.get("stop_reason")
    require(stop_reason in {"eos_token", "max_new_tokens", "generation_stopped_other"}, f"{context}: invalid stop reason")
    if stop_reason == "max_new_tokens":
        require(len(response_ids) == expected_cap, f"{context}: capped response length mismatch")
    else:
        require(len(response_ids) <= expected_cap, f"{context}: response exceeds configured cap")
    require(record.get("attack_success_label") == "unjudged_semantic", f"{context}: semantic label unexpectedly present")
    require(record.get("user_task_success_label") == "unjudged_or_proxy_only", f"{context}: utility label unexpectedly present")


def _balanced_slices(row_count: int, shard_count: int) -> list[slice]:
    require(shard_count >= 1, "shards must be positive")
    require(shard_count <= row_count, "shards cannot exceed remainder rows")
    base, extra = divmod(row_count, shard_count)
    slices: list[slice] = []
    start = 0
    for index in range(shard_count):
        size = base + (1 if index < extra else 0)
        slices.append(slice(start, start + size))
        start += size
    return slices


def _verify_model(model_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = config.get("checkpoint")
    require(isinstance(checkpoint, Mapping), "config checkpoint is missing")
    expected_files = checkpoint.get("expected_files")
    require(isinstance(expected_files, Mapping) and expected_files, "checkpoint expected_files is missing")
    actual: dict[str, str] = {}
    for filename, expected in expected_files.items():
        path = model_path / str(filename)
        require(path.is_file(), f"checkpoint file is missing: {path}")
        actual[str(filename)] = file_sha256(path)
        require(actual[str(filename)] == expected, f"checkpoint hash mismatch: {filename}")
    model_config = read_json(model_path / "config.json")
    require(model_config.get("architectures") == [checkpoint.get("expected_model_class")], "checkpoint model class mismatch")
    return {
        "path": str(model_path),
        "snapshot_revision": checkpoint.get("snapshot_revision"),
        "model_id": checkpoint.get("model_id"),
        "expected_files": actual,
    }


def _code_bundle(project_root: Path) -> dict[str, dict[str, str]]:
    paths = {
        "runner": project_root / "tools/run_stage3_frontier_collection.py",
        "attack_pilot": project_root / "src/r_opcd/attack_pilot.py",
        "attention_quarantine": project_root / "src/r_opcd/attention_quarantine.py",
        "frontier_collection": project_root / "src/r_opcd/frontier_collection.py",
        "model_adapter": project_root / "src/r_opcd/model_adapter.py",
    }
    for path in paths.values():
        require(path.is_file(), f"rollout code file is missing: {path}")
    return {name: {"path": str(path.resolve()), "sha256": file_sha256(path)} for name, path in sorted(paths.items())}


def prepare_interrupted_recovery(
    *,
    source_path: Path,
    original_shard_paths: Sequence[Path],
    partial_paths: Sequence[Path],
    config_path: Path,
    model_path: Path,
    output_dir: Path,
    shard_count: int,
    expected_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Authenticate interrupted prefixes and materialize exact dynamic remainders."""

    source_path = source_path.expanduser().resolve()
    original_shard_paths = [path.expanduser().resolve() for path in original_shard_paths]
    partial_paths = [path.expanduser().resolve() for path in partial_paths]
    config_path = config_path.expanduser().resolve()
    model_path = model_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    require(source_path.is_file(), f"authoritative source is missing: {source_path}")
    require(config_path.is_file(), f"config is missing: {config_path}")
    require(len(original_shard_paths) == len(partial_paths) > 0, "original shards and partials must have equal nonzero length")
    require(len(set(original_shard_paths)) == len(original_shard_paths), "original shard paths repeat")
    require(len(set(partial_paths)) == len(partial_paths), "partial paths repeat")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")

    source_hash = file_sha256(source_path)
    if expected_source_sha256 is not None:
        require(source_hash == expected_source_sha256, "authoritative source SHA-256 mismatch")
    sources = read_jsonl(source_path)
    source_ids = [str(row.get("id", "")) for row in sources]
    require(all(source_ids), "authoritative source contains an empty ID")
    require(len(source_ids) == len(set(source_ids)), "authoritative source IDs repeat")
    source_by_id = dict(zip(source_ids, sources, strict=True))
    config = read_json(config_path)
    require(config.get("generation", {}).get("do_sample") is False, "only deterministic greedy recovery is supported")
    require(config.get("prompt", {}).get("student_view") == "ordinary_attacked_prompt", "unexpected Student prompt contract")
    require(config.get("checkpoint", {}).get("model_id") == "Qwen/Qwen3-8B", "unexpected model ID")
    checkpoint_evidence = _verify_model(model_path, config)
    project_root = Path(__file__).resolve().parents[2]
    code = _code_bundle(project_root)
    policy = {
        "checkpoint": config["checkpoint"],
        "student_adapter": None,
        "prompt": config["prompt"],
        "generation": config["generation"],
    }

    original_owner: dict[str, int] = {}
    processed_records: dict[str, dict[str, Any]] = {}
    original_entries: list[dict[str, Any]] = []
    for index, (shard_path, partial_path) in enumerate(zip(original_shard_paths, partial_paths, strict=True)):
        require(shard_path.is_file(), f"original shard is missing: {shard_path}")
        require(partial_path.is_file(), f"partial generations file is missing: {partial_path}")
        require(not (partial_path.parent / "completion.json").exists(), f"partial unexpectedly has completion.json: {partial_path.parent}")
        require(not (partial_path.parent / "failure.json").exists(), f"partial unexpectedly has failure.json: {partial_path.parent}")
        before_hash = file_sha256(partial_path)
        shard_rows = read_jsonl(shard_path)
        partial_rows = read_jsonl(partial_path)
        after_hash = file_sha256(partial_path)
        require(before_hash == after_hash, f"partial changed while being read: {partial_path}")
        shard_ids = [str(row.get("id", "")) for row in shard_rows]
        require(all(shard_ids), f"original shard {index} contains an empty ID")
        require(len(shard_ids) == len(set(shard_ids)), f"original shard {index} IDs repeat")
        require(len(partial_rows) <= len(shard_rows), f"partial {index} exceeds original shard")
        for row, source_id in zip(shard_rows, shard_ids, strict=True):
            require(source_id in source_by_id, f"original shard {index} has unknown source ID {source_id}")
            require(canonical_sha256(row) == canonical_sha256(source_by_id[source_id]), f"original shard {index} row differs from authoritative source: {source_id}")
            require(source_id not in original_owner, f"source ID appears in multiple original shards: {source_id}")
            original_owner[source_id] = index
        partial_ids = [str(row.get("source_id", "")) for row in partial_rows]
        require(partial_ids == shard_ids[: len(partial_rows)], f"partial {index} is not a continuous prefix of its original shard")
        require(len(partial_ids) == len(set(partial_ids)), f"partial {index} source IDs repeat")
        for row_number, (record, source) in enumerate(zip(partial_rows, shard_rows, strict=False), start=1):
            validate_student_generation(record, source, config, context=f"partial {index} row {row_number}")
            source_id = str(source["id"])
            require(source_id not in processed_records, f"duplicate processed source ID: {source_id}")
            processed_records[source_id] = record
        original_entries.append({
            "index": index,
            "source": {"path": str(shard_path), "sha256": file_sha256(shard_path), "rows": len(shard_rows), "source_ids_sha256": canonical_sha256(shard_ids)},
            "partial": {"path": str(partial_path), "sha256": before_hash, "rows": len(partial_rows), "source_ids_sha256": canonical_sha256(partial_ids)},
            "continuous_prefix_verified": True,
            "remainder_rows": len(shard_rows) - len(partial_rows),
        })

    require(set(original_owner) == set(source_ids), "original shards do not exactly partition authoritative source IDs")
    remaining_rows = [row for row in sources if str(row["id"]) not in processed_records]
    remaining_ids = [str(row["id"]) for row in remaining_rows]
    require(len(processed_records) + len(remaining_rows) == len(sources), "processed/remainder count mismatch")
    require(set(processed_records).isdisjoint(remaining_ids), "processed and remainder IDs overlap")
    slices = _balanced_slices(len(remaining_rows), shard_count)

    output_dir.mkdir(parents=True)
    remainder_path = output_dir / "remainder_source.jsonl"
    write_jsonl(remainder_path, remaining_rows)
    shard_entries: list[dict[str, Any]] = []
    width = max(2, len(str(shard_count)))
    for index, shard_slice in enumerate(slices):
        rows = remaining_rows[shard_slice]
        ids = remaining_ids[shard_slice]
        path = output_dir / f"remainder_source-shard-{index:0{width}d}-of-{shard_count:0{width}d}.jsonl"
        write_jsonl(path, rows)
        shard_entries.append({"index": index, "path": str(path), "sha256": file_sha256(path), "rows": len(rows), "first_source_id": ids[0], "last_source_id": ids[-1], "source_ids_sha256": canonical_sha256(ids)})

    manifest: dict[str, Any] = {
        "schema": PREPARE_SCHEMA,
        "status": "pass",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "interrupted_run_status": "incomplete_without_completion_or_failure_record",
        "authoritative_source": {"path": str(source_path), "sha256": source_hash, "rows": len(sources), "source_ids_sha256": canonical_sha256(source_ids)},
        "config": {"path": str(config_path), "sha256": file_sha256(config_path), "schema": config.get("schema")},
        "checkpoint": checkpoint_evidence,
        "model_policy_bundle": policy,
        "model_policy_bundle_sha256": canonical_sha256(policy),
        "asserted_interrupted_run_code_bundle": code,
        "asserted_interrupted_run_code_bundle_sha256": canonical_sha256(code),
        "code_evidence_boundary": "The interrupted runner emitted no completion record. These are the frozen current rollout-code hashes asserted for recovery; record-level policy fields are independently validated, but the partial JSONL cannot itself prove historical code hashes.",
        "original_shards": original_entries,
        "processed_prefix": {"rows": len(processed_records), "source_ids_sha256_in_authoritative_order": canonical_sha256([source_id for source_id in source_ids if source_id in processed_records])},
        "remainder": {"path": str(remainder_path), "sha256": file_sha256(remainder_path), "rows": len(remaining_rows), "source_ids_sha256": canonical_sha256(remaining_ids)},
        "remainder_sharding": {"strategy": "deterministic_contiguous_balanced_in_authoritative_order_v1", "shard_count": shard_count, "shards": shard_entries},
        "validation": {"partials_stable_while_read": True, "partials_are_exact_continuous_original_shard_prefixes": True, "partial_source_ids_unique": True, "original_shards_exactly_partition_authoritative_source": True, "partial_records_match_source_and_frozen_greedy_base_policy": True, "processed_and_remainder_disjoint": True, "processed_plus_remainder_exactly_cover_source": True, "remainder_authoritative_order_preserved": True},
        "semantic_labels": "not_read_or_inferred",
        "evidence_boundary": "interrupted-prefix integrity and exact remainder construction only; no original completion is synthesized and no model outcome is inferred",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
