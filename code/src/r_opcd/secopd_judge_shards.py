"""Deterministic SecOPD judge sharding and fail-closed assembly.

The judge runner writes the complete source case into every judgment row.  This
module treats that duplication as an integrity check: source fields must remain
identical, one judge/model/config identity must be used throughout, and every
``(case_id, target)`` pair must be present exactly once before any output is
created.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CASE_SCHEMA = "r-opcd-secopd-independent-au-case-v1"
JUDGE_COMPLETION_SCHEMA = "r-opcd-stage3-independent-au-judge-completion-v1"
SHARD_MANIFEST_SCHEMA = "r-opcd-secopd-judge-shards-manifest-v1"
ASSEMBLY_MANIFEST_SCHEMA = "r-opcd-secopd-judge-shard-assembly-manifest-v1"
ASSEMBLY_COMPLETION_SCHEMA = "r-opcd-secopd-judge-shard-assembly-completion-v1"
TARGETS = ("attack", "task")
ACCEPTED_RUN_STATUSES = {"pass", "completed_with_parse_errors"}
ROW_IDENTITY_FIELDS = (
    "judge_id",
    "judge_model_id",
    "judge_snapshot_revision",
    "verifier_version",
    "prompt_revision",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


@dataclass(frozen=True)
class JsonlRecord:
    value: dict[str, Any]
    raw_line: str


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def read_jsonl_records(path: Path) -> list[JsonlRecord]:
    records: list[JsonlRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"expected a JSON object at {path}:{line_number}")
            records.append(JsonlRecord(value=value, raw_line=line))
    return records


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [record.value for record in read_jsonl_records(path)]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _validate_cases(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    require(bool(rows), "cases JSONL is empty")
    case_ids: list[str] = []
    for index, row in enumerate(rows):
        require(row.get("schema") == CASE_SCHEMA, f"case {index} has the wrong schema")
        case_id = row.get("case_id")
        require(
            isinstance(case_id, str) and bool(case_id.strip()),
            f"case {index} lacks a non-empty case_id",
        )
        case_ids.append(case_id)
    require(len(case_ids) == len(set(case_ids)), "case IDs are not unique")
    return case_ids


def _write_raw_jsonl(path: Path, records: Sequence[JsonlRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        for record in records:
            handle.write(record.raw_line)
            if not record.raw_line.endswith(("\n", "\r")):
                handle.write("\n")


def prepare_judge_shards(
    *,
    cases_path: Path,
    output_dir: Path,
    num_shards: int,
) -> dict[str, Any]:
    """Partition cases by source index modulo ``num_shards``.

    Assignment has no random state.  Counts differ by at most one, order inside
    each shard follows source order, and the original JSON line is copied rather
    than re-serialized.
    """

    cases_path = cases_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    require(cases_path.is_file(), f"cases JSONL does not exist: {cases_path}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    require(type(num_shards) is int and num_shards > 0, "num_shards must be positive")

    records = read_jsonl_records(cases_path)
    case_ids = _validate_cases([record.value for record in records])
    require(
        num_shards <= len(records),
        "num_shards cannot exceed the number of cases (empty shards are forbidden)",
    )
    shards: list[list[JsonlRecord]] = [[] for _ in range(num_shards)]
    shard_indices: list[list[int]] = [[] for _ in range(num_shards)]
    for source_index, record in enumerate(records):
        shard_index = source_index % num_shards
        shards[shard_index].append(record)
        shard_indices[shard_index].append(source_index)

    sizes = [len(shard) for shard in shards]
    require(max(sizes) - min(sizes) <= 1, "internal error: shards are not balanced")
    output_dir.mkdir(parents=True)
    width = max(2, len(str(num_shards)))
    shard_entries: list[dict[str, Any]] = []
    observed_case_ids: list[str] = []
    for shard_index, shard in enumerate(shards):
        filename = (
            f"cases-shard-{shard_index:0{width}d}-of-{num_shards:0{width}d}.jsonl"
        )
        path = output_dir / filename
        _write_raw_jsonl(path, shard)
        written = read_jsonl(path)
        expected = [record.value for record in shard]
        require(written == expected, f"written shard content changed: {filename}")
        shard_case_ids = [str(row["case_id"]) for row in written]
        observed_case_ids.extend(shard_case_ids)
        indices = shard_indices[shard_index]
        shard_entries.append(
            {
                "index": shard_index,
                "path": filename,
                "rows": len(written),
                "sha256": file_sha256(path),
                "source_indices": {
                    "first": indices[0],
                    "last": indices[-1],
                    "stride": num_shards,
                },
                "case_ids_sha256": canonical_sha256(shard_case_ids),
            }
        )

    require(
        len(observed_case_ids) == len(set(observed_case_ids)) == len(case_ids),
        "internal error: shard case IDs overlap",
    )
    require(set(observed_case_ids) == set(case_ids), "internal error: shard coverage differs")
    manifest = {
        "schema": SHARD_MANIFEST_SCHEMA,
        "status": "pass",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "assignment": "source_order_round_robin_v1",
        "source": {
            "path": str(cases_path),
            "sha256": file_sha256(cases_path),
            "rows": len(records),
            "case_ids_in_source_order_sha256": canonical_sha256(case_ids),
        },
        "num_shards": num_shards,
        "shard_size_min": min(sizes),
        "shard_size_max": max(sizes),
        "balanced_within_one": True,
        "case_ids_exactly_covered": True,
        "case_ids_non_overlapping": True,
        "row_content_preserved": True,
        "shards": shard_entries,
        "evidence_boundary": "judge inputs only; no A/U outcome inferred",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _required_file(root: Path, filename: str) -> Path:
    path = root / filename
    require(path.is_file(), f"missing {filename}: {root}")
    return path.resolve()


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


def _nonempty_identity(record: Mapping[str, Any], *, artifact: Path) -> tuple[str, ...]:
    identity: list[str] = []
    for field in ROW_IDENTITY_FIELDS:
        value = record.get(field)
        require(
            isinstance(value, str) and bool(value.strip()),
            f"{artifact}: judgment row lacks non-empty {field}",
        )
        identity.append(value)
    return tuple(identity)


def _validate_case_copy(
    record: Mapping[str, Any],
    case: Mapping[str, Any],
    *,
    artifact: Path,
) -> None:
    case_id = str(case["case_id"])
    for key, expected in case.items():
        require(key in record, f"{artifact}: judgment omits source field {key}: {case_id}")
        require(
            record[key] == expected,
            f"{artifact}: judgment/source field mismatch for {key}: {case_id}",
        )


def _targets_in_record(record: Mapping[str, Any], *, artifact: Path) -> tuple[str, ...]:
    present: list[str] = []
    for target in TARGETS:
        if target in record:
            require(
                isinstance(record[target], Mapping),
                f"{artifact}: {target} payload must be an object: {record.get('case_id')}",
            )
            present.append(target)
    require(bool(present), f"{artifact}: judgment row has neither attack nor task payload")
    return tuple(present)


def _completion_bundle(completion: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "judge_id",
        "config_sha256",
        "model_registry_sha256",
        "judge_spec",
        "judge_config",
        "prompt_revision",
        "system_prompt_sha256",
        "verifier_contract_sha256_at_start",
        "code_sha256_at_start",
    )
    for field in required:
        require(completion.get(field) is not None, f"completion lacks {field}")
    return {field: completion[field] for field in required}


@dataclass(frozen=True)
class JudgmentArtifact:
    requested_path: Path
    judgments_path: Path
    rows: list[dict[str, Any]]
    completion_path: Path | None
    completion: dict[str, Any] | None
    targets: tuple[str, ...]
    completion_bundle: dict[str, Any] | None
    provenance: dict[str, Any]


def _load_artifact(
    requested_path: Path,
    *,
    case_by_id: Mapping[str, Mapping[str, Any]],
) -> JudgmentArtifact:
    requested = requested_path.expanduser().resolve()
    if requested.is_dir():
        root = requested
        judgments_path = _required_file(root, "judgments.jsonl")
        completion_path: Path | None = _required_file(root, "completion.json")
    else:
        require(requested.is_file(), f"judgment input does not exist: {requested}")
        judgments_path = requested
        root = requested.parent
        sibling = root / "completion.json"
        completion_path = (
            sibling.resolve()
            if requested.name == "judgments.jsonl" and sibling.is_file()
            else None
        )

    rows = read_jsonl(judgments_path)
    require(bool(rows), f"judgments JSONL is empty: {judgments_path}")
    row_targets = [_targets_in_record(row, artifact=judgments_path) for row in rows]
    union_targets = tuple(target for target in TARGETS if any(target in item for item in row_targets))

    completion: dict[str, Any] | None = None
    bundle: dict[str, Any] | None = None
    provenance: dict[str, Any] = {
        "requested_path": str(requested),
        "judgments": str(judgments_path),
        "judgments_sha256": file_sha256(judgments_path),
        "records": len(rows),
        "targets": list(union_targets),
        "completion_verified": False,
    }
    if completion_path is not None:
        completion = read_json(completion_path)
        require(
            completion.get("schema") == JUDGE_COMPLETION_SCHEMA,
            f"{root}: wrong judge completion schema",
        )
        require(
            completion.get("status") in ACCEPTED_RUN_STATUSES,
            f"{root}: judge run did not complete",
        )
        require(
            completion.get("judgments_sha256") == file_sha256(judgments_path),
            f"{root}: judgments file hash mismatch",
        )
        require(completion.get("records") == len(rows), f"{root}: record count mismatch")
        declared_judgments = _declared_file(
            root, completion.get("judgments"), label="judgments"
        )
        require(
            declared_judgments == judgments_path,
            f"{root}: completion declares a different judgments file",
        )
        declared_targets = completion.get("targets")
        require(
            isinstance(declared_targets, list)
            and bool(declared_targets)
            and len(declared_targets) == len(set(declared_targets))
            and set(declared_targets).issubset(TARGETS),
            f"{root}: invalid completion targets",
        )
        require(
            all(set(targets) == set(declared_targets) for targets in row_targets),
            f"{root}: row targets differ from completion targets",
        )
        union_targets = tuple(target for target in TARGETS if target in declared_targets)

        sample_path = _declared_file(root, completion.get("sample"), label="sample")
        require(
            completion.get("sample_sha256") == file_sha256(sample_path),
            f"{root}: sample file hash mismatch",
        )
        sample_rows = read_jsonl(sample_path)
        sample_ids = [str(row.get("case_id", "")) for row in sample_rows]
        judgment_ids = [str(row.get("case_id", "")) for row in rows]
        require(sample_ids == judgment_ids, f"{root}: sample/judgment order or IDs differ")
        require(len(sample_ids) == len(set(sample_ids)), f"{root}: sample IDs repeat")
        for sample_row, case_id in zip(sample_rows, sample_ids, strict=True):
            require(case_id in case_by_id, f"{root}: sample contains unknown case {case_id}")
            require(
                sample_row == case_by_id[case_id],
                f"{root}: sample row differs from authoritative cases: {case_id}",
            )

        config_path = _declared_file(root, completion.get("config"), label="config")
        require(
            completion.get("config_sha256") == file_sha256(config_path),
            f"{root}: config file hash mismatch",
        )
        registry_path = _declared_file(
            root, completion.get("model_registry"), label="model registry"
        )
        require(
            completion.get("model_registry_sha256") == file_sha256(registry_path),
            f"{root}: model registry file hash mismatch",
        )
        bundle = _completion_bundle(completion)
        provenance.update(
            {
                "completion_verified": True,
                "completion": str(completion_path),
                "completion_sha256": file_sha256(completion_path),
                "sample": str(sample_path),
                "sample_sha256": file_sha256(sample_path),
                "config": str(config_path),
                "config_sha256": file_sha256(config_path),
                "model_registry": str(registry_path),
                "model_registry_sha256": file_sha256(registry_path),
                "status": completion["status"],
                "git_head_before_run": completion.get("git_head_before_run"),
            }
        )
    return JudgmentArtifact(
        requested_path=requested,
        judgments_path=judgments_path,
        rows=rows,
        completion_path=completion_path,
        completion=completion,
        targets=union_targets,
        completion_bundle=bundle,
        provenance=provenance,
    )


def assemble_judge_shards(
    *,
    cases_path: Path,
    judgment_inputs: Sequence[Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Validate and assemble one judge's target/shard outputs in source order."""

    cases_path = cases_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    require(cases_path.is_file(), f"cases JSONL does not exist: {cases_path}")
    require(bool(judgment_inputs), "at least one judge run or judgments path is required")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    requested = [path.expanduser().resolve() for path in judgment_inputs]
    require(len(requested) == len(set(requested)), "judgment inputs repeat")

    cases = read_jsonl(cases_path)
    case_ids = _validate_cases(cases)
    case_by_id = dict(zip(case_ids, cases, strict=True))
    artifacts = [
        _load_artifact(path, case_by_id=case_by_id)
        for path in requested
    ]

    identities: set[tuple[str, ...]] = set()
    completion_bundles: dict[str, dict[str, Any]] = {}
    target_payloads: dict[tuple[str, str], Any] = {}
    target_owners: dict[tuple[str, str], str] = {}
    shared_by_case: dict[str, dict[str, Any]] = {}

    for artifact in artifacts:
        if artifact.completion_bundle is not None:
            completion_bundles[
                canonical_sha256(artifact.completion_bundle)
            ] = artifact.completion_bundle
        for record in artifact.rows:
            case_id = record.get("case_id")
            require(
                isinstance(case_id, str) and case_id in case_by_id,
                f"{artifact.judgments_path}: unknown or empty case_id {case_id!r}",
            )
            _validate_case_copy(
                record,
                case_by_id[case_id],
                artifact=artifact.judgments_path,
            )
            identity = _nonempty_identity(record, artifact=artifact.judgments_path)
            identities.add(identity)
            shared = {
                key: value
                for key, value in record.items()
                if key not in TARGETS and key not in ROW_IDENTITY_FIELDS
            }
            prior_shared = shared_by_case.setdefault(case_id, shared)
            require(
                prior_shared == shared,
                f"non-target judgment fields differ across shards: {case_id}",
            )
            for target in _targets_in_record(record, artifact=artifact.judgments_path):
                key = (case_id, target)
                require(
                    key not in target_payloads,
                    f"duplicate judgment for case/target {case_id}/{target}; "
                    f"first={target_owners.get(key)} second={artifact.judgments_path}",
                )
                target_payloads[key] = record[target]
                target_owners[key] = str(artifact.judgments_path)

    require(len(identities) == 1, "judgment shards mix judge/model/config identities")
    identity = next(iter(identities))
    require(
        len(completion_bundles) <= 1,
        "judgment shard model/config completion bundles are inconsistent",
    )
    if completion_bundles:
        completion_bundle = next(iter(completion_bundles.values()))
        require(
            completion_bundle["judge_id"] == identity[0],
            "completion/row judge IDs differ",
        )
        require(
            completion_bundle["prompt_revision"] == identity[4],
            "completion/row prompt revisions differ",
        )
        spec = completion_bundle["judge_spec"]
        require(isinstance(spec, Mapping), "completion judge_spec must be an object")
        require(
            spec.get("model_id") == identity[1]
            and spec.get("snapshot_revision") == identity[2],
            "completion/row judge model identities differ",
        )
    else:
        completion_bundle = None

    expected_keys = {(case_id, target) for case_id in case_ids for target in TARGETS}
    observed_keys = set(target_payloads)
    missing = sorted(expected_keys - observed_keys)
    extra = sorted(observed_keys - expected_keys)
    require(
        observed_keys == expected_keys,
        "judgments do not cover every case/target exactly once; "
        f"missing={missing[:10]} extra={extra[:10]}",
    )

    merged: list[dict[str, Any]] = []
    for case_id in case_ids:
        row = dict(shared_by_case[case_id])
        row.update(dict(zip(ROW_IDENTITY_FIELDS, identity, strict=True)))
        row["attack"] = target_payloads[(case_id, "attack")]
        row["task"] = target_payloads[(case_id, "task")]
        merged.append(row)

    output_dir.mkdir(parents=True)
    judgments_out = output_dir / "judgments.jsonl"
    write_jsonl(judgments_out, merged)
    manifest = {
        "schema": ASSEMBLY_MANIFEST_SCHEMA,
        "status": "pass",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(cases_path),
            "sha256": file_sha256(cases_path),
            "rows": len(cases),
            "case_ids_in_source_order_sha256": canonical_sha256(case_ids),
        },
        "judge_identity": dict(zip(ROW_IDENTITY_FIELDS, identity, strict=True)),
        "row_model_config_bundle_sha256": canonical_sha256(identity),
        "completion_bundle": completion_bundle,
        "completion_bundle_sha256": (
            canonical_sha256(completion_bundle) if completion_bundle is not None else None
        ),
        "inputs": [artifact.provenance for artifact in artifacts],
        "input_artifacts": len(artifacts),
        "completion_verified_artifacts": sum(
            artifact.completion is not None for artifact in artifacts
        ),
        "judgments_only_artifacts": sum(
            artifact.completion is None for artifact in artifacts
        ),
        "records": len(merged),
        "targets": list(TARGETS),
        "target_records": {target: len(cases) for target in TARGETS},
        "case_ids_exactly_covered": True,
        "case_targets_exactly_once": True,
        "source_order_restored": True,
        "source_case_fields_verified": True,
        "judgments": {
            "path": str(judgments_out),
            "sha256": file_sha256(judgments_out),
            "rows": len(merged),
        },
        "evidence_boundary": "single-judge raw A/U judgments only; no cross-judge consensus inferred",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    completion = {
        "schema": ASSEMBLY_COMPLETION_SCHEMA,
        "status": "pass",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "judge_id": identity[0],
        "judge_model_id": identity[1],
        "judge_snapshot_revision": identity[2],
        "verifier_version": identity[3],
        "prompt_revision": identity[4],
        "records": len(merged),
        "targets": list(TARGETS),
        "cases": str(cases_path),
        "cases_sha256": file_sha256(cases_path),
        "judgments": str(judgments_out),
        "judgments_sha256": file_sha256(judgments_out),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "input_artifacts": len(artifacts),
        "evidence_boundary": manifest["evidence_boundary"],
    }
    completion_path = output_dir / "completion.json"
    completion_path.write_text(
        json.dumps(completion, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return completion


# Clear aliases for callers that use the noun form in orchestration code.
prepare_secopd_judge_shards = prepare_judge_shards
assemble_secopd_judge_shards = assemble_judge_shards

