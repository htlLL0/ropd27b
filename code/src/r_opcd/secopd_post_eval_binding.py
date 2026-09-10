"""CPU-only binding audit for the fixed SecOPD post-eval panel.

The spec identifies existing completed artifacts. Paths in the spec are relative
to the spec file; paths embedded in artifacts are relative to their owning file.
This audit recomputes cases, consensus labels, and comparison via their existing
builders, and binds all post outputs to the dynamically selected final adapter.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import json
import sys

from r_opcd.frontier_collection import canonical_sha256
from r_opcd.secopd_judge_shards import file_sha256, read_json, read_jsonl, require

SPEC_SCHEMA = "r-opcd-secopd-post-eval-binding-spec-v1"
REPORT_SCHEMA = "r-opcd-secopd-post-eval-binding-audit-v1"


def _path(owner: Path, value: Any) -> Path:
    require(isinstance(value, str) and bool(value), f"missing path in {owner}")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else owner.parent / path).resolve()


def _bound(owner: Path, obj: Mapping[str, Any], key: str,
           expected: Path | None = None, hash_key: str | None = None) -> Path:
    path = _path(owner, obj.get(key))
    require(path.is_file(), f"{key} file missing: {path}")
    if expected is not None:
        require(path == expected.resolve(), f"{key} path binding mismatch")
    require(file_sha256(path) == obj.get(hash_key or f"{key}_sha256"),
            f"{key} SHA-256 mismatch")
    return path


def _input(owner: Path, payload: Mapping[str, Any], expected: Path) -> None:
    _bound(owner, payload, "path", expected, "sha256")


def _adapter(value: Any, expected: Mapping[str, str], owner: Path) -> None:
    require(isinstance(value, Mapping), f"missing post student_adapter: {owner}")
    require(_path(owner, value.get("path")) == Path(expected["path"]),
            f"student_adapter path mismatch: {owner}")
    for key in ("adapter_config_sha256", "adapter_model_sha256"):
        require(value.get(key) == expected[key], f"student_adapter {key} mismatch: {owner}")


def _passed(path: Path, schema: str) -> dict[str, Any]:
    payload = read_json(path)
    require(payload.get("schema") == schema, f"schema mismatch: {path}")
    require(payload.get("status") == "pass", f"artifact did not pass: {path}")
    return payload


def audit_post_eval_binding(spec_path: Path) -> dict[str, Any]:
    """Validate the full existing post chain, without writing or GPU work."""
    # Match other project audits: reuse tools as modules, not parallel logic.
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tools.prepare_secopd_au_cases import build_cases
    from tools.assemble_secopd_au_labels import assemble
    from tools.compare_secopd_dev_pre_post import build_report

    spec_path = spec_path.resolve()
    spec = read_json(spec_path)
    require(spec.get("schema") == SPEC_SCHEMA, "binding spec schema mismatch")
    paths = {
        key: _path(spec_path, spec.get(key))
        for key in (
            "training_dir", "training_audit", "source", "post_rollouts",
            "post_cases_manifest", "post_labels_completion", "base_labels",
            "base_clean_nll_completion", "post_clean_nll_completion", "comparison",
        )
    }
    evidence: dict[str, Any] = {}

    def record(name: str, path: Path) -> None:
        evidence[name] = {"path": str(path), "sha256": file_sha256(path)}

    for key, path in paths.items():
        if key != "training_dir":
            record(key, path)
    record("spec", spec_path)

    completion_path = paths["training_dir"] / "completion.json"
    completion = _passed(completion_path, "r-opcd-stage3o-training-completion-v1")
    steps = completion.get("completed_steps")
    require(type(steps) is int and steps > 0, "invalid training completed_steps")
    require(completion.get("base_unchanged") is True, "training base changed")
    require(completion.get("decision") == "fixed_current_rollout_training_cycle_complete",
            "training cycle is not complete")
    if "configured_total_steps" in completion:
        require(steps == completion["configured_total_steps"], "training is partial")
        require(completion.get("requested_max_steps") is None, "training was max-step limited")
    checkpoint = paths["training_dir"] / f"checkpoint_step{steps:04d}"
    state_path = checkpoint / "trainer_state.json"
    state = read_json(state_path)
    require(state.get("step") == steps, "final trainer state step mismatch")
    require(isinstance(completion.get("final_trainable_sha256"), str)
            and state.get("trainable_sha256") == completion["final_trainable_sha256"],
            "final trainer state trainable hash mismatch")
    audit = _passed(paths["training_audit"], "r-opcd-stage3o-fixed-rollout-training-audit-v1")
    require(_path(paths["training_audit"], audit.get("run_dir")) == paths["training_dir"],
            "training audit run_dir mismatch")
    require(audit.get("completion_sha256") == file_sha256(completion_path),
            "training audit completion SHA-256 mismatch")
    require(audit.get("absolute_final_step", audit.get("completed_steps")) == steps,
            "training audit final step mismatch")
    adapter_dir = checkpoint / "adapter"
    adapter = {
        "path": str(adapter_dir),
        "adapter_model_sha256": file_sha256(adapter_dir / "adapter_model.safetensors"),
        "adapter_config_sha256": file_sha256(adapter_dir / "adapter_config.json"),
    }
    require(audit.get("checkpoint_adapter_sha256", {}).get(str(steps))
            == adapter["adapter_model_sha256"], "training audit adapter SHA-256 mismatch")
    record("training_completion", completion_path)
    record("trainer_state", state_path)
    record("adapter_model", adapter_dir / "adapter_model.safetensors")
    record("adapter_config", adapter_dir / "adapter_config.json")

    require(file_sha256(paths["source"]) == spec.get("expected_source_sha256"),
            "fixed source SHA-256 mismatch")
    source = read_jsonl(paths["source"])
    require(len(source) == 120, "fixed dev source must contain exactly 120 records")
    rollouts = read_jsonl(paths["post_rollouts"])
    cases_manifest_path = paths["post_cases_manifest"]
    manifest = _passed(cases_manifest_path, "r-opcd-secopd-independent-au-cases-manifest-v1")
    _bound(cases_manifest_path, manifest, "source", paths["source"])
    _bound(cases_manifest_path, manifest, "rollouts", paths["post_rollouts"])
    cases_path = _bound(cases_manifest_path, manifest, "cases")
    cases = read_jsonl(cases_path)
    require(manifest.get("records") == 120, "cases manifest count mismatch")
    require(cases == build_cases(source, rollouts), "cases do not match source/rollout rebuild")
    for row in rollouts:
        trajectory = row.get("student_trajectory", row)
        _adapter(trajectory.get("student_adapter"), adapter, paths["post_rollouts"])
        require(trajectory.get("response_token_ids_sha256")
                == canonical_sha256(trajectory.get("response_token_ids")),
                "rollout response token hash mismatch")
    record("post_cases", cases_path)

    judge_values = spec.get("post_judge_completions")
    require(isinstance(judge_values, list) and len(judge_values) == 2,
            "exactly two post_judge_completions are required")
    judge_paths: list[Path] = []
    judge_rows: list[list[dict[str, Any]]] = []
    judge_identities: set[tuple[str, str]] = set()
    for index, value in enumerate(judge_values, 1):
        owner = _path(spec_path, value)
        judge = read_json(owner)
        require(judge.get("schema") == "r-opcd-stage3-independent-au-judge-completion-v1",
                "judge completion schema mismatch (supply full-panel judge runs)")
        require(judge.get("status") in {"pass", "completed_with_parse_errors"},
                "judge run did not complete")
        require(judge.get("records") == 120 and set(judge.get("targets", [])) == {"attack", "task"},
                "judge panel/targets mismatch")
        _bound(owner, judge, "sample", cases_path)
        judgments = _bound(owner, judge, "judgments")
        _bound(owner, judge, "config")
        rows = read_jsonl(judgments)
        require(len(rows) == len(cases), "judge row count mismatch")
        model = judge.get("judge_spec", {})
        identity = (model.get("model_id"), model.get("snapshot_revision"))
        require(all(isinstance(part, str) and part for part in identity),
                "judge model identity missing")
        judge_identities.add(identity)
        for case, row in zip(cases, rows, strict=True):
            require(all(row.get(key) == val for key, val in case.items()),
                    f"judge source case drift: {case['case_id']}")
            require(row.get("judge_id") == judge.get("judge_id")
                    and row.get("judge_model_id") == identity[0]
                    and row.get("judge_snapshot_revision") == identity[1],
                    "judge row/completion identity mismatch")
        judge_paths.append(judgments)
        judge_rows.append(rows)
        record(f"post_judge_{index}_completion", owner)
        record(f"post_judge_{index}_judgments", judgments)
    require(len(judge_identities) == 2, "judges must use independent model identities")

    labels_owner = paths["post_labels_completion"]
    labels_completion = _passed(labels_owner, "r-opcd-secopd-two-judge-au-completion-v1")
    labels_path = _bound(labels_owner, labels_completion, "labels")
    _input(labels_owner, labels_completion.get("inputs", {}).get("cases", {}), cases_path)
    for index, judge_path in enumerate(judge_paths, 1):
        _input(labels_owner, labels_completion.get("inputs", {}).get(f"judge_{index}", {}), judge_path)
    require(labels_completion.get("records") == 120, "labels completion count mismatch")
    rebuilt_labels = assemble(cases, judge_rows[0], judge_rows[1],
                              judge_1_path=judge_paths[0], judge_2_path=judge_paths[1])
    require(read_jsonl(labels_path) == rebuilt_labels, "consensus labels rebuild mismatch")
    record("post_labels", labels_path)

    for name in ("base", "post"):
        owner = paths[f"{name}_clean_nll_completion"]
        nll = _passed(owner, "r-opcd-stage3p-clean-utility-nll-completion-v1")
        require(nll.get("data_sha256") == spec.get("expected_clean_data_sha256"),
                "fixed clean data SHA-256 mismatch")
        _bound(owner, nll, "data")
        _bound(owner, nll, "config")
        records_path = _bound(owner, nll, "records_file", hash_key="records_sha256")
        if name == "post":
            _adapter(nll.get("student_adapter"), adapter, owner)
        else:
            require(nll.get("student_adapter") is None, "base clean NLL unexpectedly uses adapter")
        record(f"{name}_clean_nll_records", records_path)

    rebuilt_report = build_report(
        paths["base_labels"], labels_path, paths["base_clean_nll_completion"],
        paths["post_clean_nll_completion"],
    )
    report = read_json(paths["comparison"])
    for payload in (report, rebuilt_report):
        payload.pop("created_at_utc", None)
    require(report == rebuilt_report, "comparison rebuild/hash binding mismatch")
    return {
        "schema": REPORT_SCHEMA,
        "status": "pass",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_steps": steps,
        "checkpoint": str(checkpoint),
        "student_adapter": adapter,
        "records": 120,
        "checks": {
            "dynamic_final_checkpoint": True,
            "training_audit_adapter_hash": True,
            "fixed_source_hash": True,
            "zero_cap_source_rollout_cases_rebuilt": True,
            "all_post_rows_bound_to_final_adapter": True,
            "judge_cases_and_completions_bound": True,
            "consensus_rebuilt": True,
            "fixed_clean_nll_adapter_and_hashes_bound": True,
            "comparison_rebuilt": True,
        },
        "inputs": evidence,
        "evidence_boundary": "fixed 120-row development binding audit; not locked-test evidence",
    }
