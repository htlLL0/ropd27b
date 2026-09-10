"""Reproducible SecOPD Alpaca transfer data for the R-OPCD training path.

This module deliberately separates two notions of split:

* ``secopd_original_split`` reproduces SecOPD's published construction exactly:
  NumPy ``default_rng(0)`` permutes the full clean source, empty ``input`` rows
  are then removed, and the first ten percent are validation rows.
* ``split`` is the active R-OPCD partition.  It inherits the already-frozen
  Alpaca-cleaned assignment from ``external_injection_v1`` so that importing a
  new SecOPD attack realization cannot move a previously seen task across the
  train/dev boundary.

SecOPD's online global NumPy RNG was not frozen by its split seed.  The transfer
therefore uses a separate, explicit pairing seed and records this as a
reproducibility adaptation rather than claiming byte-identical online samples.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from unicodedata import category

import numpy as np

from r_opcd.attack_pilot import build_base_messages
from r_opcd.external_injection_data import (
    LOGICAL_CASE_SCHEMA,
    canonical_sha256,
    split_for_group,
    text_sha256,
    validate_logical_case,
)
from r_opcd.training_mixture import (
    ATTACK_CANDIDATE_SCHEMA,
    UTILITY_CONTROL_SCHEMA,
    validate_utility_control,
)


SOURCE_NAMESPACE = "secopd_alpaca_transfer_v1"
SOURCE_DATASET = "secopd_alpaca_metasecalign_transfer"
SOURCE_FAMILY = "secopd_alpaca_transfer"
TASK_NAME = "SecOPDAlpacaInstructionFollowing"
TASK_TYPE = "instruction_following"
UNIFIED_TASK = "text"

SECOPD_SPLIT_SEED = 0
SECOPD_VAL_FRACTION = 0.1
ACTIVE_SPLIT_SEED = 20260902
ACTIVE_SPLIT_SOURCE = "struq_alpaca_cleaned"
DEFAULT_PAIRING_SEED = 20260904
DEFAULT_RESPONSE_CAP_TOKENS = 256
DEFAULT_SHARDS = 3

EXPECTED_CLEAN_ROWS = 51_760
EXPECTED_CLEAN_ELIGIBLE_ROWS = 19_157
EXPECTED_CLEAN_UNIQUE_GROUPS = 19_154
EXPECTED_ACTIVE_TRAIN_GROUPS = 17_249
EXPECTED_ACTIVE_DEV_GROUPS = 1_905
EXPECTED_INJECTION_ROWS = 52_002

SECOPD_REPO_URL = "https://github.com/pppyb/SecOPD.git"
SECOPD_COMMIT = "571502a2a315c4b8820dd878d4569e2a2222cb88"

# Actual downloaded material consumed by this transfer.
CLEAN_SOURCE_URL = (
    "https://huggingface.co/datasets/yahma/alpaca-cleaned/resolve/"
    "817b724584ee023ffcdb10a207033a55bcd45c47/"
    "default/train/0000.parquet"
)
CLEAN_SOURCE_REVISION = "817b724584ee023ffcdb10a207033a55bcd45c47"
CLEAN_SOURCE_SHA256 = (
    "d11b9d13a0f13c77924f0d6ceeb83baaebb3aa3eb141fce8bd2c8e033bedf420"
)

# A separately verified, content-equivalent upstream JSON reference.  It is
# provenance evidence only and must never be reported as the consumed file.
CLEAN_UPSTREAM_CONTENT_REFERENCE_URL = (
    "https://huggingface.co/datasets/yahma/alpaca-cleaned/resolve/"
    "261118de5d7aec0b71d1fc878d4ea4ebbc2bf6ad/"
    "alpaca_data_cleaned.json"
)
CLEAN_UPSTREAM_CONTENT_REFERENCE_REVISION = (
    "261118de5d7aec0b71d1fc878d4ea4ebbc2bf6ad"
)
CLEAN_UPSTREAM_CONTENT_REFERENCE_ORIGIN_URL = (
    "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/"
    "d03c782bffd50ceeb1f4ef3c020129229ec4698c/"
    "alpaca_data_cleaned.json"
)
CLEAN_UPSTREAM_CONTENT_REFERENCE_SHA256 = (
    "bd844b8247a0f543804b6ce0882b0aaec4bbf5e8d66167df6213a0f1e4fe878b"
)

INJECTION_SOURCE_URL = (
    "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/"
    "761dc5bfbdeeffa89b8bff5d038781a4055f796a/alpaca_data.json"
)
INJECTION_SOURCE_FLOATING_URL = (
    "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/"
    "alpaca_data.json"
)
INJECTION_SOURCE_REVISION = "761dc5bfbdeeffa89b8bff5d038781a4055f796a"
INJECTION_SOURCE_SHA256 = (
    "2eddafc6b977608d778aaab8dfc7e50e547b3af9826dfb9e909d9fc362e4a419"
)

FIXED_PAIRING_NOTE = (
    "Fixed pairing is a transfer reproducibility adaptation: SecOPD samples "
    "injections, attack branches, positions, and completion delimiters with "
    "global NumPy RNG state that is not fixed by its clean split seed."
)
FIXED_PAIRING_NOTE_ZH = (
    "固定 pairing 是移植可复现性改造；SecOPD 原始实现的注入样本、攻击分支、"
    "位置和 completion delimiter 使用未被 clean split seed 固定的全局 NumPy RNG。"
)

# Verbatim token pools used by SecOPD's Meta-SecAlign completion constructor.
OTHER_DELM_TOKENS: dict[str, tuple[str, ...]] = {
    "mark": (
        "{s}",
        "|{s}|",
        "<{s}>",
        "[{s}]",
        "<|{s}|>",
        "[|{s}|]",
        "<[{s}]>",
        "'''{s}'''",
        "***{s}***",
    ),
    "inst": ("Command", "Rule", "Prompt", "Task"),
    "inpt": ("Data", "Context", "Text"),
    "resp": ("Output", "Answer", "Reply"),
    "user": ("", "Prompter ", "User ", "Human "),
    "asst": ("", "Assistant ", "Chatbot ", "Bot ", "GPT ", "AI "),
}


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_list(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"expected a JSON list: {path}")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"{path} row {index} is not an object")
        rows.append(dict(item))
    return rows


def load_clean_records(path: Path) -> list[dict[str, Any]]:
    """Load the pinned yahma material without confusing Parquet and JSON hashes."""

    if path.suffix.lower() == ".json":
        return load_json_list(path)
    if path.suffix.lower() != ".parquet":
        raise ValueError(f"unsupported clean source format: {path}")
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - production dependency check.
        raise RuntimeError("reading the pinned clean source requires pyarrow") from exc
    table = parquet.read_table(path, columns=["instruction", "input", "output"])
    rows = table.to_pylist()
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"Parquet clean source contains a non-object row: {path}")
    return [dict(row) for row in rows]


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{line_number} is not an object")
            yield dict(value)


def normalized_task_digest(instruction: str, input_text: str) -> str:
    """Match the existing StruQ loader's canonical prompt-group identity."""

    return canonical_sha256([instruction.strip(), input_text.strip()])


def contains_unicode_standalone_literal(text: str, literal: str) -> bool:
    """Return whether ``literal`` occurs outside surrounding Unicode words."""

    if not literal:
        return False

    def is_word_character(character: str) -> bool:
        unicode_category = category(character)
        return unicode_category[0] in {"L", "M", "N"} or unicode_category == "Pc"

    search_start = 0
    while (match_start := text.find(literal, search_start)) >= 0:
        match_end = match_start + len(literal)
        left_is_clear = (
            not is_word_character(literal[0])
            or match_start == 0
            or not is_word_character(text[match_start - 1])
        )
        right_is_clear = (
            not is_word_character(literal[-1])
            or match_end == len(text)
            or not is_word_character(text[match_end])
        )
        if left_is_clear and right_is_clear:
            return True
        search_start = match_start + 1
    return False


def _alpaca_text_fields(
    item: Mapping[str, Any], *, source: str, index: int
) -> tuple[str, str, str]:
    missing = sorted({"instruction", "input", "output"} - set(item))
    if missing:
        raise ValueError(f"{source} row {index} missing fields: {missing}")
    values = (item["instruction"], item["input"], item["output"])
    if not all(isinstance(value, str) for value in values):
        raise ValueError(f"{source} row {index} has non-string Alpaca fields")
    return values  # type: ignore[return-value]


def reproduce_secopd_clean_split(
    clean_records: Sequence[Mapping[str, Any]],
    *,
    seed: int = SECOPD_SPLIT_SEED,
    val_fraction: float = SECOPD_VAL_FRACTION,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reproduce SecOPD: permute all rows, then filter empty ``input`` rows."""

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must lie strictly between zero and one")
    # Validate every row before indexing it.  SecOPD assumes this source schema.
    raw_fields = [
        _alpaca_text_fields(item, source="clean", index=index)
        for index, item in enumerate(clean_records)
    ]
    order = np.random.default_rng(seed).permutation(len(clean_records))
    eligible_indices = [
        int(index) for index in order if raw_fields[int(index)][1] != ""
    ]
    n_val = int(len(eligible_indices) * val_fraction)
    rows: list[dict[str, Any]] = []
    normalized_empty = 0
    for rank, raw_index in enumerate(eligible_indices):
        raw_instruction, raw_input, raw_output = raw_fields[raw_index]
        instruction = raw_instruction.strip()
        input_text = raw_input.strip()
        output = raw_output.strip()
        if not instruction or not input_text or not output:
            normalized_empty += 1
            raise ValueError(
                "SecOPD-eligible clean row becomes empty after R-OPCD normalization: "
                f"index {raw_index}"
            )
        split = "dev" if rank < n_val else "train"
        rows.append(
            {
                "source_record_id": f"alpaca-cleaned-{raw_index:05d}",
                "source_raw_index": raw_index,
                "secopd_permutation_rank": rank,
                "secopd_original_split": split,
                "instruction": instruction,
                "input": input_text,
                "output": output,
                "raw_instruction": raw_instruction,
                "raw_input": raw_input,
                "raw_output": raw_output,
                "normalized_task_digest": normalized_task_digest(
                    raw_instruction, raw_input
                ),
            }
        )
    return rows, {
        "raw_rows": len(clean_records),
        "permutation_seed": seed,
        "filter_order": "full_permutation_then_raw_input_nonempty_filter",
        "eligible_nonempty_input_rows": len(rows),
        "normalized_empty_rows": normalized_empty,
        "val_fraction": val_fraction,
        "val_rows": n_val,
        "train_rows": len(rows) - n_val,
    }


def deduplicate_clean_groups(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Choose one deterministic representative per normalized clean task."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["normalized_task_digest"])].append(dict(row))
    unique: list[dict[str, Any]] = []
    conflicting_outputs = 0
    cross_secopd_split_groups = 0
    for digest, members in sorted(grouped.items()):
        outputs = {str(member["output"]) for member in members}
        if len(outputs) != 1:
            conflicting_outputs += 1
            raise ValueError(f"clean prompt group has conflicting outputs: {digest}")
        members.sort(
            key=lambda member: (
                int(member["secopd_permutation_rank"]),
                int(member["source_raw_index"]),
            )
        )
        representative = dict(members[0])
        original_splits = sorted(
            {str(member["secopd_original_split"]) for member in members}
        )
        if len(original_splits) > 1:
            cross_secopd_split_groups += 1
        representative["secopd_original_splits"] = original_splits
        representative["duplicate_source_record_ids"] = [
            str(member["source_record_id"]) for member in members[1:]
        ]
        unique.append(representative)
    return unique, {
        "eligible_rows": len(rows),
        "unique_normalized_prompt_groups": len(unique),
        "duplicate_rows_removed": len(rows) - len(unique),
        "conflicting_output_groups": conflicting_outputs,
        "groups_crossing_secopd_original_split": cross_secopd_split_groups,
    }


def frozen_split_assignments_from_rows(
    train_rows: Iterable[Mapping[str, Any]],
    dev_rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Extract the frozen Alpaca target assignment from existing StruQ rows."""

    assignments: dict[str, str] = {}
    source_rows = 0
    duplicate_rows = 0
    for split, rows in (("train", train_rows), ("dev", dev_rows)):
        for row in rows:
            target_digest = str(row.get("target_source_digest", ""))
            if not target_digest.startswith("struq:"):
                continue
            if row.get("source_dataset") != "struq_alpaca_cleaned_adaptation":
                continue
            source_rows += 1
            digest = target_digest.removeprefix("struq:")
            observed = normalized_task_digest(
                str(row.get("user_query", "")), str(row.get("clean_context", ""))
            )
            if observed != digest:
                raise ValueError(
                    "existing StruQ target digest no longer matches normalized prompt"
                )
            previous = assignments.get(digest)
            if previous is not None:
                duplicate_rows += 1
                if previous != split:
                    raise ValueError(
                        f"existing frozen task crosses train/dev: {digest}"
                    )
            assignments[digest] = split
    if not assignments:
        raise ValueError("no frozen StruQ Alpaca assignments were found")
    return assignments, {
        "source_rows_scanned": source_rows,
        "unique_normalized_prompt_groups": len(assignments),
        "duplicate_attack_rows_collapsed": duplicate_rows,
        "train_groups": sum(value == "train" for value in assignments.values()),
        "dev_groups": sum(value == "dev" for value in assignments.values()),
    }


def load_frozen_split_assignments(
    train_path: Path, dev_path: Path
) -> tuple[dict[str, str], dict[str, Any]]:
    return frozen_split_assignments_from_rows(
        read_jsonl(train_path), read_jsonl(dev_path)
    )


def assign_active_splits(
    clean_groups: Sequence[Mapping[str, Any]],
    frozen_assignments: Mapping[str, str],
    *,
    seed: int = ACTIVE_SPLIT_SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Inherit old assignments; hash only groups absent from the old package."""

    rows: list[dict[str, Any]] = []
    mapped = 0
    unmapped = 0
    mapping_hash_disagreements = 0
    original_agreements = 0
    original_disagreements = 0
    for item in clean_groups:
        row = dict(item)
        digest = str(row["normalized_task_digest"])
        hashed = split_for_group(ACTIVE_SPLIT_SOURCE, digest, seed=seed)
        if digest in frozen_assignments:
            split = str(frozen_assignments[digest])
            if split not in {"train", "dev"}:
                raise ValueError(f"invalid frozen split for {digest}: {split}")
            mapped += 1
            assignment_source = "inherited_external_injection_v1"
            mapping_hash_disagreements += split != hashed
        else:
            split = hashed
            unmapped += 1
            assignment_source = "fallback_existing_group_hash_rule"
        original = str(row["secopd_original_split"])
        original_agreements += split == original
        original_disagreements += split != original
        row["split"] = split
        row["active_split_assignment"] = assignment_source
        rows.append(row)
    return rows, {
        "active_split_seed": seed,
        "active_split_source_namespace": ACTIVE_SPLIT_SOURCE,
        "mapped_to_existing_groups": mapped,
        "unmapped_groups": unmapped,
        "mapping_vs_group_hash_disagreements": mapping_hash_disagreements,
        "active_train_groups": sum(row["split"] == "train" for row in rows),
        "active_dev_groups": sum(row["split"] == "dev" for row in rows),
        "secopd_original_split_agreements": original_agreements,
        "secopd_original_split_disagreements": original_disagreements,
    }


def normalize_injection_pool(
    injection_records: Sequence[Mapping[str, Any]],
    *,
    seed: int = ACTIVE_SPLIT_SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    """Normalize Stanford Alpaca rows while preserving row-level sampling mass."""

    rows: list[dict[str, Any]] = []
    ref_instruction_response: dict[str, str] = {}
    groups: dict[str, set[str]] = defaultdict(set)
    for index, item in enumerate(injection_records):
        raw_instruction, raw_input, raw_output = _alpaca_text_fields(
            item, source="injection", index=index
        )
        if not raw_instruction.strip():
            raise ValueError(f"injection row {index} has an empty instruction")
        digest = normalized_task_digest(raw_instruction, raw_input)
        split = split_for_group(ACTIVE_SPLIT_SOURCE, digest, seed=seed)
        rows.append(
            {
                "source_record_id": f"stanford-alpaca-{index:05d}",
                "source_raw_index": index,
                "instruction": raw_instruction,
                "input": raw_input,
                "output": raw_output,
                "pairing_eligible": bool(raw_output.strip()),
                "normalized_task_digest": digest,
                "split": split,
            }
        )
        groups[digest].add(raw_output.strip())
        # This intentionally matches SecOPD's ordinary dict-comprehension:
        # later rows with the same exact instruction overwrite earlier rows.
        ref_instruction_response[raw_instruction] = raw_output
    return rows, {
        "raw_rows": len(injection_records),
        "valid_rows": len(rows),
        "pairing_eligible_nonempty_output_rows": sum(
            bool(row["pairing_eligible"]) for row in rows
        ),
        "empty_output_rows_excluded_from_pairing": sum(
            not bool(row["pairing_eligible"]) for row in rows
        ),
        "unique_normalized_prompt_groups": len(groups),
        "duplicate_rows": len(rows) - len(groups),
        "conflicting_output_groups": sum(len(outputs) > 1 for outputs in groups.values()),
        "train_rows": sum(row["split"] == "train" for row in rows),
        "dev_rows": sum(row["split"] == "dev" for row in rows),
        "train_groups": sum(
            split_for_group(ACTIVE_SPLIT_SOURCE, digest, seed=seed) == "train"
            for digest in groups
        ),
        "dev_groups": sum(
            split_for_group(ACTIVE_SPLIT_SOURCE, digest, seed=seed) == "dev"
            for digest in groups
        ),
    }, ref_instruction_response


def create_completion_injection(
    response: str,
    instruction: str,
    input_text: str,
    *,
    rng: np.random.Generator,
) -> str:
    """Seedable port of SecOPD's Meta-SecAlign completion-body renderer."""

    mark = str(rng.choice(OTHER_DELM_TOKENS["mark"])) + ":"

    def sample_delimiter(name: str) -> str:
        role_name = "user" if name in {"inst", "inpt"} else "asst"
        role = str(rng.choice(OTHER_DELM_TOKENS[role_name]))
        delimiter = str(rng.choice(OTHER_DELM_TOKENS[name]))
        value = role + delimiter
        probability = float(rng.random())
        if probability < 1.0 / 3.0:
            return value.upper()
        if probability < 2.0 / 3.0:
            return value.lower()
        return value

    text = mark.format(s=sample_delimiter("resp")) + "\n\n" + response
    text += (
        "\n\n"
        + mark.format(s=sample_delimiter("inst"))
        + "\n\n"
        + instruction
    )
    if input_text != "":
        text += (
            "\n\n"
            + mark.format(s=sample_delimiter("inpt"))
            + "\n\n"
            + input_text
        )
    return text


def attack_plan_counts(size: int) -> dict[str, int]:
    if size < 1:
        raise ValueError("attack plan requires at least one row")
    completion = size // 10
    straightforward = size - completion
    prepend = straightforward // 2
    append = straightforward - prepend
    return {
        "straightforward_prepend": prepend,
        "straightforward_append": append,
        "completion": completion,
    }


def _attack_plan(size: int, rng: np.random.Generator) -> list[str]:
    counts = attack_plan_counts(size)
    plan = [
        kind for kind, count in counts.items() for _ in range(count)
    ]
    return [plan[int(index)] for index in rng.permutation(len(plan))]


def _stable_group_order(
    rows: Sequence[Mapping[str, Any]], *, seed: int, purpose: str
) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            sha256(
                f"{seed}|{purpose}|{row['normalized_task_digest']}".encode("utf-8")
            ).hexdigest(),
            str(row["normalized_task_digest"]),
        ),
    )


def select_pilot_targets(
    clean_groups: Sequence[Mapping[str, Any]],
    *,
    split: str,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    if split not in {"train", "dev"}:
        raise ValueError("split must be train or dev")
    if count < 1:
        raise ValueError("pilot count must be positive")
    eligible = [dict(row) for row in clean_groups if row["split"] == split]
    if count > len(eligible):
        raise ValueError(
            f"requested {count} {split} targets but only {len(eligible)} exist"
        )
    return _stable_group_order(
        eligible, seed=seed, purpose=f"{SOURCE_NAMESPACE}:{split}:targets"
    )[:count]


def _pairing_rng(seed: int, split: str) -> np.random.Generator:
    split_offset = 0 if split == "train" else 1
    sequence = np.random.SeedSequence([seed, split_offset, 0x5EC0])
    return np.random.default_rng(sequence)


def build_attack_rows(
    targets: Sequence[Mapping[str, Any]],
    injection_rows: Sequence[Mapping[str, Any]],
    ref_instruction_response: Mapping[str, str],
    *,
    split: str,
    pairing_seed: int,
    secopd_commit: str = SECOPD_COMMIT,
) -> list[dict[str, Any]]:
    """Build one deterministically paired SecOPD attack realization per target."""

    if not targets:
        raise ValueError("cannot build attacks without targets")
    if any(str(row.get("split")) != split for row in targets):
        raise ValueError("target split mismatch")
    pool = sorted(
        (
            dict(row)
            for row in injection_rows
            if row.get("split") == split and bool(row.get("pairing_eligible"))
        ),
        key=lambda row: (int(row["source_raw_index"]), row["normalized_task_digest"]),
    )
    if len(pool) < 2:
        raise ValueError(f"injection pool for {split} is too small")
    rng = _pairing_rng(pairing_seed, split)
    plan = _attack_plan(len(targets), rng)
    ordered_targets = _stable_group_order(
        targets, seed=pairing_seed, purpose=f"{SOURCE_NAMESPACE}:{split}:pairing"
    )
    rows: list[dict[str, Any]] = []
    for target, planned_kind in zip(ordered_targets, plan, strict=True):
        for _attempt in range(max(100, len(pool) * 2)):
            injector = pool[int(rng.integers(0, len(pool)))]
            target_digest = str(target["normalized_task_digest"])
            injection_digest = str(injector["normalized_task_digest"])
            if target_digest == injection_digest:
                continue
            injection_goal = str(injector["instruction"]) + " " + str(
                injector["input"]
            )
            if not injection_goal.strip():
                continue
            if planned_kind.startswith("straightforward_") and (
                contains_unicode_standalone_literal(
                    injection_goal, str(target["output"])
                )
            ):
                continue
            completion_fake_response: str | None = None
            completion_fake_response_source: str | None = None
            if planned_kind == "straightforward_prepend":
                malicious_span = injection_goal
                contaminated = malicious_span + " " + str(target["input"])
                attack_kind = "straightforward"
                attack_position = "start"
            elif planned_kind == "straightforward_append":
                malicious_span = injection_goal
                contaminated = str(target["input"]) + " " + malicious_span
                attack_kind = "straightforward"
                attack_position = "end"
            elif planned_kind == "completion":
                raw_instruction = str(target["raw_instruction"])
                if raw_instruction in ref_instruction_response:
                    completion_fake_response = str(
                        ref_instruction_response[raw_instruction]
                    )
                    completion_fake_response_source = (
                        "injection_pool_instruction_lookup"
                    )
                else:
                    completion_fake_response = str(target["raw_output"])
                    completion_fake_response_source = "clean_target_output_fallback"
                malicious_span = create_completion_injection(
                    completion_fake_response,
                    str(injector["instruction"]),
                    str(injector["input"]),
                    rng=rng,
                )
                contaminated = str(target["input"]) + "\n\n" + malicious_span
                attack_kind = "completion"
                attack_position = "end"
            else:  # pragma: no cover - internal plan construction is exhaustive.
                raise AssertionError(planned_kind)
            if contaminated.count(malicious_span) != 1:
                continue
            row = _build_attack_row(
                target=target,
                injector=injector,
                split=split,
                pairing_seed=pairing_seed,
                secopd_commit=secopd_commit,
                planned_kind=planned_kind,
                attack_kind=attack_kind,
                attack_position=attack_position,
                injection_goal=injection_goal,
                malicious_span=malicious_span,
                contaminated_context=contaminated,
                completion_fake_response=completion_fake_response,
                completion_fake_response_source=completion_fake_response_source,
            )
            rows.append(row)
            break
        else:
            raise ValueError(
                "could not find an exact-once, cross-task injection for target "
                f"{target['source_record_id']}"
            )
    return sorted(rows, key=lambda row: str(row["id"]))


def _build_attack_row(
    *,
    target: Mapping[str, Any],
    injector: Mapping[str, Any],
    split: str,
    pairing_seed: int,
    secopd_commit: str,
    planned_kind: str,
    attack_kind: str,
    attack_position: str,
    injection_goal: str,
    malicious_span: str,
    contaminated_context: str,
    completion_fake_response: str | None,
    completion_fake_response_source: str | None,
) -> dict[str, Any]:
    clean_context = str(target["input"])
    user_query = str(target["instruction"])
    target_answer = str(target["output"])
    attack_target_answer = str(injector["output"]).strip()
    start = contaminated_context.find(malicious_span)
    target_digest = str(target["normalized_task_digest"])
    injection_digest = str(injector["normalized_task_digest"])
    base_digest = canonical_sha256([TASK_NAME, user_query, clean_context])
    row_digest = canonical_sha256(
        [
            SOURCE_NAMESPACE,
            split,
            pairing_seed,
            target_digest,
            str(injector["source_record_id"]),
            planned_kind,
            malicious_span,
        ]
    )
    attack_name = f"secopd-metasecalign-{planned_kind.replace('_', '-')}-v1"
    is_completion = attack_kind == "completion"
    row: dict[str, Any] = {
        "schema_version": LOGICAL_CASE_SCHEMA,
        "mixture_schema_version": ATTACK_CANDIDATE_SCHEMA,
        "id": f"r-opcd::{SOURCE_NAMESPACE}::{split}::{row_digest}",
        "raw_id": (
            f"{target['source_record_id']}::{injector['source_record_id']}::"
            f"{attack_name}"
        ),
        "split": split,
        "split_namespace": SOURCE_NAMESPACE,
        "split_seed": ACTIVE_SPLIT_SEED,
        "secopd_split_seed": SECOPD_SPLIT_SEED,
        "secopd_original_split": target["secopd_original_split"],
        "secopd_original_splits": list(target["secopd_original_splits"]),
        "secopd_permutation_rank": int(target["secopd_permutation_rank"]),
        "active_split_assignment": target["active_split_assignment"],
        "pairing_seed": pairing_seed,
        "source_dataset": SOURCE_DATASET,
        "source_revision": secopd_commit,
        "secopd_commit": secopd_commit,
        "target_source_id": target["source_record_id"],
        "target_source_digest": "struq:" + target_digest,
        "target_union_group_digest": target_digest,
        "normalized_task_digest": target_digest,
        "injection_source_id": injector["source_record_id"],
        "injection_source_digest": "stanford-alpaca:" + injection_digest,
        "injection_union_group_digest": injection_digest,
        "base_task_id": f"r-opcd::{SOURCE_NAMESPACE}::base::{base_digest}",
        "base_task_digest": base_digest,
        "task_name": TASK_NAME,
        "task_type": TASK_TYPE,
        "unified_task": UNIFIED_TASK,
        "max_new_tokens": DEFAULT_RESPONSE_CAP_TOKENS,
        "user_query": user_query,
        "clean_context": clean_context,
        "sanitized_context": clean_context,
        "contaminated_context": contaminated_context,
        "malicious_span": malicious_span,
        "malicious_span_char_start": start,
        "malicious_span_char_end": start + len(malicious_span),
        "malicious_span_sha256": text_sha256(malicious_span),
        "risk_label": "context_injection",
        "risk_weight": 3.0,
        "attack_str": injection_goal,
        "attack_name": attack_name,
        "attack_kind": attack_kind,
        "secopd_attack_plan_cell": planned_kind,
        "attack_category": (
            "completion_boundary_escape" if is_completion else "cross_task_instruction"
        ),
        "attack_position": attack_position,
        "attack_type": "secopd_metasecalign_transfer",
        "secopd_injection_goal": injection_goal,
        "target_answer": target_answer,
        "target_answers": [target_answer],
        "attack_target_answer": attack_target_answer,
        "attack_target_answers": [attack_target_answer],
        "completion_fake_response_policy": (
            "secopd_faithful" if is_completion else None
        ),
        "completion_fake_response_source": completion_fake_response_source,
        "completion_fake_response": completion_fake_response,
        "completion_contains_clean_target": (
            contains_unicode_standalone_literal(malicious_span, target_answer)
            if is_completion
            else False
        ),
        "adaptation_note": (
            "SecOPD Meta-SecAlign payload transferred into yesterday's R-OPCD "
            "ordinary attacked Student/T- and attention-quarantined T+ renderer; "
            "pairing RNG is explicitly frozen for reproducibility."
        ),
        "mixture_namespace": SOURCE_NAMESPACE,
        "mixture_seed": pairing_seed,
        "mixture_branch": "attack_candidate",
        "mixture_source_family": SOURCE_FAMILY,
        "mixture_selection_cell": planned_kind,
        "attack_gate_status": "pending_current_student_A_U_verification",
        "recoverability_c_status": "pending_frozen_tplus_calibration",
    }
    validate_secopd_attack_row(row)
    row["mixture_record_sha256"] = canonical_sha256(row)
    return row


def validate_secopd_attack_row(row: Mapping[str, Any]) -> None:
    validate_logical_case(row)
    required = {
        "mixture_schema_version",
        "secopd_original_split",
        "normalized_task_digest",
        "target_union_group_digest",
        "injection_union_group_digest",
        "attack_target_answer",
        "attack_target_answers",
        "attack_kind",
        "completion_fake_response_policy",
        "completion_contains_clean_target",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"SecOPD attack row missing fields: {missing}")
    if row["mixture_schema_version"] != ATTACK_CANDIDATE_SCHEMA:
        raise ValueError("SecOPD attack candidate schema changed")
    if row["unified_task"] != UNIFIED_TASK or row["task_type"] != TASK_TYPE:
        raise ValueError("SecOPD R-OPCD task contract changed")
    if row["risk_label"] != "context_injection":
        raise ValueError("SecOPD attack row must be context_injection")
    if row["attack_target_answers"] != [row["attack_target_answer"]]:
        raise ValueError("attack_target_answers must contain exactly one answer")
    if not str(row["attack_target_answer"]).strip():
        raise ValueError("attack target answer must be non-empty for A/U verification")
    if row["attack_str"] != row["secopd_injection_goal"]:
        raise ValueError("A/U attack_str must be the SecOPD injection goal")
    if row["target_union_group_digest"] == row["injection_union_group_digest"]:
        raise ValueError("target and injection must be different normalized tasks")
    if row["attack_kind"] == "completion":
        if row["completion_fake_response_policy"] != "secopd_faithful":
            raise ValueError("completion row lost faithful SecOPD marker")
        if row.get("completion_fake_response") is None:
            raise ValueError("completion row lost its fake response provenance")
        if row["malicious_span"] == row["secopd_injection_goal"]:
            raise ValueError("completion q1 must cover the full rendered body")
    else:
        if row["completion_fake_response_policy"] is not None:
            raise ValueError("straightforward row has a completion policy")
        if contains_unicode_standalone_literal(
            str(row["malicious_span"]), str(row["target_answer"])
        ):
            raise ValueError("straightforward q1 contains a clean-target literal")


def build_matched_utility_rows(
    attacks: Sequence[Mapping[str, Any]], *, pairing_seed: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attack in attacks:
        validate_secopd_attack_row(attack)
        clean_source = dict(attack)
        clean_source["contaminated_context"] = attack["clean_context"]
        messages = build_base_messages(clean_source)
        if len(messages) != 1 or messages[0]["role"] != "user":
            raise ValueError("clean utility must render as one ordinary user message")
        digest = canonical_sha256(
            [SOURCE_NAMESPACE, attack["id"], "matched_clean_counterpart"]
        )
        row: dict[str, Any] = {
            "schema_version": UTILITY_CONTROL_SCHEMA,
            "id": (
                f"r-opcd::{SOURCE_NAMESPACE}::{attack['split']}::utility::{digest}"
            ),
            "split": attack["split"],
            "mixture_namespace": SOURCE_NAMESPACE,
            "mixture_seed": pairing_seed,
            "mixture_branch": "benign_utility_control",
            "mixture_source_family": SOURCE_FAMILY,
            "control_role": "matched_clean_counterpart",
            "source_id": attack["id"],
            "source_row_sha256": canonical_sha256(dict(attack)),
            "base_task_id": attack["base_task_id"],
            "base_task_digest": attack["base_task_digest"],
            "normalized_task_digest": attack["normalized_task_digest"],
            "task_name": attack["task_name"],
            "task_type": attack["task_type"],
            "unified_task": attack["unified_task"],
            "max_new_tokens": DEFAULT_RESPONSE_CAP_TOKENS,
            "user_query": attack["user_query"],
            "clean_context": attack["clean_context"],
            "instruction": messages[0]["content"],
            "response": attack["target_answer"],
            "target_answer": attack["target_answer"],
            "target_answers": attack["target_answers"],
            "has_q1": False,
            "attack_gate": 0.0,
            "recoverability_c": 0.0,
            "teacher_pair_required": False,
            "training_objective": "ordinary_utility_sft_only",
            "matched_attack_candidate_ids": [attack["id"]],
        }
        validate_utility_control(row)
        row["mixture_record_sha256"] = canonical_sha256(row)
        rows.append(row)
    return sorted(rows, key=lambda row: str(row["id"]))


def estimated_rollout_load(
    row: Mapping[str, Any], *, response_cap_tokens: int = DEFAULT_RESPONSE_CAP_TOKENS
) -> int:
    # A tokenizer-free deterministic proxy.  Response allowance is intentionally
    # identical for every row because this arm is a single ``text:256`` task.
    prompt_char_tokens = (
        len(str(row["user_query"])) + len(str(row["contaminated_context"])) + 3
    ) // 4
    return prompt_char_tokens + response_cap_tokens


def load_balanced_shards(
    rows: Sequence[Mapping[str, Any]],
    *,
    shard_count: int = DEFAULT_SHARDS,
    response_cap_tokens: int = DEFAULT_RESPONSE_CAP_TOKENS,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, int]]]:
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    shards: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    loads = [0 for _ in range(shard_count)]
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (-estimated_rollout_load(row, response_cap_tokens=response_cap_tokens), str(row["id"])),
    )
    for row in ordered:
        shard_index = min(range(shard_count), key=lambda index: (loads[index], index))
        shards[shard_index].append(row)
        loads[shard_index] += estimated_rollout_load(
            row, response_cap_tokens=response_cap_tokens
        )
    for shard in shards:
        shard.sort(key=lambda row: str(row["id"]))
    report = [
        {"shard": index, "rows": len(shard), "estimated_load": loads[index]}
        for index, shard in enumerate(shards)
    ]
    return shards, report


def select_smoke_rows(
    train_rows: Sequence[Mapping[str, Any]],
    dev_rows: Sequence[Mapping[str, Any]],
    *,
    count: int = 12,
) -> list[dict[str, Any]]:
    """Select a deterministic split x attack-plan-cell smoke panel."""

    all_rows = [dict(row) for row in (*train_rows, *dev_rows)]
    if count < 1 or count > len(all_rows):
        raise ValueError("invalid smoke row count")
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        cells[(str(row["split"]), str(row["secopd_attack_plan_cell"]))].append(row)
    for rows in cells.values():
        rows.sort(key=lambda row: str(row["id"]))
    selected: list[dict[str, Any]] = []
    depth = 0
    while len(selected) < count:
        progressed = False
        for key in sorted(cells):
            if depth < len(cells[key]):
                selected.append(cells[key][depth])
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise ValueError("smoke selection underflow")
        depth += 1
    return selected


def audit_transfer_package(
    attacks_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    utilities_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    attacks = [
        dict(row)
        for split in ("train", "dev")
        for row in attacks_by_split.get(split, ())
    ]
    utilities = [
        dict(row)
        for split in ("train", "dev")
        for row in utilities_by_split.get(split, ())
    ]
    if not attacks or len(attacks) != len(utilities):
        raise ValueError("transfer requires one utility row per attack row")
    if len({row["id"] for row in attacks}) != len(attacks):
        raise ValueError("attack IDs are not unique")
    if len({row["id"] for row in utilities}) != len(utilities):
        raise ValueError("utility IDs are not unique")
    for row in attacks:
        validate_secopd_attack_row(row)
    for row in utilities:
        validate_utility_control(row)

    attack_ids = {str(row["id"]) for row in attacks}
    matched_ids = [
        str(identifier)
        for row in utilities
        for identifier in row["matched_attack_candidate_ids"]
    ]
    if set(matched_ids) != attack_ids or len(matched_ids) != len(attack_ids):
        raise ValueError("utility-to-attack matching is not one-to-one")

    def split_values(split: str, field: str) -> set[str]:
        return {
            str(row[field]) for row in attacks if str(row["split"]) == split
        }

    target_overlap = split_values("train", "target_union_group_digest") & split_values(
        "dev", "target_union_group_digest"
    )
    injection_overlap = split_values(
        "train", "injection_union_group_digest"
    ) & split_values("dev", "injection_union_group_digest")
    train_union = split_values("train", "target_union_group_digest") | split_values(
        "train", "injection_union_group_digest"
    )
    dev_union = split_values("dev", "target_union_group_digest") | split_values(
        "dev", "injection_union_group_digest"
    )
    union_overlap = train_union & dev_union
    if target_overlap or injection_overlap or union_overlap:
        raise ValueError("normalized Alpaca task leakage across active train/dev")

    split_reports: dict[str, Any] = {}
    for split in ("train", "dev"):
        split_attacks = [row for row in attacks if row["split"] == split]
        split_utilities = [row for row in utilities if row["split"] == split]
        split_reports[split] = {
            "attack_rows": len(split_attacks),
            "utility_rows": len(split_utilities),
            "attack_plan_cells": dict(
                sorted(
                    Counter(
                        str(row["secopd_attack_plan_cell"])
                        for row in split_attacks
                    ).items()
                )
            ),
            "attack_kinds": dict(
                sorted(Counter(str(row["attack_kind"]) for row in split_attacks).items())
            ),
            "attack_positions": dict(
                sorted(
                    Counter(str(row["attack_position"]) for row in split_attacks).items()
                )
            ),
            "completion_rows": sum(
                row["attack_kind"] == "completion" for row in split_attacks
            ),
            "completion_rows_containing_clean_target": sum(
                bool(row["completion_contains_clean_target"])
                for row in split_attacks
                if row["attack_kind"] == "completion"
            ),
        }
    old_dev_in_new_train = sum(
        row["split"] == "train"
        and row["active_split_assignment"] == "inherited_external_injection_v1"
        and row["target_union_group_digest"]
        in split_values("dev", "target_union_group_digest")
        for row in attacks
    )
    if old_dev_in_new_train:
        raise ValueError("existing dev task entered SecOPD train pilot")
    return {
        "status": "pass",
        "attack_rows": len(attacks),
        "utility_rows": len(utilities),
        "splits": split_reports,
        "train_dev_overlap": {
            "target_union_groups": len(target_overlap),
            "injection_union_groups": len(injection_overlap),
            "combined_clean_injection_union_groups": len(union_overlap),
            "base_task_ids": len(
                split_values("train", "base_task_id")
                & split_values("dev", "base_task_id")
            ),
        },
        "existing_dev_tasks_in_new_train": old_dev_in_new_train,
        "malicious_span_exact_once_failures": 0,
        "utility_q1_rows": 0,
        "utility_nonzero_c_rows": 0,
    }


def prepare_secopd_transfer(
    clean_records: Sequence[Mapping[str, Any]],
    injection_records: Sequence[Mapping[str, Any]],
    frozen_assignments: Mapping[str, str],
    *,
    pilot_train: int,
    pilot_dev: int,
    pairing_seed: int = DEFAULT_PAIRING_SEED,
    shard_count: int = DEFAULT_SHARDS,
    secopd_commit: str = SECOPD_COMMIT,
) -> dict[str, Any]:
    """Prepare source/utility rows without running a model or assigning A/U."""

    secopd_rows, secopd_split_report = reproduce_secopd_clean_split(clean_records)
    clean_groups, clean_dedup_report = deduplicate_clean_groups(secopd_rows)
    active_groups, active_split_report = assign_active_splits(
        clean_groups, frozen_assignments
    )
    injection_rows, injection_report, reference_lookup = normalize_injection_pool(
        injection_records
    )
    clean_group_digests = {
        str(row["normalized_task_digest"]) for row in active_groups
    }
    injection_group_digests = {
        str(row["normalized_task_digest"]) for row in injection_rows
    }
    targets = {
        "train": select_pilot_targets(
            active_groups, split="train", count=pilot_train, seed=pairing_seed
        ),
        "dev": select_pilot_targets(
            active_groups, split="dev", count=pilot_dev, seed=pairing_seed
        ),
    }
    attacks = {
        split: build_attack_rows(
            targets[split],
            injection_rows,
            reference_lookup,
            split=split,
            pairing_seed=pairing_seed,
            secopd_commit=secopd_commit,
        )
        for split in ("train", "dev")
    }
    utilities = {
        split: build_matched_utility_rows(
            attacks[split], pairing_seed=pairing_seed
        )
        for split in ("train", "dev")
    }
    shards: dict[str, list[list[dict[str, Any]]]] = {}
    shard_reports: dict[str, list[dict[str, int]]] = {}
    for split in ("train", "dev"):
        shards[split], shard_reports[split] = load_balanced_shards(
            attacks[split], shard_count=shard_count
        )
    smoke_count = min(12, len(attacks["train"]) + len(attacks["dev"]))
    smoke = select_smoke_rows(
        attacks["train"], attacks["dev"], count=smoke_count
    )
    audit = audit_transfer_package(attacks, utilities)
    return {
        "attacks": attacks,
        "utilities": utilities,
        "shards": shards,
        "smoke": smoke,
        "reports": {
            "secopd_original_split": secopd_split_report,
            "clean_group_deduplication": clean_dedup_report,
            "active_split": active_split_report,
            "injection_pool": injection_report,
            "source_overlap": {
                "clean_unique_groups": len(clean_group_digests),
                "injection_unique_groups": len(injection_group_digests),
                "normalized_instruction_input_groups_in_both_sources": len(
                    clean_group_digests & injection_group_digests
                ),
                "partition_key": (
                    "canonical_sha256([instruction.strip(), input.strip()])"
                ),
            },
            "shards": shard_reports,
            "audit": audit,
        },
    }
