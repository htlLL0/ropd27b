"""Deterministic Stage 3 on-policy and compromised-prefix collection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import floor
from typing import Any, Mapping, Sequence

from r_opcd.attack_pilot import POSITIONS, TASK_ORDER, build_base_messages
from r_opcd.attention_quarantine import tokenize_attention_quarantined_prompt


ATTACK_CATEGORIES = (
    "Alphanumeric Substitution",
    "Content Creation",
    "Information Retrieval",
    "Language Translation",
    "Learning and Tutoring",
    "Programming Help",
)
PREFIX_FRACTIONS = (0.25, 0.50, 0.75)


@dataclass(frozen=True)
class PromptTokenization:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    rendered_sha256: str
    attention_mask_sha256: str
    quarantined_token_indices: tuple[int, ...] = ()

    @property
    def prompt_tokens(self) -> int:
        return len(self.input_ids)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def attacked_dev_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    attacked = [
        dict(row)
        for row in rows
        if row.get("split") == "dev" and row.get("risk_label") == "context_injection"
    ]
    if len(attacked) != 280:
        raise ValueError(f"expected 280 attacked dev rows, observed {len(attacked)}")
    if len({str(row["id"]) for row in attacked}) != len(attacked):
        raise ValueError("attacked dev IDs are not unique")
    return sorted(attacked, key=lambda row: str(row["id"]))


def select_frontier_smoke(
    rows: Sequence[Mapping[str, Any]], *, seed: int = 20260902
) -> list[dict[str, Any]]:
    """Choose one row per task/category with eight rows at each attack position."""

    attacked = attacked_dev_rows(rows)
    selected: list[dict[str, Any]] = []
    used_bases_by_task: dict[str, set[str]] = {task: set() for task in TASK_ORDER}
    for task_index, task in enumerate(TASK_ORDER):
        for category_index, category in enumerate(ATTACK_CATEGORIES):
            desired_position = POSITIONS[(task_index + category_index) % len(POSITIONS)]
            candidates = [
                row
                for row in attacked
                if row["unified_task"] == task
                and row["attack_category"] == category
                and row["attack_position"] == desired_position
            ]
            if not candidates:
                raise ValueError(
                    f"missing smoke cell: {task}/{category}/{desired_position}"
                )
            used_bases = used_bases_by_task[task]
            candidates.sort(
                key=lambda row: (
                    str(row["base_task_id"]) in used_bases,
                    sha256(f"{seed}:{row['id']}".encode("utf-8")).hexdigest(),
                )
            )
            chosen = dict(candidates[0])
            selected.append(chosen)
            used_bases.add(str(chosen["base_task_id"]))

    if len(selected) != 24 or len({row["id"] for row in selected}) != 24:
        raise ValueError("frontier smoke must contain 24 unique rows")
    position_counts = {
        position: sum(row["attack_position"] == position for row in selected)
        for position in POSITIONS
    }
    if set(position_counts.values()) != {8}:
        raise ValueError(f"smoke position balance changed: {position_counts}")
    return selected


def select_prefix_smoke(smoke_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Select two rows per task while retaining all six attack categories."""

    by_cell = {
        (str(row["unified_task"]), str(row["attack_category"])): dict(row)
        for row in smoke_rows
    }
    selected: list[dict[str, Any]] = []
    for task_index, task in enumerate(TASK_ORDER):
        for offset in range(2):
            category = ATTACK_CATEGORIES[(2 * task_index + offset) % 6]
            selected.append(by_cell[(task, category)])
    if len(selected) != 8 or len({row["id"] for row in selected}) != 8:
        raise ValueError("prefix smoke must contain eight unique rows")
    if {row["attack_category"] for row in selected} != set(ATTACK_CATEGORIES):
        raise ValueError("prefix smoke lost attack-category coverage")
    return selected


def tokenize_student_prompt(tokenizer: Any, row: Mapping[str, Any]) -> PromptTokenization:
    rendered = tokenizer.apply_chat_template(
        build_base_messages(row),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("Student prompt rendered empty")
    encoded = tokenizer(rendered, add_special_tokens=False)
    input_ids = _flat_ints(encoded["input_ids"])
    raw_mask = encoded.get("attention_mask", [1] * len(input_ids))
    attention_mask = _flat_ints(raw_mask)
    if len(input_ids) != len(attention_mask) or any(value != 1 for value in attention_mask):
        raise ValueError("Student prompt must have a full all-ones attention mask")
    return PromptTokenization(
        input_ids=tuple(input_ids),
        attention_mask=tuple(attention_mask),
        rendered_sha256=sha256(rendered.encode("utf-8")).hexdigest(),
        attention_mask_sha256=sha256(bytes(attention_mask)).hexdigest(),
    )


def tokenize_teacher_plus_prompt(
    tokenizer: Any, row: Mapping[str, Any]
) -> PromptTokenization:
    result = tokenize_attention_quarantined_prompt(
        tokenizer,
        row,
        template_kwargs={"enable_thinking": False},
    )
    return PromptTokenization(
        input_ids=result.input_ids,
        attention_mask=result.attention_mask,
        rendered_sha256=result.rendered_sha256,
        attention_mask_sha256=result.attention_mask_sha256,
        quarantined_token_indices=result.quarantined_token_indices,
    )


def prefix_token_lengths(
    response_content_token_ids: Sequence[int],
    *,
    fractions: Sequence[float] = PREFIX_FRACTIONS,
) -> tuple[int, ...]:
    """Return distinct interior prefix depths using frozen response token IDs."""

    token_count = len(response_content_token_ids)
    if token_count < 2:
        return ()
    lengths: list[int] = []
    for fraction in fractions:
        if not 0.0 < fraction < 1.0:
            raise ValueError("prefix fractions must lie strictly between zero and one")
        length = min(token_count - 1, max(1, floor(token_count * fraction)))
        if length not in lengths:
            lengths.append(length)
    return tuple(lengths)


def strip_terminal_tokens(
    token_ids: Sequence[int], *, eos_token_ids: Sequence[int], pad_token_id: int | None
) -> tuple[int, ...]:
    terminal_ids = {int(value) for value in eos_token_ids}
    if pad_token_id is not None:
        terminal_ids.add(int(pad_token_id))
    content = [int(value) for value in token_ids]
    while content and content[-1] in terminal_ids:
        content.pop()
    return tuple(content)


def generation_stop_reason(
    token_ids: Sequence[int], *, eos_token_ids: Sequence[int], max_new_tokens: int
) -> str:
    eos = {int(value) for value in eos_token_ids}
    if token_ids and int(token_ids[-1]) in eos:
        return "eos_token"
    if len(token_ids) >= max_new_tokens:
        return "max_new_tokens"
    return "generation_stopped_other"


def append_shared_prefix(
    prompt: PromptTokenization, prefix_token_ids: Sequence[int]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not prefix_token_ids:
        raise ValueError("shared prefix must be non-empty")
    ids = prompt.input_ids + tuple(int(value) for value in prefix_token_ids)
    mask = prompt.attention_mask + (1,) * len(prefix_token_ids)
    if len(ids) != len(mask):
        raise RuntimeError("prefix append broke input/mask alignment")
    return ids, mask


def position_ids_preserving_quarantine(
    attention_mask: Sequence[int],
) -> tuple[int, ...]:
    """Keep absolute token slots for an unpadded mid-sequence quarantine.

    Zeros in this mask are q1 key-mask entries, not padding. They must not
    collapse RoPE positions for q1 or for any later prefix/response token.
    """

    if not attention_mask:
        raise ValueError("attention mask must be non-empty")
    if any(value not in (0, 1, False, True) for value in attention_mask):
        raise ValueError("attention mask must be binary")
    if not any(attention_mask):
        raise ValueError("attention mask must retain at least one token")
    return tuple(range(len(attention_mask)))


def _flat_ints(value: Any) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if value and isinstance(value[0], Sequence):
        value = value[0]
    return [int(item) for item in value]
