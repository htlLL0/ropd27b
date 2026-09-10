"""Exact CPU-only input-length audit for independent SecOPD A/U judges."""

from __future__ import annotations

from hashlib import sha256
from math import ceil
from statistics import fmean
from typing import Any, Mapping, Sequence

from r_opcd.independent_verifiers import (
    INDEPENDENT_VERIFIER_VERSION,
    SYSTEM_PROMPT,
    build_independent_prompt,
)


LENGTH_ROW_SCHEMA = "r-opcd-secopd-au-judge-input-length-row-v1"
REPORT_SCHEMA = "r-opcd-secopd-au-judge-input-length-audit-v1"
TARGETS = ("attack", "task")


def nearest_rank(values: Sequence[int], probability: float) -> int:
    """Return the deterministic nearest-rank percentile for positive integers."""

    if not values:
        raise ValueError("percentile input must be non-empty")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("percentile probability must lie in [0, 1]")
    ordered = sorted(int(value) for value in values)
    if any(value < 1 for value in ordered):
        raise ValueError("input lengths must be positive")
    if probability == 0.0:
        return ordered[0]
    return ordered[min(len(ordered) - 1, ceil(probability * len(ordered)) - 1)]


def renderer_contract(prompt_revision: str) -> dict[str, Any]:
    """Describe the exact render/tokenize path shared with the GPU judge runner."""

    return {
        "schema": "r-opcd-secopd-independent-au-renderer-contract-v1",
        "independent_verifier_version": INDEPENDENT_VERIFIER_VERSION,
        "prompt_revision": prompt_revision,
        "targets": list(TARGETS),
        "system_prompt_sha256": sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "prompt_builder": (
            "r_opcd.independent_verifiers.build_independent_prompt"
        ),
        "messages": [
            {"role": "system", "content_source": "SYSTEM_PROMPT"},
            {"role": "user", "content_source": "build_independent_prompt"},
        ],
        "apply_chat_template": {
            "tokenize": False,
            "add_generation_prompt": True,
            "judge_chat_template_kwargs_applied": True,
        },
        "tokenizer_call": {
            "add_special_tokens": False,
            "padding": True,
            "return_tensors": "pt",
        },
        "length_measurement": "encoded.attention_mask.sum(dim=1)",
        "model_forward_calls": 0,
    }


def validate_case_ids(cases: Sequence[Mapping[str, Any]]) -> list[str]:
    """Validate the runner's unique-case identity contract."""

    if not cases:
        raise ValueError("cases must be non-empty")
    case_ids: list[str] = []
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise ValueError(f"case {index} must be a mapping")
        raw_case_id = case.get("case_id")
        if not isinstance(raw_case_id, str) or not raw_case_id.strip():
            raise ValueError(f"case {index} has an empty case_id")
        case_ids.append(raw_case_id)
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("case IDs must be unique")
    return case_ids


def _attention_lengths(encoded: Mapping[str, Any], expected: int) -> list[int]:
    if not isinstance(encoded, Mapping) or "attention_mask" not in encoded:
        raise ValueError("tokenizer output lacks attention_mask")
    attention_mask = encoded["attention_mask"]
    try:
        totals = attention_mask.sum(dim=1)
    except Exception as error:
        raise ValueError("attention_mask must support sum(dim=1)") from error
    raw_totals = totals.tolist() if hasattr(totals, "tolist") else totals
    if (
        not isinstance(raw_totals, Sequence)
        or isinstance(raw_totals, (str, bytes))
    ):
        raise ValueError("attention_mask sums must be a sequence")
    lengths = [int(value) for value in raw_totals]
    if len(lengths) != expected:
        raise ValueError("tokenizer attention_mask batch dimension mismatch")
    if any(value < 1 for value in lengths):
        raise ValueError("tokenizer produced an empty judge input")
    return lengths


def summarize_target_rows(
    rows: Sequence[Mapping[str, Any]], *, max_input_tokens: int
) -> dict[str, Any]:
    """Summarize one judge/target route and retain every cap-relevant ID."""

    if not rows:
        raise ValueError("target rows must be non-empty")
    lengths = [int(row["input_tokens"]) for row in rows]
    maximum = max(lengths)
    worst = [row for row in rows if int(row["input_tokens"]) == maximum]
    over_cap = [
        row for row in rows if int(row["input_tokens"]) > max_input_tokens
    ]
    quantiles = {
        "method": "nearest_rank",
        "p50": nearest_rank(lengths, 0.50),
        "p90": nearest_rank(lengths, 0.90),
        "p95": nearest_rank(lengths, 0.95),
        "p99": nearest_rank(lengths, 0.99),
        "p100": maximum,
    }
    return {
        "count": len(lengths),
        "min": min(lengths),
        "mean": fmean(lengths),
        "quantiles": quantiles,
        "p50_nearest_rank": quantiles["p50"],
        "p90_nearest_rank": quantiles["p90"],
        "p95_nearest_rank": quantiles["p95"],
        "p99_nearest_rank": quantiles["p99"],
        "max": maximum,
        "at_cap": sum(value == max_input_tokens for value in lengths),
        "over_cap": len(over_cap),
        "minimum_headroom": max_input_tokens - maximum,
        "worst_case_ids": [str(row["case_id"]) for row in worst],
        "worst_case_indices_zero_based": [
            int(row["case_index_zero_based"]) for row in worst
        ],
        "over_cap_case_ids": [str(row["case_id"]) for row in over_cap],
    }


def audit_judge_input_lengths(
    *,
    tokenizer: Any,
    cases: Sequence[Mapping[str, Any]],
    judge_id: str,
    prompt_revision: str,
    chat_template_kwargs: Mapping[str, Any] | None,
    batch_size: int,
    max_input_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Render and tokenize all case x target inputs for one configured judge."""

    if not isinstance(judge_id, str) or not judge_id.strip():
        raise ValueError("judge_id must be a non-empty string")
    if not isinstance(prompt_revision, str) or not prompt_revision:
        raise ValueError("prompt_revision must be a non-empty string")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be positive")
    if type(max_input_tokens) is not int or max_input_tokens < 1:
        raise ValueError("max_input_tokens must be positive")
    if chat_template_kwargs is not None and not isinstance(
        chat_template_kwargs, Mapping
    ):
        raise ValueError("chat_template_kwargs must be a mapping")

    case_ids = validate_case_ids(cases)
    template_kwargs = dict(chat_template_kwargs or {})
    all_rows: list[dict[str, Any]] = []
    target_summaries: dict[str, dict[str, Any]] = {}

    for target in TARGETS:
        target_rows: list[dict[str, Any]] = []
        for start in range(0, len(cases), batch_size):
            batch = cases[start : start + batch_size]
            user_prompts: list[str] = []
            rendered_prompts: list[str] = []
            for case in batch:
                user_prompt = build_independent_prompt(
                    case, target, prompt_revision=prompt_revision
                )
                rendered = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                    **template_kwargs,
                )
                if not isinstance(rendered, str) or not rendered:
                    raise ValueError("chat template produced an empty rendered prompt")
                user_prompts.append(user_prompt)
                rendered_prompts.append(rendered)

            encoded = tokenizer(
                rendered_prompts,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            )
            input_lengths = _attention_lengths(encoded, len(batch))
            for offset, (
                case_id,
                user_prompt,
                rendered_prompt,
                input_tokens,
            ) in enumerate(
                zip(
                    case_ids[start : start + len(batch)],
                    user_prompts,
                    rendered_prompts,
                    input_lengths,
                    strict=True,
                )
            ):
                row = {
                    "schema": LENGTH_ROW_SCHEMA,
                    "judge_id": judge_id,
                    "target": target,
                    "case_index_zero_based": start + offset,
                    "case_id": case_id,
                    "input_tokens": input_tokens,
                    "max_input_tokens": max_input_tokens,
                    "within_cap": input_tokens <= max_input_tokens,
                    "prompt_sha256": sha256(
                        user_prompt.encode("utf-8")
                    ).hexdigest(),
                    "rendered_prompt_sha256": sha256(
                        rendered_prompt.encode("utf-8")
                    ).hexdigest(),
                }
                target_rows.append(row)
                all_rows.append(row)
        target_summaries[target] = summarize_target_rows(
            target_rows, max_input_tokens=max_input_tokens
        )

    expected_rows = len(cases) * len(TARGETS)
    if len(all_rows) != expected_rows:
        raise RuntimeError("judge input audit did not cover the full case x target grid")
    return all_rows, {
        "case_count": len(cases),
        "target_count": len(TARGETS),
        "input_instance_count": len(all_rows),
        "targets": target_summaries,
        "model_forward_calls": 0,
    }
