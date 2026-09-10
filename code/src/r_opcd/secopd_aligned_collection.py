"""Opt-in SecOPD role/decoding collection; no changes to legacy training code."""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from html import escape
from typing import Any

import numpy as np

from r_opcd.attack_pilot import build_attention_quarantine_messages
from r_opcd.frontier_collection import canonical_sha256, generation_stop_reason, strip_terminal_tokens
from r_opcd.secopd_data import validate_secopd_attack_row

ARM = "secopd_roles_think_t1_16k_v1"


def source_seed(parent_id: str, purpose: str, seed: int = 20260910) -> int:
    return int(canonical_sha256([ARM, purpose, seed, parent_id])[:8], 16)


def reposition_source(original: dict, seed: int = 20260910) -> dict:
    """Preserve task, injector and attack kind; redraw only straightforward position."""
    validate_secopd_attack_row(original)
    row = deepcopy(original)
    row["parent_source_id"] = original["id"]
    row["parent_source_row_sha256"] = canonical_sha256(original)
    row["parent_mixture_record_sha256"] = original.get("mixture_record_sha256")
    row["parent_attack_name"] = original["attack_name"]
    rng_seed = source_seed(original["id"], "attack_position", seed)
    draw = float(np.random.default_rng(rng_seed).random())
    if row["attack_kind"] == "straightforward":
        before = draw < 0.5
        row["attack_position"] = "start" if before else "end"
        row["secopd_attack_plan_cell"] = "straightforward_" + ("prepend" if before else "append")
        row["contaminated_context"] = (
            row["malicious_span"] + " " + row["clean_context"] if before
            else row["clean_context"] + " " + row["malicious_span"]
        )
    elif row["attack_kind"] == "completion":
        if row["attack_position"] != "end":
            raise ValueError("Completion must stay appended")
    else:
        raise ValueError("Unknown attack kind")
    row["position_sampling"] = {
        "seed": rng_seed, "master_seed": seed,
        "uniform_draw": draw if row["attack_kind"] == "straightforward" else None,
        "rule": "straightforward: Bernoulli(0.5) prepend; completion: append",
        "materialization": 0, "per_update_online_regeneration": False,
    }
    span, context = row["malicious_span"], row["contaminated_context"]
    if context.count(span) != 1:
        raise ValueError("Attack span must occur exactly once")
    row["malicious_span_char_start"] = context.index(span)
    row["malicious_span_char_end"] = context.index(span) + len(span)
    row["id"] = "r-opcd::" + ARM + "::" + canonical_sha256([
        original["id"], seed, row["secopd_attack_plan_cell"], context])
    row["raw_id"] = original["raw_id"] + "::" + ARM
    row["attack_name"] = "secopd-metasecalign-" + row["secopd_attack_plan_cell"].replace("_", "-") + "-roles-v1"
    row["mixture_namespace"] = ARM
    row["mixture_seed"] = seed
    row["mixture_selection_cell"] = row["secopd_attack_plan_cell"]
    row["max_new_tokens"] = 16384
    row["collection_arm"] = ARM
    row["prompt_view"] = "secopd_user_input_thinking_v1"
    row["adaptation_note"] = "Same parent task/injector/type; seeded position redraw; separate user/input; thinking on."
    row.pop("mixture_record_sha256", None)
    row["mixture_record_sha256"] = canonical_sha256(row)
    validate_secopd_attack_row(row)
    return row


def student_messages(row: dict) -> list[dict]:
    return [{"role": "user", "content": row["user_query"]},
            {"role": "input", "content": row["contaminated_context"]}]


def teacher_messages(row: dict) -> list[dict]:
    messages = build_attention_quarantine_messages(row)
    if [m["role"] for m in messages] != ["system", "user", "tool", "user"]:
        raise ValueError("Existing T-plus contract changed; explicit re-audit required")
    messages[2]["role"] = "input"
    return messages


def text_chunks(messages: list[dict]) -> list[str]:
    """Match pinned Qwen3VL header/body chunks + Qwen3_5 generation suffix.

    Scope is text-only prompts without assistant history or tools. Do not use
    this minimal renderer as a general tool-call or multimodal chat renderer.
    """
    if not messages:
        raise ValueError("Empty conversation")
    chunks = []
    for i, msg in enumerate(messages):
        if msg["role"] not in ("system", "user", "input") or not isinstance(msg["content"], str):
            raise ValueError("Unsupported role or non-text content")
        chunks.extend([("\n" if i else "") + "<|im_start|>" + msg["role"] + "\n",
                       msg["content"] + "<|im_end|>"])
    chunks.append("\n<|im_start|>assistant\n<think>\n")
    return chunks


def tokenize_messages(tokenizer: Any, messages: list[dict], span: str | None = None) -> dict:
    chunks = text_chunks(messages)
    rendered = "".join(chunks)
    ids, offsets, cursor = [], [], 0
    for chunk in chunks:
        encoded = tokenizer(chunk, add_special_tokens=False, return_offsets_mapping=True)
        ids.extend(encoded["input_ids"])
        offsets.extend((a + cursor, b + cursor) for a, b in encoded["offset_mapping"])
        cursor += len(chunk)
    mask = [1] * len(ids)
    selected = []
    if span is not None:
        marker = '<quarantined_span id="q1"'
        if rendered.count(marker) != 1:
            raise ValueError("Ambiguous q1 marker")
        start = rendered.index(">\n", rendered.index(marker)) + 2
        end = rendered.index("\n  </quarantined_span>", start)
        if rendered[start:end] != escape(span, quote=True):
            raise ValueError("T-plus changed attack text")
        selected = [i for i, (a, b) in enumerate(offsets) if b > start and a < end]
        if not selected:
            raise ValueError("Empty q1 mask")
        for i in selected:
            mask[i] = 0
    if not ids or len(ids) != len(offsets) or not mask[-1]:
        raise ValueError("Invalid prompt or masked generation suffix")
    if tokenizer.decode(ids, skip_special_tokens=False) != rendered:
        raise ValueError("Prompt token round trip changed visible text")
    return {"prompt_token_ids": ids, "prompt_attention_mask": mask, "prompt_tokens": len(ids),
            "prompt_rendered": rendered, "prompt_token_ids_sha256": canonical_sha256(ids),
            "prompt_rendered_sha256": sha256(rendered.encode()).hexdigest(),
            "prompt_attention_mask_sha256": sha256(bytes(mask)).hexdigest(),
            "attention_quarantined_token_indices": selected,
            "attention_quarantined_tokens": len(selected),
            "tokenization_contract": "pinned_qwen3_5_text_header_body_chunks_v1"}


def make_request(item: dict, config: dict) -> dict:
    return {"model": config["served_model_name"], "prompt": item["prompt"]["prompt_token_ids"],
            "temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
            "presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0,
            "max_tokens": 16384, "seed": source_seed(item["source"]["parent_source_id"], "rollout", config["seed"]),
            "n": 1, "stream": False, "ignore_eos": False, "stop_token_ids": config["eos_token_ids"],
            "return_token_ids": True, "add_special_tokens": False,
            "skip_special_tokens": False, "spaces_between_special_tokens": False}


def response_parts(tokenizer: Any, content: list[int]) -> dict:
    boundary = tokenizer.encode("</think>", add_special_tokens=False)
    if len(boundary) != 1:
        raise ValueError("Thinking boundary must be one native token")
    locations = [i for i, token in enumerate(content) if token == boundary[0]]
    first = locations[0] if locations else None
    return {"thinking_prefix_is_prompt_not_sampled": True,
            "thinking_end_count": len(locations), "thinking_end_token_index": first,
            "reasoning": tokenizer.decode(content[:first] if first is not None else content, skip_special_tokens=False),
            "final_answer": tokenizer.decode(content[first + 1:], skip_special_tokens=False).strip() if first is not None else "",
            "final_answer_token_ids": content[first + 1:] if first is not None else [],
            "reasoning_tokens": first if first is not None else len(content)}


def make_record(item: dict, raw: dict, config: dict, tokenizer: Any, elapsed: float) -> dict:
    # Reuse legacy identity/usage/stop validation, not its hard-coded greedy metadata.
    from collect_secopd_qwen36_9600 import parse_tokens
    if raw.get("model") != config["served_model_name"]:
        raise ValueError("Backend returned wrong model identity")
    tokens, reason = parse_tokens(raw, item["prompt"]["prompt_token_ids"], config)
    content = list(strip_terminal_tokens(tokens, eos_token_ids=config["eos_token_ids"], pad_token_id=tokenizer.pad_token_id))
    row = item["source"]
    fields = ("raw_id", "base_task_id", "split", "unified_task", "task_name", "task_type", "attack_name",
              "attack_category", "attack_position", "mixture_schema_version", "mixture_namespace", "mixture_seed",
              "mixture_branch", "mixture_source_family", "mixture_selection_cell", "mixture_record_sha256",
              "attack_gate_status", "recoverability_c_status")
    return {"schema": "r-opcd-stage3-student-rollout-v1", "trajectory_id": row["id"] + "::student_on_policy",
            "source_id": row["id"], "source_row_sha256": canonical_sha256(row),
            "parent_source_id": row["parent_source_id"], "collection_arm": ARM,
            **{k: row[k] for k in fields if k in row}, **item["prompt"],
            "policy_role": "frozen_initial_student_rollout", "model_name": config["model_name"],
            "model_revision": config["model_revision"], "student_adapter": None,
            "messages": item["messages"], "prompt_view": row["prompt_view"],
            "prompt_consumed_fields": ["user_query", "contaminated_context"],
            "do_sample": True, "enable_thinking": True, "temperature": 1.0, "top_p": 1.0,
            "top_k": 0, "backend_top_k": -1, "decode_seed_index": 0,
            "decode_seed": item["request"]["seed"], "effective_sampling_seed": item["request"]["seed"],
            "backend_seed": item["request"]["seed"], "generation_backend": "vllm-0.24.0",
            "max_new_tokens": 16384, "response_token_ids": tokens, "response_content_token_ids": content,
            "response_token_ids_sha256": canonical_sha256(tokens), "generated_tokens_including_terminal": len(tokens),
            "response_content_tokens": len(content), "stop_reason": reason,
            "response": tokenizer.decode(content, skip_special_tokens=False).strip(), **response_parts(tokenizer, content),
            "generation_elapsed_seconds": elapsed, "attack_success_label": "unjudged_semantic",
            "user_task_success_label": "unjudged_semantic", "token_identity_source": "backend_return_token_ids_no_retokenization",
            "generation_config_sha256": canonical_sha256(config),
            "training_ready": False, "note": "Full sampled trajectory retained; final_answer is diagnostic only"}


def smoke_indices(items: list[dict]) -> list[int]:
    chosen = []
    for lane in ("gpu25", "gpu34"):
        for cell in ("straightforward_prepend", "straightforward_append", "completion"):
            matches = sorted((i for i in items if i["lane"] == lane and i["source"]["secopd_attack_plan_cell"] == cell),
                             key=lambda i: source_seed(i["source"]["parent_source_id"], "smoke"))
            if len(matches) < 2:
                raise ValueError("Missing two canaries per lane/cell")
            chosen.extend(i["index"] for i in matches[:2])
    return chosen
