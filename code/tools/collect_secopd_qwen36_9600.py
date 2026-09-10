#!/usr/bin/env python3
"""Isolated native-tokenizer gate and exact-token vLLM collection; no training."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time
import urllib.request

from r_opcd.attack_pilot import build_base_messages, local_diagnostics
from r_opcd.frontier_collection import (
    canonical_sha256, generation_stop_reason, strip_terminal_tokens,
    tokenize_student_prompt, tokenize_teacher_plus_prompt,
)


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    h = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def emit(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    handle.flush()


def prompt_record(prompt):
    return {
        "prompt_token_ids": list(prompt.input_ids),
        "prompt_attention_mask": list(prompt.attention_mask),
        "prompt_tokens": prompt.prompt_tokens,
        "prompt_token_ids_sha256": canonical_sha256(prompt.input_ids),
        "prompt_rendered_sha256": prompt.rendered_sha256,
        "prompt_attention_mask_sha256": prompt.attention_mask_sha256,
        "attention_quarantined_token_indices": list(prompt.quarantined_token_indices),
        "attention_quarantined_tokens": len(prompt.quarantined_token_indices),
    }


def choose_smoke(rows):
    selected = []
    for cell in ("straightforward_prepend", "straightforward_append", "completion"):
        members = [r for r in rows if r["source"]["secopd_attack_plan_cell"] == cell]
        selected.extend(sorted(members, key=lambda r: canonical_sha256([20260909, r["source"]["id"]]))[:4])
    if len(selected) != 12:
        raise ValueError("Smoke requires four records per construction cell")
    return selected


def make_request(prompt_ids, config):
    return {
        "model": config["served_model_name"], "prompt": list(prompt_ids),
        "temperature": 0.0, "top_p": 1.0, "top_k": -1,
        "max_tokens": config["max_new_tokens"], "seed": config["seed"],
        "n": 1, "stream": False, "ignore_eos": False,
        "stop_token_ids": config["eos_token_ids"],
        "return_token_ids": True, "add_special_tokens": False,
        "skip_special_tokens": True, "spaces_between_special_tokens": False,
    }


def parse_tokens(raw, prompt_ids, config):
    choices = raw.get("choices", [])
    if len(choices) != 1:
        raise ValueError("Expected one backend choice")
    choice = choices[0]
    if choice.get("prompt_token_ids") != prompt_ids:
        raise ValueError("Backend consumed different prompt token IDs")
    tokens = choice.get("token_ids")
    if not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError("Backend did not return actual generated token IDs")
    if raw["usage"]["completion_tokens"] != len(tokens):
        raise ValueError("Backend token IDs/usage mismatch")
    if raw["usage"]["prompt_tokens"] != len(prompt_ids):
        raise ValueError("Backend prompt usage mismatch")
    reason = generation_stop_reason(tokens, eos_token_ids=config["eos_token_ids"], max_new_tokens=config["max_new_tokens"])
    if choice["finish_reason"] == "length":
        if len(tokens) != config["max_new_tokens"] or reason != "max_new_tokens":
            raise ValueError("Unexpected context truncation or inconsistent cap state")
    elif choice["finish_reason"] == "stop":
        if reason != "eos_token":
            raise ValueError("Backend omitted terminal token or used an undeclared stop")
    else:
        raise ValueError(f"Unexpected finish reason: {choice['finish_reason']}")
    return tokens, reason


def prepare(args, config, tokenizer):
    from audit_secopd_transfer_prompts import audit_prompt_row
    source_path = args.data / "pilot_train/attack_candidates.jsonl"
    rows = read_rows(source_path)
    if len(rows) != 9600 or len({r["id"] for r in rows}) != 9600:
        raise ValueError("Expected exactly 9600 unique source cases")
    if digest(source_path) != config["source_sha256"]:
        raise ValueError("Source hash changed")
    probe = tokenizer("audit", add_special_tokens=False)["input_ids"]
    counts = Counter()
    lengths, plus_lengths = [], []
    with (args.output / "prepared.jsonl").open("x") as prepared, (args.output / "teacher_views.jsonl").open("x") as views:
        for index, row in enumerate(rows):
            checked = audit_prompt_row(tokenizer, row, probe_response_ids=probe)
            prompt = tokenize_student_prompt(tokenizer, row)
            plus = tokenize_teacher_plus_prompt(tokenizer, row)
            if prompt.prompt_tokens + config["max_new_tokens"] > config["max_model_len"]:
                raise ValueError(f"No prompt truncation allowed: {row['id']}")
            emit(prepared, {"index": index, "source": row, "prompt": prompt_record(prompt),
                            "messages": build_base_messages(row),
                            "request": make_request(prompt.input_ids, config)})
            emit(views, {"source_id": row["id"], "source_row_sha256": canonical_sha256(row),
                         "teacher_plus": prompt_record(plus), "audit": checked,
                         "evidence_boundary": "tokenizer_and_mask_only_no_model_forward"})
            lengths.append(prompt.prompt_tokens)
            plus_lengths.append(plus.prompt_tokens)
            counts[row["secopd_attack_plan_cell"]] += 1
            if (index + 1) % 250 == 0:
                save(args.output / "progress.json", {"stage": "cpu_prompt_gate", "completed": index + 1, "total": len(rows), "updated_at_utc": now()})
                print(f"native prompt gate {index + 1}/{len(rows)}", flush=True)
    report = {"status": "pass", "rows": len(rows), "counts": counts,
              "source_sha256": config["source_sha256"], "prepared_sha256": digest(args.output / "prepared.jsonl"),
              "teacher_views_sha256": digest(args.output / "teacher_views.jsonl"),
              "max_student_prompt_tokens": max(lengths), "mean_student_prompt_tokens": sum(lengths) / len(lengths),
              "max_teacher_plus_prompt_tokens": max(plus_lengths),
              "model_revision": config["model_revision"], "completed_at_utc": now(),
              "checks": ["Student_equals_Tminus", "exact_q1_only_mask", "shared_prefix_unmasked", "offline_reference_mutation_invariant", "no_prompt_truncation"],
              "not_confirmed": ["27B_quarantined_forward_correctness", "A_U_labels", "training_readiness"]}
    save(args.output / "prompt_gate.json", report)
    print(json.dumps(report), flush=True)


def make_record(item, raw, config, tokenizer, elapsed):
    row, prompt = item["source"], item["prompt"]
    tokens, reason = parse_tokens(raw, prompt["prompt_token_ids"], config)
    content = list(strip_terminal_tokens(tokens, eos_token_ids=config["eos_token_ids"], pad_token_id=tokenizer.pad_token_id))
    response = tokenizer.decode(content, skip_special_tokens=True).strip()
    provenance_fields = ("raw_id", "base_task_id", "split", "unified_task", "task_name", "task_type", "attack_name",
                         "attack_category", "attack_position", "mixture_schema_version", "mixture_namespace",
                         "mixture_seed", "mixture_branch", "mixture_source_family", "mixture_selection_cell",
                         "mixture_record_sha256", "attack_gate_status", "recoverability_c_status")
    return {"schema": "r-opcd-stage3-student-rollout-v1", "trajectory_id": row["id"] + "::student_on_policy",
            "source_id": row["id"], "source_row_sha256": canonical_sha256(row),
            **{field: row[field] for field in provenance_fields if field in row}, **prompt,
            "policy_role": "current_student_on_policy", "model_name": config["model_name"],
            "model_revision": config["model_revision"], "student_adapter": None,
            "decode_seed_index": 0, "decode_seed": None, "effective_sampling_seed": None,
            "do_sample": False, "temperature": 0.0, "top_p": 1.0, "top_k": 0,
            "backend_top_k": -1, "backend_seed": config["seed"], "generation_backend": "vllm-0.24.0",
            "prompt_view": "ordinary_attacked_prompt", "prompt_consumed_fields": ["user_query", "contaminated_context"],
            "max_new_tokens": config["max_new_tokens"], "response_token_ids": tokens,
            "response_content_token_ids": content, "response_token_ids_sha256": canonical_sha256(tokens),
            "generated_tokens_including_terminal": len(tokens), "response_content_tokens": len(content),
            "stop_reason": reason, "response": response, "generation_elapsed_seconds": elapsed,
            "attack_success_label": "unjudged_semantic", "user_task_success_label": "unjudged_or_proxy_only",
            "offline_reference_not_in_prompt": {"target_answer": row.get("target_answer"), "target_answers": row.get("target_answers")},
            "local_diagnostics_are_not_semantic_labels": local_diagnostics(row, response),
            "token_identity_source": "backend_return_token_ids_no_retokenization",
            "generation_config_sha256": canonical_sha256(config)}


def generate(args, config, tokenizer):
    gate = json.loads((args.output / "prompt_gate.json").read_text())
    if gate["status"] != "pass" or digest(args.output / "prepared.jsonl") != gate["prepared_sha256"]:
        raise ValueError("Missing or changed preparation gate")
    items = read_rows(args.output / "prepared.jsonl")
    cache = args.output / "cases"
    cache.mkdir(exist_ok=False)
    completed = {}
    counts = Counter()
    started = time.monotonic()

    def one(item):
        case_dir = cache / f"{item['index']:05d}"
        case_dir.mkdir(exist_ok=False)
        save(case_dir / "request.json", {"source_id": item["source"]["id"], "messages": item["messages"], "body": item["request"]})
        before = time.monotonic()
        request = urllib.request.Request("http://127.0.0.1:8000/v1/completions", data=json.dumps(item["request"]).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=config["request_timeout_seconds"]) as reply:
                raw = json.load(reply)
            save(case_dir / "raw_response.json", raw)
            record = make_record(item, raw, config, tokenizer, time.monotonic() - before)
            save(case_dir / "trajectory.json", record)
            return item["index"], record
        except Exception as exc:
            save(case_dir / "error.json", {"error_type": type(exc).__name__, "message": str(exc), "at_utc": now(), "semantic_label": None})
            raise

    def progress(stage):
        elapsed = time.monotonic() - started
        value = {"stage": stage, "completed": len(completed), "total": len(items),
                 "counts": counts, "elapsed_seconds": elapsed, "updated_at_utc": now(),
                 "estimated_remaining_seconds": elapsed / len(completed) * (len(items) - len(completed)) if completed else None}
        save(args.output / "progress.json", value)
        return value

    def run_items(batch, stage):
        iterator = iter(batch)
        with ThreadPoolExecutor(max_workers=config["concurrency"]) as executor:
            pending = {executor.submit(one, item) for item in [next(iterator, None) for _ in range(config["concurrency"])] if item is not None}
            while pending:
                done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                if not done:
                    progress(stage)
                for future in done:
                    index, record = future.result()
                    if index in completed:
                        raise ValueError("Duplicate source index")
                    completed[index] = record
                    counts[record["stop_reason"]] += 1
                    counts["empty_response"] += not bool(record["response"])
                    emit(journal, {"index": index, "source_id": record["source_id"], "stop_reason": record["stop_reason"], "tokens": record["generated_tokens_including_terminal"], "at_utc": now()})
                    progress(stage)
                    if len(completed) % 10 == 0:
                        print(f"{stage}: {len(completed)}/{len(items)} {dict(counts)}", flush=True)
                    item = next(iterator, None)
                    if item is not None:
                        pending.add(executor.submit(one, item))

    with (args.output / "completed.jsonl").open("x") as journal:
        try:
            smoke = choose_smoke(items)
            save(args.output / "smoke_selection.json", [{"index": item["index"], "source_id": item["source"]["id"], "cell": item["source"]["secopd_attack_plan_cell"]} for item in smoke])
            run_items(smoke, "smoke")
            if any(not row["response"] or row["stop_reason"] != "eos_token" for row in completed.values()):
                raise ValueError("Smoke contains empty/capped outputs; inspect before scaling")
            save(args.output / "smoke_gate.json", {"status": "pass", "count": 12, "note": "Format/token/EOS gate, not A/U evaluation", "completed_at_utc": now()})
            run_items([item for item in items if item["index"] not in completed], "full_generation")
            if set(completed) != set(range(9600)):
                raise ValueError("Incomplete exact source coverage")
            with (args.output / "generations.jsonl").open("x") as handle:
                for index in range(9600):
                    emit(handle, completed[index])
            save(args.output / "generation_summary.json", {**progress("generation_complete"),
                 "generations_sha256": digest(args.output / "generations.jsonl"), "training_ready": False,
                 "pending": ["cap_and_empty_disposition", "independent_A_U_labels", "27B_Tplus_forward_gate", "27B_reliability_calibration"]})
        except Exception as exc:
            save(args.output / "failure.json", {**progress("failed"), "type": type(exc).__name__, "error": str(exc)})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "generate"))
    parser.add_argument("--model", type=Path, default=Path("/model"))
    parser.add_argument("--data", type=Path, default=Path("/data"))
    parser.add_argument("--output", type=Path, default=Path("/output"))
    args = parser.parse_args()
    config = json.loads((args.output / "config.json").read_text())
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.eos_token_id not in config["eos_token_ids"]:
        raise ValueError("Tokenizer/model EOS mismatch")
    {"prepare": prepare, "generate": generate}[args.mode](args, config, tokenizer)


if __name__ == "__main__":
    main()
