"""Unlabeled clean-T+ OPCD primitives, with a bounded full-vocabulary head.

No semantic label, gold answer, benign SFT, or quarantine prompt is consumed.
All three views score the SAME realized student response with causal alignment.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from r_opcd.objective import compute_full_vocab_objective
from r_opcd.secopd_aligned_collection import tokenize_messages


def clean_messages(source):
    return [{"role": "user", "content": source["user_query"]},
            {"role": "input", "content": source["clean_context"]}]


def clean_prompt(source, tokenizer):
    # clean_context is the original upstream input, not sanitized_context/gold.
    return tokenize_messages(tokenizer, clean_messages(source))


def training_example(item, plus_prompt, response, *, max_sequence_tokens, vocab_size):
    attacked = item["prompt"]["prompt_token_ids"]
    clean = plus_prompt["prompt_token_ids"]
    if not response or any(type(t) is not int or not 0 <= t < vocab_size for t in response):
        raise ValueError("Empty or invalid response token IDs")
    if any(len(p) + len(response) > max_sequence_tokens for p in (attacked, clean)):
        raise ValueError("Sequence limit exceeded; silent prompt/response truncation is forbidden")
    if not all(item["prompt"]["prompt_attention_mask"]) or not all(plus_prompt["prompt_attention_mask"]):
        raise ValueError("Clean-teacher variant requires ordinary unmasked contexts")
    return {"index": item["index"], "source_id": item["source"]["id"],
            "student_prompt_ids": attacked, "teacher_minus_prompt_ids": attacked,
            "teacher_plus_prompt_ids": clean, "response_ids": list(response)}


def response_hidden(model, prompt_ids, response_ids):
    """Skip the enormous whole-sequence LM head; predict y[t] at |prompt|-1+t."""
    base = model.get_base_model()
    decoder = base.model.language_model
    device = base.lm_head.weight.device
    ids = torch.tensor([prompt_ids + response_ids[:-1]], device=device, dtype=torch.long)
    output = decoder(input_ids=ids, attention_mask=torch.ones_like(ids),
                     use_cache=False, return_dict=True)
    hidden = output.last_hidden_state[:, len(prompt_ids) - 1:, :]
    if hidden.shape[1] != len(response_ids):
        raise RuntimeError("Causal response alignment failed")
    return hidden


def chunked_backward(student, minus, plus, head, objective, chunk_size):
    """Exact global token denominator; accumulate dL/dhidden then ONE decoder VJP.

The target and masks are recomputed without a decoder forward. This avoids
retaining [all response tokens, vocabulary] distributions and avoids repeatedly
backpropagating through the transformer once per vocabulary chunk.
"""
    if student.shape != minus.shape or student.shape != plus.shape or chunk_size < 1:
        raise ValueError("Mismatched hidden states or invalid chunk size")
    if any(p.requires_grad for p in head.parameters()):
        raise ValueError("The shared LM head must remain frozen")
    if minus.requires_grad or plus.requires_grad:
        raise ValueError("Teacher states must be detached")
    leaf = student.detach().float().requires_grad_(True)
    ones = torch.ones(student.shape[0], device=student.device)

    def part(start):
        sl = slice(start, start + chunk_size)
        s = F.linear(leaf[:, sl].to(head.weight.dtype), head.weight, head.bias)
        with torch.no_grad():
            m = head(minus[:, sl])
            p = head(plus[:, sl])
        return compute_full_vocab_objective(s, m, p, attack_gate=ones,
                    reliability_weight=ones, **objective)

    numerator = selected = disagreement = 0.0
    with torch.no_grad():
        for start in range(0, student.shape[1], chunk_size):
            result = part(start)
            numerator += result.numerator.item()
            selected += result.selected_mass.item()
            disagreement += result.disagreement.sum().item()
            del result
    denominator = selected + objective.get("eps_loss", 1e-8)
    for start in range(0, student.shape[1], chunk_size):
        result = part(start)
        (result.numerator / denominator).backward()
        del result
    if leaf.grad is None or not torch.isfinite(leaf.grad).all():
        raise RuntimeError("Missing or nonfinite hidden gradient")
    student.backward(leaf.grad.to(student.dtype))
    return {"loss": numerator / denominator, "selected_tokens": selected,
            "response_tokens": student.shape[1],
            "mean_teacher_js": disagreement / (student.shape[0] * student.shape[1]),
            "attack_gate": 1, "c": 1, "sft_loss": 0}


def train_step(model, optimizer, example, config):
    optimizer.zero_grad(set_to_none=True)
    response = example["response_ids"]
    model.eval()
    with model.disable_adapter(), torch.no_grad():
        minus = response_hidden(model, example["teacher_minus_prompt_ids"], response)
        plus = response_hidden(model, example["teacher_plus_prompt_ids"], response)
    model.train()
    student = response_hidden(model, example["student_prompt_ids"], response)
    metrics = chunked_backward(student, minus, plus, model.get_base_model().lm_head,
                               config["objective"], config["runtime"]["response_chunk_size"])
    params = [p for p in model.parameters() if p.requires_grad]
    if not params or any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
        raise RuntimeError("Missing or nonfinite LoRA gradient")
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise RuntimeError("A frozen parameter received a gradient")
    norm = torch.nn.utils.clip_grad_norm_(params, config["optimizer"]["max_grad_norm"], error_if_nonfinite=True)
    metrics["grad_norm"] = norm.item()
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in params):
        raise RuntimeError("Nonfinite adapter parameters after optimizer update")
    return metrics


def attach_adapter(base, config):
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", **config["lora"]))
    # No dropout makes repeated teacher/student comparisons and resumes stable.
    if config["lora"]["lora_dropout"] != 0:
        raise ValueError("This frozen-view contract requires zero LoRA dropout")
    if config["runtime"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in n or ".language_model." not in n for n in trainable):
        raise RuntimeError("Only text-decoder LoRA parameters may train")
    return model
