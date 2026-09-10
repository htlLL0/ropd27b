"""Reusable exact-versus-compressed gate for frozen response logits."""

from __future__ import annotations

from dataclasses import asdict
import gc
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from r_opcd.diagnostics import compare_objectives
from r_opcd.distributions import compute_uniform_tail_topk_objective
from r_opcd.objective import compute_full_vocab_objective, js_divergence_from_logits


def evaluate_compression_gate(
    teacher_minus_logits: Tensor,
    teacher_plus_logits: Tensor,
    response_mask: Tensor,
    *,
    objective_config: Mapping[str, Any],
    topk_per_view: Sequence[int],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    """Evaluate predeclared K values on valid response tokens only."""

    response_mask = response_mask.bool()
    valid_disagreement = js_divergence_from_logits(
        teacher_plus_logits, teacher_minus_logits
    )[response_mask]
    if not valid_disagreement.numel():
        raise RuntimeError("no valid response tokens for compression gate")
    tau_disagreement = float(valid_disagreement.median().item())
    objective_kwargs = {
        "attack_gate": torch.ones(teacher_minus_logits.shape[0]),
        "reliability_weight": torch.ones(teacher_minus_logits.shape[:2]),
        "tau_disagreement": tau_disagreement,
        "delta_min": float(objective_config["delta_min"]),
        "delta_max": float(objective_config["delta_max"]),
        "eta": float(objective_config["eta"]),
        "lambda_forward": float(objective_config["lambda_forward"]),
    }

    exact_student = teacher_minus_logits.clone().requires_grad_(True)
    exact = compute_full_vocab_objective(
        exact_student,
        teacher_minus_logits,
        teacher_plus_logits,
        response_mask=response_mask,
        **objective_kwargs,
    )
    exact_gradient = torch.autograd.grad(exact.loss, exact_student)[0]
    exact_corrective_count = int((exact.corrective_mask & response_mask).sum())
    if exact_corrective_count == 0:
        raise RuntimeError("exact objective selected no valid corrective tokens")

    rows = []
    for topk in topk_per_view:
        approximate_student = teacher_minus_logits.clone().requires_grad_(True)
        approximate = compute_uniform_tail_topk_objective(
            approximate_student,
            teacher_minus_logits,
            teacher_plus_logits,
            response_mask=response_mask,
            topk=int(topk),
            **objective_kwargs,
        )
        approximate_gradient = torch.autograd.grad(
            approximate.objective.loss, approximate_student
        )[0]
        metrics = compare_objectives(
            exact,
            approximate.objective,
            exact_gradient=exact_gradient,
            approximate_gradient=approximate_gradient,
            valid_mask=response_mask,
        )
        support_counts = approximate.bundles.student.support_mask.sum(dim=-1)
        tail_masses = torch.stack(
            [
                approximate.bundles.student.tail_log_mass.exp(),
                approximate.bundles.teacher_minus.tail_log_mass.exp(),
                approximate.bundles.teacher_plus.tail_log_mass.exp(),
            ]
        )
        valid_tail_masses = tail_masses[:, response_mask]
        passed = all(
            [
                metrics.alpha_sign_agreement
                >= thresholds["alpha_sign_agreement_min"],
                metrics.corrective_mask_jaccard
                >= thresholds["corrective_mask_jaccard_min"],
                metrics.gradient_cosine >= thresholds["gradient_cosine_min"],
                metrics.hybrid_kl_mean_relative_error
                <= thresholds["hybrid_kl_mean_relative_error_max"],
            ]
        )
        rows.append(
            {
                "topk_per_view": int(topk),
                "mean_union_support": float(
                    support_counts[response_mask].float().mean().item()
                ),
                "mean_tail_mass": float(valid_tail_masses.mean().item()),
                "max_tail_mass": float(valid_tail_masses.max().item()),
                **asdict(metrics),
                "passed": passed,
            }
        )
        del approximate, approximate_student, approximate_gradient
        gc.collect()

    accepted_topks = [row["topk_per_view"] for row in rows if row["passed"]]
    selected_topk = min(accepted_topks) if accepted_topks else None
    return {
        "valid_response_token_count": int(response_mask.sum().item()),
        "tau_disagreement": tau_disagreement,
        "exact_corrective_count": exact_corrective_count,
        "thresholds": dict(thresholds),
        "compression_results": rows,
        "selected_topk": selected_topk,
        "compression_gate_passed": selected_topk is not None,
        "decision": (
            "advance_to_stage2d_design"
            if selected_topk is not None
            else "stop_uniform_tail_compressed_path"
        ),
    }
