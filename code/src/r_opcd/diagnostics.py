"""Exact-versus-compressed objective comparison metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from r_opcd.objective import RopcdObjectiveOutput


@dataclass(frozen=True)
class ApproximationMetrics:
    disagreement_max_abs_error: float
    alpha_sign_agreement: float
    corrective_mask_jaccard: float
    exact_corrective_count: int
    approximate_corrective_count: int
    hybrid_kl_mean_relative_error: float
    gradient_cosine: float


def compare_objectives(
    exact: RopcdObjectiveOutput,
    approximate: RopcdObjectiveOutput,
    *,
    exact_gradient: Tensor,
    approximate_gradient: Tensor,
    valid_mask: Tensor | None = None,
    eps: float = 1.0e-12,
) -> ApproximationMetrics:
    """Summarize the diagnostics required before accepting a compressed path."""

    if exact_gradient.shape != approximate_gradient.shape:
        raise ValueError("gradient shapes must match")
    if valid_mask is None:
        valid = torch.ones_like(exact.corrective_mask, dtype=torch.bool)
    else:
        if valid_mask.shape != exact.corrective_mask.shape:
            raise ValueError("valid_mask must match the token dimensions")
        valid = valid_mask.to(device=exact.corrective_mask.device, dtype=torch.bool)
    if not valid.any():
        raise ValueError("valid_mask must select at least one token")

    exact_mask = exact.corrective_mask & valid
    approximate_mask = approximate.corrective_mask & valid
    union = exact_mask | approximate_mask
    intersection = exact_mask & approximate_mask
    mask_jaccard = (
        1.0
        if not union.any()
        else float(intersection.sum().item() / union.sum().item())
    )

    exact_sign = torch.sign(exact.alpha)
    approximate_sign = torch.sign(approximate.alpha)
    sign_agreement = float(
        (exact_sign[valid] == approximate_sign[valid]).float().mean().item()
    )

    hybrid_scale = exact.hybrid_kl[valid].abs().mean().item() + eps
    hybrid_error = (
        exact.hybrid_kl.detach()[valid] - approximate.hybrid_kl.detach()[valid]
    ).abs().mean().item() / hybrid_scale

    gradient_valid = valid.to(device=exact_gradient.device)
    exact_flat = exact_gradient.detach().float()[gradient_valid].reshape(-1)
    approximate_flat = (
        approximate_gradient.detach().float()[gradient_valid].reshape(-1)
    )
    exact_norm = torch.linalg.vector_norm(exact_flat)
    approximate_norm = torch.linalg.vector_norm(approximate_flat)
    if exact_norm <= eps and approximate_norm <= eps:
        gradient_cosine = 1.0
    elif exact_norm <= eps or approximate_norm <= eps:
        gradient_cosine = 0.0
    else:
        raw_cosine = float(
            torch.dot(exact_flat, approximate_flat).item()
            / (exact_norm.item() * approximate_norm.item())
        )
        gradient_cosine = max(-1.0, min(1.0, raw_cosine))

    return ApproximationMetrics(
        disagreement_max_abs_error=float(
            (exact.disagreement[valid] - approximate.disagreement[valid])
            .abs()
            .max()
            .item()
        ),
        alpha_sign_agreement=sign_agreement,
        corrective_mask_jaccard=mask_jaccard,
        exact_corrective_count=int(exact_mask.sum().item()),
        approximate_corrective_count=int(approximate_mask.sum().item()),
        hybrid_kl_mean_relative_error=float(hybrid_error),
        gradient_cosine=gradient_cosine,
    )
