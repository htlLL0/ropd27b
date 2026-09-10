"""Tri-view top-k distribution reference with explicit tail mass."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor
import torch.nn.functional as F

from r_opcd.objective import RopcdObjectiveOutput, compute_full_vocab_objective


@dataclass(frozen=True)
class DistributionBundle:
    view_name: str
    support_indices: Tensor
    support_log_probabilities: Tensor
    tail_log_mass: Tensor
    response_mask: Tensor
    vocab_size: int

    @property
    def support_mask(self) -> Tensor:
        return self.support_indices >= 0

    @property
    def total_probability_mass(self) -> Tensor:
        support_mass = torch.where(
            self.support_mask,
            self.support_log_probabilities.exp(),
            torch.zeros_like(self.support_log_probabilities),
        ).sum(dim=-1)
        return support_mass + self.tail_log_mass.exp()


@dataclass(frozen=True)
class TriViewBundles:
    student: DistributionBundle
    teacher_minus: DistributionBundle
    teacher_plus: DistributionBundle


@dataclass(frozen=True)
class TopKObjectiveOutput:
    objective: RopcdObjectiveOutput
    bundles: TriViewBundles
    reconstructed_student_log_probabilities: Tensor
    reconstructed_teacher_minus_log_probabilities: Tensor
    reconstructed_teacher_plus_log_probabilities: Tensor


def tri_view_union_topk_indices(
    student_logits: Tensor,
    teacher_minus_logits: Tensor,
    teacher_plus_logits: Tensor,
    *,
    topk: int,
) -> Tensor:
    """Return a deterministic sorted union of the three per-token top-k sets."""

    _validate_logit_triplet(student_logits, teacher_minus_logits, teacher_plus_logits)
    vocab_size = student_logits.shape[-1]
    if topk <= 0 or topk > vocab_size:
        raise ValueError(f"topk must be in [1, {vocab_size}]")

    own_indices = [
        torch.topk(logits.detach(), k=topk, dim=-1).indices
        for logits in (student_logits, teacher_minus_logits, teacher_plus_logits)
    ]
    combined = torch.cat(own_indices, dim=-1)
    max_support = min(3 * topk, vocab_size)
    output = torch.full(
        (*student_logits.shape[:-1], max_support),
        -1,
        dtype=torch.long,
        device=student_logits.device,
    )

    for batch_index in range(student_logits.shape[0]):
        for time_index in range(student_logits.shape[1]):
            unique = torch.unique(
                combined[batch_index, time_index], sorted=True
            )
            output[batch_index, time_index, : unique.numel()] = unique
    return output


def gather_distribution_bundle(
    logits: Tensor,
    support_indices: Tensor,
    *,
    response_mask: Tensor,
    view_name: str,
) -> DistributionBundle:
    """Gather exact support log-probabilities and exact complement log mass."""

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, time, vocab]")
    if support_indices.shape[:2] != logits.shape[:2] or support_indices.ndim != 3:
        raise ValueError("support indices must have shape [batch, time, support]")
    if response_mask.shape != logits.shape[:2]:
        raise ValueError("response_mask must match [batch, time]")
    vocab_size = logits.shape[-1]
    valid_support = support_indices >= 0
    if torch.any(support_indices[valid_support] >= vocab_size):
        raise ValueError("support index exceeds vocabulary size")

    log_probabilities = F.log_softmax(logits.float(), dim=-1)
    safe_indices = support_indices.clamp_min(0)
    gathered = torch.gather(log_probabilities, dim=-1, index=safe_indices)
    negative_infinity = torch.full_like(gathered, float("-inf"))
    support_log_probabilities = torch.where(
        valid_support, gathered, negative_infinity
    )

    dense_support_mask = torch.zeros_like(log_probabilities, dtype=torch.bool)
    flat_mask = dense_support_mask.reshape(-1, vocab_size)
    flat_indices = support_indices.reshape(-1, support_indices.shape[-1])
    row_indices = torch.arange(flat_indices.shape[0], device=logits.device)
    for slot in range(flat_indices.shape[-1]):
        slot_indices = flat_indices[:, slot]
        slot_valid = slot_indices >= 0
        flat_mask[row_indices[slot_valid], slot_indices[slot_valid]] = True

    tail_log_mass = torch.logsumexp(
        log_probabilities.masked_fill(dense_support_mask, float("-inf")),
        dim=-1,
    )
    bundle = DistributionBundle(
        view_name=view_name,
        support_indices=support_indices,
        support_log_probabilities=support_log_probabilities,
        tail_log_mass=tail_log_mass,
        response_mask=response_mask,
        vocab_size=vocab_size,
    )
    _validate_bundle(bundle)
    full_numeric_mass = log_probabilities.exp().sum(dim=-1)
    if not torch.allclose(
        bundle.total_probability_mass,
        full_numeric_mass,
        atol=2e-6,
        rtol=2e-6,
    ):
        raise ValueError("support and tail do not preserve full-softmax mass")
    return bundle


def build_tri_view_bundles(
    student_logits: Tensor,
    teacher_minus_logits: Tensor,
    teacher_plus_logits: Tensor,
    *,
    response_mask: Tensor,
    topk: int,
) -> TriViewBundles:
    """Build three bundles on one shared Student/T-/T+ support."""

    support_indices = tri_view_union_topk_indices(
        student_logits,
        teacher_minus_logits,
        teacher_plus_logits,
        topk=topk,
    )
    return TriViewBundles(
        student=gather_distribution_bundle(
            student_logits,
            support_indices,
            response_mask=response_mask,
            view_name="student",
        ),
        teacher_minus=gather_distribution_bundle(
            teacher_minus_logits.detach(),
            support_indices,
            response_mask=response_mask,
            view_name="teacher_minus",
        ),
        teacher_plus=gather_distribution_bundle(
            teacher_plus_logits.detach(),
            support_indices,
            response_mask=response_mask,
            view_name="teacher_plus",
        ),
    )


def reconstruct_uniform_tail_log_probabilities(bundle: DistributionBundle) -> Tensor:
    """Reconstruct a diagnostic dense distribution with uniform tail tokens.

    This is a comparison device, not the production compressed loss. It keeps
    selected probabilities exact and spreads the exact complement mass evenly
    over unselected vocabulary items.
    """

    _validate_bundle(bundle)
    valid_support = bundle.support_mask
    support_count = valid_support.sum(dim=-1)
    tail_count = bundle.vocab_size - support_count
    safe_tail_count = tail_count.clamp_min(1)
    tail_log_probability = bundle.tail_log_mass - safe_tail_count.log()
    tail_log_probability = torch.where(
        tail_count > 0,
        tail_log_probability,
        torch.full_like(tail_log_probability, float("-inf")),
    )
    dense = tail_log_probability.unsqueeze(-1).expand(
        *tail_log_probability.shape, bundle.vocab_size
    ).clone()

    flat_dense = dense.reshape(-1, bundle.vocab_size)
    flat_indices = bundle.support_indices.reshape(
        -1, bundle.support_indices.shape[-1]
    )
    flat_values = bundle.support_log_probabilities.reshape_as(flat_indices)
    row_indices = torch.arange(flat_indices.shape[0], device=dense.device)
    for slot in range(flat_indices.shape[-1]):
        slot_indices = flat_indices[:, slot]
        slot_valid = slot_indices >= 0
        flat_dense[row_indices[slot_valid], slot_indices[slot_valid]] = flat_values[
            slot_valid, slot
        ]

    mass = dense.exp().sum(dim=-1)
    if not torch.allclose(
        mass,
        bundle.total_probability_mass,
        atol=_mass_tolerance(bundle.vocab_size, mass.dtype),
        rtol=0.0,
    ):
        raise RuntimeError("reconstruction does not preserve bundle probability mass")
    return dense


def compute_uniform_tail_topk_objective(
    student_logits: Tensor,
    teacher_minus_logits: Tensor,
    teacher_plus_logits: Tensor,
    *,
    response_mask: Tensor,
    topk: int,
    **objective_kwargs,
) -> TopKObjectiveOutput:
    """Compute a dense diagnostic objective reconstructed from tri-view top-k."""

    bundles = build_tri_view_bundles(
        student_logits,
        teacher_minus_logits,
        teacher_plus_logits,
        response_mask=response_mask,
        topk=topk,
    )
    reconstructed_student = reconstruct_uniform_tail_log_probabilities(
        bundles.student
    )
    reconstructed_minus = reconstruct_uniform_tail_log_probabilities(
        bundles.teacher_minus
    )
    reconstructed_plus = reconstruct_uniform_tail_log_probabilities(
        bundles.teacher_plus
    )
    objective = compute_full_vocab_objective(
        reconstructed_student,
        reconstructed_minus,
        reconstructed_plus,
        response_mask=response_mask,
        **objective_kwargs,
    )
    return TopKObjectiveOutput(
        objective=objective,
        bundles=bundles,
        reconstructed_student_log_probabilities=reconstructed_student,
        reconstructed_teacher_minus_log_probabilities=reconstructed_minus,
        reconstructed_teacher_plus_log_probabilities=reconstructed_plus,
    )


def _validate_logit_triplet(*logits: Tensor) -> None:
    if len(logits) != 3:
        raise ValueError("expected Student, T-, and T+ logits")
    if any(tensor.ndim != 3 for tensor in logits):
        raise ValueError("all logits must have shape [batch, time, vocab]")
    shapes = {tuple(tensor.shape) for tensor in logits}
    devices = {tensor.device for tensor in logits}
    if len(shapes) != 1:
        raise ValueError(f"tri-view logit shape mismatch: {sorted(shapes)}")
    if len(devices) != 1:
        raise ValueError("tri-view logits must share one device")
    if any(not torch.isfinite(tensor).all() for tensor in logits):
        raise ValueError("tri-view logits must be finite")


def _validate_bundle(bundle: DistributionBundle) -> None:
    indices = bundle.support_indices
    log_probabilities = bundle.support_log_probabilities
    if indices.shape != log_probabilities.shape or indices.ndim != 3:
        raise ValueError("bundle support tensors must have matching rank-3 shapes")
    if bundle.tail_log_mass.shape != indices.shape[:2]:
        raise ValueError("tail_log_mass shape mismatch")
    if bundle.response_mask.shape != indices.shape[:2]:
        raise ValueError("response_mask shape mismatch")
    valid = indices >= 0
    if torch.any(indices[valid] >= bundle.vocab_size):
        raise ValueError("bundle contains an out-of-range token index")
    if not torch.isfinite(log_probabilities[valid]).all():
        raise ValueError("valid support log-probabilities must be finite")
    if torch.isnan(bundle.tail_log_mass).any():
        raise ValueError("tail log mass must not be NaN")
    mass_tolerance = _mass_tolerance(
        bundle.vocab_size, bundle.support_log_probabilities.dtype
    )
    if not torch.allclose(
        bundle.total_probability_mass,
        torch.ones_like(bundle.tail_log_mass),
        atol=mass_tolerance,
        rtol=0.0,
    ):
        raise ValueError("bundle probability mass does not sum to one")


def _mass_tolerance(vocab_size: int, dtype: torch.dtype) -> float:
    """Bound accumulated float error without hiding support/tail partition loss."""

    epsilon = torch.finfo(dtype).eps
    return max(2.0e-6, 8.0 * math.sqrt(vocab_size) * epsilon)
