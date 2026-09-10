"""Full-vocabulary V3 geometry with an optional teacher-selected local loss.

This module is intentionally independent of VeRL and distributed workers. It is
the numerical authority that later compressed/distributed implementations must
match on small batches.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass(frozen=True)
class RopcdObjectiveOutput:
    loss: Tensor
    numerator: Tensor
    direction: Tensor
    disagreement: Tensor
    relevance_mask: Tensor
    alpha: Tensor
    deficit: Tensor
    corrective_mask: Tensor
    target_probabilities: Tensor
    forward_kl: Tensor
    reverse_kl: Tensor
    hybrid_kl: Tensor
    denominator: Tensor
    selected_mass: Tensor
    loss_support_indices: Tensor | None = None
    support_mass_student: Tensor | None = None
    support_mass_target: Tensor | None = None
    support_mass_teacher_plus: Tensor | None = None


def centered_logits(logits: Tensor) -> Tensor:
    """Remove the vocabulary-wise additive degree of freedom."""

    _check_logits(logits)
    return logits - logits.mean(dim=-1, keepdim=True)


def js_divergence_from_logits(logits_a: Tensor, logits_b: Tensor) -> Tensor:
    """Token-wise Jensen-Shannon divergence over the full vocabulary."""

    _check_same_shape(logits_a, logits_b)
    log_a = F.log_softmax(logits_a.float(), dim=-1)
    log_b = F.log_softmax(logits_b.float(), dim=-1)
    prob_a = log_a.exp()
    prob_b = log_b.exp()
    mixture = 0.5 * (prob_a + prob_b)
    log_mixture = mixture.clamp_min(torch.finfo(mixture.dtype).tiny).log()
    return 0.5 * (
        (prob_a * (log_a - log_mixture)).sum(dim=-1)
        + (prob_b * (log_b - log_mixture)).sum(dim=-1)
    )


def compute_full_vocab_objective(
    student_logits: Tensor,
    teacher_minus_logits: Tensor,
    teacher_plus_logits: Tensor,
    *,
    attack_gate: Tensor,
    reliability_weight: Tensor,
    response_mask: Tensor | None = None,
    tau_disagreement: float = 0.0,
    delta_min: float = 0.0,
    delta_max: float = 1.0,
    eta: float = 1.0,
    lambda_forward: float = 0.5,
    eps_alpha: float = 1.0e-8,
    eps_loss: float = 1.0e-8,
    teacher_plus_top_k: int | None = None,
) -> RopcdObjectiveOutput:
    """Compute the exact V3 corrective objective on `[batch, time, vocab]` logits.

    Target construction is detached in its entirety. ``reliability_weight`` is
    the frozen-T+ trajectory score ``c_t``.  It scales only the loss numerator;
    excluding it from both the target and denominator preserves the intended
    target geometry and reduces correction strength when T+ assigns low support
    to the realized Student trajectory.

    When ``teacher_plus_top_k`` is set, only the final Hybrid-KL comparison
    uses T+'s top-k support. Student and the ORIGINAL detached corrected target
    are separately renormalized there. Direction, JS, alpha, deficit, masks,
    reliability, and the full target remain full-vocabulary quantities.
    """

    _check_same_shape(student_logits, teacher_minus_logits, teacher_plus_logits)
    if student_logits.ndim != 3:
        raise ValueError("expected logits with shape [batch, time, vocab]")
    if not 0.0 <= lambda_forward <= 1.0:
        raise ValueError("lambda_forward must be in [0, 1]")
    if tau_disagreement < 0.0:
        raise ValueError("tau_disagreement must be non-negative")
    if delta_max < 0.0 or delta_min < 0.0:
        raise ValueError("delta thresholds must be non-negative")
    if teacher_plus_top_k is not None and (
        type(teacher_plus_top_k) is not int
        or not 1 <= teacher_plus_top_k <= student_logits.shape[-1]
    ):
        raise ValueError("teacher_plus_top_k must be an integer inside the vocabulary")

    batch_size, time_steps, _ = student_logits.shape
    device = student_logits.device
    work_dtype = torch.float32

    teacher_minus = teacher_minus_logits.detach().to(dtype=work_dtype)
    teacher_plus = teacher_plus_logits.detach().to(dtype=work_dtype)
    student = student_logits.to(dtype=work_dtype)

    r_student = centered_logits(student)
    r_minus = centered_logits(teacher_minus)
    r_plus = centered_logits(teacher_plus)
    direction = (r_plus - r_minus).detach()

    disagreement = js_divergence_from_logits(teacher_plus, teacher_minus).detach()
    relevance_mask = disagreement > tau_disagreement

    with torch.no_grad():
        p_student_target = F.softmax(student.detach(), dim=-1)
        p_minus = F.softmax(teacher_minus, dim=-1)
        p_plus = F.softmax(teacher_plus, dim=-1)
        geometry_weight = ((p_student_target + p_minus + p_plus) / 3.0).detach()

        student_offset = r_student.detach() - r_minus
        numerator = (geometry_weight * student_offset * direction).sum(dim=-1)
        squared_norm = (geometry_weight * direction.square()).sum(dim=-1)
        alpha = numerator / (squared_norm + eps_alpha)
        deficit = torch.clamp(F.relu(1.0 - alpha), min=0.0, max=delta_max)
        positive_deficit = deficit > delta_min
        corrective_mask = relevance_mask & positive_deficit

        target_logits = r_student.detach() + eta * deficit.unsqueeze(-1) * direction
        target_log_probabilities = F.log_softmax(target_logits, dim=-1).detach()
        target_probabilities = target_log_probabilities.exp().detach()

    support_indices = None
    mass_student = mass_target = mass_plus = None
    loss_target_log_probabilities = target_log_probabilities
    loss_target_probabilities = target_probabilities
    if teacher_plus_top_k is None:
        student_log_probabilities = F.log_softmax(r_student, dim=-1)
    else:
        # The set is selected ONLY by frozen T+, never by Student or q_t.
        support_indices = torch.topk(
            teacher_plus, k=teacher_plus_top_k, dim=-1, sorted=True
        ).indices.detach()
        # Raw Student logits avoid a spurious full-vocabulary autograd path
        # through centering; local softmax is invariant to that common shift.
        student_log_probabilities = F.log_softmax(
            student.gather(-1, support_indices), dim=-1
        )
        loss_target_log_probabilities = F.log_softmax(
            target_logits.gather(-1, support_indices), dim=-1
        ).detach()
        loss_target_probabilities = loss_target_log_probabilities.exp()
        with torch.no_grad():
            mass_student = p_student_target.gather(-1, support_indices).sum(-1)
            mass_target = target_probabilities.gather(-1, support_indices).sum(-1)
            mass_plus = p_plus.gather(-1, support_indices).sum(-1)
    student_probabilities = student_log_probabilities.exp()
    forward_kl = (
        loss_target_probabilities * (loss_target_log_probabilities - student_log_probabilities)
    ).sum(dim=-1)
    reverse_kl = (
        student_probabilities * (student_log_probabilities - loss_target_log_probabilities)
    ).sum(dim=-1)
    hybrid_kl = lambda_forward * forward_kl + (1.0 - lambda_forward) * reverse_kl

    gate = _expand_token_tensor(
        attack_gate,
        name="attack_gate",
        batch_size=batch_size,
        time_steps=time_steps,
        device=device,
        dtype=work_dtype,
    )
    reliability = _expand_token_tensor(
        reliability_weight,
        name="reliability_weight",
        batch_size=batch_size,
        time_steps=time_steps,
        device=device,
        dtype=work_dtype,
    ).detach()
    if torch.any((reliability < 0.0) | (reliability > 1.0)):
        raise ValueError("reliability_weight must be in [0, 1]")

    if response_mask is None:
        valid_response = torch.ones(
            (batch_size, time_steps), device=device, dtype=work_dtype
        )
    else:
        valid_response = _expand_token_tensor(
            response_mask,
            name="response_mask",
            batch_size=batch_size,
            time_steps=time_steps,
            device=device,
            dtype=work_dtype,
        )

    selected = gate * corrective_mask.to(work_dtype) * valid_response
    denominator = selected.sum() + eps_loss
    numerator = (selected * reliability * hybrid_kl).sum()
    loss = numerator / denominator

    return RopcdObjectiveOutput(
        loss=loss,
        numerator=numerator,
        direction=direction,
        disagreement=disagreement,
        relevance_mask=relevance_mask,
        alpha=alpha.detach(),
        deficit=deficit.detach(),
        corrective_mask=corrective_mask,
        target_probabilities=target_probabilities,
        forward_kl=forward_kl,
        reverse_kl=reverse_kl,
        hybrid_kl=hybrid_kl,
        denominator=denominator.detach(),
        selected_mass=selected.sum().detach(),
        loss_support_indices=support_indices,
        support_mass_student=mass_student,
        support_mass_target=mass_target,
        support_mass_teacher_plus=mass_plus,
    )


def _check_logits(logits: Tensor) -> None:
    if not torch.is_floating_point(logits):
        raise TypeError("logits must be floating point")
    if logits.shape[-1] < 2:
        raise ValueError("vocabulary dimension must contain at least two tokens")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")


def _check_same_shape(*logits: Tensor) -> None:
    if not logits:
        raise ValueError("at least one tensor is required")
    for tensor in logits:
        _check_logits(tensor)
    shapes = {tuple(tensor.shape) for tensor in logits}
    if len(shapes) != 1:
        raise ValueError(f"logit shape mismatch: {sorted(shapes)}")


def _expand_token_tensor(
    value: Tensor,
    *,
    name: str,
    batch_size: int,
    time_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    value = torch.as_tensor(value, device=device, dtype=dtype)
    if value.shape == (batch_size,):
        value = value[:, None].expand(batch_size, time_steps)
    elif value.shape != (batch_size, time_steps):
        raise ValueError(
            f"{name} must have shape [{batch_size}] or [{batch_size}, {time_steps}], "
            f"got {list(value.shape)}"
        )
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")
    return value
