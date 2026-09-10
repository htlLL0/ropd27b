"""Frozen-T+ trajectory reliability scores for R-OPCD.

The reliability path is deliberately parameter free.  It consumes the same
T+ teacher-forcing logits already needed by the corrective objective and never
adds a model forward pass or a trainable checkpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import Tensor
import torch.nn.functional as F


CALIBRATION_SCHEMA = "r-opcd-frozen-tplus-reliability-calibration-v1"


@dataclass(frozen=True)
class ReliabilityCalibration:
    """Frozen scalar calibration for the trajectory-NLL reliability score."""

    mean_nll: float
    std_nll: float
    window_size: int = 32
    tau_r: float = 0.0
    temperature_r: float = 1.0
    epsilon: float = 1.0e-6
    sample_count: int | None = None

    def __post_init__(self) -> None:
        values = torch.tensor(
            [
                self.mean_nll,
                self.std_nll,
                self.tau_r,
                self.temperature_r,
                self.epsilon,
            ],
            dtype=torch.float64,
        )
        if not torch.isfinite(values).all():
            raise ValueError("reliability calibration values must be finite")
        if self.mean_nll < 0.0:
            raise ValueError("mean_nll must be non-negative")
        if self.std_nll < 0.0:
            raise ValueError("std_nll must be non-negative")
        if self.window_size < 1:
            raise ValueError("window_size must be positive")
        if self.temperature_r <= 0.0:
            raise ValueError("temperature_r must be positive")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if self.sample_count is not None and self.sample_count < 1:
            raise ValueError("sample_count must be positive when provided")

    def to_dict(self) -> dict[str, Any]:
        return {"schema": CALIBRATION_SCHEMA, **asdict(self)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReliabilityCalibration":
        if payload.get("schema") != CALIBRATION_SCHEMA:
            raise ValueError("reliability calibration schema mismatch")
        return cls(
            mean_nll=float(payload["mean_nll"]),
            std_nll=float(payload["std_nll"]),
            window_size=int(payload.get("window_size", 32)),
            tau_r=float(payload.get("tau_r", 0.0)),
            temperature_r=float(payload.get("temperature_r", 1.0)),
            epsilon=float(payload.get("epsilon", 1.0e-6)),
            sample_count=(
                None
                if payload.get("sample_count") is None
                else int(payload["sample_count"])
            ),
        )


@dataclass(frozen=True)
class ReliabilityScoreOutput:
    """Inspectable intermediates for the frozen reliability calculation."""

    token_nll: Tensor
    window_mean_nll: Tensor
    history_count: Tensor
    standardized_nll: Tensor
    weight: Tensor


def student_token_nll_from_teacher_plus(
    teacher_plus_logits: Tensor,
    response_token_ids: Tensor,
    *,
    response_mask: Tensor | None = None,
) -> Tensor:
    """Return each actual Student token's NLL under frozen T+ logits."""

    if teacher_plus_logits.ndim != 3:
        raise ValueError("teacher_plus_logits must have shape [batch, time, vocab]")
    batch_size, time_steps, vocab_size = teacher_plus_logits.shape
    token_ids = torch.as_tensor(response_token_ids, device=teacher_plus_logits.device)
    if token_ids.shape != (batch_size, time_steps):
        raise ValueError("response_token_ids must match the logits batch/time shape")
    if token_ids.dtype == torch.bool or torch.is_floating_point(token_ids):
        raise TypeError("response_token_ids must contain integer token IDs")
    token_ids = token_ids.to(dtype=torch.long)
    if torch.any((token_ids < 0) | (token_ids >= vocab_size)):
        raise ValueError("response_token_ids contain an out-of-vocabulary ID")

    mask = _validated_mask(
        response_mask,
        shape=(batch_size, time_steps),
        device=teacher_plus_logits.device,
    )
    with torch.no_grad():
        log_probabilities = F.log_softmax(
            teacher_plus_logits.detach().float(), dim=-1
        )
        token_nll = -torch.gather(
            log_probabilities, dim=-1, index=token_ids.unsqueeze(-1)
        ).squeeze(-1)
        token_nll = torch.where(mask, token_nll, torch.zeros_like(token_nll))
    return token_nll.detach()


def previous_window_mean_nll(
    token_nll: Tensor,
    *,
    response_mask: Tensor | None = None,
    window_size: int = 32,
) -> tuple[Tensor, Tensor]:
    """Average NLL over the previous ``window_size`` response positions.

    Position ``t`` uses only Student tokens strictly before ``t``.  Consequently
    the first valid response position has no history and a count of zero.
    """

    if token_nll.ndim != 2:
        raise ValueError("token_nll must have shape [batch, time]")
    if not torch.is_floating_point(token_nll):
        raise TypeError("token_nll must be floating point")
    if not torch.isfinite(token_nll).all():
        raise ValueError("token_nll must be finite")
    if window_size < 1:
        raise ValueError("window_size must be positive")

    batch_size, time_steps = token_nll.shape
    mask = _validated_mask(
        response_mask, shape=(batch_size, time_steps), device=token_nll.device
    )
    values = token_nll.detach().float() * mask.to(dtype=torch.float32)
    counts = mask.to(dtype=torch.float32)
    value_prefix = F.pad(values.cumsum(dim=-1), (1, 0))
    count_prefix = F.pad(counts.cumsum(dim=-1), (1, 0))
    end = torch.arange(time_steps, device=token_nll.device)
    start = (end - window_size).clamp_min(0)
    window_sum = value_prefix[:, end] - value_prefix[:, start]
    history_count = count_prefix[:, end] - count_prefix[:, start]
    window_mean = window_sum / history_count.clamp_min(1.0)
    window_mean = torch.where(
        mask & (history_count > 0), window_mean, torch.zeros_like(window_mean)
    )
    return window_mean.detach(), history_count.detach()


def reliability_from_teacher_plus_logits(
    teacher_plus_logits: Tensor,
    response_token_ids: Tensor,
    calibration: ReliabilityCalibration,
    *,
    response_mask: Tensor | None = None,
) -> ReliabilityScoreOutput:
    """Map frozen-T+ sliding-window NLL to detached continuous weights.

    The first valid token has no preceding Student token.  We assign it the
    calibrated neutral value ``z=0`` (and therefore ``c=0.5`` under the default
    ``tau_r=0``), while excluding it from calibration statistics.
    """

    token_nll = student_token_nll_from_teacher_plus(
        teacher_plus_logits,
        response_token_ids,
        response_mask=response_mask,
    )
    return reliability_from_token_nll(
        token_nll,
        calibration,
        response_mask=response_mask,
    )


def reliability_from_token_nll(
    token_nll: Tensor,
    calibration: ReliabilityCalibration,
    *,
    response_mask: Tensor | None = None,
) -> ReliabilityScoreOutput:
    """Map precomputed T+ token NLLs to reliability without another softmax."""

    if token_nll.ndim != 2:
        raise ValueError("token_nll must have shape [batch, time]")
    window_mean, history_count = previous_window_mean_nll(
        token_nll,
        response_mask=response_mask,
        window_size=calibration.window_size,
    )
    mask = _validated_mask(
        response_mask,
        shape=tuple(token_nll.shape),
        device=token_nll.device,
    )
    has_history = mask & (history_count > 0)
    with torch.no_grad():
        standardized = (window_mean - calibration.mean_nll) / (
            calibration.std_nll + calibration.epsilon
        )
        standardized = torch.where(
            has_history, standardized, torch.zeros_like(standardized)
        )
        weight = torch.sigmoid(
            -(standardized - calibration.tau_r) / calibration.temperature_r
        )
        weight = torch.where(mask, weight, torch.zeros_like(weight))
    return ReliabilityScoreOutput(
        token_nll=token_nll.detach(),
        window_mean_nll=window_mean.detach(),
        history_count=history_count.detach(),
        standardized_nll=standardized.detach(),
        weight=weight.detach(),
    )


class ReliabilityCalibrationAccumulator:
    """Streaming population moments over valid sliding-window NLL values."""

    def __init__(self, *, window_size: int = 32) -> None:
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.trajectory_count = 0

    def update(
        self,
        teacher_plus_logits: Tensor,
        response_token_ids: Tensor,
        *,
        response_mask: Tensor | None = None,
    ) -> None:
        token_nll = student_token_nll_from_teacher_plus(
            teacher_plus_logits,
            response_token_ids,
            response_mask=response_mask,
        )
        window_mean, history_count = previous_window_mean_nll(
            token_nll,
            response_mask=response_mask,
            window_size=self.window_size,
        )
        mask = _validated_mask(
            response_mask,
            shape=tuple(token_nll.shape),
            device=teacher_plus_logits.device,
        )
        values = window_mean[mask & (history_count > 0)].double().cpu()
        self.trajectory_count += int(mask.any(dim=-1).sum().item())
        if values.numel() == 0:
            return
        batch_count = int(values.numel())
        batch_mean = float(values.mean().item())
        batch_m2 = float(((values - batch_mean) ** 2).sum().item())
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        self.m2 += batch_m2 + delta * delta * self.count * batch_count / total
        self.count = total

    def finalize(
        self,
        *,
        tau_r: float = 0.0,
        temperature_r: float = 1.0,
        epsilon: float = 1.0e-6,
    ) -> ReliabilityCalibration:
        if self.count < 1:
            raise ValueError("calibration contains no position with prior-token history")
        population_variance = max(0.0, self.m2 / self.count)
        return ReliabilityCalibration(
            mean_nll=self.mean,
            std_nll=population_variance**0.5,
            window_size=self.window_size,
            tau_r=tau_r,
            temperature_r=temperature_r,
            epsilon=epsilon,
            sample_count=self.count,
        )


def _validated_mask(
    response_mask: Tensor | None,
    *,
    shape: tuple[int, int],
    device: torch.device,
) -> Tensor:
    if response_mask is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    mask = torch.as_tensor(response_mask, device=device)
    if mask.shape != shape:
        raise ValueError("response_mask must match the logits batch/time shape")
    if torch.is_floating_point(mask) and not torch.isfinite(mask).all():
        raise ValueError("response_mask must be finite")
    if torch.any((mask != 0) & (mask != 1)):
        raise ValueError("response_mask must be binary")
    return mask.bool()
