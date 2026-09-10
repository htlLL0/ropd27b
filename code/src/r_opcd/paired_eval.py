"""Paired attack-label comparison utilities for T- and T+ views."""

from __future__ import annotations

from collections import Counter
import math
import random
from typing import Any, Mapping, Sequence


def transition_label(tminus_success: bool, tplus_success: bool) -> str:
    if tminus_success and tplus_success:
        return "both_attack_success"
    if tminus_success and not tplus_success:
        return "attack_suppression_flip"
    if not tminus_success and tplus_success:
        return "attack_regression_flip"
    return "both_attack_fail"


def summarize_paired(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("paired records must not be empty")
    counts = Counter(str(row["transition"]) for row in records)
    n = len(records)
    tminus_successes = sum(bool(row["tminus_attack_success"]) for row in records)
    tplus_successes = sum(bool(row["tplus_attack_success"]) for row in records)
    suppression = counts["attack_suppression_flip"]
    regression = counts["attack_regression_flip"]
    return {
        "n": n,
        "tminus_attack_successes": tminus_successes,
        "tminus_asr": tminus_successes / n,
        "tplus_attack_successes": tplus_successes,
        "tplus_asr": tplus_successes / n,
        "asr_delta_tplus_minus_tminus": (tplus_successes - tminus_successes) / n,
        "transitions": {
            name: counts[name]
            for name in (
                "both_attack_success",
                "attack_suppression_flip",
                "attack_regression_flip",
                "both_attack_fail",
            )
        },
        "attack_suppression_rate_given_tminus_success": (
            suppression / tminus_successes if tminus_successes else None
        ),
        "net_attack_suppression_flips": suppression - regression,
        "mcnemar_exact_two_sided_p": mcnemar_exact_two_sided_p(
            suppression, regression
        ),
    }


def paired_bootstrap_delta(
    records: Sequence[Mapping[str, Any]], *, seed: int, samples: int
) -> dict[str, Any]:
    if not records:
        raise ValueError("paired records must not be empty")
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    differences = [
        int(bool(row["tplus_attack_success"]))
        - int(bool(row["tminus_attack_success"]))
        for row in records
    ]
    rng = random.Random(seed)
    n = len(differences)
    estimates = sorted(
        sum(differences[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(samples)
    )
    return {
        "seed": seed,
        "samples": samples,
        "confidence": 0.95,
        "lower": _quantile(estimates, 0.025),
        "upper": _quantile(estimates, 0.975),
    }


def mcnemar_exact_two_sided_p(suppression: int, regression: int) -> float:
    if suppression < 0 or regression < 0:
        raise ValueError("discordant counts must be nonnegative")
    discordant = suppression + regression
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k) for k in range(min(suppression, regression) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a quantile of an empty sequence")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)
