"""Independent attack and task verifier composition."""

from __future__ import annotations

from typing import Any, Protocol

from r_opcd.contracts import EvaluationResult, PairedInjectionSample, VerifierResult


class AttackVerifier(Protocol):
    def verify(self, sample: PairedInjectionSample, trajectory: Any) -> VerifierResult:
        """Return whether the attacker's goal succeeded."""


class TaskVerifier(Protocol):
    def verify(self, sample: PairedInjectionSample, trajectory: Any) -> VerifierResult:
        """Return whether the original user task succeeded."""


class CompositeEvaluator:
    """Evaluate A and U independently and preserve both raw results."""

    def __init__(self, attack_verifier: AttackVerifier, task_verifier: TaskVerifier):
        self._attack_verifier = attack_verifier
        self._task_verifier = task_verifier

    def evaluate(self, sample: PairedInjectionSample, trajectory: Any) -> EvaluationResult:
        attack = self._attack_verifier.verify(sample, trajectory)
        task = self._task_verifier.verify(sample, trajectory)
        return EvaluationResult(attack=attack, task=task)
