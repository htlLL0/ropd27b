"""Versioned, JSON-friendly Stage 1 data and evaluator contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class PairedInjectionSample:
    """A clean/attacked pair that shares one original user task."""

    sample_id: str
    base_task_id: str
    attack_id: str
    user_task: str
    history: list[Mapping[str, Any]]
    clean_observation: Any
    injected_content: Any
    clean_input: Any
    attacked_input: Any
    task_verifier_spec: Mapping[str, Any]
    attack_verifier_spec: Mapping[str, Any]
    split: str
    source: str
    schema_version: str = "r-opcd-paired-v1"

    def __post_init__(self) -> None:
        required_text = {
            "sample_id": self.sample_id,
            "base_task_id": self.base_task_id,
            "attack_id": self.attack_id,
            "user_task": self.user_task,
            "split": self.split,
            "source": self.source,
        }
        empty = [name for name, value in required_text.items() if not value.strip()]
        if empty:
            raise ValueError(f"empty required fields: {empty}")
        if self.split not in {"train", "dev", "test"}:
            raise ValueError(f"unsupported split: {self.split}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PairedInjectionSample":
        return cls(**dict(value))


@dataclass(frozen=True)
class VerifierResult:
    """One independent verifier result; errors are never coerced to booleans."""

    value: bool | None
    reason: str
    verifier_version: str
    raw_evidence: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        if self.error is None and self.value is None:
            raise ValueError("a successful verifier result must provide a boolean value")
        if self.error is not None and self.value is not None:
            raise ValueError("an errored verifier result must not provide a boolean value")


@dataclass(frozen=True)
class EvaluationResult:
    """Joint A/U evaluation without collapsing the two outcomes."""

    attack: VerifierResult
    task: VerifierResult

    @property
    def is_valid(self) -> bool:
        return self.attack.error is None and self.task.error is None

    @property
    def attack_succeeded(self) -> bool:
        if not self.is_valid:
            raise RuntimeError("cannot read attack outcome from an invalid evaluation")
        assert self.attack.value is not None
        return self.attack.value

    @property
    def user_task_succeeded(self) -> bool:
        if not self.is_valid:
            raise RuntimeError("cannot read task outcome from an invalid evaluation")
        assert self.task.value is not None
        return self.task.value

    @property
    def recovered(self) -> bool:
        return (not self.attack_succeeded) and self.user_task_succeeded
