"""R-OPCD V3 reference implementation."""

from r_opcd.contracts import EvaluationResult, PairedInjectionSample, VerifierResult
from r_opcd.distributions import DistributionBundle, TriViewBundles
from r_opcd.model_adapter import ParameterIntegrity, ResponseLogits
from r_opcd.objective import RopcdObjectiveOutput, compute_full_vocab_objective
from r_opcd.prompt_views import PromptViewPair, TeacherForcingBatch
from r_opcd.tokenizer_adapter import AlignedTeacherBatches, TokenizerIdentity

__all__ = [
    "AlignedTeacherBatches",
    "DistributionBundle",
    "EvaluationResult",
    "PairedInjectionSample",
    "ParameterIntegrity",
    "PromptViewPair",
    "RopcdObjectiveOutput",
    "ResponseLogits",
    "TeacherForcingBatch",
    "TokenizerIdentity",
    "TriViewBundles",
    "VerifierResult",
    "compute_full_vocab_objective",
]
