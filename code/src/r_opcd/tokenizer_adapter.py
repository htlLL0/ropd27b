"""Real chat-template adapter for aligned T-/T+ teacher forcing."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from torch import Tensor

from r_opcd.prompt_views import (
    PromptView,
    PromptViewPair,
    TeacherForcingBatch,
    assert_response_alignment,
    build_teacher_forcing_batch,
)
from r_opcd.attention_quarantine import tokenize_attention_quarantined_prompt
from r_opcd.attack_pilot import build_base_messages


class ChatTokenizer(Protocol):
    """Small protocol implemented by Hugging Face chat tokenizers."""

    name_or_path: str
    pad_token_id: int | None
    eos_token_id: int | None
    chat_template: str | None

    def __len__(self) -> int: ...

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> Any: ...

    def __call__(self, text: str, **kwargs: Any) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class TokenizerIdentity:
    name_or_path: str
    tokenizer_class: str
    vocab_size: int
    tokenizer_length: int
    pad_token_id: int
    eos_token_id: int | None
    chat_template_sha256: str


@dataclass(frozen=True)
class AlignedTeacherBatches:
    teacher_minus: TeacherForcingBatch
    teacher_plus: TeacherForcingBatch


@dataclass(frozen=True)
class AlignedRecoveryBatches:
    """Current Student, ordinary T-, and attention-quarantined T+ batches."""

    student: TeacherForcingBatch
    teacher_minus: TeacherForcingBatch
    teacher_plus: TeacherForcingBatch


def load_local_hf_tokenizer(snapshot_path: str | Path) -> ChatTokenizer:
    """Load a tokenizer from an existing directory without any Hub fallback."""

    path = Path(snapshot_path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"tokenizer snapshot does not exist: {path}")
    if not (path / "tokenizer_config.json").is_file():
        raise FileNotFoundError("tokenizer_config.json is missing from snapshot")

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "transformers is required; install the project stage2b extra"
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(
        str(path),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        raise ValueError("tokenizer does not define a chat template")
    return tokenizer


def describe_tokenizer(tokenizer: ChatTokenizer) -> TokenizerIdentity:
    """Return the tokenizer identity fields needed by a pilot manifest."""

    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer pad_token_id must be defined")
    if not tokenizer.chat_template:
        raise ValueError("tokenizer chat_template must be defined")
    tokenizer_length = int(len(tokenizer))
    raw_vocab_size = getattr(tokenizer, "vocab_size", None)
    vocab_size = tokenizer_length if raw_vocab_size is None else int(raw_vocab_size)
    return TokenizerIdentity(
        name_or_path=str(tokenizer.name_or_path),
        tokenizer_class=type(tokenizer).__name__,
        vocab_size=vocab_size,
        tokenizer_length=tokenizer_length,
        pad_token_id=int(tokenizer.pad_token_id),
        eos_token_id=(
            None if tokenizer.eos_token_id is None else int(tokenizer.eos_token_id)
        ),
        chat_template_sha256=sha256(
            tokenizer.chat_template.encode("utf-8")
        ).hexdigest(),
    )


def verify_snapshot_files(
    snapshot_path: str | Path, expected_hashes: Mapping[str, str]
) -> dict[str, str]:
    """Verify every declared tokenizer file and return its observed SHA-256."""

    root = Path(snapshot_path).expanduser().resolve()
    observed: dict[str, str] = {}
    for relative_name, expected_hash in sorted(expected_hashes.items()):
        path = root / relative_name
        if not path.is_file():
            raise FileNotFoundError(f"required tokenizer file is missing: {path}")
        digest = sha256(path.read_bytes()).hexdigest()
        if digest != expected_hash:
            raise ValueError(
                f"tokenizer file hash mismatch for {relative_name}: "
                f"expected {expected_hash}, observed {digest}"
            )
        observed[relative_name] = digest
    return observed


def tokenize_prompt_view(
    tokenizer: ChatTokenizer,
    view: PromptView,
    *,
    add_generation_prompt: bool = True,
    template_kwargs: Mapping[str, Any] | None = None,
) -> list[int]:
    """Apply one real chat template and return an unpadded prompt token list."""

    kwargs = dict(template_kwargs or {})
    tokenized = tokenizer.apply_chat_template(
        list(view.messages),
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )
    return _normalize_token_ids(tokenized, field_name=f"{view.name} prompt")


def render_prompt_view(
    tokenizer: ChatTokenizer,
    view: PromptView,
    *,
    add_generation_prompt: bool = True,
    template_kwargs: Mapping[str, Any] | None = None,
) -> str:
    """Render a view for template-equivalence checks without tokenization."""

    kwargs = dict(template_kwargs or {})
    rendered = tokenizer.apply_chat_template(
        list(view.messages),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError(f"{view.name} rendered prompt must be a non-empty string")
    return rendered


def tokenize_response_texts(
    tokenizer: ChatTokenizer, response_texts: Sequence[str]
) -> list[list[int]]:
    """Encode fixture responses once; production must pass rollout IDs instead."""

    response_ids: list[list[int]] = []
    for response in response_texts:
        if not response:
            raise ValueError("fixture response text must be non-empty")
        encoded = tokenizer(response, add_special_tokens=False)
        if "input_ids" not in encoded:
            raise ValueError("tokenizer response output lacks input_ids")
        response_ids.append(
            _normalize_token_ids(encoded["input_ids"], field_name="response")
        )
    return response_ids


def build_aligned_teacher_batches(
    tokenizer: ChatTokenizer,
    view_pairs: Sequence[PromptViewPair],
    response_token_ids: Sequence[Sequence[int]],
    *,
    add_generation_prompt: bool = True,
    template_kwargs: Mapping[str, Any] | None = None,
) -> AlignedTeacherBatches:
    """Build T-/T+ inputs while reusing one authoritative response tensor."""

    if len(view_pairs) != len(response_token_ids):
        raise ValueError("view-pair and response batch sizes must match")
    if not view_pairs:
        raise ValueError("aligned Teacher batch must be non-empty")
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer pad_token_id must be defined")

    minus_prompt_ids = [
        tokenize_prompt_view(
            tokenizer,
            pair.teacher_minus,
            add_generation_prompt=add_generation_prompt,
            template_kwargs=template_kwargs,
        )
        for pair in view_pairs
    ]
    plus_prompt_ids = [
        tokenize_prompt_view(
            tokenizer,
            pair.teacher_plus,
            add_generation_prompt=add_generation_prompt,
            template_kwargs=template_kwargs,
        )
        for pair in view_pairs
    ]
    teacher_minus = build_teacher_forcing_batch(
        minus_prompt_ids,
        response_token_ids,
        pad_token_id=int(tokenizer.pad_token_id),
        view_name="teacher_minus",
    )
    teacher_plus = build_teacher_forcing_batch(
        plus_prompt_ids,
        response_token_ids,
        pad_token_id=int(tokenizer.pad_token_id),
        view_name="teacher_plus",
    )
    assert_response_alignment(teacher_minus, teacher_plus)
    return AlignedTeacherBatches(
        teacher_minus=teacher_minus,
        teacher_plus=teacher_plus,
    )


def build_attention_quarantine_recovery_batches(
    tokenizer: ChatTokenizer,
    rows: Sequence[Mapping[str, Any]],
    response_token_ids: Sequence[Sequence[int]],
    *,
    add_generation_prompt: bool = True,
    template_kwargs: Mapping[str, Any] | None = None,
) -> AlignedRecoveryBatches:
    """Build three aligned views for the latest exact-span training oracle.

    Student and T- receive the identical ordinary attacked prompt.  T+ receives
    the Stage 2L authenticated serialization and its q1-body attention mask.
    All three score the exact same on-policy Student response token IDs.
    """

    if len(rows) != len(response_token_ids):
        raise ValueError("row and response batch sizes must match")
    if not rows:
        raise ValueError("recovery batch must be non-empty")
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer pad_token_id must be defined")

    kwargs = dict(template_kwargs or {})
    ordinary_prompt_ids = [
        _normalize_token_ids(
            tokenizer.apply_chat_template(
                build_base_messages(row),
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                **kwargs,
            ),
            field_name="ordinary attacked prompt",
        )
        for row in rows
    ]
    privileged_prompts = [
        tokenize_attention_quarantined_prompt(
            tokenizer,
            row,
            add_generation_prompt=add_generation_prompt,
            template_kwargs=kwargs,
        )
        for row in rows
    ]
    privileged_prompt_ids = [list(prompt.input_ids) for prompt in privileged_prompts]
    privileged_prompt_masks = [
        list(prompt.attention_mask) for prompt in privileged_prompts
    ]

    batch_kwargs = {
        "response_token_ids": response_token_ids,
        "pad_token_id": int(tokenizer.pad_token_id),
    }
    student = build_teacher_forcing_batch(
        ordinary_prompt_ids,
        view_name="student",
        **batch_kwargs,
    )
    teacher_minus = build_teacher_forcing_batch(
        ordinary_prompt_ids,
        view_name="teacher_minus",
        **batch_kwargs,
    )
    teacher_plus = build_teacher_forcing_batch(
        privileged_prompt_ids,
        view_name="teacher_plus",
        prompt_attention_masks=privileged_prompt_masks,
        **batch_kwargs,
    )
    assert_response_alignment(student, teacher_minus)
    assert_response_alignment(teacher_minus, teacher_plus)
    if not student.prompt_fingerprint == teacher_minus.prompt_fingerprint:
        raise RuntimeError("Student and T- ordinary prompt IDs diverged")
    return AlignedRecoveryBatches(
        student=student,
        teacher_minus=teacher_minus,
        teacher_plus=teacher_plus,
    )


def _normalize_token_ids(value: Any, *, field_name: str) -> list[int]:
    if isinstance(value, Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValueError(f"{field_name} mapping lacks input_ids")
        value = value["input_ids"]
    if (
        isinstance(value, Sequence)
        and len(value) == 1
        and isinstance(value[0], Sequence)
        and not isinstance(value[0], (str, bytes))
    ):
        value = value[0]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} token IDs must be a sequence")
    ids = [int(token_id) for token_id in value]
    if not ids:
        raise ValueError(f"{field_name} token IDs must be non-empty")
    if any(token_id < 0 for token_id in ids):
        raise ValueError(f"{field_name} token IDs must be non-negative")
    return ids
