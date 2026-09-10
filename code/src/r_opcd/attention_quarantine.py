"""Reusable Stage 2L exact-span attention-quarantine tokenization.

The serialized prompt retains the malicious span byte for byte after XML
escaping.  Only token positions overlapping the ``q1`` body receive a zero in
the privileged Teacher's prompt attention mask.  Keeping this logic outside a
generation runner lets teacher-forcing and generation share one fail-closed
implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from html import escape as xml_escape
from typing import Any, Mapping, Protocol, Sequence

from r_opcd.attack_pilot import build_attention_quarantine_messages


class OffsetChatTokenizer(Protocol):
    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> Any: ...

    def __call__(self, text: str, **kwargs: Any) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class AttentionQuarantinedPrompt:
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    quarantined_token_indices: tuple[int, ...]
    rendered_sha256: str
    attention_mask_sha256: str

    @property
    def prompt_tokens(self) -> int:
        return len(self.input_ids)

    @property
    def quarantined_tokens(self) -> int:
        return len(self.quarantined_token_indices)


def tokenize_attention_quarantined_prompt(
    tokenizer: OffsetChatTokenizer,
    row: Mapping[str, Any],
    *,
    add_generation_prompt: bool = True,
    template_kwargs: Mapping[str, Any] | None = None,
) -> AttentionQuarantinedPrompt:
    """Tokenize one privileged prompt and mask exactly the rendered q1 body."""

    kwargs = dict(template_kwargs or {})
    messages = build_attention_quarantine_messages(row)
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("attention-quarantine prompt must render to non-empty text")

    opening_start = rendered.index('<quarantined_span id="q1"')
    content_start = rendered.index(">\n", opening_start) + 2
    content_end = rendered.index("\n  </quarantined_span>", content_start)
    expected_attack = xml_escape(str(row["malicious_span"]), quote=True)
    if rendered[content_start:content_end] != expected_attack:
        raise ValueError("q1 rendered content no longer matches malicious_span")

    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    if "input_ids" not in encoded or "offset_mapping" not in encoded:
        raise ValueError("fast tokenizer must return input_ids and offset_mapping")
    input_ids = _flat_int_sequence(encoded["input_ids"], name="input_ids")
    offsets = _flat_offsets(encoded["offset_mapping"])
    if len(offsets) != len(input_ids):
        raise ValueError("offset mapping length does not match input_ids")

    raw_mask = encoded.get("attention_mask", [1] * len(input_ids))
    attention_mask = _flat_int_sequence(raw_mask, name="attention_mask")
    if len(attention_mask) != len(input_ids):
        raise ValueError("attention mask length does not match input_ids")
    if any(value != 1 for value in attention_mask):
        raise ValueError("tokenizer emitted a nontrivial unpadded prompt mask")

    quarantined = tuple(
        index
        for index, (left, right) in enumerate(offsets)
        if right > content_start and left < content_end
    )
    if not quarantined:
        raise ValueError("attention quarantine masked no q1 tokens")
    for index in quarantined:
        attention_mask[index] = 0
    if not any(attention_mask):
        raise ValueError("attention quarantine removed the entire prompt")

    frozen_mask = tuple(attention_mask)
    return AttentionQuarantinedPrompt(
        input_ids=tuple(input_ids),
        attention_mask=frozen_mask,
        quarantined_token_indices=quarantined,
        rendered_sha256=sha256(rendered.encode("utf-8")).hexdigest(),
        attention_mask_sha256=sha256(bytes(frozen_mask)).hexdigest(),
    )


def _flat_int_sequence(value: Any, *, name: str) -> list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if (
        isinstance(value, Sequence)
        and len(value) == 1
        and isinstance(value[0], Sequence)
        and not isinstance(value[0], (str, bytes))
    ):
        value = value[0]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    return [int(item) for item in value]


def _flat_offsets(value: Any) -> list[tuple[int, int]]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if (
        isinstance(value, Sequence)
        and len(value) == 1
        and isinstance(value[0], Sequence)
        and value[0]
        and isinstance(value[0][0], Sequence)
    ):
        value = value[0]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("offset_mapping must be a sequence")
    offsets: list[tuple[int, int]] = []
    for item in value:
        if not isinstance(item, Sequence) or len(item) != 2:
            raise ValueError("each offset must contain exactly two integers")
        left, right = int(item[0]), int(item[1])
        if left < 0 or right < left:
            raise ValueError("token offsets must be ordered and non-negative")
        offsets.append((left, right))
    return offsets
