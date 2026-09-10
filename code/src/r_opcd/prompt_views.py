"""Tokenizer-independent prompt views and response-alignment contracts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Literal, Mapping, Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class PromptView:
    name: str
    messages: tuple[Mapping[str, Any], ...]
    fingerprint: str


@dataclass(frozen=True)
class PromptViewPair:
    teacher_minus: PromptView
    teacher_plus: PromptView


@dataclass(frozen=True)
class TeacherForcingBatch:
    """A padded prompt plus a shared, right-padded Student response."""

    view_name: str
    prompt_fingerprint: str
    input_ids: Tensor
    attention_mask: Tensor
    position_ids: Tensor
    responses: Tensor
    response_mask: Tensor
    prompt_width: int

    @property
    def response_logit_slice(self) -> slice:
        """Causal-logit slice that predicts the response token sequence."""

        response_width = self.responses.shape[1]
        return slice(self.prompt_width - 1, self.prompt_width - 1 + response_width)


def build_prompt_views(
    messages: Sequence[Mapping[str, Any]],
    safe_context: str,
    *,
    system_message_policy: Literal["merge", "prepend", "reject"] = "merge",
    template_default_system_context: str | None = None,
) -> PromptViewPair:
    """Construct same-input T- and T+ views; only T+ receives `safe_context`."""

    if not safe_context.strip():
        raise ValueError("safe_context must be non-empty")
    if not messages:
        raise ValueError("messages must be non-empty")
    if system_message_policy not in {"merge", "prepend", "reject"}:
        raise ValueError("unsupported system_message_policy")
    if (
        template_default_system_context is not None
        and not template_default_system_context.strip()
    ):
        raise ValueError("template_default_system_context must be non-empty")

    original_messages = deepcopy(list(messages))
    system_positions = [
        index
        for index, message in enumerate(original_messages)
        if message.get("role") == "system"
    ]
    if any(index != 0 for index in system_positions):
        raise ValueError("system messages must appear only in the leading position")

    if system_positions and system_message_policy == "reject":
        raise ValueError("the original conversation already has a system message")
    if system_positions and system_message_policy == "merge":
        minus_list = deepcopy(original_messages)
        original_system_content = original_messages[0].get("content")
        if not isinstance(original_system_content, str):
            raise ValueError("the leading system content must be a string")
        plus_list = deepcopy(original_messages)
        plus_list[0] = dict(plus_list[0])
        plus_list[0]["content"] = (
            f"{original_system_content.rstrip()}\n\n{safe_context.strip()}"
        )
    elif not system_positions and template_default_system_context is not None:
        baseline_system = template_default_system_context.strip()
        minus_list = [
            {"role": "system", "content": baseline_system},
            *deepcopy(original_messages),
        ]
        plus_list = [
            {
                "role": "system",
                "content": f"{baseline_system}\n\n{safe_context.strip()}",
            },
            *deepcopy(original_messages),
        ]
    else:
        minus_list = deepcopy(original_messages)
        plus_list = [
            {"role": "system", "content": safe_context.strip()},
            *deepcopy(original_messages),
        ]
    minus_messages = tuple(minus_list)
    plus_messages = tuple(plus_list)

    return PromptViewPair(
        teacher_minus=PromptView(
            name="teacher_minus",
            messages=minus_messages,
            fingerprint=_fingerprint_messages(minus_messages),
        ),
        teacher_plus=PromptView(
            name="teacher_plus",
            messages=plus_messages,
            fingerprint=_fingerprint_messages(plus_messages),
        ),
    )


def build_teacher_forcing_batch(
    prompt_token_ids: Sequence[Sequence[int]],
    response_token_ids: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    view_name: str,
    prompt_attention_masks: Sequence[Sequence[int]] | None = None,
) -> TeacherForcingBatch:
    """Left-pad prompts and right-pad responses without coupling their lengths.

    ``prompt_attention_masks`` is optional because ordinary Student and T- views
    attend every prompt token.  Privileged views such as the Stage 2L exact-span
    attention quarantine pass an explicit binary mask.  Response tokens always
    remain visible: the quarantine is a prompt-side Teacher privilege, not a
    way to suppress the shared on-policy response being scored.
    """

    if len(prompt_token_ids) != len(response_token_ids):
        raise ValueError("prompt and response batch sizes must match")
    if not prompt_token_ids:
        raise ValueError("teacher-forcing batch must be non-empty")
    if any(len(prompt) == 0 for prompt in prompt_token_ids):
        raise ValueError("each prompt must contain at least one token")
    if any(len(response) == 0 for response in response_token_ids):
        raise ValueError("each response must contain at least one token")
    if prompt_attention_masks is not None:
        if len(prompt_attention_masks) != len(prompt_token_ids):
            raise ValueError("prompt-mask and prompt batch sizes must match")
        for prompt, prompt_mask in zip(
            prompt_token_ids, prompt_attention_masks, strict=True
        ):
            if len(prompt_mask) != len(prompt):
                raise ValueError("each prompt mask must match its prompt length")
            if any(value not in (0, 1, False, True) for value in prompt_mask):
                raise ValueError("prompt attention masks must be binary")
            if not any(prompt_mask):
                raise ValueError("each prompt mask must retain at least one token")

    batch_size = len(prompt_token_ids)
    prompt_width = max(len(prompt) for prompt in prompt_token_ids)
    response_width = max(len(response) for response in response_token_ids)
    sequence_width = prompt_width + response_width

    input_ids = torch.full(
        (batch_size, sequence_width), pad_token_id, dtype=torch.long
    )
    attention_mask = torch.zeros_like(input_ids)
    position_ids = torch.zeros_like(input_ids)
    responses = torch.full(
        (batch_size, response_width), pad_token_id, dtype=torch.long
    )
    response_mask = torch.zeros_like(responses)

    for row, (prompt, response) in enumerate(
        zip(prompt_token_ids, response_token_ids, strict=True)
    ):
        prompt_start = prompt_width - len(prompt)
        input_ids[row, prompt_start:prompt_width] = torch.tensor(prompt)
        if prompt_attention_masks is None:
            attention_mask[row, prompt_start:prompt_width] = 1
        else:
            attention_mask[row, prompt_start:prompt_width] = torch.tensor(
                prompt_attention_masks[row], dtype=torch.long
            )

        response_end = prompt_width + len(response)
        input_ids[row, prompt_width:response_end] = torch.tensor(response)
        attention_mask[row, prompt_width:response_end] = 1
        position_ids[row, prompt_start:response_end] = torch.arange(
            len(prompt) + len(response), dtype=torch.long
        )
        responses[row, : len(response)] = torch.tensor(response)
        response_mask[row, : len(response)] = 1

    prompt_fingerprint = sha256(
        json.dumps(
            [list(prompt) for prompt in prompt_token_ids],
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return TeacherForcingBatch(
        view_name=view_name,
        prompt_fingerprint=prompt_fingerprint,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        responses=responses,
        response_mask=response_mask,
        prompt_width=prompt_width,
    )


def assert_response_alignment(
    teacher_minus: TeacherForcingBatch, teacher_plus: TeacherForcingBatch
) -> None:
    """Fail if two Teacher views are not scoring the identical Student response."""

    if teacher_minus.responses.shape != teacher_plus.responses.shape:
        raise ValueError("response tensor shapes differ across Teacher views")
    if not torch.equal(teacher_minus.responses, teacher_plus.responses):
        raise ValueError("response token IDs differ across Teacher views")
    if not torch.equal(teacher_minus.response_mask, teacher_plus.response_mask):
        raise ValueError("response masks differ across Teacher views")


def _fingerprint_messages(messages: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(
        list(messages), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()
