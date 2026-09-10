"""Frozen causal-LM response-logit extraction for no-update pilots."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Protocol, Sequence

import torch
from torch import Tensor, nn

from r_opcd.prompt_views import TeacherForcingBatch


class CausalLMOutput(Protocol):
    logits: Tensor


@dataclass(frozen=True)
class ParameterIntegrity:
    parameter_count: int
    buffer_element_count: int
    trainable_parameter_count: int
    gradient_parameter_count: int
    structure_sha256: str
    sentinel_sha256: str
    version_counters: tuple[int, ...]


@dataclass(frozen=True)
class ResponseLogits:
    view_name: str
    logits: Tensor
    response_mask: Tensor
    source_dtype: str
    source_device: str


@torch.inference_mode()
def greedy_generate_preserving_positions(
    model: nn.Module,
    *,
    input_ids: Tensor,
    attention_mask: Tensor,
    max_new_tokens: int,
    eos_token_ids: Sequence[int],
) -> Tensor:
    """Greedily decode one sequence without collapsing quarantined positions.

    Transformers derives positions from a 2D attention mask when calling
    ``generate``. That treats q1 zeros as padding and shifts all later RoPE
    positions. This narrow cache loop instead assigns positions from the
    complete serialized token sequence while retaining the q1 key mask.
    """

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("position-preserving generation requires one input row")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention mask must match input IDs")
    mask_values = attention_mask.detach().cpu().flatten().tolist()
    if any(value not in (0, 1) for value in mask_values):
        raise ValueError("attention mask must be binary")
    if not bool(attention_mask.any()):
        raise ValueError("attention mask must retain at least one token")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    eos = {int(value) for value in eos_token_ids}
    if not eos:
        raise ValueError("at least one EOS token ID is required")

    prompt_tokens = input_ids.shape[1]
    position_ids = torch.arange(
        prompt_tokens, dtype=torch.long, device=input_ids.device
    ).unsqueeze(0)
    outputs: Any = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
    )
    if not hasattr(outputs, "logits") or not hasattr(outputs, "past_key_values"):
        raise ValueError("causal model output must contain logits and past_key_values")
    past_key_values = outputs.past_key_values
    generated: list[Tensor] = []

    for offset in range(max_new_tokens):
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)
        if int(next_token.item()) in eos:
            break
        if offset + 1 == max_new_tokens:
            break

        attention_mask = torch.cat(
            [
                attention_mask,
                attention_mask.new_ones((attention_mask.shape[0], 1)),
            ],
            dim=-1,
        )
        next_position = torch.tensor(
            [[prompt_tokens + offset]], dtype=torch.long, device=input_ids.device
        )
        outputs = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            position_ids=next_position,
            past_key_values=past_key_values,
            use_cache=True,
        )
        if not hasattr(outputs, "logits") or not hasattr(outputs, "past_key_values"):
            raise ValueError("cached causal model output is incomplete")
        past_key_values = outputs.past_key_values

    return torch.cat(generated, dim=-1)


def _sample_from_logits(
    logits: Tensor,
    *,
    generator: torch.Generator,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Tensor:
    """Sample one token after deterministic temperature/top-k/top-p filtering."""

    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError("sampling expects one row of logits")
    if temperature <= 0:
        raise ValueError("sampling temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if top_k < 0:
        raise ValueError("top_k must be nonnegative")
    filtered = logits.float() / temperature
    if top_k:
        retained = min(top_k, filtered.shape[-1])
        threshold = torch.topk(filtered, retained, dim=-1).values[..., -1, None]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))
    if top_p < 1:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative = sorted_probabilities.cumsum(dim=-1)
        remove = cumulative - sorted_probabilities >= top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        filtered = torch.full_like(filtered, float("-inf"))
        filtered.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    probabilities = torch.softmax(filtered, dim=-1)
    if not torch.isfinite(probabilities).all() or not bool((probabilities.sum(-1) > 0).all()):
        raise ValueError("filtered sampling probabilities are invalid")
    return torch.multinomial(probabilities, 1, generator=generator)


@torch.inference_mode()
def sample_generate_preserving_positions(
    model: nn.Module,
    *,
    input_ids: Tensor,
    attention_mask: Tensor,
    max_new_tokens: int,
    eos_token_ids: Sequence[int],
    generator: torch.Generator,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Tensor:
    """Sample one cached sequence while retaining absolute serialized positions."""

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("position-preserving generation requires one input row")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention mask must match input IDs")
    mask_values = attention_mask.detach().cpu().flatten().tolist()
    if any(value not in (0, 1) for value in mask_values):
        raise ValueError("attention mask must be binary")
    if not bool(attention_mask.any()):
        raise ValueError("attention mask must retain at least one token")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    eos = {int(value) for value in eos_token_ids}
    if not eos:
        raise ValueError("at least one EOS token ID is required")

    prompt_tokens = input_ids.shape[1]
    position_ids = torch.arange(
        prompt_tokens, dtype=torch.long, device=input_ids.device
    ).unsqueeze(0)
    outputs: Any = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
    )
    if not hasattr(outputs, "logits") or not hasattr(outputs, "past_key_values"):
        raise ValueError("causal model output must contain logits and past_key_values")
    past_key_values = outputs.past_key_values
    generated: list[Tensor] = []
    for offset in range(max_new_tokens):
        next_token = _sample_from_logits(
            outputs.logits[:, -1, :],
            generator=generator,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        generated.append(next_token)
        if int(next_token.item()) in eos or offset + 1 == max_new_tokens:
            break
        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
            dim=-1,
        )
        next_position = torch.tensor(
            [[prompt_tokens + offset]], dtype=torch.long, device=input_ids.device
        )
        outputs = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            position_ids=next_position,
            past_key_values=past_key_values,
            use_cache=True,
        )
        if not hasattr(outputs, "logits") or not hasattr(outputs, "past_key_values"):
            raise ValueError("cached causal model output is incomplete")
        past_key_values = outputs.past_key_values
    return torch.cat(generated, dim=-1)


def capture_parameter_integrity(model: nn.Module) -> ParameterIntegrity:
    """Capture cheap mutation sentinels without copying full model weights."""

    structure_hasher = sha256()
    sentinel_hasher = sha256()
    parameter_count = 0
    trainable_count = 0
    gradient_count = 0
    versions: list[int] = []

    for name, parameter in model.named_parameters():
        parameter_count += parameter.numel()
        trainable_count += parameter.numel() if parameter.requires_grad else 0
        gradient_count += parameter.numel() if parameter.grad is not None else 0
        versions.append(parameter._version)
        descriptor = (
            f"{name}|{tuple(parameter.shape)}|{parameter.dtype}|{parameter.numel()}"
        ).encode("utf-8")
        structure_hasher.update(descriptor)
        sentinel_hasher.update(descriptor)
        flat = parameter.detach().reshape(-1)
        if flat.numel():
            indices = sorted({0, flat.numel() // 2, flat.numel() - 1})
            sample = flat[indices].float().cpu().contiguous().numpy().tobytes()
            sentinel_hasher.update(sample)

    buffer_element_count = 0
    for name, buffer in model.named_buffers():
        buffer_element_count += buffer.numel()
        versions.append(buffer._version)
        descriptor = (
            f"buffer:{name}|{tuple(buffer.shape)}|{buffer.dtype}|{buffer.numel()}"
        ).encode("utf-8")
        structure_hasher.update(descriptor)
        sentinel_hasher.update(descriptor)
        flat = buffer.detach().reshape(-1)
        if flat.numel():
            indices = sorted({0, flat.numel() // 2, flat.numel() - 1})
            sample = flat[indices].float().cpu().contiguous().numpy().tobytes()
            sentinel_hasher.update(sample)

    return ParameterIntegrity(
        parameter_count=parameter_count,
        buffer_element_count=buffer_element_count,
        trainable_parameter_count=trainable_count,
        gradient_parameter_count=gradient_count,
        structure_sha256=structure_hasher.hexdigest(),
        sentinel_sha256=sentinel_hasher.hexdigest(),
        version_counters=tuple(versions),
    )


def assert_parameter_integrity_unchanged(
    before: ParameterIntegrity, after: ParameterIntegrity
) -> None:
    """Fail closed on any observed parameter mutation or gradient allocation."""

    if before.parameter_count != after.parameter_count:
        raise RuntimeError("model parameter count changed during no-update forward")
    if before.buffer_element_count != after.buffer_element_count:
        raise RuntimeError("model buffer count changed during no-update forward")
    if before.structure_sha256 != after.structure_sha256:
        raise RuntimeError("model parameter structure changed during no-update forward")
    if before.sentinel_sha256 != after.sentinel_sha256:
        raise RuntimeError("model parameter sentinel changed during no-update forward")
    if before.version_counters != after.version_counters:
        raise RuntimeError("model parameter version changed during no-update forward")
    if before.trainable_parameter_count or after.trainable_parameter_count:
        raise RuntimeError("frozen model contains trainable parameters")
    if before.gradient_parameter_count or after.gradient_parameter_count:
        raise RuntimeError("frozen model accumulated parameter gradients")


@torch.no_grad()
def extract_response_logits(
    model: nn.Module,
    batch: TeacherForcingBatch,
    *,
    device: torch.device | str,
    output_device: torch.device | str = "cpu",
    output_dtype: torch.dtype = torch.float32,
) -> ResponseLogits:
    """Run one frozen view and retain only logits predicting response tokens."""

    if model.training:
        raise ValueError("model must be in eval mode")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("model parameters must have requires_grad=False")

    source_device = torch.device(device)
    outputs: Any = model(
        input_ids=batch.input_ids.to(source_device),
        attention_mask=batch.attention_mask.to(source_device),
        position_ids=batch.position_ids.to(source_device),
        use_cache=False,
    )
    if not hasattr(outputs, "logits"):
        raise ValueError("causal model output does not contain logits")
    full_logits = outputs.logits
    if full_logits.ndim != 3 or full_logits.shape[:2] != batch.input_ids.shape:
        raise ValueError("causal model logits do not match the input dimensions")
    response_logits = full_logits[:, batch.response_logit_slice, :]
    if response_logits.shape[:2] != batch.responses.shape:
        raise ValueError("response-logit slice does not match response dimensions")
    if not torch.isfinite(response_logits).all():
        raise ValueError("response logits must be finite")

    return ResponseLogits(
        view_name=batch.view_name,
        logits=response_logits.to(device=output_device, dtype=output_dtype).contiguous(),
        response_mask=batch.response_mask.to(output_device).bool().contiguous(),
        source_dtype=str(full_logits.dtype),
        source_device=str(full_logits.device),
    )
