"""Small trainable output adapter and checkpoint helpers for runtime smokes."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn


class OutputLogitAdapter(nn.Module):
    """Low-rank correction over frozen response hidden states.

    This is deliberately a bounded integration adapter, not the final parameter-
    efficient training architecture.  It lets the real Qwen3-8B hidden/logit path,
    R-OPCD objective, optimizer, checkpoint, resume, and rollback execute together
    without allocating gradients for the frozen 8B base model.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        *,
        rank: int,
        seed: int,
        init_std: float = 0.01,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or vocab_size < 2 or rank < 1:
            raise ValueError("hidden_size, vocab_size, and rank must be positive")
        if init_std <= 0.0:
            raise ValueError("init_std must be positive")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.down = nn.Parameter(
            torch.randn(hidden_size, rank, generator=generator, dtype=torch.float32)
            * init_std
        )
        self.up = nn.Parameter(torch.zeros(rank, vocab_size, dtype=torch.float32))

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.shape[-1] != self.down.shape[0]:
            raise ValueError("hidden-state width differs from adapter input width")
        work = hidden_states.float()
        return (work @ self.down) @ self.up


def trainable_state_sha256(module: nn.Module) -> str:
    digest = sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def save_smoke_checkpoint(
    path: Path,
    *,
    adapter: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    metadata: Mapping[str, Any],
) -> None:
    if step < 0:
        raise ValueError("checkpoint step must be non-negative")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "r-opcd-stage3-runtime-smoke-checkpoint-v1",
            "step": step,
            "adapter_state": adapter.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metadata": dict(metadata),
        },
        path,
    )


def load_smoke_checkpoint(
    path: Path,
    *,
    adapter: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "r-opcd-stage3-runtime-smoke-checkpoint-v1":
        raise ValueError("runtime smoke checkpoint schema mismatch")
    adapter.load_state_dict(payload["adapter_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    return payload
