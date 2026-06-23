from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from . import gfx1250_backend
from .megakernel import ACT_IDENTITY, _validate_common


@dataclass(frozen=True)
class RoutingMetadata:
    slice_sizes: torch.Tensor
    ragged_metadata: object
    gather_idx: torch.Tensor
    scatter_idx: torch.Tensor
    route_probs: torch.Tensor
    top_idx: torch.Tensor


def _column_major_last2(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.transpose(-1, -2).contiguous().transpose(-1, -2)


def build_routing_metadata(
    tokens: torch.Tensor,
    gate_weights: torch.Tensor,
    *,
    top_k: int,
    expert_capacity: Optional[int] = None,
) -> RoutingMetadata:
    if top_k not in (1, 2):
        raise ValueError("staging routing currently supports top_k in {1, 2}")
    if tokens.device.type != "cuda":
        raise ValueError("staging routing expects HIP/CUDA tensors on device")

    s = tokens.shape[0]
    e = gate_weights.shape[1]
    logits = tokens.float() @ gate_weights.float()
    probs = torch.softmax(logits, dim=1)
    vals, idxs = torch.topk(probs, top_k, dim=1)
    vals = vals / vals.sum(dim=1, keepdim=True)

    route_pos = torch.arange(s * top_k, device=tokens.device, dtype=torch.int32)
    token_ids = (route_pos // top_k).to(torch.int32)
    expert_ids = idxs.reshape(-1).to(torch.int32)
    route_probs = vals.reshape(-1).to(torch.float32)
    order = torch.argsort(expert_ids, stable=True)

    sorted_expert_ids = expert_ids[order]
    slice_sizes = torch.bincount(sorted_expert_ids, minlength=e).to(torch.int32)
    if expert_capacity is not None and bool((slice_sizes > expert_capacity).any().item()):
        raise NotImplementedError("decomposed Gluon staging path does not drop capacity-overflow routes yet")

    module = gfx1250_backend.load_moe_backend()
    ragged_metadata = module.make_ragged_tensor_metadata(slice_sizes, route_pos.numel())
    return RoutingMetadata(
        slice_sizes=slice_sizes,
        ragged_metadata=ragged_metadata,
        gather_idx=token_ids[order].contiguous(),
        scatter_idx=route_pos[order].contiguous(),
        route_probs=route_probs,
        top_idx=idxs,
    )


def forward_decomposed_staging(
    tokens: torch.Tensor,
    gate_weights: torch.Tensor,
    expert_up: torch.Tensor,
    expert_down: torch.Tensor,
    bias_up: torch.Tensor,
    bias_down: torch.Tensor,
    *,
    top_k: int,
    expert_capacity: Optional[int] = None,
    activation: int = ACT_IDENTITY,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 256,
    num_buffers: int = 2,
    schedule: str = "baseline",
    pingpong: bool = False,
    num_warps: int = 4,
) -> torch.Tensor:
    """Multi-dispatch Gluon staging path for porting the TDM/WMMA compute body.

    This is intentionally separate from `forward_megakernel`; it is a bring-up
    helper for validating upstream Gluon MoE matmul integration.
    """
    if activation != ACT_IDENTITY:
        raise NotImplementedError("decomposed Gluon staging currently supports identity activation only")
    _validate_common(
        tokens,
        gate_weights,
        expert_up,
        expert_down,
        bias_up,
        bias_down,
        top_k=top_k,
        expert_capacity=expert_capacity,
        activation=activation,
        expert_up_v=None,
        bias_up_v=None,
        block_h=None,
        num_warps=num_warps,
    )

    routing = build_routing_metadata(
        tokens,
        gate_weights,
        top_k=top_k,
        expert_capacity=expert_capacity,
    )
    module = gfx1250_backend.load_moe_backend()
    expert_up_cm = _column_major_last2(expert_up)
    expert_down_cm = _column_major_last2(expert_down)
    matmul_kwargs = dict(
        num_buffers=num_buffers,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        schedule=schedule,
        pingpong=pingpong,
        num_warps=num_warps,
    )

    hidden, _ = module.matmul(
        tokens,
        expert_up_cm,
        bias_up.float(),
        routing.ragged_metadata,
        gather_indx=routing.gather_idx,
        **matmul_kwargs,
    )
    combined_rows, _ = module.matmul(
        hidden,
        expert_down_cm,
        bias_down.float(),
        routing.ragged_metadata,
        scatter_indx=routing.scatter_idx,
        **matmul_kwargs,
    )

    s, h = tokens.shape
    weighted = combined_rows.float() * routing.route_probs[:, None]
    return weighted.view(s, top_k, h).sum(dim=1)


__all__ = [
    "RoutingMetadata",
    "build_routing_metadata",
    "forward_decomposed_staging",
]
