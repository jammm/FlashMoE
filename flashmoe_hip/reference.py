"""
PyTorch-based reference implementation for FlashMoE on HIP/ROCm.

Replaces the cuBLASDx-based CUDA reference with pure PyTorch operations
that work on AMD GPUs via ROCm's PyTorch backend.
"""
from .jit import ContextHandle, InitArgs, MLPType, ActivationType, ForwardArgs
import torch
import torch.nn.functional as F


def _get_activation(act_type: ActivationType):
    if act_type == ActivationType.IDENTITY:
        return lambda x, alpha, beta: x
    elif act_type == ActivationType.SILU:
        return lambda x, alpha, beta: alpha * F.silu(x * beta)
    elif act_type == ActivationType.GELU:
        return lambda x, alpha, beta: F.gelu(x)
    elif act_type == ActivationType.RELU:
        return lambda x, alpha, beta: F.relu(x)
    else:
        raise ValueError(f"Unknown activation type: {act_type}")


def _dtype_from_enum(dt):
    from .jit import DataType
    if dt == DataType.BF16:
        return torch.bfloat16
    elif dt == DataType.FP16:
        return torch.float16
    elif dt == DataType.FP32:
        return torch.float32
    elif dt == DataType.FP64:
        return torch.float64
    raise ValueError(f"Unknown data type: {dt}")


class _RefContext:
    """Holds reference implementation state."""
    def __init__(self, arg: InitArgs):
        self.S = arg.tokens_per_rank
        self.H = arg.token_dim
        self.I = arg.ffn_size
        self.E = arg.num_experts
        self.EC = arg.expert_peer_capacity
        self.top_k = arg.top_k
        self.mlp_type = arg.mlp_type
        self.act_type = arg.act_type
        self.data_type = arg.data_type
        self.device_id = arg.device_id
        self.dtype = _dtype_from_enum(arg.data_type)


def initialize(arg: InitArgs) -> ContextHandle:
    torch.cuda.set_device(arg.device_id)
    ctx = _RefContext(arg)

    class _RefModule:
        """Mimics the compiled module interface with PyTorch ops."""
        def __init__(self, ctx):
            self._ctx = ctx

    mod = _RefModule(ctx)
    return ContextHandle(mod, ctx)


class RefForwardArgs:
    expert_up: int
    expert_down: int
    bias_up: int
    bias_down: int
    ref_input: int       # workspace [roundEC, H]
    ref_interim0: int    # workspace [roundEC, I]
    ref_interim1: int    # workspace [roundEC, H]
    ref_out: int         # output [S, H]
    expert_up_v: int = 0
    bias_up_v: int = 0

    def __init__(self,
                 expert_up: int,
                 expert_down: int,
                 bias_up: int,
                 bias_down: int,
                 ref_input: int,
                 ref_interim0: int,
                 ref_interim1: int,
                 ref_out: int,
                 *,
                 expert_up_v: int,
                 bias_up_v: int):
        self.ref_input = ref_input
        self.ref_interim0 = ref_interim0
        self.ref_interim1 = ref_interim1
        self.ref_out = ref_out
        self.expert_up_v = expert_up_v
        self.bias_up = bias_up
        self.bias_down = bias_down
        self.bias_up_v = bias_up_v
        self.expert_up = expert_up
        self.expert_down = expert_down


def forward(handle: ContextHandle, token_ids_ptr: int, f_args: ForwardArgs, r_args: RefForwardArgs) -> None:
    """
    Pure PyTorch reference MoE forward pass.

    For each expert:
      1. Gather tokens routed to this expert
      2. GEMM0: tokens @ expert_up + bias_up, with activation
         (if gated: also compute tokens @ expert_up_v + bias_up_v, then gate)
      3. GEMM1: intermediate @ expert_down + bias_down
      4. Combine (scatter) results back, weighted by routing probability for top_k > 1
    """
    ctx = handle.context
    S, H, I, E = ctx.S, ctx.H, ctx.I, ctx.E
    EC = ctx.EC
    top_k = ctx.top_k
    dtype = ctx.dtype
    device = torch.device(f"cuda:{ctx.device_id}")

    from math import ceil
    roundEC = ceil(EC / 16) * 16  # conservative rounding

    # Reconstruct tensors from data pointers
    tokens = torch.tensor([], dtype=dtype, device=device)
    tokens.set_(torch.Storage._new_shared(S * H * tokens.element_size(), device=device),
                0, (S, H))
    # For the reference, we work directly with PyTorch tensors passed by the caller.
    # The pointer-based interface is for compatibility with the JIT compiled path.
    # In practice, callers should use forward_torch() below for the reference.
    raise NotImplementedError(
        "Pointer-based forward() requires unsafe memory reinterpretation. "
        "Use forward_torch() with PyTorch tensors directly for the reference implementation."
    )


def forward_torch(
    ctx: _RefContext,
    tokens: torch.Tensor,        # [S, H]
    expert_counts: torch.Tensor,  # [E] int32
    expert_up: torch.Tensor,      # [num_local_experts, H, I]
    expert_down: torch.Tensor,    # [num_local_experts, I, H]
    bias_up: torch.Tensor,        # [num_local_experts, I]
    bias_down: torch.Tensor,      # [num_local_experts, H]
    token_ids: torch.Tensor,      # [total_routed] with .tokenIdx and .probability
    moe_out: torch.Tensor,        # [S, H] output
    *,
    expert_up_v: torch.Tensor = None,  # [num_local_experts, H, I] for gated MLP
    bias_up_v: torch.Tensor = None,    # [num_local_experts, I] for gated MLP
    swish_alpha: float = 1.0,
    swish_beta: float = 1.0,
) -> None:
    """
    Pure PyTorch reference MoE forward. Operates on PyTorch tensors directly.

    This avoids pointer reinterpretation and is suitable for correctness testing.
    """
    S, H, I, E = ctx.S, ctx.H, ctx.I, ctx.E
    EC = ctx.EC
    top_k = ctx.top_k
    act_fn = _get_activation(ctx.act_type)
    is_gated = ctx.mlp_type == MLPType.GATED

    moe_out.zero_()

    offset = 0
    for expert_idx in range(E):
        count = min(int(expert_counts[expert_idx].item()), EC)
        if count == 0:
            continue

        # Gather tokens routed to this expert
        expert_token_ids = token_ids[offset:offset + count]
        token_indices = expert_token_ids[:, 0].long()  # tokenIdx field
        token_probs = expert_token_ids[:, 1].float() if top_k > 1 else None

        gathered = tokens[token_indices]  # [count, H]

        # GEMM0: up-projection
        # gathered [count, H] @ expert_up[expert_idx] [H, I] -> [count, I]
        up_result = gathered @ expert_up[expert_idx] + bias_up[expert_idx]

        if is_gated:
            gate_val = act_fn(up_result, swish_alpha, swish_beta)
            # V projection
            v_result = gathered @ expert_up_v[expert_idx] + bias_up_v[expert_idx]
            intermediate = v_result * gate_val
        else:
            intermediate = act_fn(up_result, swish_alpha, swish_beta)

        # GEMM1: down-projection
        # intermediate [count, I] @ expert_down[expert_idx] [I, H] -> [count, H]
        down_result = intermediate @ expert_down[expert_idx] + bias_down[expert_idx]

        # Combine / scatter back
        if top_k == 1:
            moe_out[token_indices] = down_result
        else:
            # Weighted accumulation
            weighted = (down_result * token_probs.unsqueeze(1)).to(moe_out.dtype)
            moe_out.index_add_(0, token_indices, weighted)

        offset += count
