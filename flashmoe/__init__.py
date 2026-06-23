from __future__ import annotations

import importlib
import os
import sys


def _normalize_backend(value: str) -> str:
    backend = value.strip().lower()
    if backend in ("auto", ""):
        return "auto"
    if backend in ("cuda", "nvidia"):
        return "cuda"
    if backend in ("hip", "amd", "rocm"):
        return "hip"
    raise ValueError(
        "FLASHMOE_BACKEND must be one of auto, cuda, nvidia, hip, amd, or rocm"
    )


def _detect_backend() -> str:
    requested = _normalize_backend(os.environ.get("FLASHMOE_BACKEND", "auto"))
    if requested != "auto":
        return requested
    try:
        import torch

        if getattr(torch.version, "hip", None):
            return "hip"
    except Exception:
        pass
    return "cuda"


BACKEND = _detect_backend()


def backend_name() -> str:
    return BACKEND


if BACKEND == "hip":
    try:
        _backend = importlib.import_module("flashmoe_hip")
        _jit = importlib.import_module("flashmoe_hip.jit")
        _router = importlib.import_module("flashmoe_hip.router")
        _reference = importlib.import_module("flashmoe_hip.reference")
        _cb = importlib.import_module("flashmoe_hip.cb")
        _bindings = importlib.import_module("flashmoe_hip.bindings")
    except Exception as exc:
        raise RuntimeError(
            "FlashMoE selected the HIP backend but flashmoe_hip failed to load"
        ) from exc

    sys.modules[__name__ + ".jit"] = _jit
    sys.modules[__name__ + ".router"] = _router
    sys.modules[__name__ + ".reference"] = _reference
    sys.modules[__name__ + ".cb"] = _cb
    sys.modules[__name__ + ".bindings"] = _bindings

    router = _router
    reference = _reference
    InitArgs = _jit.InitArgs
    ContextHandle = _jit.ContextHandle
    Topology = _jit.Topology
    MLPType = _jit.MLPType
    ActivationType = _jit.ActivationType
    DataType = _jit.DataType
    ForwardArgs = _jit.ForwardArgs
    get_local_rank = _cb.get_local_rank
    initialize = _backend.initialize
    forward = _backend.forward
    finalize = _backend.finalize
else:
    from . import reference, router
    from .cb import get_local_rank
    from .jit import (
        ActivationType,
        ContextHandle,
        DataType,
        ForwardArgs,
        InitArgs,
        MLPType,
        Topology,
    )

    SHOULD_FINALIZE_NVSHMEM = False

    def initialize(arg: InitArgs) -> ContextHandle:
        from . import cb
        from .bindings import flashmoe_bindings
        from .jit import _get_compiled
        import cuda.core as cuda
        import nvshmem.core as nvshmem

        assert arg.ep_rank is None or (
            (arg.rank_map is not None)
            and (arg.ep_world is not None)
            and (arg.expert_map is not None)
            and (arg.num_local_experts is not None)
            and (arg.my_pe is not None)
        ), "if rank is set, then so should all dependent metadata"
        if arg.ep_rank is None:
            global SHOULD_FINALIZE_NVSHMEM
            SHOULD_FINALIZE_NVSHMEM = True
            cb.initialize()
            arg.ep_rank = cb.get_rank()
            arg.my_pe = cb.get_rank()
            arg.ep_world = cb.get_world_size()
            assert arg.num_experts % arg.ep_world == 0
            arg.num_local_experts = arg.num_experts // arg.ep_world
            arg.expert_map = []
            for i in range(arg.num_experts):
                arg.expert_map.append(i // arg.num_local_experts)
            arg.rank_map = []
            for i in range(arg.ep_world):
                arg.rank_map.append(i)
        nvshmem.sync_all(stream=cuda.Stream.from_handle(arg.stream_ptr))

        def detect_topo():
            assert nvshmem.init_status() == nvshmem.InitStatus.STATUS_IS_INITIALIZED
            if nvshmem.team_n_pes(nvshmem.Teams.TEAM_SHARED) == nvshmem.n_pes():
                return Topology.NVLINK_ONLY
            return Topology.MIXED

        arg.topo = detect_topo()

        mod_prefix = "flashmoe_moe"
        mod_name = (
            f"{mod_prefix}_s{arg.tokens_per_rank}_h{arg.token_dim}_i{arg.ffn_size}"
            f"_e{arg.num_experts}_ec{arg.expert_peer_capacity}_k{arg.top_k}"
            f"_topo{arg.topo}_mt{arg.mlp_type}_dt{arg.data_type}_act{arg.act_type}_arch{arg.gpu_arch}"
        )

        src = flashmoe_bindings.substitute(
            arch=arg.gpu_arch,
            s=arg.tokens_per_rank,
            h=arg.token_dim,
            i=arg.ffn_size,
            e=arg.num_experts,
            ec=arg.expert_peer_capacity,
            tk=arg.top_k,
            mod_name=mod_name,
            topo=arg.topo,
            mt=arg.mlp_type,
            act=arg.act_type,
            dt=arg.data_type,
        )
        mod = _get_compiled(arg, src, mod_prefix, mod_name)
        ctx = mod.initialize(
            num_experts=arg.num_experts,
            expert_peer_capacity=arg.expert_peer_capacity,
            ep_world=arg.ep_world,
            my_pe=arg.my_pe,
            ep_rank=arg.ep_rank,
            local_rank=arg.device_id,
            num_local_experts=arg.num_local_experts,
            expert_map=arg.expert_map,
            rank_map=arg.rank_map,
            stream_ptr=arg.stream_ptr,
        )
        return ContextHandle(mod, ctx)

    def forward(handle: ContextHandle, args: ForwardArgs) -> None:
        handle.mod.forward(
            handle.context,
            tokens=args.tokens,
            expert_counts=args.expert_counts,
            local_expert_up=args.local_expert_up,
            local_expert_up_v=args.local_expert_up_v,
            local_bias_up=args.local_bias_up,
            local_bias_up_v=args.local_bias_up_v,
            local_expert_down=args.local_expert_down,
            local_bias_down=args.local_bias_down,
            moe_out=args.moe_out,
            swish_alpha=args.swish_alpha,
            swish_beta=args.swish_beta,
            stream_ptr=args.stream_ptr,
        )

    def finalize(handle: ContextHandle, stream_ptr: int) -> None:
        handle.mod.finalize(handle.context, stream_ptr)
        global SHOULD_FINALIZE_NVSHMEM
        if SHOULD_FINALIZE_NVSHMEM:
            SHOULD_FINALIZE_NVSHMEM = False
            import cuda.core as cuda
            import nvshmem.core as nvshmem

            dev = cuda.Device(get_local_rank())
            dev.sync()
            if nvshmem.init_status() == nvshmem.InitStatus.STATUS_IS_INITIALIZED:
                nvshmem.finalize()


__all__ = [
    "BACKEND",
    "backend_name",
    "router",
    "reference",
    "InitArgs",
    "ContextHandle",
    "Topology",
    "MLPType",
    "ActivationType",
    "DataType",
    "ForwardArgs",
    "get_local_rank",
    "initialize",
    "forward",
    "finalize",
]
