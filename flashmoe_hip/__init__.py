from . import router
from .jit import InitArgs, ContextHandle, Topology, MLPType, ActivationType, DataType, ForwardArgs
from .cb import get_local_rank
from . import reference

SHOULD_FINALIZE_ROCSHMEM = False
def initialize(arg: InitArgs) -> ContextHandle:
    from .jit import _get_compiled
    from .bindings import flashmoe_bindings
    from . import cb
    import torch
    assert arg.ep_rank is None or ((arg.rank_map is not None)
        and (arg.ep_world is not None) and (arg.expert_map is not None)
        and (arg.num_local_experts is not None)
        and (arg.my_pe is not None)), "if rank is set, then so should all dependent metadata"
    if arg.ep_rank is None:
        global SHOULD_FINALIZE_ROCSHMEM
        SHOULD_FINALIZE_ROCSHMEM = True
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

    cb.sync_all(stream_ptr=arg.stream_ptr)

    def detect_topo():
        """Detect interconnect topology for AMD GPUs.
        XGMI_ONLY: all peers connected via XGMI (Infinity Fabric).
        MIXED: some peers on XGMI, others on PCIe/network.
        """
        try:
            import subprocess
            result = subprocess.run(
                ["rocm-smi", "--showtopo"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                output = result.stdout
                has_xgmi = "XGMI" in output
                has_pcie = "PCIE" in output or "PCIe" in output
                if has_xgmi and not has_pcie:
                    return Topology.XGMI_ONLY
                elif has_xgmi:
                    return Topology.MIXED
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        # If we can't detect, assume local ranks on visible devices use the
        # fast local fabric path.
        if arg.ep_world is not None:
            num_devices = torch.cuda.device_count()
            if arg.ep_world <= num_devices:
                return Topology.XGMI_ONLY
        return Topology.MIXED
    arg.topo = detect_topo()

    mod_prefix = "flashmoe_moe"
    mod_name = (f"{mod_prefix}_s{arg.tokens_per_rank}_h{arg.token_dim}_i{arg.ffn_size}"
            f"_e{arg.num_experts}_ec{arg.expert_peer_capacity}_k{arg.top_k}"
            f"_topo{arg.topo}_mt{arg.mlp_type}_dt{arg.data_type}_act{arg.act_type}_arch{arg.gpu_arch}")

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
        dt=arg.data_type
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
        stream_ptr=arg.stream_ptr
    )
    return ContextHandle(mod, ctx)

def forward(handle: ContextHandle, args: ForwardArgs) -> None:
    handle.mod.forward(handle.context,
                       tokens=args.tokens,
                       expert_counts=args.expert_counts,
                       local_expert_up=args.local_expert_up,
                       local_expert_up_v=args.local_expert_up_v or 0,
                       local_bias_up=args.local_bias_up,
                       local_bias_up_v=args.local_bias_up_v or 0,
                       local_expert_down=args.local_expert_down,
                       local_bias_down=args.local_bias_down,
                       moe_out=args.moe_out,
                       swish_alpha=args.swish_alpha,
                       swish_beta=args.swish_beta,
                       stream_ptr=args.stream_ptr)

def finalize(handle: ContextHandle, stream_ptr: int) -> None:
    handle.mod.finalize(handle.context, stream_ptr)
    global SHOULD_FINALIZE_ROCSHMEM
    if SHOULD_FINALIZE_ROCSHMEM:
        SHOULD_FINALIZE_ROCSHMEM = False
        import torch
        dev = torch.cuda.device(get_local_rank())
        torch.cuda.synchronize(dev)
