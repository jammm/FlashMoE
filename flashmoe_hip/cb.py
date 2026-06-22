# Communication backend for HIP/ROCm
# Uses torch.distributed or MPI (rocSHMEM Python bindings are not yet available)

IS_INITIALIZED = False
_RANK = -1
_WORLD_SIZE = -1

def has_package(name: str):
    import importlib.util
    return importlib.util.find_spec(name) is not None

def get_local_rank() -> int:
    import os
    if has_package("torch") and os.environ.get("LOCAL_RANK") is not None:
        return int(os.environ.get("LOCAL_RANK"))
    elif has_package("mpi4py"):
        import mpi4py.MPI as MPI
        import torch
        return MPI.COMM_WORLD.Get_rank() % torch.cuda.device_count()
    else:
        raise RuntimeError("At least one of {torch, mpi4py} must be available")

def initialize() -> None:
    import torch
    global IS_INITIALIZED, _RANK, _WORLD_SIZE
    if IS_INITIALIZED:
        return
    initialized = False
    local_rank = get_local_rank()
    torch.cuda.set_device(local_rank)
    if has_package("torch"):
        import torch.distributed as dist
        if dist.is_initialized():
            _RANK = dist.get_rank()
            _WORLD_SIZE = dist.get_world_size()
            dist.barrier()
            initialized = True
    if not initialized and has_package("mpi4py"):
        import mpi4py.MPI as MPI
        _RANK = MPI.COMM_WORLD.Get_rank()
        _WORLD_SIZE = MPI.COMM_WORLD.Get_size()
        MPI.COMM_WORLD.Barrier()
        initialized = True
    IS_INITIALIZED = initialized
    if not initialized:
        raise RuntimeError("At least one of {torch.distributed, mpi4py} must be initialized")

def get_rank() -> int:
    assert IS_INITIALIZED, "Communication backend not initialized"
    return _RANK

def get_world_size() -> int:
    assert IS_INITIALIZED, "Communication backend not initialized"
    return _WORLD_SIZE

def sync_all(stream_ptr: int) -> None:
    import torch
    stream = torch.cuda.ExternalStream(stream_ptr)
    stream.synchronize()
    if not IS_INITIALIZED:
        return
    if has_package("torch"):
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()
    elif has_package("mpi4py"):
        import mpi4py.MPI as MPI
        MPI.COMM_WORLD.Barrier()
