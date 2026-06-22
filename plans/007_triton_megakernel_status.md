# Triton megakernel status

## Implemented path

`flashmoe_triton.forward_megakernel` now provides a one-body fused Triton MoE kernel for gfx1250.

The kernel performs, inside a single Triton kernel body:

- router projection for one token
- softmax and top-1/top-2 expert selection
- expert capacity accounting via `expert_counts`
- up projection
- optional gated/SwiGLU V projection
- activation
- down projection
- local top-k combine into the final output row

This avoids the baseline Triton path's global `routed_tokens` and intermediate `[E, round_ec, I]` materialization. It is intentionally scalar/SIMT and correctness-first: one Triton program owns a token and computes the full hidden output vector.

## Current host-visible dispatch caveat

The fused kernel itself is a single Triton dispatch for the MoE work, but the Python wrapper still clears `expert_counts` before launch with `expert_counts.zero_()`. A fully single-dispatch production wrapper needs either:

- a persistent context with counts reset by a safe in-kernel global initialization protocol, or
- an API contract where the caller supplies already-reset per-forward scratch.

An in-kernel startup flag where program 0 zeroes `expert_counts` and other
programs wait on a volatile flag still needs a safer initialization protocol
before it can replace the host-side reset.

## Validation

Use the Triton-distributed venv and disable PyTorch's CUDA caching allocator for clean process teardown:

```bash
source /jam/triton_dist_venv/bin/activate
ROCM_ROOT=$(/jam/venv/bin/rocm-sdk path --root)
export ROCM_ROOT ROCM_PATH=$ROCM_ROOT ROCM_HOME=$ROCM_ROOT HIP_HOME=$ROCM_ROOT
export PATH=$ROCM_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_ROOT/lib:${LD_LIBRARY_PATH:-}
export TRITON_CODEGEN_BACKENDS=amd
export PYTHONPATH=/jam/flashmoe:${PYTHONPATH:-}
export PYTORCH_NO_CUDA_MEMORY_CACHING=1
python /jam/flashmoe/tests/triton_moe_smoke.py --case megakernel --s 16 --h 64 --i 64 --e 4 --top-k 2 --expert-capacity 16
```

Keep command output, expert-count dumps, and error statistics out of git.
Compare locally against CPU references and store run logs only in scratch
space.

## Remaining parity work

This is not yet the paper-parity persistent TDM/tiled design. The next implementation step is to replace the scalar inner loops with tiled up/down projection work:

- map persistent CTAs to token/expert tiles rather than one token per program
- use Triton's gfx1250 TDM descriptor path for global-to-shared movement
- use mbarrier/named-barrier equivalents from the upstream gfx1250 lowering instead of semaphore assumptions
- keep routing, dispatch, expert compute, and combine in one kernel body
- avoid the host-side `expert_counts.zero_()` by moving scratch lifecycle into an initialized persistent context or a verified in-kernel initialization protocol
