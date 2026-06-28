# FlashMoE Port Context

This file is a sanitized continuation note for agents working on this branch.
Do not add hardware specifications, benchmark numbers, profiler output, or raw
device-query logs here.

## Project Overview

FlashMoE is a fused distributed Mixture-of-Experts system based on the public
paper and upstream repository. The port work preserves the paper/CUDA execution
model while adding HIP and Gluon implementation paths.

High-level target name: MI450.

## Repository Structure

- `csrc/include/flashmoe/`: original CUDA headers. Do not modify unless the
  user explicitly asks for upstream changes.
- `csrc/include/flashmoe/hip/`: HIP backend and CUDA compatibility layer.
- `flashmoe_hip/`: HIP Python module and JIT bindings.
- `flashmoe_gluon/`: local Gluon megakernel and experimental Gluon rocSHMEM
  helpers.
- `docs/`: sanitized implementation status docs.
- `plans/`: sanitized local plans and runbooks.

## Current Status

- HIP is the distributed paper/CUDA parity baseline.
- Gluon is the local megakernel performance and readability path.
- Gluon rocSHMEM is guarded because its protocol is not parity.
- vLLM integration is paused until a supported runtime path is selected.
- The local Qwen-like precomputed-routing Gluon smoke passes isolated
  finite-output validation.

## Hard Rules

- Do not push unless explicitly requested.
- Do not bypass the scheduler or task model to improve a local metric.
- Do not add direct atomic or CAS-based distributed workarounds in place of
  rocSHMEM protocol behavior.
- Compare distributed changes against the HIP parity path.
- Keep raw benchmark and profiler data out of committed files.

## Environment

Use `/jam/venv` for Python work. For normal Python runs, unset
`LD_LIBRARY_PATH`, `ROCM_HOME`, and `ROCM_PATH` and rely on the venv wheel
preload behavior:

```bash
env -u LD_LIBRARY_PATH -u ROCM_HOME -u ROCM_PATH \
  PATH="/jam/venv/bin:$PATH" PYTHONPATH=/jam/FlashMoE \
  /jam/venv/bin/python <script>
```

Set `HSA_DISABLE_COREDUMP_ON_EXCEPTION=1` for crash-prone repros.

Build selectors for this branch:

- `GPU_TARGETS=gfx1250`
- `CMAKE_HIP_ARCHITECTURES=gfx1250`
- `ARCH=1250`
- `FLASHMOE_HIP_ARCH=1250`
- `PYTORCH_ROCM_ARCH=gfx1250` when a Python extension path consults PyTorch's
  target environment

The HIP JIT path in `flashmoe_hip/jit.py` passes `GPU_TARGETS=gfx1250` and
`ARCH=1250` into CMake.

## Validation Order

1. Static Python compilation and `git diff --check`.
2. Isolated local Gluon smoke for Qwen-like precomputed routing.
3. HIP CPU-reference smoke for parity-sensitive behavior.
4. rocprof only after the same smoke passes without profiling.

Do not store run output in this file. Summarize only pass/fail/source-level
status in docs.
