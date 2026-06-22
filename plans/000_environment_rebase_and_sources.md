# Environment, Rebase, and Source Notes

Date: 2026-06-21

## Repository State

- Working tree: `/jam/flashmoe`
- Active branch: `jam/hip_port`
- Added upstream remote: `https://github.com/osayamenja/FlashMoE`
- Rebased `jam/hip_port` onto `upstream/main` successfully.
- Current upstream base after rebase: `9cc0c32 Update citation`
- Local branch is ahead of the fork remote because rebase rewrote the HIP-port commits. Do not push unless explicitly requested.

## ROCm / TheRock Environment

Use the TheRock wheels installed in `/jam/venv`.

```bash
source /jam/venv/bin/activate
export ROCM_ROOT="$(rocm-sdk path --root)"
export PATH="$(rocm-sdk path --bin):$PATH"
export CMAKE_PREFIX_PATH="$(rocm-sdk path --cmake):$CMAKE_PREFIX_PATH"
export LD_LIBRARY_PATH="$ROCM_ROOT/lib:$LD_LIBRARY_PATH"
```

Resolve ROCm paths through `rocm-sdk path` rather than committing machine-specific
toolchain paths or version output.

The local TheRock guidance in `/jam/TheRock/CLAUDE.md` describes TheRock as a CMake super-project, but for this task the important practical point is that ROCm comes from the installed wheel layout rather than `/opt/rocm`.

## GPU State

The implementation target is `gfx1250`. Runtime device queries and benchmark
logs should stay out of git; use local scratch files for run output when needed.

## Validation Notes

Direct HIP:

- Build direct HIP tests with host-side references before depending on Python
  framework results.

PyTorch:

- Do not use PyTorch matmul as the correctness oracle until the local stack is
  independently validated.
- Prefer HIP tests with host-computed references for now.

Compile maintenance:

- Keep tests synchronized with template signatures and gfx1250 WMMA tile shapes.

## External Sources Studied

- Paper PDF: https://arxiv.org/pdf/2506.04667
- Project page: https://flash-moe.github.io/
- Upstream repository: https://github.com/osayamenja/FlashMoE
- Public TheRock HIP/TDM headers and tests in the local ROCm checkout
- Public Triton/LLVM gfx1250 lowering code in `/jam/triton`
