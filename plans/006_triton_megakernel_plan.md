# Triton FlashMoE path for gfx1250

## Current implementation

`flashmoe_triton` is a functional Triton implementation for gfx1250 using the TheRock Triton/PyTorch stack in `/jam/venv`.

The validated path is split into three Triton dispatches:

1. `_router_kernel`
   - Computes router logits, softmax, top-1/top-2 selection, route probabilities, expert counts.
   - Materializes accepted route payloads into `routed_tokens[E, round_ec, H]`.
   - Zero-initializes route slots so capacity-padding work is harmless.

2. `_up_project_kernel`
   - Computes expert up projection from `routed_tokens`.
   - Supports ungated MLP and gated/SwiGLU-style MLP via `expert_up_v` and `bias_up_v`.
   - Applies `identity`, `relu`, `silu`, or sigmoid-approximate `gelu`.
   - Stores fp32 intermediate `[E, round_ec, I]`.

3. `_down_project_kernel`
   - Computes expert down projection from the intermediate.
   - Converts intermediate to fp16 for the dot path to match the HIP test/reference quantization.
   - Applies `bias_down`, route probability, and atomically combines into fp32 output.

This is not the final persistent, single-dispatch megakernel. It is a correct Triton baseline that exercises router, per-expert projection, gated path, and combine on gfx1250.

## Why this is split today

The stable Triton baseline currently materializes intermediates between stages:

- Direct `tl.dot` from sparse token-id gathered rows is avoided by materializing `routed_tokens`.
- Up-projection and down-projection are split by an intermediate buffer.
- Scalar/SIMT fallback work remains a correctness and compiler-isolation path.

The passing probes are:

- route table load plus fp32 atomic combine
- routed-token up projection
- manual and dot-based down projection from an intermediate buffer
- full split router/up/down path

So the stable gfx1250 Triton route is currently to materialize the intermediate between the two expert GEMMs.

## Triton-distributed / rocSHMEM status

I cloned `https://github.com/ByteDance-Seed/Triton-distributed` at commit `2b4c24b`.

Relevant findings:

- It has a HIP rocSHMEM device API at `python/triton_dist/language/extra/hip/librocshmem_device.py`.
- The generic shmem proxy selects rocSHMEM on HIP via `TRITON_DIST_SHMEM_BACKEND=rocshmem`.
- It includes AMD EP/all-to-all kernels and AMD tests.
- It is not a drop-in package with the current TheRock venv. Importing it with `PYTHONPATH=/jam/Triton-distributed/python` fails because TheRock Triton lacks `triton._C.libtriton.distributed`.
- `/jam/venv` does not have `pyrocshmem`.
- TheRock’s ROCm SDK does include rocSHMEM headers, `librocshmem.a`, and `librocshmem_device_gfx1250.bc`, so an isolated Triton-distributed build should be able to reuse the local rocSHMEM installation instead of rebuilding all of rocSHMEM.

Recommended next path for distributed parity:

1. Build Triton-distributed in an isolated environment, not destructively inside `/jam/venv`.
2. Build/install only `pyrocshmem` against TheRock’s rocSHMEM if possible.
3. Validate `python/triton_dist/test/amd/test_rocshmem_api.py` with two local processes pinned to the single GPU.
4. Port the FlashMoE communication/scheduling layer onto `triton_dist.language.extra.libshmem_device`.
5. Revisit a true persistent one-dispatch body once the Triton compiler/runtime fault is isolated or a Triton-distributed codegen path avoids it.

## Validation

Smoke commands:

```bash
source /jam/venv/bin/activate
export PYTHONPATH=/jam/flashmoe:${PYTHONPATH:-}

python tests/triton_moe_smoke.py --case forward
python tests/triton_moe_smoke.py --case forward --top-k 1 --expert-capacity 16
python tests/triton_moe_smoke.py --case forward --s 32 --h 64 --i 64 --e 16 --top-k 2 --expert-capacity 8
```

Keep command output, counts, and error statistics out of git. Compare locally
against CPU references and store run logs only in scratch space.
