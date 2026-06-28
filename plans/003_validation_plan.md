# Validation Plan

## Environment

For Python validation, use the venv and unset explicit ROCm path overrides:

```bash
env -u LD_LIBRARY_PATH -u ROCM_HOME -u ROCM_PATH \
  PATH="/jam/venv/bin:$PATH" PYTHONPATH=/jam/FlashMoE \
  /jam/venv/bin/python <script>
```

Use `HSA_DISABLE_COREDUMP_ON_EXCEPTION=1` for crash-prone kernel debugging so
failed experiments do not fill the workspace with core dumps.

## Validation Order

1. Static checks
   - `python -m py_compile` for touched Python modules and tests.
   - `git diff --check`.

2. Local Gluon smoke
   - Run the isolated Qwen-like precomputed-routing smoke.
   - Require finite output on repeat calls before using the result for further
     profiling or integration work.

3. HIP parity smoke
   - Run the HIP CPU-reference smoke for identity, activation, and gated cases.
   - Treat HIP as the distributed paper/CUDA parity baseline.

4. Gluon rocSHMEM
   - The public entry point is guarded because it is not parity.
   - Do not use it as a passing distributed validation signal unless the run is
     explicitly marked experimental.

5. vLLM
   - Paused for now. Do not use vLLM results to validate the Gluon kernel until
     the supported runtime path is settled.

## Result Recording Policy

Committed docs may say that a check passed, failed, or is guarded. They must
not include:

- benchmark numbers;
- profile-counter values;
- raw timing output;
- runtime device-query output;
- hardware specifications;
- full debug-state dumps.

Keep raw logs and profiler artifacts in local scratch storage only.
