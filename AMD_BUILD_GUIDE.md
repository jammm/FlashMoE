# FlashMoE HIP Build Guide

This is a sanitized build guide for the HIP port. Keep hardware specifications,
benchmark numbers, raw profiler output, and device-query output out of this
file.

## Environment

Use the active venv for Python work:

```bash
source /jam/venv/bin/activate
```

For normal Python validation, do not force ROCm library paths. Prefer:

```bash
env -u LD_LIBRARY_PATH -u ROCM_HOME -u ROCM_PATH \
  PATH="/jam/venv/bin:$PATH" PYTHONPATH=/jam/FlashMoE \
  /jam/venv/bin/python <script>
```

Build-only commands may need toolchain discovery through the installed ROCm
SDK. Keep exact local paths in scratch notes unless they are required by source
configuration.

The HIP target selector for this branch is `gfx1250`. The numeric architecture
value passed through the HIP JIT is `1250`.

## What Was Ported

The HIP backend mirrors the upstream FlashMoE structure:

- CUDA compatibility wrappers live under `csrc/include/flashmoe/hip/`.
- `flashmoe_hip/` mirrors the original Python module layout.
- The HIP path preserves the paper/CUDA persistent-kernel actor model:
  routing metadata, dispatch, scheduler/subscriber roles, processor tasks,
  communication signaling, and combine.
- rocSHMEM is the distributed communication backend for the HIP parity path.

## Build Guidance

Use the repository build system where possible rather than ad hoc commands.
The useful target-specific build knobs are:

- `GPU_TARGETS=gfx1250`
- `CMAKE_HIP_ARCHITECTURES=gfx1250`
- `ARCH=1250`
- `FLASHMOE_HIP_ARCH=1250`
- `PYTORCH_ROCM_ARCH=gfx1250` for Python extension builds that consult PyTorch
  environment settings

These are build selectors, not benchmark or device-spec data.

### Python Package

Install the package editable into the active venv:

```bash
env -u LD_LIBRARY_PATH -u ROCM_HOME -u ROCM_PATH \
  PATH="/jam/venv/bin:$PATH" \
  /jam/venv/bin/python -m pip install -e /jam/FlashMoE
```

### HIP JIT Path

The normal HIP path JIT-builds generated bindings when `flashmoe_hip.initialize`
or `flashmoe_hip.router.initialize` is called. The JIT writes a generated
`.hip.cpp` file and invokes CMake with:

```text
-DGENERATED_SRC=<generated .hip.cpp>
-DFLASHMOE_KERNELS_SOURCE=/jam/FlashMoE/csrc
-DTARGET_MODULE_NAME=<generated module name>
-DGPU_TARGETS=gfx1250
-DARCH=1250
-DCMAKE_BUILD_TYPE=Release
```

Use this import check to verify the package and CMake entry point are visible:

```bash
env -u LD_LIBRARY_PATH -u ROCM_HOME -u ROCM_PATH \
  PATH="/jam/venv/bin:$PATH" PYTHONPATH=/jam/FlashMoE \
  /jam/venv/bin/python - <<'PY'
import flashmoe_hip
print("flashmoe_hip import ok")
PY
```

Use the HIP CPU-reference smoke to force the JIT path through a real build:

```bash
HSA_DISABLE_COREDUMP_ON_EXCEPTION=1 PYTHONUNBUFFERED=1 \
FLASHMOE_BACKEND=hip \
env -u LD_LIBRARY_PATH -u ROCM_HOME -u ROCM_PATH \
  PATH="/jam/venv/bin:$PATH" PYTHONPATH=/jam/FlashMoE \
  /jam/venv/bin/python -u tests/hip_megakernel_cpu_reference.py \
    --s <tokens> --h <hidden> --i <intermediate> --e <experts> --dtype fp16
```

Keep the shape values local to the run; do not commit the output.

### Direct CMake Path

For direct HIP/CMake experiments, use the HIP CMake file and pass the target
selector explicitly. The HIP CMake file is named `CMakeLists_hip.txt`, so use a
temporary source copy unless you intentionally rename files in the repo:

```bash
rm -rf /tmp/flashmoe_hip_csrc /tmp/flashmoe_hip_build
mkdir -p /tmp/flashmoe_hip_csrc
cp -a /jam/FlashMoE/csrc/. /tmp/flashmoe_hip_csrc/
cp /tmp/flashmoe_hip_csrc/CMakeLists_hip.txt /tmp/flashmoe_hip_csrc/CMakeLists.txt

cmake -S /tmp/flashmoe_hip_csrc -B /tmp/flashmoe_hip_build \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGPU_TARGETS=gfx1250 \
  -DCMAKE_HIP_ARCHITECTURES=gfx1250

cmake --build /tmp/flashmoe_hip_build --parallel
```

If a local CMake invocation needs ROCm or rocSHMEM package roots, pass them as
CMake cache variables from local environment discovery. Do not hardcode machine
paths in committed docs.

### Standalone `setup_hip.py`

`setup_hip.py` exists for standalone extension experiments. Prefer the JIT path
above for parity validation. If you use `setup_hip.py`, set the same target
selectors explicitly in the environment and keep the build transcript local:

```bash
PYTORCH_ROCM_ARCH=gfx1250 \
FLASHMOE_HIP_ARCH=1250 \
env -u LD_LIBRARY_PATH -u ROCM_HOME -u ROCM_PATH \
  PATH="/jam/venv/bin:$PATH" \
  /jam/venv/bin/python setup_hip.py build_ext --inplace
```

## Validation

Prefer validation that checks source behavior rather than recording runtime
measurements:

- compile touched HIP and Python bindings;
- run correctness tests against CPU or host references;
- run HIP parity smokes before comparing Gluon distributed behavior;
- keep rocSHMEM tests clearly separated from local-only Gluon smokes.

Do not commit benchmark output, profiler CSVs, hardware-query dumps, or exact
run measurements.

## Troubleshooting

- If Python runs cannot find ROCm libraries, first verify the venv install and
  wheel preload behavior before setting process-wide library paths.
- If a crash-prone kernel repro is expected, set
  `HSA_DISABLE_COREDUMP_ON_EXCEPTION=1` to avoid filling the workspace.
- If rocSHMEM linking fails, check build flags and rocSHMEM installation
  locally, then document only the source-level fix.
