# Triton FlashMoE Plan

## Current Position

The Triton path is a correctness and compiler-isolation reference. It is not
the current paper-parity distributed path and is not the active local
megakernel optimization target.

The stable Triton baseline materializes routing/intermediate state across
separate dispatches. That is useful for debugging, but it is not the final
persistent FlashMoE architecture.

## Status

- Keep Triton as a fallback reference for isolated routing and expert math.
- Do not use Triton-distributed or vLLM branch experiments as validation for
  the current Gluon megakernel.
- Keep any Triton-distributed rocSHMEM work isolated from `/jam/venv` unless a
  supported branch is selected.

## Direction

If Triton work resumes, the target remains a persistent one-body MoE kernel
that preserves routing, scheduling, expert compute, communication, and combine
inside the paper-style execution model.

Do not record benchmark numbers, raw profile output, or hardware details in
this plan.
