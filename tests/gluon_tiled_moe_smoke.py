from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flashmoe_gluon as fmg
from tests.gluon_moe_smoke import _ACTIVATIONS, _make_gated_inputs, _reference
from tests.triton_moe_smoke import _make_inputs


SHAPES = ((64, 128), (128, 64), (128, 128))
QUICK_CASES = (
    (64, 128, 2, False),
    (128, 64, 2, False),
    (128, 128, 2, True),
)


def _run_case(
    *,
    s: int,
    h: int,
    i: int,
    e: int,
    top_k: int,
    gated: bool,
    expert_capacity: int,
    num_programs: int,
    rtol: float,
    atol: float,
    seed: int,
) -> None:
    activation = "silu" if gated else "identity"
    tokens, gate, up, down, bias_up, bias_down = _make_inputs(s, h, i, e, seed=seed)
    up_v = None
    bias_up_v = None
    if gated:
        up_v, bias_up_v = _make_gated_inputs(s, h, i, e, seed=seed + 1000)

    ref_out = _reference(
        tokens,
        gate,
        up,
        down,
        bias_up,
        bias_down,
        top_k,
        expert_capacity,
        activation=activation,
        up_v_h=up_v,
        bias_up_v_h=bias_up_v,
    )
    out = fmg.forward_megakernel(
        tokens.cuda(),
        gate.cuda(),
        up.cuda(),
        down.cuda(),
        bias_up.cuda(),
        bias_down.cuda(),
        top_k=top_k,
        expert_capacity=expert_capacity,
        activation=_ACTIVATIONS[activation],
        expert_up_v=up_v.cuda() if up_v is not None else None,
        bias_up_v=bias_up_v.cuda() if bias_up_v is not None else None,
        num_programs=num_programs,
    )
    torch.cuda.synchronize()
    got = out.cpu()
    diff = (got - ref_out).abs()
    finite = bool(torch.isfinite(got).all())
    close = bool(torch.allclose(got, ref_out, rtol=rtol, atol=atol))
    print(
        "gluon_tiled_moe",
        f"h={h}",
        f"i={i}",
        f"top_k={top_k}",
        f"gated={int(gated)}",
        "max_abs",
        float(diff.max()),
        "mean_abs",
        float(diff.mean()),
        "finite",
        finite,
        "close",
        close,
    )
    if not finite or not close:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s", type=int, default=8)
    parser.add_argument("--e", type=int, default=2)
    parser.add_argument("--expert-capacity", type=int, default=8)
    parser.add_argument("--num-programs", type=int, default=2)
    parser.add_argument("--rtol", type=float, default=8e-2)
    parser.add_argument("--atol", type=float, default=8e-3)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    torch.cuda.set_device(0)
    if args.full:
        cases = [
            (h, i, top_k, gated)
            for h, i in SHAPES
            for top_k in (1, 2)
            for gated in (False, True)
        ]
    else:
        cases = list(QUICK_CASES)

    for case_id, (h, i, top_k, gated) in enumerate(cases):
        _run_case(
            s=args.s,
            h=h,
            i=i,
            e=args.e,
            top_k=top_k,
            gated=gated,
            expert_capacity=args.expert_capacity,
            num_programs=args.num_programs,
            rtol=args.rtol,
            atol=args.atol,
            seed=31 + case_id,
        )


if __name__ == "__main__":
    main()
