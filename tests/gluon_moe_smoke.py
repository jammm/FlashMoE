from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flashmoe_gluon as fmg
from tests.triton_moe_smoke import _make_inputs, _reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s", type=int, default=16)
    parser.add_argument("--h", type=int, default=64)
    parser.add_argument("--i", type=int, default=64)
    parser.add_argument("--e", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--expert-capacity", type=int, default=16)
    parser.add_argument("--backend-status", action="store_true")
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--allow-scalar-debug", action="store_true")
    parser.add_argument("--allow-decomposed-staging", action="store_true")
    args = parser.parse_args()

    if args.backend_status or args.no_run:
        status = fmg.backend_status()
        print(
            "gluon_backend",
            "available",
            status.available,
            "reason",
            status.reason,
            "root",
            status.triton_root,
            "moe_example",
            status.moe_example,
        )
        if args.no_run:
            return

    torch.cuda.set_device(0)
    tokens, gate, up, down, bias_up, bias_down = _make_inputs(args.s, args.h, args.i, args.e)
    ref_out, ref_counts = _reference(
        tokens,
        gate,
        up,
        down,
        bias_up,
        bias_down,
        args.top_k,
        args.expert_capacity,
    )

    if args.allow_scalar_debug:
        if args.top_k != 1:
            raise ValueError("--allow-scalar-debug requires --top-k 1")
        out = fmg.forward_scalar_top1_debug(
            tokens.cuda(),
            gate.cuda(),
            up.cuda(),
            down.cuda(),
            bias_up.cuda(),
            bias_down.cuda(),
            expert_capacity=args.expert_capacity,
        )
    elif args.allow_decomposed_staging:
        out = fmg.forward_decomposed_staging(
            tokens.cuda(),
            gate.cuda(),
            up.cuda(),
            down.cuda(),
            bias_up.cuda(),
            bias_down.cuda(),
            top_k=args.top_k,
            expert_capacity=args.expert_capacity,
        )
    else:
        out = fmg.forward_megakernel(
            tokens.cuda(),
            gate.cuda(),
            up.cuda(),
            down.cuda(),
            bias_up.cuda(),
            bias_down.cuda(),
            top_k=args.top_k,
            expert_capacity=args.expert_capacity,
        )
    torch.cuda.synchronize()
    got = out.cpu()
    diff = (got - ref_out).abs()
    print(
        "gluon_megakernel",
        "max_abs",
        float(diff.max()),
        "mean_abs",
        float(diff.mean()),
        "finite",
        bool(torch.isfinite(got).all()),
    )


if __name__ == "__main__":
    main()
