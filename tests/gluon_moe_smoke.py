from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flashmoe_gluon as fmg
from tests.triton_moe_smoke import _make_inputs


_ACTIVATIONS = {
    "identity": fmg.ACT_IDENTITY,
    "silu": fmg.ACT_SILU,
    "gelu": fmg.ACT_GELU,
    "relu": fmg.ACT_RELU,
}


def _activation(x: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "identity":
        return x
    if activation == "silu":
        return x * torch.sigmoid(x)
    if activation == "gelu":
        return torch.nn.functional.gelu(x)
    if activation == "relu":
        return torch.relu(x)
    raise AssertionError(activation)


def _make_gated_inputs(
    s: int,
    h: int,
    i: int,
    e: int,
    *,
    seed: int = 19,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    up_v = (torch.rand((e, h, i), generator=generator) * 0.2 - 0.1).half()
    bias_up_v = (torch.rand((e, i), generator=generator) * 0.02 - 0.01).half()
    return up_v, bias_up_v


def _reference(
    tokens_h: torch.Tensor,
    gate_h: torch.Tensor,
    up_h: torch.Tensor,
    down_h: torch.Tensor,
    bias_up_h: torch.Tensor,
    bias_down_h: torch.Tensor,
    top_k: int,
    expert_capacity: int,
    *,
    activation: str,
    up_v_h: torch.Tensor | None = None,
    bias_up_v_h: torch.Tensor | None = None,
) -> torch.Tensor:
    tokens = tokens_h.float()
    gate = gate_h.float()
    up = up_h.float()
    down = down_h.float()
    bias_up = bias_up_h.float()
    bias_down = bias_down_h.float()
    up_v = up_v_h.float() if up_v_h is not None else None
    bias_up_v = bias_up_v_h.float() if bias_up_v_h is not None else None

    probs = torch.softmax(tokens @ gate, dim=1)
    vals, idxs = torch.topk(probs, top_k, dim=1)
    vals = vals / vals.sum(dim=1, keepdim=True)

    s, h = tokens.shape
    e = gate.shape[1]
    out = torch.zeros((s, h), dtype=torch.float32)
    routes: list[list[tuple[int, float]]] = [[] for _ in range(e)]
    for token_id in range(s):
        for route_id in range(top_k):
            expert_id = int(idxs[token_id, route_id])
            routes[expert_id].append((token_id, float(vals[token_id, route_id])))

    for expert_id in range(e):
        for token_id, prob in routes[expert_id][:expert_capacity]:
            hidden = tokens[token_id] @ up[expert_id] + bias_up[expert_id]
            hidden = _activation(hidden, activation)
            if up_v is not None:
                assert bias_up_v is not None
                hidden = hidden * (tokens[token_id] @ up_v[expert_id] + bias_up_v[expert_id])
            result = hidden @ down[expert_id] + bias_down[expert_id]
            out[token_id] += result * prob
    return out


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
    parser.add_argument("--activation", choices=tuple(_ACTIVATIONS), default="identity")
    parser.add_argument("--gated", action="store_true")
    parser.add_argument("--rtol", type=float, default=8e-2)
    parser.add_argument("--atol", type=float, default=8e-3)
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
    up_v = None
    bias_up_v = None
    if args.gated:
        up_v, bias_up_v = _make_gated_inputs(args.s, args.h, args.i, args.e)
    ref_out = _reference(
        tokens,
        gate,
        up,
        down,
        bias_up,
        bias_down,
        args.top_k,
        args.expert_capacity,
        activation=args.activation,
        up_v_h=up_v,
        bias_up_v_h=bias_up_v,
    )

    if args.allow_scalar_debug:
        if args.top_k != 1:
            raise ValueError("--allow-scalar-debug requires --top-k 1")
        if args.gated or args.activation != "identity":
            raise ValueError("--allow-scalar-debug only supports vanilla identity")
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
        if args.gated or args.activation != "identity":
            raise ValueError("--allow-decomposed-staging only supports vanilla identity")
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
            activation=_ACTIVATIONS[args.activation],
            expert_up_v=up_v.cuda() if up_v is not None else None,
            bias_up_v=bias_up_v.cuda() if bias_up_v is not None else None,
        )
    torch.cuda.synchronize()
    got = out.cpu()
    diff = (got - ref_out).abs()
    finite = bool(torch.isfinite(got).all())
    close = bool(torch.allclose(got, ref_out, rtol=args.rtol, atol=args.atol))
    print(
        "gluon_megakernel",
        "max_abs",
        float(diff.max()),
        "mean_abs",
        float(diff.mean()),
        "finite",
        finite,
        "close",
        close,
        "rtol",
        args.rtol,
        "atol",
        args.atol,
    )
    if not finite or not close:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
