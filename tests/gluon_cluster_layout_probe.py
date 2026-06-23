from __future__ import annotations

import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton._C.libtriton.gluon_ir import make_cga_layout


def _hashable_cga_layout(layout):
    return tuple(tuple(row) for row in layout)


@gluon.jit
def _slice_expand_probe(out, CGA_LAYOUT_M: gl.constexpr):
    layout: gl.constexpr = gl.amd.AMDWMMALayout(
        3,
        True,
        ((0, 1), (1, 0)),
        (),
        (16, 16, 32),
        CGA_LAYOUT_M,
    )
    rows = gl.arange(0, 32, layout=gl.SliceLayout(1, layout))
    cols = gl.arange(0, 64, layout=gl.SliceLayout(0, layout))
    rows_2d = gl.expand_dims(rows, 1)
    cols_2d = gl.expand_dims(cols, 0)
    cols_vals = gl.convert_layout(cols_2d, layout)
    vals = gl.full((32, 64), 0.0, gl.float32, layout=layout) + cols_vals.to(gl.float32)
    offsets = gl.convert_layout(rows_2d * 64 + cols_2d, layout)
    gl.store(out + offsets, vals)


@gluon.jit
def _full_probe(out, CGA_LAYOUT_M: gl.constexpr):
    layout: gl.constexpr = gl.amd.AMDWMMALayout(
        3,
        True,
        ((0, 1), (1, 0)),
        (),
        (16, 16, 32),
        CGA_LAYOUT_M,
    )
    rows = gl.arange(0, 32, layout=gl.SliceLayout(1, layout))
    cols = gl.arange(0, 64, layout=gl.SliceLayout(0, layout))
    rows_2d = gl.expand_dims(rows, 1)
    cols_2d = gl.expand_dims(cols, 0)
    vals = gl.full((32, 64), 7.0, gl.float32, layout=layout)
    offsets = gl.convert_layout(rows_2d * 64 + cols_2d, layout)
    gl.store(out + offsets, vals)


@gluon.jit
def _cga_scalar_count(counter, CGA_LAYOUT_M: gl.constexpr):
    layout: gl.constexpr = gl.amd.AMDWMMALayout(
        3,
        True,
        ((0, 1), (1, 0)),
        (),
        (16, 16, 32),
        CGA_LAYOUT_M,
    )
    vals = gl.full((32, 64), 0.0, gl.float32, layout=layout)
    if vals.shape[0] == 32:
        gl.atomic_add(counter, 1, sem="relaxed", scope="gpu")


def main() -> None:
    torch.cuda.set_device(0)
    out = torch.empty((32, 64), device="cuda", dtype=torch.float32)
    cga_layout_m = _hashable_cga_layout(make_cga_layout([2, 1], [2, 1], [0, 1]))
    _full_probe[(1,)](
        out,
        CGA_LAYOUT_M=cga_layout_m,
        num_warps=4,
        num_ctas=2,
    )
    torch.cuda.synchronize()
    got = out.cpu()
    print("cluster_full", "first", float(got[0, 1]), "last", float(got[31, 63]))

    _slice_expand_probe[(1,)](
        out,
        CGA_LAYOUT_M=cga_layout_m,
        num_warps=4,
        num_ctas=2,
    )
    torch.cuda.synchronize()
    got = out.cpu()
    print("cluster_slice_expand", "first", float(got[0, 1]), "last", float(got[31, 63]))

    counter = torch.zeros((1,), device="cuda", dtype=torch.int32)
    _cga_scalar_count[(2,)](
        counter,
        CGA_LAYOUT_M=cga_layout_m,
        num_warps=4,
        num_ctas=2,
    )
    torch.cuda.synchronize()
    print("cga_scalar_count", int(counter.cpu()[0]))


if __name__ == "__main__":
    main()
