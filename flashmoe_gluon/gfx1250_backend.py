from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Optional


_MOE_MODULE = "moe_gfx1250"
_DEFAULT_TRITON_ROOT = Path("/jam/triton")


@dataclass(frozen=True)
class BackendStatus:
    available: bool
    reason: str
    triton_root: Optional[Path]
    moe_example: Optional[Path]


def _candidate_triton_roots() -> list[Path]:
    roots: list[Path] = []
    env_root = os.environ.get("FLASHMOE_TRITON_ROOT") or os.environ.get("TRITON_ROOT")
    if env_root:
        roots.append(Path(env_root))
    roots.append(_DEFAULT_TRITON_ROOT)
    repo_relative = Path(__file__).resolve().parents[2] / "triton"
    roots.append(repo_relative)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.expanduser()
        if resolved not in seen:
            deduped.append(resolved)
            seen.add(resolved)
    return deduped


def _paths_for(root: Path) -> tuple[Path, Path, Path, Path]:
    python_path = root / "python"
    triton_kernels_path = python_path / "triton_kernels"
    examples_path = root / "third_party" / "amd" / "python" / "examples" / "gluon"
    moe_example = examples_path / f"{_MOE_MODULE}.py"
    return python_path, triton_kernels_path, examples_path, moe_example


def locate_backend(triton_root: Optional[Path | str] = None) -> BackendStatus:
    roots = [Path(triton_root)] if triton_root is not None else _candidate_triton_roots()
    for root in roots:
        python_path, triton_kernels_path, examples_path, moe_example = _paths_for(root)
        if not python_path.exists():
            continue
        if not (triton_kernels_path / "triton_kernels").exists():
            continue
        if not moe_example.exists():
            continue
        if not (examples_path / "gfx1250_utils.py").exists():
            continue
        return BackendStatus(True, "ok", root, moe_example)
    searched = ", ".join(str(root) for root in roots)
    return BackendStatus(False, f"could not locate gfx1250 Gluon MoE backend under: {searched}", None, None)


def _prepend_once(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def load_moe_backend(triton_root: Optional[Path | str] = None) -> ModuleType:
    status = locate_backend(triton_root)
    if not status.available or status.triton_root is None:
        raise RuntimeError(status.reason)

    python_path, triton_kernels_path, examples_path, _ = _paths_for(status.triton_root)
    _prepend_once(python_path)
    _prepend_once(examples_path)
    _prepend_once(triton_kernels_path)
    return importlib.import_module(_MOE_MODULE)


def backend_status(triton_root: Optional[Path | str] = None) -> BackendStatus:
    status = locate_backend(triton_root)
    if not status.available:
        return status
    try:
        load_moe_backend(status.triton_root)
    except Exception as exc:
        return BackendStatus(False, f"failed to import {_MOE_MODULE}: {exc}", status.triton_root, status.moe_example)
    return status


def matmul(*args: Any, triton_root: Optional[Path | str] = None, **kwargs: Any) -> Any:
    """Call the upstream gfx1250 Gluon MoE matmul helper.

    This is a staging helper, not the FlashMoE megakernel entry point. It lets us
    reuse and port the TDM/WMMA body without adding Triton-distributed.
    """
    module = load_moe_backend(triton_root)
    return module.matmul(*args, **kwargs)


__all__ = [
    "BackendStatus",
    "backend_status",
    "load_moe_backend",
    "locate_backend",
    "matmul",
]
