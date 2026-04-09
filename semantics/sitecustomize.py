from __future__ import annotations

import sys
from pathlib import Path


def _looks_like_argcausaldisco(path: Path) -> bool:
    return (
        (path / "__init__.py").exists()
        and (path / "abapc.py").exists()
        and (path / "utils").is_dir()
    )


def _add_to_syspath(path: Path) -> None:
    resolved = str(path.resolve())
    if path.exists() and resolved not in sys.path:
        sys.path.insert(0, resolved)


def _resolve_argcausaldisco_root(gradual_root: Path) -> Path | None:
    embedded_root = gradual_root.parent
    if _looks_like_argcausaldisco(embedded_root):
        return embedded_root

    local_clone = gradual_root / "ArgCausalDisco"
    if _looks_like_argcausaldisco(local_clone):
        return local_clone

    return None


GRADUAL_ROOT = Path(__file__).resolve().parent
ARGCAUSALDISCO_ROOT = _resolve_argcausaldisco_root(GRADUAL_ROOT)

if ARGCAUSALDISCO_ROOT is not None:
    _add_to_syspath(ARGCAUSALDISCO_ROOT.parent)
    _add_to_syspath(ARGCAUSALDISCO_ROOT)

for dependency_name in ("GradualABA", "aspforaba", "notears", "py-causal"):
    for dependency_path in (
        GRADUAL_ROOT / dependency_name,
        GRADUAL_ROOT.parent / dependency_name,
        GRADUAL_ROOT.parent.parent / dependency_name,
    ):
        if dependency_path.exists():
            _add_to_syspath(dependency_path)
            break
