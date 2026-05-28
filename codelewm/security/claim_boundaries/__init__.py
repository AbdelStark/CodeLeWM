"""Reusable claim-boundary text fragments referenced by manifests and cards.

A claim boundary is a verbatim block of text that scopes the kinds of public
claim a downstream artifact is allowed to make. Boundaries are versioned by
filename and are loaded as plain text so they can be embedded into dataset
cards, model cards, and JSON manifests without diverging.

The module exposes a small loader that returns the text by name. The loader
does not parse or render the boundary; it returns the exact bytes from disk
so downstream consumers can hash or fingerprint the boundary.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = [
    "ClaimBoundaryError",
    "available_claim_boundaries",
    "claim_boundary_fingerprint",
    "load_claim_boundary",
]


_BOUNDARIES_DIR = Path(__file__).resolve().parent


class ClaimBoundaryError(LookupError):
    """Raised when a requested claim boundary file is not registered."""


def _boundary_path(name: str) -> Path:
    if not name or any(ch in name for ch in "/\\"):
        raise ClaimBoundaryError(f"invalid claim boundary name: {name!r}")
    path = _BOUNDARIES_DIR / f"{name}.md"
    if not path.is_file():
        raise ClaimBoundaryError(f"claim boundary not found: {name!r}")
    return path


def load_claim_boundary(name: str) -> str:
    """Return the verbatim text of the named claim boundary.

    ``name`` is the filename stem (e.g. ``execution_substrate.v1``). The
    function reads the file as UTF-8 and returns it unchanged. Callers must
    not mutate the text before embedding it into a manifest or card.
    """

    path = _boundary_path(name)
    return path.read_text(encoding="utf-8")


def claim_boundary_fingerprint(name: str) -> str:
    """Return the SHA-256 hex digest of the named claim boundary."""

    text = load_claim_boundary(name)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def available_claim_boundaries() -> tuple[str, ...]:
    """Return the registered claim-boundary names, sorted."""

    return tuple(
        sorted(p.stem for p in _BOUNDARIES_DIR.glob("*.md") if p.is_file())
    )
