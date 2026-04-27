"""
family_registry.py — Centralised registry of supported Hamiltonian families.

Each entry captures the canonical name, parameter conventions, boundary support,
and natural-language keyword hints for one Hamiltonian family.  The registry is
the single source of truth for the NL interface layer; the parser, validator, and
report generator all look here rather than hard-coding family names.

Parameter conventions match ``hamiltonians.build_hamiltonian_spec()`` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# FamilySpec
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

@dataclass
class FamilySpec:
    """
    Metadata for a single supported Hamiltonian family.

    Attributes
    ----------
    canonical_name   : The exact string accepted by ``hamiltonians.build_hamiltonian_spec()``.
    aliases          : Alternative names / abbreviations (case-insensitive lookup).
    required_params  : Parameters that must be supplied; parser raises a warning if absent.
    optional_params  : Parameters with sensible defaults; parser uses defaults when absent.
    boundary_support : Boundary conditions the family supports (currently only ``"obc"``).
    short_description: One-sentence physical description for display / explanation generation.
    keyword_hints    : Words / phrases in user text that suggest this family.
    """

    canonical_name: str
    aliases: List[str] = field(default_factory=list)
    required_params: List[str] = field(default_factory=list)
    optional_params: Dict[str, float] = field(default_factory=dict)
    boundary_support: List[str] = field(default_factory=lambda: ["obc"])
    short_description: str = ""
    keyword_hints: List[str] = field(default_factory=list)


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Registry entries
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

_REGISTRY: Dict[str, FamilySpec] = {
    "tfim": FamilySpec(
        canonical_name="tfim",
        aliases=[
            "transverse field ising",
            "transverse-field ising",
            "transverse ising",
            "tf ising",
            "tfim",
        ],
        required_params=["J", "h"],
        optional_params={"J": 1.0, "h": 0.5},
        boundary_support=["obc"],
        short_description=(
            "Transverse-field Ising model: H = -J Σ Z_i Z_{i+1} - h Σ X_i (OBC). "
            "Paradigmatic model for a quantum phase transition between ferromagnetic "
            "and paramagnetic phases at |h/J| = 1."
        ),
        keyword_hints=[
            "transverse field",
            "transverse-field",
            "ising",
            "tfim",
            "ferromagnetic",
            "quantum critical",
            "ZZ coupling",
            "X field",
        ],
    ),

    "ising_general": FamilySpec(
        canonical_name="ising_general",
        aliases=[
            "ising general",
            "ising_general",
            "general ising",
            "longitudinal ising",
            "longitudinal-field ising",
        ],
        required_params=["J", "hx", "hz"],
        optional_params={"J": 1.0, "hx": 0.5, "hz": 0.0},
        boundary_support=["obc"],
        short_description=(
            "General / longitudinal Ising model: H = -J Σ ZZ - hx Σ X - hz Σ Z (OBC). "
            "Extends TFIM with an additional longitudinal (Z) field hz; hz=0 recovers TFIM."
        ),
        keyword_hints=[
            "longitudinal field",
            "longitudinal-field",
            "ising general",
            "general ising",
            "ZZ",
            "X field",
            "Z field",
            "hx",
            "hz",
        ],
    ),

    "xxz": FamilySpec(
        canonical_name="xxz",
        aliases=[
            "xxz",
            "xxz model",
            "anisotropic heisenberg",
            "anisotropic-heisenberg",
            "xxz heisenberg",
        ],
        required_params=["J", "delta"],
        optional_params={"J": 1.0, "delta": 1.0},
        boundary_support=["obc"],
        short_description=(
            "XXZ model: H = J Σ (X_i X_{i+1} + Y_i Y_{i+1} + δ Z_i Z_{i+1}) (OBC). "
            "Anisotropic Heisenberg chain; delta=0 → XX model, delta=1 → isotropic Heisenberg."
        ),
        keyword_hints=[
            "xxz",
            "anisotropic",
            "delta",
            "anisotropy",
            "XX YY ZZ",
            "spin chain",
            "exchange anisotropy",
        ],
    ),

    "heisenberg": FamilySpec(
        canonical_name="heisenberg",
        aliases=[
            "heisenberg",
            "heisenberg model",
            "isotropic heisenberg",
            "isotropic-heisenberg",
            "xxx",
            "xxx model",
        ],
        required_params=["J"],
        optional_params={"J": 1.0},
        boundary_support=["obc"],
        short_description=(
            "Isotropic Heisenberg model: H = J Σ (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1}) (OBC). "
            "SU(2)-symmetric spin-1/2 chain; special case of XXZ with delta=1."
        ),
        keyword_hints=[
            "heisenberg",
            "isotropic",
            "SU(2)",
            "xxx",
            "spin-1/2",
            "antiferromagnet",
            "ferromagnet",
            "exchange coupling",
        ],
    ),
}


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Helper functions
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def get_supported_families() -> List[str]:
    """Return the canonical names of all registered Hamiltonian families."""
    return list(_REGISTRY.keys())


def get_family_spec(name: str) -> Optional[FamilySpec]:
    """
    Return the ``FamilySpec`` for ``name``, or ``None`` if not found.

    Lookup is case-insensitive and checks both canonical names and aliases.

    Args:
        name : Canonical name or alias string.

    Returns:
        Matching ``FamilySpec``, or ``None``.
    """
    key = name.strip().lower()

    # Direct canonical lookup
    if key in _REGISTRY:
        return _REGISTRY[key]

    # Alias scan
    for spec in _REGISTRY.values():
        if key in [a.lower() for a in spec.aliases]:
            return spec

    return None


def is_supported_family(name: str) -> bool:
    """
    Return ``True`` iff ``name`` resolves to a registered family (canonical or alias).

    Args:
        name : Name or alias to check (case-insensitive).
    """
    return get_family_spec(name) is not None


__all__ = [
    "FamilySpec",
    "get_supported_families",
    "get_family_spec",
    "is_supported_family",
]
