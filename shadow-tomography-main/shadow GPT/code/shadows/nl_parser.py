"""
nl_parser.py — Rule-based natural-language parser for Hamiltonian descriptions.

Converts free-text descriptions of quantum spin-chain Hamiltonians into
ParsedHamiltonian objects (defined in nl_schema.py).  All detection is
rule-based: regex for structure, alias/keyword lookup against the family
registry, and param-pattern inference as a final fallback.  No learned
components, no LLM.

Detection pipeline (in order)
------------------------------
parse_hamiltonian_text(text)
  1. normalize_text           — lowercase, collapse whitespace
  2. extract_n_qubits         — regex scan for qubit/site count
  3. extract_boundary         — keyword scan for OBC / PBC
  4. extract_numeric_params   — regex for J / h / hx / hz / delta
  5. detect_family            — staged alias → keyword → param inference
  6. validation               — missing required params, unsupported boundary
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .nl_schema import ParsedHamiltonian
from .family_registry import get_family_spec, get_supported_families, is_supported_family


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Compiled constants
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

# Parameters ordered longest-first so the regex alternation never shadows a
# prefix: "hx" / "hz" must be tried before "h".
_PARAM_NAMES_ORDERED: List[str] = ["delta", "hx", "hz", "J", "h"]

# Canonical casing for each parameter (all matching is case-insensitive).
_PARAM_CANONICAL: Dict[str, str] = {
    "delta": "delta",
    "hx": "hx",
    "hz": "hz",
    "j": "J",
    "h": "h",
}

# Numeric literal: optional sign, integer or decimal, optional scientific exponent.
_NUM_PAT = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"

# PARAM = VALUE with an optional-alpha-underscore lookbehind to prevent matching
# "ahx=0.3" or "whz=1".  Case-insensitive so "J=1" and "j=1" both work.
_PARAM_RE = re.compile(
    r"(?<![a-zA-Z_])("
    + "|".join(re.escape(p) for p in _PARAM_NAMES_ORDERED)
    + r")\s*=\s*("
    + _NUM_PAT
    + r")",
    re.IGNORECASE,
)

# Qubit / site count patterns (tried in order; first match wins).
_NQUBIT_PATTERNS: List[re.Pattern] = [
    re.compile(r"n_qubits\s*=\s*(\d+)\b", re.IGNORECASE),
    re.compile(r"(\d+)\s*-?\s*qubits?\b", re.IGNORECASE),
    re.compile(r"(\d+)\s*-?\s*sites?\b", re.IGNORECASE),
    re.compile(r"(\d+)\s*-?\s*spins?\b", re.IGNORECASE),
    re.compile(r"\bn\s*=\s*(\d+)\b", re.IGNORECASE),
]

_PBC_RE = re.compile(r"\b(periodic|pbc|periodic[\s-]boundary)\b", re.IGNORECASE)
_OBC_RE = re.compile(r"\b(open|obc|open[\s-]boundary)\b", re.IGNORECASE)


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Public extraction helpers
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def normalize_text(text: str) -> str:
    """
    Lowercase and collapse horizontal whitespace.

    Hyphens are preserved so that downstream patterns can match both
    "4-qubit" and "4 qubit" forms.  Alias matching normalises hyphens
    separately when comparing against registry entries.
    """
    text = text.lower()
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_n_qubits(text: str) -> Optional[int]:
    """
    Return the integer qubit count from *text*, or ``None`` if not found.

    Recognised forms (case-insensitive, hyphens optional):
        ``4 qubits``, ``4-qubit``, ``4 sites``, ``4 spins``,
        ``N=4``, ``n = 4``, ``n_qubits=4``.

    *text* may be raw or already normalised.
    """
    for pat in _NQUBIT_PATTERNS:
        m = pat.search(text)
        if m:
            return int(m.group(1))
    return None


def extract_boundary(text: str) -> str:
    """
    Return ``"pbc"`` or ``"obc"`` based on keywords in *text*.

    Defaults to ``"obc"`` when neither keyword is found.

    *text* may be raw or already normalised.
    """
    if _PBC_RE.search(text):
        return "pbc"
    # OBC keyword present, or default
    return "obc"


def extract_numeric_params(text: str) -> Dict[str, float]:
    """
    Extract explicit ``PARAM = VALUE`` assignments from *text*.

    Supported parameters: J, h, hx, hz, delta.
    Matching is case-insensitive; returned keys use canonical casing.

    Examples
    --------
    >>> extract_numeric_params("J=1, hx=0.7, hz=0.3")
    {'J': 1.0, 'hx': 0.7, 'hz': 0.3}
    >>> extract_numeric_params("delta = 1.5, J = 2.0")
    {'delta': 1.5, 'J': 2.0}

    *text* may be raw or already normalised.
    """
    result: Dict[str, float] = {}
    for m in _PARAM_RE.finditer(text):
        raw_name = m.group(1).lower()
        canonical = _PARAM_CANONICAL.get(raw_name)
        if canonical is not None:
            result[canonical] = float(m.group(2))
    return result


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Internal detection helpers
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def _dehyphen(text: str) -> str:
    """Replace hyphens with spaces for fuzzy alias / keyword comparison."""
    return text.replace("-", " ")


def _alias_matches(norm_text: str) -> Dict[str, int]:
    """
    Return ``{canonical_name: best_alias_length}`` for every registry family
    whose aliases appear as substrings in *norm_text*.

    Comparison is done after replacing hyphens with spaces on both sides so
    that "transverse-field" in user text matches "transverse field" alias.
    """
    dh_text = _dehyphen(norm_text)
    best: Dict[str, int] = {}
    for name in get_supported_families():
        spec = get_family_spec(name)
        if spec is None:
            continue
        for alias in spec.aliases:
            dh_alias = _dehyphen(alias.lower())
            if dh_alias in dh_text:
                if len(dh_alias) > best.get(name, 0):
                    best[name] = len(dh_alias)
    return best


def _keyword_scores(norm_text: str) -> Dict[str, int]:
    """
    Return ``{canonical_name: score}`` for keyword-hint matches.

    Score = sum of character lengths of matched hints so that longer,
    more specific phrases outweigh short single-word hints.
    """
    dh_text = _dehyphen(norm_text)
    scores: Dict[str, int] = {}
    for name in get_supported_families():
        spec = get_family_spec(name)
        if spec is None:
            continue
        total = 0
        for hint in spec.keyword_hints:
            if _dehyphen(hint.lower()) in dh_text:
                total += len(hint)
        if total > 0:
            scores[name] = total
    return scores


def _disambiguate_ising(
    norm_text: str,
    params: Dict[str, float],
    candidates: List[str],
    warnings: List[str],
) -> Tuple[Optional[str], float]:
    """
    Resolve ambiguity between ``"tfim"`` and ``"ising_general"``.

    Uses explicit longitudinal-field keywords and extracted parameter names.
    Modifies *warnings* in place; returns ``(canonical_name, confidence)``.
    """
    has_hx = "hx" in params
    has_hz = "hz" in params
    has_longitudinal = bool(re.search(r"\b(longitudinal|general)\b", norm_text))

    if has_hx or has_hz or has_longitudinal:
        return "ising_general", 0.85

    # Only "h" or nothing — lean toward TFIM
    if "h" in params or re.search(r"\b(tfim|transverse)\b", norm_text):
        if "ising_general" in candidates:
            warnings.append(
                "Text mentions 'ising' without 'general' or longitudinal-field "
                "keywords; assuming TFIM. Add 'hx'/'hz' params or 'general'/'longitudinal' "
                "qualifier to select ising_general."
            )
        return "tfim", 0.75

    # Genuinely ambiguous — default to TFIM with a stronger warning
    warnings.append(
        "Cannot distinguish TFIM from general Ising without explicit param names "
        "(hx, hz) or a qualifier ('general', 'longitudinal'). Assuming TFIM."
    )
    return "tfim", 0.55


def _disambiguate_heisenberg(
    norm_text: str,
    params: Dict[str, float],
    warnings: List[str],
) -> Tuple[Optional[str], float]:
    """
    Resolve ambiguity between ``"heisenberg"`` and ``"xxz"``.

    If ``delta`` is present in *params*, or "anisotropic" / "xxz" / "delta"
    appear in *norm_text*, returns xxz; otherwise returns heisenberg.
    Modifies *warnings* in place; returns ``(canonical_name, confidence)``.
    """
    has_delta_param = "delta" in params
    has_aniso_text = bool(re.search(r"\b(anisotropic|xxz|delta)\b", norm_text))

    if has_delta_param or has_aniso_text:
        return "xxz", 0.90
    return "heisenberg", 0.90


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Public family detector
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def detect_family(
    text: str,
    params: Dict[str, float],
) -> Tuple[Optional[str], float, List[str]]:
    """
    Identify the Hamiltonian family from *text* and extracted *params*.

    Returns ``(canonical_name_or_None, confidence_in_0_1, warnings_list)``.

    Detection is conservative: when ambiguous the function returns a lower
    confidence and adds a human-readable warning rather than overclaiming.

    Detection stages (highest confidence first)
    -------------------------------------------
    1. **Alias match** (conf 0.85–0.95): any registered alias found as a
       substring in the normalised, dehyphenated text.
    2. **Keyword-hint scoring** (conf 0.55–0.75): weighted sum of keyword
       hint phrase lengths matched per family.
    3. **Parameter-name fallback** (conf 0.50–0.65): presence of ``delta``
       → xxz; ``hx``/``hz`` → ising_general; ``h`` alone → TFIM.
    4. **Failure** → ``(None, 0.0, [warning])``.

    Disambiguation rules are applied at every stage for the two overlapping
    pairs: {tfim, ising_general} and {heisenberg, xxz}.

    *text* may be raw or already normalised (normalization is idempotent).
    """
    norm = normalize_text(text)
    warnings: List[str] = []

    # ── Stage 1: alias matching ────────────────────────────────────────────────
    alias_hits = _alias_matches(norm)

    if alias_hits:
        candidates = sorted(alias_hits, key=lambda k: -alias_hits[k])

        ising_overlap = {"tfim", "ising_general"} & set(candidates)
        heis_overlap = {"heisenberg", "xxz"} & set(candidates)

        # Both ising-family candidates present — disambiguate
        if len(ising_overlap) >= 2:
            fam, conf = _disambiguate_ising(norm, params, list(ising_overlap), warnings)
            return fam, conf, warnings

        # Both heisenberg-family candidates present — disambiguate
        if len(heis_overlap) >= 2:
            fam, conf = _disambiguate_heisenberg(norm, params, warnings)
            return fam, conf, warnings

        # Single unambiguous match
        top = candidates[0]

        # A lone "heisenberg" alias still needs XXZ disambiguation
        if top == "heisenberg":
            fam, conf = _disambiguate_heisenberg(norm, params, warnings)
            return fam, conf, warnings

        # "tfim" alias matched but params suggest ising_general
        if top == "tfim" and ("hx" in params or "hz" in params):
            warnings.append(
                "hx/hz parameters found but text matches a TFIM alias; "
                "interpreting as ising_general."
            )
            return "ising_general", 0.85, warnings

        conf = 0.95
        return top, conf, warnings

    # ── Stage 2: keyword-hint scoring ─────────────────────────────────────────
    kw_scores = _keyword_scores(norm)

    if kw_scores:
        ranked = sorted(kw_scores, key=lambda k: -kw_scores[k])
        top = ranked[0]
        second_score = kw_scores[ranked[1]] if len(ranked) > 1 else 0
        margin = kw_scores[top] - second_score

        # Ising-pair disambiguation in keyword stage
        if "tfim" in ranked[:2] and "ising_general" in ranked[:2]:
            fam, conf = _disambiguate_ising(norm, params, ranked[:2], warnings)
            return fam, conf, warnings

        # Heisenberg-pair disambiguation in keyword stage
        if top in {"heisenberg", "xxz"} or (
            len(ranked) > 1 and ranked[1] in {"heisenberg", "xxz"}
            and top in {"heisenberg", "xxz"}
        ):
            fam, conf = _disambiguate_heisenberg(norm, params, warnings)
            return fam, min(conf, 0.75), warnings

        # Clear winner
        if margin >= 3 or len(ranked) == 1:
            conf = min(0.75, 0.50 + 0.03 * margin)
            return top, conf, warnings

        # Narrow margin — pick top but warn
        warnings.append(
            f"Family detection is ambiguous (top candidates: {ranked[:3]}). "
            f"Assuming '{top}'."
        )
        return top, 0.55, warnings

    # ── Stage 3: parameter-name fallback ──────────────────────────────────────
    if "delta" in params:
        warnings.append(
            "No family keywords found; 'delta' parameter suggests XXZ model."
        )
        return "xxz", 0.65, warnings

    if "hx" in params or "hz" in params:
        warnings.append(
            "No family keywords found; 'hx'/'hz' parameter(s) suggest general Ising model."
        )
        return "ising_general", 0.60, warnings

    if "h" in params:
        warnings.append(
            "No family keywords found; only 'h' parameter present — assuming TFIM. "
            "If this is a general Ising model, use 'hx'/'hz' parameter names."
        )
        return "tfim", 0.50, warnings

    # ── Stage 4: failure ──────────────────────────────────────────────────────
    warnings.append(
        "Cannot identify the Hamiltonian family from the provided description. "
        f"Supported families: {', '.join(get_supported_families())}."
    )
    return None, 0.0, warnings


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Main entry point
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def parse_hamiltonian_text(text: str) -> ParsedHamiltonian:
    """
    Parse a free-text Hamiltonian description into a ``ParsedHamiltonian``.

    This is the main public entry point for the NL interface layer.

    The function never raises on bad input — all issues are surfaced as
    ``warnings`` inside the returned object.  Callers should check
    ``result.supported`` and ``result.warnings`` before proceeding to
    the inference stage.

    Parameters
    ----------
    text : str
        Human-readable description, e.g.
        ``"4-qubit TFIM with J=1 and h=0.8"``.

    Returns
    -------
    ParsedHamiltonian
        Fields populated from what could be extracted.  ``family`` is
        ``None`` and ``supported=False`` when the family cannot be
        identified.
    """
    raw_text = text
    norm = normalize_text(text)

    n_qubits = extract_n_qubits(norm)
    boundary = extract_boundary(norm)
    params = extract_numeric_params(norm)

    family, confidence, det_warnings = detect_family(text, params)
    warnings: List[str] = list(det_warnings)

    # ── Qubit count ────────────────────────────────────────────────────────────
    if n_qubits is None:
        warnings.append("Qubit count not specified in the description.")

    # ── Boundary support ───────────────────────────────────────────────────────
    if boundary == "pbc":
        warnings.append(
            "Periodic boundary conditions (PBC) detected, but only OBC is currently "
            "supported by the pipeline. Set boundary to 'obc' to proceed."
        )

    # ── Missing required parameters ────────────────────────────────────────────
    if family is not None:
        spec = get_family_spec(family)
        if spec is not None:
            for p in spec.required_params:
                if p not in params:
                    warnings.append(
                        f"Required parameter '{p}' not found in description "
                        f"for family '{family}'."
                    )

    # ── Support flag ───────────────────────────────────────────────────────────
    # All four conditions must hold for the result to be actionable downstream:
    # known family, registered, OBC boundary, and all required params present.
    _missing_required: List[str] = []
    if family is not None:
        spec = get_family_spec(family)
        if spec is not None:
            _missing_required = [p for p in spec.required_params if p not in params]

    supported = (
        family is not None
        and is_supported_family(family)
        and boundary != "pbc"
        and len(_missing_required) == 0
    )

    return ParsedHamiltonian(
        raw_text=raw_text,
        family=family,
        params=params,
        n_qubits=n_qubits,
        boundary=boundary,
        supported=supported,
        confidence=confidence,
        warnings=warnings,
    )


__all__ = [
    "normalize_text",
    "extract_n_qubits",
    "extract_boundary",
    "extract_numeric_params",
    "detect_family",
    "parse_hamiltonian_text",
]
