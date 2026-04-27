"""
llm_interface.py — Optional LLM-assisted layer for the NL Hamiltonian interface.

Architecture rule
-----------------
The LLM may only assist with TWO things:
  1. Converting freer-form text into the existing ``ParsedHamiltonian`` schema
     (more flexible parsing than the rule-based ``nl_parser``).
  2. Lightly rewriting the deterministic ``ExplanationResult`` text into more
     natural prose.

The LLM must NOT predict any physics.  Every physical number comes from the
structured pipeline:
    parse → ParsedHamiltonian → backend → PropertyResult → report_generator
                                                         → ExplanationResult

LLM availability
----------------
The LLM layer is entirely optional.  If the ``anthropic`` package is not
installed, or no API key is configured, all functions fall back cleanly to the
rule-based deterministic system.  No exception propagates to the caller unless
they explicitly pass ``fallback=False``.

Provider
--------
Uses the Anthropic Messages API (``claude-haiku-4-5-20251001`` by default).
Configure via:
  - ``LLMConfig(api_key="sk-ant-...")`` passed directly, or
  - ``ANTHROPIC_API_KEY`` environment variable.

Public API
----------
LLMConfig                          — lightweight configuration dataclass.
LLMUnavailableError                — raised when the LLM cannot be reached
                                     and ``fallback=False``.

parse_hamiltonian_with_llm(text, config)
    → ParsedHamiltonian (LLM-assisted; falls back to rule-based parser).

rewrite_explanation_with_llm(explanation, config)
    → ExplanationResult with more natural prose; falls back to original.

analyze_hamiltonian_nl(text, backend, use_llm_parse, use_llm_rewrite, ...)
    → ExplanationResult — thin end-to-end wrapper with full fallback chain.
"""

from __future__ import annotations

import json
import os
import re
import warnings as _warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .nl_schema import ExplanationResult, ParsedHamiltonian, PropertyResult
from .family_registry import get_family_spec, get_supported_families


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Availability guard
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

try:
    import anthropic as _anthropic
    _ANTHROPIC_AVAILABLE = True
except ModuleNotFoundError:
    _anthropic = None          # type: ignore[assignment]
    _ANTHROPIC_AVAILABLE = False


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Configuration & errors
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

@dataclass
class LLMConfig:
    """
    Configuration for the optional LLM layer.

    Attributes
    ----------
    model : str
        Anthropic model ID to use for parsing and rewriting.
        Defaults to ``claude-haiku-4-5-20251001`` (fast, inexpensive).
    max_tokens : int
        Maximum tokens in the LLM response (default 512).
    temperature : float
        Sampling temperature (default 0.0 for deterministic output).
    api_key : str or None
        Anthropic API key.  If None, reads from the ``ANTHROPIC_API_KEY``
        environment variable.
    timeout : float
        Request timeout in seconds (default 30.0).
    """

    model: str = "claude-haiku-4-5-20251001"
    max_tokens: int = 512
    temperature: float = 0.0
    api_key: Optional[str] = None
    timeout: float = 30.0


class LLMUnavailableError(RuntimeError):
    """
    Raised when the LLM layer is unavailable and ``fallback=False``.

    Possible causes:
    - ``anthropic`` package not installed.
    - ``ANTHROPIC_API_KEY`` not set and no key in ``LLMConfig``.
    - API call failed (network error, quota exceeded, etc.).
    """


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Internal LLM call helper
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def _resolve_api_key(config: LLMConfig) -> str:
    """Return the API key from config or environment, or raise ``LLMUnavailableError``."""
    key = config.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise LLMUnavailableError(
            "No Anthropic API key configured. "
            "Set ANTHROPIC_API_KEY or pass LLMConfig(api_key='sk-ant-...')."
        )
    return key


def _call_llm(prompt: str, config: LLMConfig) -> str:
    """
    Call the Anthropic Messages API and return the text response.

    Raises
    ------
    LLMUnavailableError
        If the ``anthropic`` package is missing, the API key is absent, or
        the API call fails.
    """
    if not _ANTHROPIC_AVAILABLE:
        raise LLMUnavailableError(
            "The 'anthropic' package is not installed. "
            "Install it with: pip install anthropic"
        )

    api_key = _resolve_api_key(config)

    try:
        client = _anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=config.model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as exc:
        raise LLMUnavailableError(f"LLM API call failed: {exc}") from exc


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# JSON extraction helper
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def _extract_json(text: str) -> Dict[str, Any]:
    """
    Extract the first JSON object from *text*.

    The LLM may wrap its JSON in markdown fences (```json … ```) or plain text.
    This helper strips fences and then calls ``json.loads``.

    Raises ``ValueError`` if no valid JSON object is found.
    """
    # Strip markdown fences if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    # Try to find the first { ... } block.
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return json.loads(brace_match.group(0))

    raise ValueError(f"No JSON object found in LLM response: {text!r}")


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Parsing prompt & response handler
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def _build_parse_prompt(text: str) -> str:
    """
    Build the LLM prompt for Hamiltonian parsing.

    The prompt instructs the model to extract structured fields and return
    only a JSON object — no prose, no commentary.
    """
    families_block = "\n".join(
        f"  - {f}" for f in get_supported_families()
    )
    return f"""\
You are a physics assistant that extracts structured information from \
descriptions of quantum spin-chain Hamiltonians.

Supported Hamiltonian families and their required parameters:
  - tfim           : J (ZZ coupling), h (transverse field)
  - ising_general  : J (ZZ coupling), hx (transverse/X field), hz (longitudinal/Z field)
  - xxz            : J (exchange coupling), delta (ZZ anisotropy)
  - heisenberg     : J (exchange coupling)

Task: read the description below and return ONLY a single JSON object with \
these exact fields:
{{
  "family": "<canonical name from the list above, or null if unrecognised>",
  "n_qubits": <integer, or null if not specified>,
  "params": {{"<param_name>": <float>, ...}},
  "boundary": "obc" or "pbc"
}}

Rules:
- Return null for family if you are not confident.
- Include only the params you can identify from the text; do not invent values.
- Boundary defaults to "obc" unless the text explicitly says periodic or PBC.
- Return ONLY the JSON object, no explanation, no markdown prose.

Description: {text}"""


def _parse_llm_response_to_parsed_hamiltonian(
    raw_text: str,
    original_text: str,
) -> ParsedHamiltonian:
    """
    Parse the LLM's JSON response into a ``ParsedHamiltonian``.

    Validates extracted fields against the registry.  Sets ``supported``
    and ``confidence`` consistently with what the rule-based parser would do.

    Raises ``ValueError`` if the JSON cannot be decoded or is structurally invalid.
    """
    from .nl_parser import parse_hamiltonian_text  # local import to avoid circular

    data = _extract_json(raw_text)

    family: Optional[str]     = data.get("family")
    n_qubits: Optional[int]   = data.get("n_qubits")
    params_raw: Dict[str, Any] = data.get("params", {})
    boundary: str              = data.get("boundary", "obc")

    # Coerce and validate.
    if family is not None:
        family = str(family).strip().lower()
        spec = get_family_spec(family)
        if spec is None:
            family = None   # LLM hallucinated a family name

    params: Dict[str, float] = {}
    for k, v in params_raw.items():
        try:
            params[str(k)] = float(v)
        except (TypeError, ValueError):
            pass  # skip non-numeric values

    warnings: List[str] = []

    # Check required params if family is known.
    missing: List[str] = []
    if family is not None:
        spec = get_family_spec(family)
        if spec is not None:
            missing = [p for p in spec.required_params if p not in params]
            if missing:
                warnings.append(
                    f"LLM parsing: required parameters missing for '{family}': "
                    + ", ".join(missing)
                )

    if family is None:
        warnings.append("LLM parsing: Hamiltonian family not recognised.")

    if n_qubits is not None:
        try:
            n_qubits = int(n_qubits)
            if n_qubits <= 0:
                warnings.append(f"LLM parsing: n_qubits={n_qubits} is not positive; ignored.")
                n_qubits = None
        except (TypeError, ValueError):
            n_qubits = None

    supported = (
        family is not None
        and boundary != "pbc"
        and len(missing) == 0
        and n_qubits is not None
    )

    # Confidence: high if family + all params + n_qubits are present; lower otherwise.
    if supported:
        confidence = 0.90
    elif family is not None and len(missing) == 0:
        confidence = 0.70
    elif family is not None:
        confidence = 0.40
    else:
        confidence = 0.10

    return ParsedHamiltonian(
        raw_text=original_text,
        family=family,
        params=params,
        n_qubits=n_qubits,
        boundary=boundary,
        supported=supported,
        confidence=confidence,
        warnings=warnings,
    )


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Rewriting prompt & response handler
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def _build_rewrite_prompt(explanation: ExplanationResult) -> str:
    """Build the LLM prompt for explanation rewriting."""
    return f"""\
You are a physics communicator. Your job is to lightly rewrite a \
technical physics summary into clear, natural prose that a non-specialist \
can follow, while preserving every physical fact, number, and caveat exactly.

Rules:
- Do NOT change, omit, or embellish any numerical values.
- Do NOT add physics claims that were not in the original.
- Do NOT remove warnings or caveats.
- Keep both outputs concise (short_summary: 1–2 sentences; \
detailed_summary: a short paragraph).
- Return ONLY a JSON object, no markdown, no commentary.

Return format:
{{
  "short_summary": "<rewritten short summary>",
  "detailed_summary": "<rewritten detailed summary>"
}}

Original short summary:
{explanation.short_summary}

Original detailed summary:
{explanation.detailed_summary}"""


def _apply_rewrite_response(
    raw_text: str,
    original: ExplanationResult,
) -> ExplanationResult:
    """
    Parse the LLM's rewrite JSON and return a new ``ExplanationResult``.

    Warnings are always carried over unchanged from *original*.
    Falls back to *original* if the JSON cannot be parsed.
    """
    data = _extract_json(raw_text)
    short    = str(data.get("short_summary",    original.short_summary)).strip()
    detailed = str(data.get("detailed_summary", original.detailed_summary)).strip()

    if not short:
        short = original.short_summary
    if not detailed:
        detailed = original.detailed_summary

    return ExplanationResult(
        short_summary=short,
        detailed_summary=detailed,
        warnings=list(original.warnings),   # always preserved
    )


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Public: parse_hamiltonian_with_llm
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def parse_hamiltonian_with_llm(
    text: str,
    config: Optional[LLMConfig] = None,
    fallback: bool = True,
) -> ParsedHamiltonian:
    """
    Parse *text* into a ``ParsedHamiltonian`` with optional LLM assistance.

    The LLM receives a structured prompt asking for JSON with ``family``,
    ``n_qubits``, ``params``, and ``boundary`` fields.  The response is
    validated against the family registry before being returned.

    Fallback behaviour
    ------------------
    If ``fallback=True`` (default), any LLM failure — unavailable package,
    missing API key, API error, malformed JSON — silently falls back to the
    rule-based ``parse_hamiltonian_text``.

    If ``fallback=False``, raises ``LLMUnavailableError`` on LLM problems or
    ``ValueError`` on malformed responses.

    Parameters
    ----------
    text : str
        Free-text Hamiltonian description.
    config : LLMConfig or None
        LLM configuration.  Defaults are used when None.
    fallback : bool
        Whether to fall back to rule-based parsing on failure (default True).

    Returns
    -------
    ParsedHamiltonian
        Parsed object, sourced from LLM or rule-based fallback.
    """
    from .nl_parser import parse_hamiltonian_text

    cfg = config or LLMConfig()

    try:
        prompt   = _build_parse_prompt(text)
        raw      = _call_llm(prompt, cfg)
        parsed   = _parse_llm_response_to_parsed_hamiltonian(raw, text)
        parsed.warnings.insert(0, "LLM-assisted parsing used.")
        return parsed

    except LLMUnavailableError as exc:
        if not fallback:
            raise
        _warnings.warn(
            f"LLM parsing unavailable ({exc}); falling back to rule-based parser.",
            UserWarning,
            stacklevel=2,
        )
        return parse_hamiltonian_text(text)

    except Exception as exc:
        if not fallback:
            raise LLMUnavailableError(f"LLM parsing failed: {exc}") from exc
        _warnings.warn(
            f"LLM parsing failed ({exc}); falling back to rule-based parser.",
            UserWarning,
            stacklevel=2,
        )
        return parse_hamiltonian_text(text)


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Public: rewrite_explanation_with_llm
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def rewrite_explanation_with_llm(
    explanation: ExplanationResult,
    config: Optional[LLMConfig] = None,
    fallback: bool = True,
) -> ExplanationResult:
    """
    Optionally rewrite *explanation* text into more natural prose.

    The LLM is instructed to preserve all physical facts and numbers, all
    caveats, and all warnings.  The ``warnings`` field of the returned object
    is always identical to the input's — it is never touched by the LLM.

    Fallback behaviour
    ------------------
    On any LLM failure, returns *explanation* unchanged when ``fallback=True``
    (default).  Raises ``LLMUnavailableError`` when ``fallback=False``.

    Parameters
    ----------
    explanation : ExplanationResult
        Deterministic explanation produced by ``make_explanation_result``.
    config : LLMConfig or None
        LLM configuration.  Defaults are used when None.
    fallback : bool
        Whether to return the original explanation on LLM failure (default True).

    Returns
    -------
    ExplanationResult
        Rewritten explanation, or *explanation* unchanged on failure.
    """
    cfg = config or LLMConfig()

    try:
        prompt   = _build_rewrite_prompt(explanation)
        raw      = _call_llm(prompt, cfg)
        rewritten = _apply_rewrite_response(raw, explanation)
        # Tag the result so callers can tell it was LLM-rewritten.
        rewritten.warnings = list(explanation.warnings) + [
            "Short and detailed summaries were lightly rewritten by an LLM for "
            "readability. All physical values come from the deterministic backend."
        ]
        return rewritten

    except LLMUnavailableError as exc:
        if not fallback:
            raise
        _warnings.warn(
            f"LLM rewriting unavailable ({exc}); keeping deterministic explanation.",
            UserWarning,
            stacklevel=2,
        )
        return explanation

    except Exception as exc:
        if not fallback:
            raise LLMUnavailableError(f"LLM rewriting failed: {exc}") from exc
        _warnings.warn(
            f"LLM rewriting failed ({exc}); keeping deterministic explanation.",
            UserWarning,
            stacklevel=2,
        )
        return explanation


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Public: analyze_hamiltonian_nl  (thin end-to-end wrapper)
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def analyze_hamiltonian_nl(
    text: str,
    backend: str = "exact",
    use_llm_parse: bool = True,
    use_llm_rewrite: bool = True,
    llm_config: Optional[LLMConfig] = None,
    **kwargs,
) -> ExplanationResult:
    """
    End-to-end NL Hamiltonian analysis with optional LLM assistance.

    Pipeline
    --------
    1. Parse *text* → ``ParsedHamiltonian``
       - LLM-assisted if ``use_llm_parse=True`` and LLM is available.
       - Falls back to rule-based ``parse_hamiltonian_text`` automatically.
    2. Run the physics backend → ``PropertyResult``
       - ``backend="exact"``     : exact diagonalization.
       - ``backend="shadowgpt"`` : learned ShadowGPT model.
    3. Generate deterministic explanation → ``ExplanationResult``
    4. Optionally rewrite explanation with LLM
       - Only if ``use_llm_rewrite=True`` and LLM is available.
       - Falls back to deterministic explanation automatically.

    Parameters
    ----------
    text : str
        Free-text Hamiltonian description.
    backend : {"exact", "shadowgpt"}
        Which physics backend to use (default ``"exact"``).
    use_llm_parse : bool
        Attempt LLM-assisted parsing (default True; falls back if unavailable).
    use_llm_rewrite : bool
        Attempt LLM-assisted rewriting of the final explanation
        (default True; falls back if unavailable).
    llm_config : LLMConfig or None
        Shared LLM configuration for both parse and rewrite steps.
    **kwargs
        Forwarded to the backend (e.g. ``checkpoint_dir``, ``n_shadows``
        for the ShadowGPT backend).

    Returns
    -------
    ExplanationResult
        Populated with short summary, detailed summary, and warnings.

    Raises
    ------
    ValueError
        If the parsed Hamiltonian is not supported by the requested backend.
    FileNotFoundError
        If ``backend="shadowgpt"`` and the checkpoint directory is not found.
    """
    from .nl_parser import parse_hamiltonian_text
    from .inference_engine import evaluate_exact, evaluate_with_shadowgpt
    from .report_generator import make_explanation_result

    # ── Step 1: Parse ─────────────────────────────────────────────────────────
    if use_llm_parse:
        parsed = parse_hamiltonian_with_llm(text, config=llm_config, fallback=True)
    else:
        parsed = parse_hamiltonian_text(text)

    # ── Step 2: Backend ───────────────────────────────────────────────────────
    if backend == "exact":
        result = evaluate_exact(parsed)
    elif backend == "shadowgpt":
        result = evaluate_with_shadowgpt(parsed, **kwargs)
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Choose 'exact' or 'shadowgpt'."
        )

    # ── Step 3: Deterministic explanation ─────────────────────────────────────
    explanation = make_explanation_result(result)

    # ── Step 4: Optional LLM rewrite ──────────────────────────────────────────
    if use_llm_rewrite:
        explanation = rewrite_explanation_with_llm(
            explanation, config=llm_config, fallback=True
        )

    return explanation


__all__ = [
    "LLMConfig",
    "LLMUnavailableError",
    "parse_hamiltonian_with_llm",
    "rewrite_explanation_with_llm",
    "analyze_hamiltonian_nl",
]
