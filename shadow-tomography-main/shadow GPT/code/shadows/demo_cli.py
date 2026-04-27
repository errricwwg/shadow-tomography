"""
demo_cli.py — Command-line demo for the ShadowGPT natural-language Hamiltonian interface.

Runs the full NL pipeline from terminal:
    parse → physics backend → report → (optional LLM rewrite)

Usage examples
--------------
# Exact backend (default):
python demo_cli.py --text "4-qubit TFIM with J=1 and h=0.8"

# Show full backend notes:
python demo_cli.py --text "4-qubit XXZ with J=1 and delta=0.5" --show-notes

# With LLM-assisted parsing and rewriting (requires ANTHROPIC_API_KEY):
python demo_cli.py --text "a four-site transverse Ising chain, J=1, h=0.6" \\
    --use-llm-parse --use-llm-rewrite

# ShadowGPT learned backend:
python demo_cli.py --text "4-qubit TFIM J=1 h=0.8" --backend shadowgpt \\
    --checkpoint-dir /path/to/checkpoints --n-shadows 300 --seed 42

# Interactive mode (loop until empty input / quit):
python demo_cli.py --backend exact --interactive
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from typing import Optional

# Ensure the parent directory (code root) is on sys.path so that
# `import shadows` works whether the script is run from the code root
# or from inside the shadows/ package directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Output helpers
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

_WIDTH = 65
_INDENT = "  "


def _header(title: str) -> None:
    print(f"\n{'=' * _WIDTH}")
    print(f"  {title}")
    print(f"{'=' * _WIDTH}")


def _field(label: str, value) -> None:
    label_str = f"{label}:"
    if value is None:
        print(f"{_INDENT}{label_str:<20} (not available)")
    elif isinstance(value, float):
        print(f"{_INDENT}{label_str:<20} {value:.6f}")
    elif isinstance(value, list) and len(value) == 0:
        print(f"{_INDENT}{label_str:<20} (none)")
    elif isinstance(value, list):
        print(f"{_INDENT}{label_str}")
        for item in value:
            wrapped = textwrap.fill(
                str(item),
                width=_WIDTH - 4,
                initial_indent=_INDENT * 2 + "- ",
                subsequent_indent=_INDENT * 2 + "  ",
            )
            print(wrapped)
    else:
        print(f"{_INDENT}{label_str:<20} {value}")


def _separator() -> None:
    print(f"{'─' * _WIDTH}")


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Section printers
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def _print_parsed(parsed) -> None:
    _header("Parsed Hamiltonian")
    _field("family",     parsed.family     or "(not identified)")
    _field("n_qubits",   parsed.n_qubits)
    _field("boundary",   parsed.boundary)
    _field("params",     ", ".join(f"{k}={v:.3g}" for k, v in sorted(parsed.params.items()))
                          or "(none)")
    _field("supported",  parsed.supported)
    _field("confidence", parsed.confidence)
    _field("warnings",   parsed.warnings)


def _print_result(result) -> None:
    _header("Physics Result")
    _field("family",        result.family)
    _field("n_qubits",      result.n_qubits)
    _field("params",        ", ".join(f"{k}={v:.3g}" for k, v in sorted(result.params.items())))
    _separator()
    _field("energy",        result.energy)
    _field("magnetization", result.magnetization)
    _field("correlations",  result.correlations)
    _field("renyi2_entropy",result.renyi2_entropy)


def _print_notes(result) -> None:
    _header("Backend Notes")
    if not result.notes:
        print(f"{_INDENT}(none)")
    else:
        for note in result.notes:
            wrapped = textwrap.fill(
                note,
                width=_WIDTH - 2,
                initial_indent=_INDENT,
                subsequent_indent=_INDENT + "  ",
            )
            print(wrapped)


def _print_explanation(expl) -> None:
    _header("Explanation")

    print(f"\n{_INDENT}Short summary:")
    wrapped = textwrap.fill(
        expl.short_summary,
        width=_WIDTH - 4,
        initial_indent=_INDENT * 2,
        subsequent_indent=_INDENT * 2,
    )
    print(wrapped)

    print(f"\n{_INDENT}Detailed summary:")
    for sentence in expl.detailed_summary.split(". "):
        sentence = sentence.strip().rstrip(".")
        if not sentence:
            continue
        wrapped = textwrap.fill(
            sentence + ".",
            width=_WIDTH - 4,
            initial_indent=_INDENT * 2,
            subsequent_indent=_INDENT * 2 + "  ",
        )
        print(wrapped)

    print(f"\n{_INDENT}Warnings:")
    if not expl.warnings:
        print(f"{_INDENT * 2}(none)")
    else:
        for w in expl.warnings:
            wrapped = textwrap.fill(
                w,
                width=_WIDTH - 4,
                initial_indent=_INDENT * 2 + "- ",
                subsequent_indent=_INDENT * 2 + "  ",
            )
            print(wrapped)


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Core pipeline
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def run_pipeline(text: str, args: argparse.Namespace) -> int:
    """
    Execute the full NL pipeline for *text* and print results.

    Returns 0 on success, 1 on error.
    """
    import warnings as _w

    # Suppress RuntimeWarnings from hamiltonians.py (existing code, not our bug).
    _w.filterwarnings("ignore", category=RuntimeWarning)

    from shadows.nl_parser import parse_hamiltonian_text
    from shadows.inference_engine import evaluate_exact, evaluate_with_shadowgpt
    from shadows.report_generator import make_explanation_result
    from shadows.llm_interface import (
        parse_hamiltonian_with_llm,
        rewrite_explanation_with_llm,
    )

    print(f"\n{'─' * _WIDTH}")
    print(f"  Input: {text!r}")
    print(f"{'─' * _WIDTH}")

    # ── Step 1: Parse ─────────────────────────────────────────────────────────
    try:
        if args.use_llm_parse:
            parsed = parse_hamiltonian_with_llm(text, fallback=True)
        else:
            parsed = parse_hamiltonian_text(text)
    except Exception as exc:
        print(f"\n[ERROR] Parsing failed: {exc}", file=sys.stderr)
        return 1

    _print_parsed(parsed)

    if not parsed.supported:
        issues = []
        if parsed.family is None:
            issues.append("Hamiltonian family not recognised.")
        if parsed.n_qubits is None:
            issues.append("Number of qubits not found.")
        if parsed.boundary == "pbc":
            issues.append("Periodic boundary conditions are not supported.")
        from shadows.family_registry import get_family_spec
        if parsed.family is not None:
            spec = get_family_spec(parsed.family)
            if spec:
                missing = [p for p in spec.required_params if p not in parsed.params]
                if missing:
                    issues.append(
                        f"Required parameters missing: {', '.join(missing)}."
                    )
        msg = "  ".join(issues) if issues else "Parsed Hamiltonian is not supported."
        print(f"\n[ERROR] Cannot run backend: {msg}", file=sys.stderr)
        print("  Tip: try a description like: '4-qubit TFIM with J=1 and h=0.8'",
              file=sys.stderr)
        return 1

    # ── Step 2: Backend ───────────────────────────────────────────────────────
    try:
        if args.backend == "exact":
            result = evaluate_exact(parsed)
        elif args.backend == "shadowgpt":
            kwargs = {}
            if args.checkpoint_dir:
                kwargs["checkpoint_dir"] = args.checkpoint_dir
            if args.device:
                kwargs["device"] = args.device
            if args.seed is not None:
                kwargs["seed"] = args.seed
            result = evaluate_with_shadowgpt(
                parsed,
                n_shadows=args.n_shadows,
                temperature=args.temperature,
                **kwargs,
            )
        else:
            print(f"[ERROR] Unknown backend: {args.backend!r}", file=sys.stderr)
            return 1
    except FileNotFoundError as exc:
        print(f"\n[ERROR] Checkpoint not found: {exc}", file=sys.stderr)
        print("  Tip: train a ShadowGPT model and set --checkpoint-dir.",
              file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\n[ERROR] Backend failed: {exc}", file=sys.stderr)
        return 1

    _print_result(result)

    if args.show_notes:
        _print_notes(result)

    # ── Step 3: Report ────────────────────────────────────────────────────────
    try:
        explanation = make_explanation_result(result)
    except Exception as exc:
        print(f"\n[ERROR] Report generation failed: {exc}", file=sys.stderr)
        return 1

    # ── Step 4: Optional LLM rewrite ──────────────────────────────────────────
    if args.use_llm_rewrite:
        try:
            explanation = rewrite_explanation_with_llm(explanation, fallback=True)
        except Exception as exc:
            # fallback=True should not raise, but be safe
            _w.warn(f"LLM rewrite step raised unexpectedly: {exc}")

    _print_explanation(explanation)
    return 0


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Interactive mode
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def run_interactive(args: argparse.Namespace) -> None:
    print(f"\n{'=' * _WIDTH}")
    print("  ShadowGPT NL Interface — Interactive Mode")
    print(f"  Backend: {args.backend}  |  LLM parse: {args.use_llm_parse}"
          f"  |  LLM rewrite: {args.use_llm_rewrite}")
    print(f"  Type a Hamiltonian description, or 'quit' / empty line to exit.")
    print(f"{'=' * _WIDTH}")

    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not text or text.lower() in ("quit", "exit", "q"):
            print("Exiting.")
            break

        run_pipeline(text, args)


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Argument parser
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="demo_cli.py",
        description=(
            "Natural-language Hamiltonian analysis via the ShadowGPT pipeline.\n"
            "Parses a free-text description, runs physics (exact or ShadowGPT backend),\n"
            "and prints a human-readable explanation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python demo_cli.py --text "4-qubit TFIM J=1 h=0.8"
              python demo_cli.py --text "6-qubit XXZ delta=0.5 J=1" --show-notes
              python demo_cli.py --text "4-qubit Heisenberg J=1" --use-llm-parse
              python demo_cli.py --backend shadowgpt --text "4-qubit TFIM J=1 h=0.8" \\
                  --checkpoint-dir /ckpts --n-shadows 300 --seed 42
              python demo_cli.py --interactive --backend exact
        """),
    )

    # ── Input ──────────────────────────────────────────────────────────────────
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text", "-t",
        metavar="TEXT",
        help="Free-text Hamiltonian description (e.g. '4-qubit TFIM with J=1 h=0.8').",
    )
    input_group.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Prompt for Hamiltonian descriptions in a loop.",
    )

    # ── Backend ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--backend", "-b",
        choices=["exact", "shadowgpt"],
        default="exact",
        help="Physics backend: 'exact' (default) or 'shadowgpt'.",
    )

    # ── LLM options ───────────────────────────────────────────────────────────
    p.add_argument(
        "--use-llm-parse",
        action="store_true",
        default=False,
        help=(
            "Use LLM-assisted parsing (requires ANTHROPIC_API_KEY). "
            "Falls back to rule-based parser if unavailable."
        ),
    )
    p.add_argument(
        "--use-llm-rewrite",
        action="store_true",
        default=False,
        help=(
            "Use LLM to rewrite the final explanation (requires ANTHROPIC_API_KEY). "
            "Falls back to deterministic explanation if unavailable."
        ),
    )

    # ── ShadowGPT backend options ─────────────────────────────────────────────
    sgpt = p.add_argument_group("ShadowGPT backend options (--backend shadowgpt)")
    sgpt.add_argument(
        "--checkpoint-dir",
        metavar="DIR",
        default=None,
        help=(
            "Root directory of per-family checkpoints "
            "({DIR}/{family}/best_gpt.pt + tokenizer_gpt.json). "
            "Overrides SHADOWGPT_CHECKPOINTS_DIR env var."
        ),
    )
    sgpt.add_argument(
        "--n-shadows",
        type=int,
        default=200,
        metavar="N",
        help="Number of synthetic shadow measurements to generate (default: 200).",
    )
    sgpt.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        metavar="T",
        help="Sampling temperature for autoregressive generation (default: 1.0).",
    )
    sgpt.add_argument(
        "--device",
        metavar="DEVICE",
        default=None,
        help="Torch device string, e.g. 'cpu', 'cuda', 'cuda:0'. Auto-detected if omitted.",
    )
    sgpt.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="SEED",
        help="Random seed for reproducible shadow generation.",
    )

    # ── Output options ─────────────────────────────────────────────────────────
    p.add_argument(
        "--show-notes",
        action="store_true",
        default=False,
        help="Print detailed backend notes from PropertyResult.",
    )

    return p


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Entry point
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.interactive:
        run_interactive(args)
        return 0
    else:
        return run_pipeline(args.text, args)


if __name__ == "__main__":
    sys.exit(main())
