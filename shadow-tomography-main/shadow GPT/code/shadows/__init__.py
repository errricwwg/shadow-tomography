"""
Classical shadows — ShadowGPT generative pipeline.

Learning objective: train a decoder-only GPT to model p(b | P, g) over
classical shadow measurements, then generate synthetic shadows to estimate
physical observables downstream.

Exports
-------
Generative model:
    ShadowGPT, GPTConfig, create_gpt_from_tokenizer

Generative dataset:
    GenerativeShadowDataset

Tokenizer:
    ShadowTokenizer, create_default_tokenizer, create_generative_tokenizer
    add_generative_tokens, build_generative_sequence, decode_generative_outcomes
    add_hamiltonian_conditioning, encode_hamiltonian_prefix
    add_multi_hamiltonian_conditioning, encode_multi_hamiltonian_prefix

Data collection & processing (infrastructure):
    ShadowCollector, ShadowMeasurement, collect_shadows_from_state
    ShadowProcessor, process_shadow_data
    ShadowConfig, create_default_config, create_pauli_config,
    create_clifford_config, create_custom_config

Natural-language interface (Step 1 — schema & registry):
    ParsedHamiltonian, PropertyResult, ExplanationResult
    FamilySpec, get_supported_families, get_family_spec, is_supported_family

Natural-language interface (Step 2 — rule-based parser):
    parse_hamiltonian_text

Natural-language interface (Step 3 — exact inference backend):
    evaluate_exact, parse_and_evaluate_exact

Natural-language interface (Step 4 — learned ShadowGPT inference backend):
    evaluate_with_shadowgpt, parse_and_evaluate_with_shadowgpt

Natural-language interface (Step 5 — report / explanation layer):
    make_short_summary, make_detailed_summary, make_explanation_result
    explain_exact, explain_with_shadowgpt

Natural-language interface (Step 6 — optional LLM-assisted layer):
    LLMConfig, LLMUnavailableError
    parse_hamiltonian_with_llm, rewrite_explanation_with_llm
    analyze_hamiltonian_nl
"""

from .config import (
    ShadowConfig,
    create_clifford_config,
    create_custom_config,
    create_default_config,
    create_pauli_config,
)
from .collector import (
    PAULI_NAMES,
    SINGLE_QUBIT_CLIFFORDS,
    ShadowCollector,
    ShadowMeasurement,
    collect_shadows_from_state,
)
from .processor import ShadowProcessor, process_shadow_data
from .tokenization import (
    ShadowTokenizer,
    TokenizationConfig,
    create_default_tokenizer,
    create_generative_tokenizer,
    add_hamiltonian_conditioning,
    encode_hamiltonian_prefix,
    add_multi_hamiltonian_conditioning,
    encode_multi_hamiltonian_prefix,
    add_generative_tokens,
    build_generative_sequence,
    decode_generative_outcomes,
)

from .nl_schema import ParsedHamiltonian, PropertyResult, ExplanationResult
from .family_registry import (
    FamilySpec,
    get_supported_families,
    get_family_spec,
    is_supported_family,
)
from .nl_parser import parse_hamiltonian_text
from .inference_engine import (
    evaluate_exact,
    parse_and_evaluate_exact,
    evaluate_with_shadowgpt,
    parse_and_evaluate_with_shadowgpt,
)
from .report_generator import (
    make_short_summary,
    make_detailed_summary,
    make_explanation_result,
    explain_exact,
    explain_with_shadowgpt,
)
from .llm_interface import (
    LLMConfig,
    LLMUnavailableError,
    parse_hamiltonian_with_llm,
    rewrite_explanation_with_llm,
    analyze_hamiltonian_nl,
)

try:
    from .datasets import GenerativeShadowDataset
    from .model import GPTConfig, ShadowGPT, create_gpt_from_tokenizer
    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    GenerativeShadowDataset = None    # type: ignore[assignment]
    GPTConfig = None                  # type: ignore[assignment]
    ShadowGPT = None                  # type: ignore[assignment]
    create_gpt_from_tokenizer = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

__all__ = [
    # Config
    "ShadowConfig",
    "create_default_config",
    "create_pauli_config",
    "create_clifford_config",
    "create_custom_config",
    # Collector
    "ShadowCollector",
    "ShadowMeasurement",
    "SINGLE_QUBIT_CLIFFORDS",
    "PAULI_NAMES",
    "collect_shadows_from_state",
    # Processor
    "ShadowProcessor",
    "process_shadow_data",
    # NL interface — Step 1 (schema & registry)
    "ParsedHamiltonian",
    "PropertyResult",
    "ExplanationResult",
    "FamilySpec",
    "get_supported_families",
    "get_family_spec",
    "is_supported_family",
    # NL interface — Step 2 (parser)
    "parse_hamiltonian_text",
    # NL interface — Step 3 (exact inference)
    "evaluate_exact",
    "parse_and_evaluate_exact",
    # NL interface — Step 4 (learned ShadowGPT inference)
    "evaluate_with_shadowgpt",
    "parse_and_evaluate_with_shadowgpt",
    # NL interface — Step 5 (report / explanation layer)
    "make_short_summary",
    "make_detailed_summary",
    "make_explanation_result",
    "explain_exact",
    "explain_with_shadowgpt",
    # NL interface — Step 6 (optional LLM-assisted layer)
    "LLMConfig",
    "LLMUnavailableError",
    "parse_hamiltonian_with_llm",
    "rewrite_explanation_with_llm",
    "analyze_hamiltonian_nl",
    # Tokenizer
    "ShadowTokenizer",
    "TokenizationConfig",
    "create_default_tokenizer",
    "create_generative_tokenizer",
    "add_hamiltonian_conditioning",
    "encode_hamiltonian_prefix",
    "add_multi_hamiltonian_conditioning",
    "encode_multi_hamiltonian_prefix",
    "add_generative_tokens",
    "build_generative_sequence",
    "decode_generative_outcomes",
]

if _TORCH_AVAILABLE:
    __all__ += [
        "GenerativeShadowDataset",
        "GPTConfig",
        "ShadowGPT",
        "create_gpt_from_tokenizer",
    ]
