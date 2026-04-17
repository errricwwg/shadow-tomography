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
