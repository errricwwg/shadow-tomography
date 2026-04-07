"""
Classical shadows module for quantum state tomography and property estimation.

This module provides:
- Classical shadow data collection from quantum states
- Shadow data processing and property estimation
- Tokenization for language model training
- Dataset management for PyTorch
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
from .tokenization import ShadowTokenizer, create_default_tokenizer

try:
    from .datasets import ShadowDataModule, ShadowDataset, create_data_module
except ModuleNotFoundError:
    ShadowDataModule = None  # type: ignore[assignment]
    ShadowDataset = None  # type: ignore[assignment]
    create_data_module = None  # type: ignore[assignment]

__all__ = [
    "ShadowConfig",
    "create_default_config",
    "create_pauli_config",
    "create_clifford_config",
    "create_custom_config",
    "ShadowCollector",
    "ShadowMeasurement",
    "SINGLE_QUBIT_CLIFFORDS",
    "PAULI_NAMES",
    "collect_shadows_from_state",
    "ShadowProcessor",
    "process_shadow_data",
    "ShadowTokenizer",
    "create_default_tokenizer",
]

if ShadowDataset is not None:
    __all__.extend([
        "ShadowDataset",
        "ShadowDataModule",
        "create_data_module",
    ])
