"""
Configuration classes for classical shadow protocols.

This module defines the ShadowConfig dataclass that contains all parameters
needed for classical shadow data collection and processing.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple
import numpy as np


@dataclass
class ShadowConfig:
    """
    Configuration for classical shadow data collection and processing.

    Measurement basis modes
    -----------------------
    "random":
        Each qubit is measured in a uniformly random Pauli basis (X, Y, or Z).
        No extra fields required.

    "pauli":
        Each qubit is measured in a Pauli basis sampled according to
        ``pauli_weights`` (weights for X, Y, Z respectively).
        Defaults to uniform [1/3, 1/3, 1/3] when not specified.

    "clifford":
        Each qubit independently receives one of the 24 single-qubit Clifford
        gates chosen uniformly at random, then is measured in the Z-basis.
        This is a *local* random Clifford protocol — not a global Clifford
        circuit. ``clifford_depth`` is reserved for future use (e.g. a global
        Clifford circuit implementation) and has no effect currently.

    "custom":
        The user supplies a ``custom_sampler`` callable with signature::

            sampler(n_qubits: int, rng: np.random.Generator)
                -> tuple[np.ndarray, list[np.ndarray]]

        It must return ``(basis_labels, unitaries)`` where:

        - ``basis_labels`` — integer array of shape ``(n_qubits,)`` labelling
          the chosen basis per qubit (the meaning is user-defined).
        - ``unitaries`` — list of ``n_qubits`` 2×2 complex unitary matrices.
          The collector applies ``unitaries[i]`` to qubit ``i`` and measures
          in the computational (Z) basis, equivalent to measuring qubit ``i``
          in the basis defined by ``unitaries[i]†``.

        In custom mode the collector also stores these per-shot unitaries
        inside each ``ShadowMeasurement`` and persists them to disk so the
        processor can later evaluate estimators exactly.

        Example — measuring every qubit in the X/Y/Z Hadamard-rotated basis::

            from shadows.collector import _PAULI_ROTATIONS

            def my_sampler(n_qubits, rng):
                basis = rng.integers(0, 3, size=n_qubits)
                unitaries = [_PAULI_ROTATIONS[b] for b in basis]
                return basis, unitaries

            config = ShadowConfig(
                n_qubits=4,
                measurement_basis="custom",
                custom_sampler=my_sampler,
            )
    """

    # ── Core parameters ───────────────────────────────────────────────────────

    n_qubits: int
    n_shadows: int = 1000

    # ── Measurement settings ──────────────────────────────────────────────────

    measurement_basis: str = "random"  # "random" | "pauli" | "clifford" | "custom"

    # "pauli" mode: weights for [X, Y, Z] — normalised automatically.
    pauli_weights: Optional[List[float]] = None

    # "clifford" mode: reserved for a future global Clifford implementation.
    # Not used by the current local single-qubit Clifford sampler.
    clifford_depth: int = 1

    # "custom" mode: callable provided by the user (see class docstring).
    custom_sampler: Optional[Callable] = None

    # ── Reproducibility ───────────────────────────────────────────────────────

    # Seed for the collector's local RNG (np.random.default_rng).
    # Does NOT set the global numpy seed.
    seed: Optional[int] = None

    # ── Parallel processing (used by external orchestration, not collector) ───

    parallel: bool = True
    n_jobs: int = -1

    # ── Processing parameters ─────────────────────────────────────────────────

    median_of_means: bool = True
    n_means: int = 10
    n_samples_per_mean: int = 100

    # ── Property estimation ───────────────────────────────────────────────────

    observables: Optional[List[str]] = None
    correlation_length: int = 2
    renyi_entropy: bool = True

    # ── Data storage ──────────────────────────────────────────────────────────

    save_shadows: bool = True
    save_observables: bool = True
    output_dir: str = "./shadow_data"

    # ─────────────────────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialisation."""
        if self.n_qubits <= 0:
            raise ValueError("n_qubits must be positive.")

        if self.n_shadows <= 0:
            raise ValueError("n_shadows must be positive.")

        valid_bases = ("random", "pauli", "clifford", "custom")
        if self.measurement_basis not in valid_bases:
            raise ValueError(
                f"Unknown measurement_basis {self.measurement_basis!r}. "
                f"Must be one of: {valid_bases}."
            )

        if self.measurement_basis == "pauli":
            if self.pauli_weights is None:
                self.pauli_weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
            weights = np.asarray(self.pauli_weights, dtype=float)
            if weights.shape != (3,) or np.any(weights < 0):
                raise ValueError(
                    "pauli_weights must be a non-negative list of length 3 "
                    "[w_X, w_Y, w_Z]."
                )
            if not np.isclose(weights.sum(), 1.0, atol=1e-8):
                # Normalise silently — common convenience.
                self.pauli_weights = list(weights / weights.sum())

        if self.measurement_basis == "custom" and self.custom_sampler is None:
            raise ValueError(
                "measurement_basis='custom' requires a 'custom_sampler' callable.\n"
                "Expected signature:\n"
                "    sampler(n_qubits: int, rng: np.random.Generator)\n"
                "        -> tuple[np.ndarray, list[np.ndarray]]\n"
                "where the first return value is an integer basis-label array of shape\n"
                "(n_qubits,) and the second is a list of n_qubits 2×2 unitary matrices."
            )

        if self.observables is None:
            self.observables = ["energy", "magnetization", "correlations"]

    # ─────────────────────────────────────────────────────────────────────────

    def get_measurement_basis_info(self) -> dict:
        """Return a summary of the measurement basis configuration."""
        info: dict = {
            "basis_type": self.measurement_basis,
            "n_qubits": self.n_qubits,
            "n_shadows": self.n_shadows,
        }
        if self.measurement_basis == "pauli":
            info["pauli_weights"] = self.pauli_weights
        elif self.measurement_basis == "clifford":
            info["note"] = (
                "Local random Clifford: each qubit independently uses one "
                "of 24 single-qubit Cliffords (uniform). clifford_depth is "
                "reserved for a future global Clifford implementation."
            )
        elif self.measurement_basis == "custom":
            info["custom_sampler"] = repr(self.custom_sampler)
        return info

    def get_processing_info(self) -> dict:
        """Return a summary of the processing configuration."""
        return {
            "median_of_means": self.median_of_means,
            "n_means": self.n_means,
            "n_samples_per_mean": self.n_samples_per_mean,
            "observables": self.observables,
            "correlation_length": self.correlation_length,
            "renyi_entropy": self.renyi_entropy,
        }

    def __repr__(self) -> str:
        return (
            f"ShadowConfig(n_qubits={self.n_qubits}, "
            f"n_shadows={self.n_shadows}, "
            f"basis={self.measurement_basis!r})"
        )


# ─── Convenience factory functions ────────────────────────────────────────────

def create_default_config(
    n_qubits: int,
    n_shadows: int = 1000,
    measurement_basis: str = "random",
) -> ShadowConfig:
    """Create a default shadow configuration."""
    return ShadowConfig(
        n_qubits=n_qubits,
        n_shadows=n_shadows,
        measurement_basis=measurement_basis,
    )


def create_pauli_config(
    n_qubits: int,
    n_shadows: int = 1000,
    pauli_weights: Optional[List[float]] = None,
) -> ShadowConfig:
    """
    Create a configuration for weighted Pauli measurements.

    Args:
        n_qubits: Number of qubits.
        n_shadows: Number of shadow measurements.
        pauli_weights: Weights for [X, Y, Z] (default: uniform [1/3, 1/3, 1/3]).
    """
    return ShadowConfig(
        n_qubits=n_qubits,
        n_shadows=n_shadows,
        measurement_basis="pauli",
        pauli_weights=pauli_weights,
    )


def create_clifford_config(
    n_qubits: int,
    n_shadows: int = 1000,
    clifford_depth: int = 1,
) -> ShadowConfig:
    """
    Create a configuration for local random Clifford measurements.

    Each qubit independently receives one of the 24 single-qubit Clifford gates,
    chosen uniformly at random.  ``clifford_depth`` is stored but not used in
    the current implementation; it is reserved for a future global Clifford
    circuit mode.
    """
    return ShadowConfig(
        n_qubits=n_qubits,
        n_shadows=n_shadows,
        measurement_basis="clifford",
        clifford_depth=clifford_depth,
    )


def create_custom_config(
    n_qubits: int,
    custom_sampler: Callable,
    n_shadows: int = 1000,
) -> ShadowConfig:
    """
    Create a configuration for a user-defined measurement basis.

    Args:
        n_qubits: Number of qubits.
        custom_sampler: Callable with signature
            ``(n_qubits, rng) -> (basis_labels, unitaries)``.
            See ``ShadowConfig`` docstring for details.
        n_shadows: Number of shadow measurements.
    """
    return ShadowConfig(
        n_qubits=n_qubits,
        n_shadows=n_shadows,
        measurement_basis="custom",
        custom_sampler=custom_sampler,
    )
