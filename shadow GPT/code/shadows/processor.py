"""
Classical shadow data processing and property estimation.

This module provides the ShadowProcessor class for processing classical shadow
measurements to estimate quantum properties.

Supported estimators
--------------------
estimate_magnetization : average Z-magnetization  (1/n) Σ_i ⟨Z_i⟩
estimate_correlations  : average ZZ two-point correlations ⟨Z_i Z_j⟩
estimate_energy        : energy ⟨H⟩ for a Pauli-decomposable Hamiltonian
estimate_renyi_entropy : Rényi-2 entropy of a bipartition (paired-shot purity)

Shadow estimator formulas
--------------------------
All estimators for local-Clifford / local-Pauli shadows use the inverse channel

    M⁻¹[U†|s⟩⟨s|U] = 3 U†|s⟩⟨s|U − I    (single qubit, d=2)

so the unbiased per-shot estimator for a traceless single-qubit observable P is

    ô_i(P) = 3 · (U_i P U_i†)[s_i, s_i]

and for a Pauli string P = ⊗_i P_i the estimator is the product ∏_i ô_i(P_i),
which is unbiased for ⟨P⟩ for ANY state (including entangled ones) because
Tr(A⊗B (C⊗D)) = Tr(AC)·Tr(BD).

Basis-mode support
------------------
The processor reconstructs rotation unitaries from stored basis labels:
    "random" / "pauli"  →  _PAULI_ROTATIONS_LOCAL[b]  (b ∈ {0,1,2} = {X,Y,Z})
    "clifford"          →  SINGLE_QUBIT_CLIFFORDS[b]   (b ∈ 0…23)
    "custom"            →  uses stored ShadowMeasurement.unitaries directly
"""

import warnings
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import os

from .config import ShadowConfig
from .collector import ShadowCollector, ShadowMeasurement, SINGLE_QUBIT_CLIFFORDS


# ─── Local Pauli constants ─────────────────────────────────────────────────────
# Reproduced here (not imported from collector) to avoid tight coupling.

_H_P = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_S_P = np.array([[1, 0], [0, 1j]], dtype=complex)
_Sdg_P = _S_P.conj().T
_I2_P = np.eye(2, dtype=complex)
_X_P = np.array([[0, 1], [1, 0]], dtype=complex)
_Z_P = np.array([[1, 0], [0, -1]], dtype=complex)

# Per-basis rotation unitaries for random/pauli mode: [X-basis, Y-basis, Z-basis]
_PAULI_ROTATIONS_LOCAL: List[np.ndarray] = [_H_P, _H_P @ _Sdg_P, _I2_P]

# Rényi-2 purity estimator: warn when subsystem size exceeds this threshold.
# Above this size the per-pair estimator has range ≈ [-4^k, 5^k], giving
# exponentially large variance and impractical error bars.
_RENYI_SUBSYSTEM_WARN_THRESHOLD: int = 4

# Optional pyclifford dependency — required only for estimate_energy().
try:
    from pyclifford import PauliPolynomial as _PauliPolynomial  # type: ignore[import]
    _PYCLIFFORD_AVAILABLE = True
except ImportError:
    _PauliPolynomial = None
    _PYCLIFFORD_AVAILABLE = False


# ─── Data class ───────────────────────────────────────────────────────────────

@dataclass
class PropertyEstimate:
    """Result of a quantum property estimation from classical shadows."""
    property_name: str
    estimate: float
    error: float
    confidence_interval: Tuple[float, float]
    n_samples: int


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _get_rotation_unitary(mode: str, b: int) -> np.ndarray:
    """
    Return the 2×2 rotation unitary that corresponds to basis label *b*
    in measurement mode *mode*.

    Raises ValueError for "custom" mode, which must use stored per-shot
    unitary data instead of basis-label reconstruction.
    """
    if mode in ("random", "pauli"):
        return _PAULI_ROTATIONS_LOCAL[b]
    elif mode == "clifford":
        return SINGLE_QUBIT_CLIFFORDS[b]
    else:
        raise ValueError(
            f"Cannot reconstruct a rotation unitary from basis mode {mode!r}. "
            "Supported modes: 'random', 'pauli', 'clifford'. "
            "For 'custom' mode use the stored per-shot unitaries attached to "
            "each ShadowMeasurement."
        )


def _check_basis_mode(mode: str, estimator_name: str) -> None:
    """Raise ValueError if *mode* is not one of the supported modes."""
    supported = ("random", "pauli", "clifford", "custom")
    if mode not in supported:
        raise ValueError(
            f"{estimator_name} does not support basis mode {mode!r}. "
            f"Supported modes: {supported}."
        )


def _get_measurement_unitaries(
    measurement: ShadowMeasurement,
    mode: str,
    n_qubits: int,
    estimator_name: str,
) -> List[np.ndarray]:
    """
    Return the per-qubit rotation unitaries used for one shot.

    For built-in modes the unitaries are reconstructed from basis labels.
    For ``custom`` mode the matrices must be stored directly on the
    ``ShadowMeasurement``.
    """
    if mode != "custom":
        return [
            _get_rotation_unitary(mode, int(measurement.basis[i]))
            for i in range(n_qubits)
        ]

    if measurement.unitaries is None:
        raise ValueError(
            f"{estimator_name} requires stored per-shot unitaries for "
            "measurement_basis='custom', but this measurement has none. "
            "Re-collect the data with the updated collector or load a .npz "
            "file that includes the 'custom_unitaries' array."
        )

    unitaries = np.asarray(measurement.unitaries, dtype=complex)
    if unitaries.shape != (n_qubits, 2, 2):
        raise ValueError(
            f"{estimator_name} expected custom unitary data with shape "
            f"({n_qubits}, 2, 2), got {unitaries.shape}."
        )
    if not np.all(np.isfinite(unitaries)):
        raise ValueError(
            f"{estimator_name} received custom unitary data containing NaN or Inf."
        )

    return [unitaries[i] for i in range(n_qubits)]


def _iter_hamiltonian_terms(hamiltonian: Any):
    """
    Yield Pauli terms from a Hamiltonian-like object.

    Supports both iterable Hamiltonians and sequence-style objects such as
    PyClifford's ``PauliPolynomial``, which may implement ``__len__`` and
    ``__getitem__`` without defining ``__iter__``.
    """
    if hasattr(hamiltonian, "__iter__"):
        yield from hamiltonian
        return

    if hasattr(hamiltonian, "__len__") and hasattr(hamiltonian, "__getitem__"):
        for i in range(len(hamiltonian)):
            yield hamiltonian[i]
        return

    raise TypeError(
        "Hamiltonian object does not support term iteration via __iter__ "
        "or sequence access (__len__ + __getitem__)."
    )


def _z_shadow_val(U: np.ndarray, s: int) -> float:
    """
    Single-qubit shadow estimator for the Z observable.

    Returns  3 · (U Z U†)[s, s],  an unbiased per-shot estimator for ⟨Z_i⟩
    when unitary U was applied to qubit i and outcome s was measured.
    """
    rotated = U @ _Z_P @ U.conj().T
    return 3.0 * float(np.real(rotated[s, s]))


def _pauli_matrix_from_bits(x_bit: int, z_bit: int) -> Optional[np.ndarray]:
    """
    Return the per-qubit symplectic Pauli matrix X^x · Z^z, or None for identity.

    pyclifford's symplectic representation stores the physical Pauli P_i of
    qubit i as  X_i^{x_i} Z_i^{z_i}  with a global phase absorbed into
    term.p.  Specifically Y = i·X·Z, so the [1,1] entry is X·Z = -iY; the
    factor i (or powers thereof) lives in term.c · (1j)^{term.p}.

        [0,0] → None  (identity, contributes factor 1)
        [1,0] → X
        [0,1] → Z
        [1,1] → X·Z = -iY  (phase compensated by term.p)
    """
    if x_bit == 0 and z_bit == 0:
        return None
    if x_bit == 1 and z_bit == 0:
        return _X_P
    if x_bit == 0 and z_bit == 1:
        return _Z_P
    return _X_P @ _Z_P   # [1,1]: XZ = [[0,-1],[1,0]] = -iY


# ─── Main processor class ─────────────────────────────────────────────────────

class ShadowProcessor:
    """
    Process classical shadow measurements to estimate quantum properties.

    All four estimators read from collector.measurements and obtain the
    per-shot rotation unitaries either by reconstructing them from built-in
    basis labels or, for ``measurement_basis='custom'``, by using the stored
    ``ShadowMeasurement.unitaries`` data. No dummy data or random placeholders
    are used.
    """

    def __init__(self, config: ShadowConfig) -> None:
        """
        Initialise the shadow processor.

        Args:
            config: Shadow configuration (shared with the collector is fine).
        """
        self.config = config
        self.estimates: Dict[str, PropertyEstimate] = {}
        # Local RNG for bootstrap; does not affect global numpy state.
        self.rng = np.random.default_rng(config.seed)

    # ── Orchestration ──────────────────────────────────────────────────────────

    def process_shadows(
        self,
        collector: ShadowCollector,
        hamiltonian: Optional[Any] = None,
    ) -> Dict[str, PropertyEstimate]:
        """
        Run all configured estimators on *collector*'s shadow measurements.

        Args:
            collector:   ShadowCollector with measurements already collected.
            hamiltonian: Hamiltonian (Operator / pyclifford PauliPolynomial) for
                         energy estimation.  Ignored when "energy" is not in
                         config.observables.

        Returns:
            Dict mapping property name → PropertyEstimate.
        """
        if not collector.measurements:
            raise ValueError("No measurements to process. Collect shadows first.")

        estimates: Dict[str, PropertyEstimate] = {}

        if "energy" in self.config.observables and hamiltonian is not None:
            estimates["energy"] = self.estimate_energy(collector, hamiltonian)

        if "magnetization" in self.config.observables:
            estimates["magnetization"] = self.estimate_magnetization(collector)

        if "correlations" in self.config.observables:
            estimates["correlations"] = self.estimate_correlations(collector)

        if self.config.renyi_entropy:
            estimates["renyi_entropy"] = self.estimate_renyi_entropy(collector)

        self.estimates = estimates
        return estimates

    # ── Estimators ────────────────────────────────────────────────────────────

    def estimate_magnetization(self, collector: ShadowCollector) -> PropertyEstimate:
        """
        Estimate the average Z-magnetization  m_z = (1/n) Σ_i ⟨Z_i⟩.

        Per-shot estimator
        ------------------
        For qubit i with rotation U_i and outcome s_i::

            ô_i = 3 · (U_i Z U_i†)[s_i, s_i]

        which averages to ⟨Z_i⟩ over many shots.  For random/pauli mode this
        simplifies to  ô_i = 3(1-2s_i)  when b_i = 2 (Z measurement), else 0.

        The per-shot magnetization  (1/n) Σ_i ô_i  is averaged over all shots
        using median-of-means (if config.median_of_means) or plain mean.

        Supported modes: "random", "pauli", "clifford", "custom".
        """
        mode = collector.config.measurement_basis
        _check_basis_mode(mode, "estimate_magnetization")

        n = self.config.n_qubits
        measurements = collector.measurements
        N = len(measurements)

        shot_vals = np.zeros(N)
        for k, m in enumerate(measurements):
            shot_unitaries = _get_measurement_unitaries(
                m,
                mode,
                n,
                "estimate_magnetization",
            )
            total = 0.0
            for i in range(n):
                U_i = shot_unitaries[i]
                total += _z_shadow_val(U_i, int(m.outcome[i]))
            shot_vals[k] = total / n

        estimate, error = self._aggregate(shot_vals)
        ci = (estimate - 2.0 * error, estimate + 2.0 * error)

        return PropertyEstimate(
            property_name="magnetization",
            estimate=estimate,
            error=error,
            confidence_interval=ci,
            n_samples=N,
        )

    def estimate_correlations(self, collector: ShadowCollector) -> PropertyEstimate:
        """
        Estimate pair-averaged ZZ correlations  (1/|P|) Σ_{(i,j)∈P} ⟨Z_i Z_j⟩.

        Pairs P contain all (i, j) with 1 ≤ j−i ≤ config.correlation_length.

        Per-shot estimator
        ------------------
        The shadow estimator for the two-site observable Z_i ⊗ Z_j is::

            ô_{ij} = ô_i · ô_j

        This is unbiased for ⟨Z_i Z_j⟩ for **any** state (including entangled
        ones) because Tr(A⊗B (C⊗D)) = Tr(AC)·Tr(BD).

        The per-shot average  (1/|P|) Σ_{(i,j)} ô_i ô_j  is then aggregated
        over shots.

        Supported modes: "random", "pauli", "clifford", "custom".
        """
        mode = collector.config.measurement_basis
        _check_basis_mode(mode, "estimate_correlations")

        n = self.config.n_qubits
        L = self.config.correlation_length
        measurements = collector.measurements
        N = len(measurements)

        pairs = [
            (i, j)
            for i in range(n)
            for j in range(i + 1, min(i + L + 1, n))
        ]
        if not pairs:
            raise ValueError(
                f"No valid pairs for n_qubits={n}, correlation_length={L}."
            )

        shot_vals = np.zeros(N)
        for k, m in enumerate(measurements):
            shot_unitaries = _get_measurement_unitaries(
                m,
                mode,
                n,
                "estimate_correlations",
            )
            z_vals = np.array([
                _z_shadow_val(
                    shot_unitaries[i],
                    int(m.outcome[i]),
                )
                for i in range(n)
            ])
            shot_vals[k] = float(np.mean([z_vals[i] * z_vals[j] for i, j in pairs]))

        estimate, error = self._aggregate(shot_vals)
        ci = (estimate - 2.0 * error, estimate + 2.0 * error)

        return PropertyEstimate(
            property_name="correlations",
            estimate=estimate,
            error=error,
            confidence_interval=ci,
            n_samples=N,
        )

    def estimate_energy(
        self,
        collector: ShadowCollector,
        hamiltonian: Any,
    ) -> PropertyEstimate:
        """
        Estimate the energy expectation value ⟨H⟩ from classical shadows.

        Hamiltonian requirements
        -------------------------
        *hamiltonian* must be a pyclifford ``PauliPolynomial`` (or the project's
        ``Operator`` subclass).  The estimator iterates over Pauli terms and
        uses the symplectic generator bits to extract per-qubit Pauli matrices.

        Per-shot estimator
        ------------------
        For  H = Σ_k c_k P_k  (Pauli decomposition), the single-shot estimator
        is::

            ô^{(H)} = Σ_k c_k · ∏_{i: P_k,i ≠ I}  3·(U_i σ_i U_i†)[s_i, s_i]

        where σ_i = X^{x_i} Z^{z_i} is the per-qubit symplectic Pauli and
        c_k = term.c · (1j)^{term.p} is the effective coefficient (always real
        for a Hermitian Hamiltonian).  Taking np.real(c_k · product) and
        averaging over shots yields an unbiased estimate of ⟨H⟩.

        Supported modes: "random", "pauli", "clifford", "custom".

        Args:
            collector:   ShadowCollector with measurements.
            hamiltonian: pyclifford PauliPolynomial or Operator.

        Returns:
            PropertyEstimate of ⟨H⟩ ± standard error.

        Raises:
            TypeError:  If *hamiltonian* lacks the PauliPolynomial interface.
            ValueError: If qubit count mismatches or basis mode unsupported.
        """
        mode = collector.config.measurement_basis
        _check_basis_mode(mode, "estimate_energy")

        # Duck-type check: must expose qubit count and support either iteration
        # or sequence-style term access.
        has_term_access = (
            hasattr(hamiltonian, "__iter__")
            or (hasattr(hamiltonian, "__len__") and hasattr(hamiltonian, "__getitem__"))
        )
        if not (hasattr(hamiltonian, "N") and has_term_access):
            raise TypeError(
                "estimate_energy() requires a pyclifford PauliPolynomial "
                "(or Operator subclass).  "
                f"Got {type(hamiltonian).__name__}.  "
                "Build the Hamiltonian with ham_tf_ising() or ham_cluster_ising()."
            )

        n = self.config.n_qubits
        if hamiltonian.N != n:
            raise ValueError(
                f"Hamiltonian has {hamiltonian.N} qubits but "
                f"config.n_qubits = {n}."
            )

        measurements = collector.measurements
        N = len(measurements)

        # ── Pre-parse Hamiltonian terms once (outside the shot loop) ───────────
        # Each entry: (c_eff, [(qubit_idx, sigma_matrix), ...])
        # c_eff is complex but Im(c_eff * product) ≈ 0 for Hermitian H.
        parsed_terms: List[Tuple[complex, List[Tuple[int, np.ndarray]]]] = []

        for term in _iter_hamiltonian_terms(hamiltonian):
            c_eff = complex(term.c) * (1j ** int(term.p))
            g = term.g.reshape(-1, 2)   # (n, 2): g[i] = [x_i, z_i]
            nonid: List[Tuple[int, np.ndarray]] = []
            for i in range(n):
                sigma = _pauli_matrix_from_bits(int(g[i, 0]), int(g[i, 1]))
                if sigma is not None:
                    nonid.append((i, sigma))
            parsed_terms.append((c_eff, nonid))

        if not parsed_terms:
            raise ValueError("Hamiltonian contains no terms.")

        # ── Per-shot energy estimates ──────────────────────────────────────────
        shot_vals = np.zeros(N)
        for k, m in enumerate(measurements):
            # Obtain per-qubit rotation unitaries for this shot
            U = _get_measurement_unitaries(m, mode, n, "estimate_energy")
            s = m.outcome   # int array, shape (n,)

            energy_k = 0.0
            for c_eff, nonid in parsed_terms:
                if not nonid:
                    # Pure identity term: no qubit factors, contribute scalar
                    energy_k += float(np.real(c_eff))
                    continue

                prod = complex(1.0)
                for qi, sigma_i in nonid:
                    rotated_diag = (U[qi] @ sigma_i @ U[qi].conj().T)[int(s[qi]), int(s[qi])]
                    prod *= 3.0 * rotated_diag

                energy_k += float(np.real(c_eff * prod))

            shot_vals[k] = energy_k

        estimate, error = self._aggregate(shot_vals)
        ci = (estimate - 2.0 * error, estimate + 2.0 * error)

        return PropertyEstimate(
            property_name="energy",
            estimate=estimate,
            error=error,
            confidence_interval=ci,
            n_samples=N,
        )

    def estimate_renyi_entropy(
        self,
        collector: ShadowCollector,
        n_subsystem: Optional[int] = None,
    ) -> PropertyEstimate:
        """
        Estimate the Rényi-2 entropy S₂ = −log Tr(ρ_A²) of a bipartition
        using the paired-shot purity estimator.

        Subsystem
        ---------
        Subsystem A = qubits 0 … n_subsystem−1 (default: n_qubits // 2).

        For **pure states** (e.g. ground states from ED/DMRG):
        - Global purity (n_subsystem = n) = 1 trivially, so S₂ = 0.
        - Half-chain purity measures **entanglement entropy** (non-trivial).

        Algorithm (Huang et al. 2020, Section IV)
        ------------------------------------------
        The N shots are split into two halves.  For the m-th pair
        (shot k, shot l = k + N/2) the per-pair estimator is::

            F_m = ∏_{i∈A}  [9 |M_i[s_k^i, s_l^i]|² − 4]

        where  M_i = U_i^{(k)} (U_i^{(l)})†.

        This is an **unbiased** estimator for Tr(ρ_A²) for any state; the proof
        follows from independence of shots k and l and E[ρ̂] = ρ.

        Rényi-2 entropy:  S₂ ≈ −log(max(mean(F_m), ε))

        ⚠ Variance warning
        -------------------
        Each factor 9|M|²−4 lies in [−4, 5], so the product F_m lies in
        [−4^k, 5^k] for subsystem size k.  The variance **grows exponentially**
        in k.  A UserWarning is emitted when k > _RENYI_SUBSYSTEM_WARN_THRESHOLD
        (default 4).  For k > 8, estimates from O(1000) shots will typically
        have error bars much larger than the estimate itself.

        Supported modes: "random", "pauli", "clifford", "custom".

        Args:
            collector:   ShadowCollector with measurements.
            n_subsystem: Subsystem size k (default n_qubits // 2).

        Returns:
            PropertyEstimate of S₂ with propagated error and 95 % CI.

        Raises:
            ValueError: If fewer than 2 measurements (need ≥ 1 pair).
        """
        mode = collector.config.measurement_basis
        _check_basis_mode(mode, "estimate_renyi_entropy")

        n = self.config.n_qubits
        measurements = collector.measurements
        N = len(measurements)

        if N < 2:
            raise ValueError(
                "estimate_renyi_entropy() requires at least 2 shadow measurements "
                f"to form one pair; got {N}."
            )

        k = (n // 2) if n_subsystem is None else int(np.clip(n_subsystem, 1, n))

        if k > _RENYI_SUBSYSTEM_WARN_THRESHOLD:
            warnings.warn(
                f"Rényi-2 purity estimator for subsystem of k={k} qubits has "
                f"exponentially large variance: per-pair range ≈ "
                f"[{-4**k:.2e}, {5**k:.2e}].  "
                f"With {N // 2} pairs the standard error may far exceed the "
                "true value.  Consider a smaller subsystem or more shots.",
                UserWarning,
                stacklevel=2,
            )

        n_pairs = N // 2
        shots_1 = measurements[:n_pairs]
        shots_2 = measurements[n_pairs: 2 * n_pairs]

        purity_ests = np.zeros(n_pairs)

        for m_idx, (sk, sl) in enumerate(zip(shots_1, shots_2)):
            U_k_all = _get_measurement_unitaries(
                sk,
                mode,
                n,
                "estimate_renyi_entropy",
            )
            U_l_all = _get_measurement_unitaries(
                sl,
                mode,
                n,
                "estimate_renyi_entropy",
            )
            term = 1.0
            for i in range(k):
                U_k = U_k_all[i]
                U_l = U_l_all[i]

                # M = U_k · U_l†;  overlap = |M[s_k, s_l]|²
                M = U_k @ U_l.conj().T
                overlap_sq = float(abs(M[int(sk.outcome[i]), int(sl.outcome[i])]) ** 2)
                term *= 9.0 * overlap_sq - 4.0

            purity_ests[m_idx] = term

        purity_mean = float(np.mean(purity_ests))
        if n_pairs > 1:
            purity_std = float(np.std(purity_ests, ddof=1) / np.sqrt(n_pairs))
        else:
            # With a single pair there is no sample variance estimate; return
            # a zero standard error rather than propagating NaN.
            purity_std = 0.0

        # Clip to physically valid range before taking logarithm.
        # (Individual pair estimates can be negative due to finite-sample noise;
        #  the true Tr(ρ²) ∈ (0, 1].)
        purity_clipped = float(np.clip(purity_mean, 1e-12, 1.0))
        s2 = -np.log(purity_clipped)

        # Error propagation through −log: δS₂ ≈ δ(purity) / purity
        s2_error = purity_std / max(purity_clipped, 1e-12)

        ci = (max(s2 - 2.0 * s2_error, 0.0), s2 + 2.0 * s2_error)

        return PropertyEstimate(
            property_name="renyi_entropy",
            estimate=s2,
            error=s2_error,
            confidence_interval=ci,
            n_samples=n_pairs,
        )

    # ── Statistical aggregation ────────────────────────────────────────────────

    def _aggregate(self, shot_vals: np.ndarray) -> Tuple[float, float]:
        """
        Reduce per-shot estimator values to (estimate, error).

        Uses median-of-means when config.median_of_means is True;
        otherwise uses plain mean ± std/√N.
        """
        if self.config.median_of_means:
            return self.median_of_means(shot_vals)
        N = len(shot_vals)
        mean = float(np.mean(shot_vals))
        err = float(np.std(shot_vals, ddof=1) / np.sqrt(N)) if N > 1 else 0.0
        return mean, err

    def median_of_means(
        self,
        data: np.ndarray,
        n_means: Optional[int] = None,
        n_samples_per_mean: Optional[int] = None,  # kept for API compat; ignored
    ) -> Tuple[float, float]:
        """
        Median-of-means estimator for robust reduction of per-shot values.

        Partitions *data* into K = min(n_means, N) non-overlapping groups of
        equal size B = N // K, computes the mean within each group, and
        returns the median of those K group means.

        The error is the MAD × 1.4826 / √K, which is a consistent estimator
        of the standard deviation of the group means under normality.

        Args:
            data:               Per-shot estimator values, shape (N,).
            n_means:            Number of groups K (default: config.n_means).
            n_samples_per_mean: Ignored; uniform partitioning is used instead.

        Returns:
            (estimate, error)
        """
        if n_means is None:
            n_means = self.config.n_means

        N = len(data)
        K = min(int(n_means), N)
        if K <= 0:
            K = 1

        B = N // K
        if B == 0:
            # Fewer points than groups — fall back to plain mean
            mean = float(np.mean(data))
            err = float(np.std(data, ddof=1) / np.sqrt(N)) if N > 1 else 0.0
            return mean, err

        groups = np.array_split(data[: K * B], K)   # K groups of size B
        group_means = np.array([float(g.mean()) for g in groups])

        estimate = float(np.median(group_means))
        # MAD-based std estimate (robust, consistent under normality)
        mad = float(np.median(np.abs(group_means - estimate)))
        error = mad * 1.4826 / np.sqrt(K)

        return estimate, error

    def bootstrap_estimate(
        self,
        data: np.ndarray,
        n_bootstrap: int = 1000,
    ) -> Tuple[float, float]:
        """
        Bootstrap standard error estimate.

        Resamples *data* with replacement *n_bootstrap* times and returns the
        original sample mean as the point estimate with the standard deviation
        of bootstrap sample means as the error.

        Args:
            data:        Per-shot estimator values, shape (N,).
            n_bootstrap: Number of bootstrap resamples.

        Returns:
            (estimate, error)
        """
        n = len(data)
        original_mean = float(np.mean(data))

        boot_means = np.array([
            float(np.mean(data[self.rng.integers(0, n, size=n)]))
            for _ in range(n_bootstrap)
        ])

        error = float(np.std(boot_means, ddof=1))
        return original_mean, error

    # ── Data access / persistence ──────────────────────────────────────────────

    def save_estimates(self, filename: Optional[str] = None) -> str:
        """
        Save property estimates to a compressed ``.npz`` file.

        Each PropertyEstimate is stored as five scalar arrays:
        ``{name}_estimate``, ``{name}_error``,
        ``{name}_ci_low``, ``{name}_ci_high``, ``{name}_n_samples``.
        Config metadata is stored as ``config_n_qubits``,
        ``config_n_shadows``, ``config_basis``.

        Args:
            filename: Output filename (auto-generated if None).

        Returns:
            Absolute path to the saved ``.npz`` file.
        """
        if not self.estimates:
            raise ValueError("No estimates to save. Call process_shadows() first.")

        if filename is None:
            filename = (
                f"estimates_n{self.config.n_qubits}_s{self.config.n_shadows}.npz"
            )

        os.makedirs(self.config.output_dir, exist_ok=True)
        filepath_stem = os.path.join(self.config.output_dir, filename)
        if filepath_stem.endswith(".npz"):
            filepath_stem = filepath_stem[:-4]

        save_dict: Dict[str, Any] = {}
        for name, est in self.estimates.items():
            save_dict[f"{name}_estimate"] = np.float64(est.estimate)
            save_dict[f"{name}_error"] = np.float64(est.error)
            save_dict[f"{name}_ci_low"] = np.float64(est.confidence_interval[0])
            save_dict[f"{name}_ci_high"] = np.float64(est.confidence_interval[1])
            save_dict[f"{name}_n_samples"] = np.int64(est.n_samples)

        save_dict["config_n_qubits"] = np.int64(self.config.n_qubits)
        save_dict["config_n_shadows"] = np.int64(self.config.n_shadows)
        save_dict["config_basis"] = np.array(self.config.measurement_basis)

        np.savez_compressed(filepath_stem, **save_dict)
        filepath_out = filepath_stem + ".npz"
        print(f"Saved {len(self.estimates)} property estimates to {filepath_out}")
        return filepath_out

    def get_summary(self) -> Dict[str, Any]:
        """
        Return a summary dict for all computed property estimates.

        Returns:
            Dict with ``n_estimates``, ``properties``, and per-property
            ``value``, ``error``, ``relative_error``, ``n_samples``.
        """
        if not self.estimates:
            return {}

        summary: Dict[str, Any] = {
            "n_estimates": len(self.estimates),
            "properties": list(self.estimates.keys()),
            "estimates": {},
        }

        for name, est in self.estimates.items():
            rel_err = (
                abs(est.error / est.estimate)
                if est.estimate != 0.0
                else float("inf")
            )
            summary["estimates"][name] = {
                "value": est.estimate,
                "error": est.error,
                "relative_error": rel_err,
                "n_samples": est.n_samples,
            }

        return summary

    def __repr__(self) -> str:
        return (
            f"ShadowProcessor(n_qubits={self.config.n_qubits}, "
            f"n_estimates={len(self.estimates)})"
        )


# ─── Convenience function ─────────────────────────────────────────────────────

def process_shadow_data(
    collector: ShadowCollector,
    config: Optional[ShadowConfig] = None,
    hamiltonian: Optional[Any] = None,
) -> ShadowProcessor:
    """
    Convenience wrapper: create a ShadowProcessor and run all estimators.

    Args:
        collector:   ShadowCollector with measurements.
        config:      Processing config (uses collector.config if None).
        hamiltonian: Hamiltonian for energy estimation (optional).

    Returns:
        ShadowProcessor with all configured estimates computed.
    """
    if config is None:
        config = collector.config
    processor = ShadowProcessor(config)
    processor.process_shadows(collector, hamiltonian)
    return processor
