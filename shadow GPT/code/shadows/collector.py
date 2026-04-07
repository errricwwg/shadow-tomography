"""
Classical shadow data collection from quantum states.

This module provides the ShadowCollector class for collecting classical shadow
measurements from dense state vectors and quimb MatrixProductState (MPS) objects,
supporting multiple measurement bases (random Pauli, weighted Pauli, local random
Clifford, and custom).

Qubit / site ordering convention
----------------------------------
Qubit 0 is the *most significant* bit (big-endian).  For an n-qubit system,
computational basis state |b_{n-1} ... b_1 b_0⟩ maps to integer
b_{n-1}·2^{n-1} + ... + b_0·2^0 and to state-vector index
b_0·2^{n-1} + ... + b_{n-1}·2^0.  Equivalently, the state vector is indexed
as sv[i] = ⟨i|ψ⟩ with i written in big-endian binary.

For MPS sampling, quimb site 0 is treated as the most significant qubit,
consistent with the dense-state convention above.

Boundary conditions for MPS sampling
--------------------------------------
Both open (OBC) and periodic (PBC) boundary conditions are supported.
The boundary type is detected automatically from the quimb MPS object
via ``mps.cyclic``.

OBC algorithm — right-canonical sequential Born-rule:
    A single right-canonical copy (OC at site 0) is prepared *once* before
    the shot loop, amortising the O(n·χ³) SVD cost.  Each shot sweeps
    left→right in O(n·χ²) without further SVDs or copies.

PBC algorithm — double-tensor right-environment:
    For a PBC MPS, ψ(b) = Tr(M₀[b₀]·…·M_{n-1}[b_{n-1}]).
    Right environments R[k] (double-tensor, χ²×χ²) are pre-computed *once*
    from the bare site tensors in O(n·χ⁶).  The key identity
        Σ_s U[s,t]·U*[s,t'] = δ_{t,t'}  for any unitary U
    ensures the transfer matrices are rotation-invariant, so R[k] does not
    depend on the per-shot measurement unitaries.
    Each shot then sweeps left→right in O(n·χ⁴).

    PBC is exact and correct for all bond dimensions, but is O(χ²) more
    expensive per shot than OBC.  A UserWarning is emitted when the max
    bond dimension exceeds _PBC_CHI_WARN_THRESHOLD (default 16).
"""

import os
import warnings
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from .config import ShadowConfig

# Optional quimb dependency — only required for sample_mps().
try:
    import quimb.tensor as _qtn
    QUIMB_AVAILABLE: bool = True
except ImportError:
    _qtn = None  # type: ignore[assignment]
    QUIMB_AVAILABLE: bool = False


# ─── Single-qubit gate constants ──────────────────────────────────────────────

_I2 = np.eye(2, dtype=complex)

_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)

_S = np.array([[1, 0], [0, 1j]], dtype=complex)
_Sdg = _S.conj().T  # S†

# Rotation unitaries that bring each Pauli eigenbasis to the Z computational basis.
#
# To measure qubit in X-basis: apply H, then measure in Z.
#   H|+⟩ = |0⟩,  H|−⟩ = |1⟩  ✓
#
# To measure qubit in Y-basis: apply HS†, then measure in Z.
#   S†|i⟩  = |+⟩,  then H|+⟩ = |0⟩  ✓
#   S†|−i⟩ = |−⟩,  then H|−⟩ = |1⟩  ✓
#
# To measure qubit in Z-basis: identity, measure in Z directly.
#
# Index 0 = X, 1 = Y, 2 = Z.
_PAULI_ROTATIONS: List[np.ndarray] = [_H, _H @ _Sdg, _I2]

PAULI_NAMES: List[str] = ["X", "Y", "Z"]


# ─── Single-qubit Clifford group (24 elements) ────────────────────────────────

def _generate_single_qubit_cliffords() -> List[np.ndarray]:
    """
    Enumerate all 24 single-qubit Clifford gates via BFS over generators {H, S}.

    The single-qubit Clifford group C₁ = ⟨H, S⟩ has exactly 24 elements modulo
    global phase.  Each element is represented as a 2×2 unitary matrix whose
    first significant entry is real positive (a canonical phase convention that
    uniquely identifies each coset of U(1) in the Clifford group).

    Returns:
        List of 24 distinct 2×2 complex unitary matrices.
    """

    def _normalize_phase(m: np.ndarray) -> np.ndarray:
        """Remove global phase: make the first nonzero entry real positive."""
        for x in m.flatten():
            if abs(x) > 1e-10:
                return m * (abs(x) / x)
        return m  # zero matrix — should never happen for a unitary

    def _matrix_key(m: np.ndarray) -> tuple:
        n = _normalize_phase(m)
        return (
            tuple(np.round(n.real.flatten(), 8))
            + tuple(np.round(n.imag.flatten(), 8))
        )

    cliffords: List[np.ndarray] = []
    seen: set = set()
    queue: List[np.ndarray] = [_I2.copy()]

    while queue:
        g = queue.pop(0)
        k = _matrix_key(g)
        if k in seen:
            continue
        seen.add(k)
        cliffords.append(_normalize_phase(g))
        # Expand with both generators; finite group guarantees BFS terminates.
        queue.append(g @ _H)
        queue.append(g @ _S)

    return cliffords


SINGLE_QUBIT_CLIFFORDS: List[np.ndarray] = _generate_single_qubit_cliffords()
assert len(SINGLE_QUBIT_CLIFFORDS) == 24, (
    f"Clifford group enumeration error: expected 24 gates, got "
    f"{len(SINGLE_QUBIT_CLIFFORDS)}."
)
_N_CLIFFORDS: int = 24


# ─── Dense-state helpers ──────────────────────────────────────────────────────

def _validate_state_vector(sv: np.ndarray, n_qubits: int) -> np.ndarray:
    """
    Validate a state vector and return a normalised complex copy.

    Checks:
    - 1D array.
    - Length exactly 2^n_qubits.
    - All entries finite.
    - Norm ≈ 1 (normalises automatically with a UserWarning if not).

    Args:
        sv: Input state vector.
        n_qubits: Expected number of qubits.

    Returns:
        Validated, normalised complex 1D array.

    Raises:
        ValueError: If the shape, dimension, or values are invalid.
    """
    sv = np.asarray(sv, dtype=complex)

    if sv.ndim != 1:
        raise ValueError(
            f"State vector must be a 1D array, got shape {sv.shape}."
        )

    expected_dim = 2 ** n_qubits
    if sv.shape[0] != expected_dim:
        raise ValueError(
            f"State vector length {sv.shape[0]} does not match "
            f"2^n_qubits = 2^{n_qubits} = {expected_dim}."
        )

    if not np.all(np.isfinite(sv)):
        raise ValueError("State vector contains NaN or Inf values.")

    norm = float(np.linalg.norm(sv))
    if norm == 0.0:
        raise ValueError("State vector has zero norm.")

    if not np.isclose(norm, 1.0, atol=1e-6):
        warnings.warn(
            f"State vector norm is {norm:.8f} (expected 1.0). "
            "Normalising automatically.",
            UserWarning,
            stacklevel=3,
        )
        sv = sv / norm

    return sv


def _apply_local_unitaries(
    state_vector: np.ndarray,
    unitaries: List[np.ndarray],
    n_qubits: int,
) -> np.ndarray:
    """
    Apply a tensor product of single-qubit unitaries to a state vector.

    Computes  (U_0 ⊗ U_1 ⊗ … ⊗ U_{n-1}) |ψ⟩  where qubit 0 is the most
    significant axis (big-endian ordering, matching numpy's default reshape).

    Complexity: O(n · 2^n) — each of the n single-qubit gates costs O(2^n).

    Args:
        state_vector: Complex 1D array of length 2^n_qubits.
        unitaries: List of n_qubits 2×2 complex unitary matrices.
        n_qubits: Number of qubits.

    Returns:
        Rotated state vector of the same length.
    """
    state = state_vector.reshape([2] * n_qubits)

    for qubit, U in enumerate(unitaries):
        # np.tensordot contracts axis 1 of U with axis `qubit` of state.
        # The result has U's axis 0 prepended (at position 0).
        state = np.tensordot(U, state, axes=([1], [qubit]))
        # Move the new axis back to the correct qubit position.
        state = np.moveaxis(state, 0, qubit)

    return state.reshape(-1)


def _int_to_bits(integer: int, n_bits: int) -> np.ndarray:
    """
    Convert an integer to a big-endian bit array of length n_bits.

    Example: _int_to_bits(6, 4) == [0, 1, 1, 0]
    """
    return np.array(
        [(integer >> (n_bits - 1 - i)) & 1 for i in range(n_bits)],
        dtype=int,
    )


def _coerce_shot_unitaries(
    unitaries: Union[List[np.ndarray], np.ndarray],
    n_qubits: int,
    *,
    context: str,
) -> np.ndarray:
    """
    Validate and coerce one shot's per-qubit unitaries to shape ``(n_qubits, 2, 2)``.

    Parameters
    ----------
    unitaries:
        Sequence or array of 2x2 complex matrices, one per qubit.
    n_qubits:
        Expected number of qubits / unitary matrices.
    context:
        Short human-readable context string for error messages.

    Returns
    -------
    ndarray
        Complex array of shape ``(n_qubits, 2, 2)``.
    """
    arr = np.asarray(unitaries, dtype=complex)
    if arr.shape != (n_qubits, 2, 2):
        raise ValueError(
            f"{context} returned unitary data with shape {arr.shape}, "
            f"expected ({n_qubits}, 2, 2)."
        )

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{context} returned unitary data containing NaN or Inf.")

    ident = np.eye(2, dtype=complex)
    for i, U in enumerate(arr):
        if not np.allclose(U @ U.conj().T, ident, atol=1e-8):
            raise ValueError(
                f"{context} returned a non-unitary 2x2 matrix for qubit {i}."
            )

    return np.array(arr, dtype=complex, copy=True)


# ─── MPS sampling helpers ─────────────────────────────────────────────────────

# PBC sampling emits a UserWarning when max bond dimension exceeds this value.
# Beyond this threshold, the O(n·χ⁶) pre-computation becomes very expensive.
_PBC_CHI_WARN_THRESHOLD: int = 16


def _is_cyclic_mps(mps: Any, n_qubits: int) -> bool:
    """
    Return ``True`` if *mps* uses periodic boundary conditions.

    Checks ``mps.cyclic`` first (quimb ≥ 1.x), then falls back to probing
    whether ``mps.bond(n-1, 0)`` exists.
    """
    if hasattr(mps, "cyclic"):
        return bool(mps.cyclic)
    if n_qubits < 2:
        return False
    try:
        mps.bond(n_qubits - 1, 0)
        return True
    except Exception:
        return False


def _prepare_canonical_mps(mps: Any) -> Any:
    """
    Return a right-canonical, normalised copy of an OBC MPS (OC at site 0).

    Creates one copy of the MPS, normalises it in-place, and calls
    ``canonize(0)`` to set the orthogonality centre at site 0 (right-canonical
    form).  The original MPS is never mutated.

    This is called **once** before the shot loop in ``sample_mps()`` so the
    O(n·χ³) SVD sweep is amortised across all shots rather than repeated
    for every shot.

    After canonization, each site tensor (except site 0) is right-isometric,
    meaning the right environment at every site contracts to the identity.
    The sequential Born-rule sweep therefore needs no further SVD or environment
    contraction — only O(n·χ²) work per shot.
    """
    psi = mps.copy()
    # quimb API compatibility: newer versions expose ``normalize()`` while
    # some older code paths used ``normalize_()``.
    if hasattr(psi, "normalize_"):
        psi.normalize_()
    else:
        psi.normalize()
    psi.canonize(0)
    return psi


def _get_site_tensor_obc(mps: Any, i: int, n_qubits: int) -> np.ndarray:
    """
    Extract site tensor *i* from an OBC MPS and transpose to canonical axis order.

    Uses quimb's named-index API to determine the correct permutation without
    hardcoding any internal axis order.

    Returned array shapes:
        i == 0          →  (d, χ_R)
        0 < i < n-1     →  (χ_L, d, χ_R)
        i == n-1        →  (χ_L, d)
        n_qubits == 1   →  (d,)
    """
    t = mps[mps.site_tag(i)]
    data: np.ndarray = t.data
    inds: List[str] = list(t.inds)
    phys_ind: str = mps.site_ind(i)

    ordered_inds: List[str] = []
    if i > 0:
        ordered_inds.append(mps.bond(i - 1, i))
    ordered_inds.append(phys_ind)
    if i < n_qubits - 1:
        ordered_inds.append(mps.bond(i, i + 1))

    perm = [inds.index(idx) for idx in ordered_inds]
    return np.transpose(data, perm)


def _get_site_tensor_pbc(mps: Any, i: int, n_qubits: int) -> np.ndarray:
    """
    Extract site tensor *i* from a PBC (cyclic) MPS, axes = (χ_L, d, χ_R).

    All sites return a 3-D array.

    - Site 0:    χ_L = periodic bond  (site n-1 ↔ site 0)
    - Site n-1:  χ_R = periodic bond  (site n-1 ↔ site 0)
    - Others:    standard left / right virtual bonds

    Uses ``mps.bond(n-1, 0)`` to identify the periodic bond index name.
    """
    n = n_qubits
    t = mps[mps.site_tag(i)]
    data: np.ndarray = t.data
    inds: List[str] = list(t.inds)
    phys_ind: str = mps.site_ind(i)

    if i == 0:
        left_bond = mps.bond(n - 1, 0)                           # periodic bond
        right_bond = mps.bond(0, 1) if n > 1 else mps.bond(n - 1, 0)
    elif i == n - 1:
        left_bond = mps.bond(n - 2, n - 1)
        right_bond = mps.bond(n - 1, 0)                          # periodic bond
    else:
        left_bond = mps.bond(i - 1, i)
        right_bond = mps.bond(i, i + 1)

    perm = [inds.index(left_bond), inds.index(phys_ind), inds.index(right_bond)]
    return np.transpose(data, perm)


def _mps_sample_obc_shot(
    canonical_mps: Any,
    unitaries: List[np.ndarray],
    n_qubits: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Single-shot Born-rule sampling for a right-canonical OBC MPS.

    Parameters
    ----------
    canonical_mps:
        Right-canonical MPS with OC at site 0, produced by
        ``_prepare_canonical_mps``.  **Never mutated.**
    unitaries:
        Per-site 2×2 rotation unitaries for this shot.
    n_qubits:
        Number of sites / qubits.
    rng:
        Local numpy RNG.

    Returns
    -------
    outcomes : ndarray of int, shape (n_qubits,)

    Algorithm
    ---------
    Sweep left → right.  At site *i*:

    1. Retrieve the site tensor and transpose axes to (χ_L, d, χ_R).
    2. Contract the accumulated left-boundary vector with the left bond:
       ``C = left_env @ A``.  At site 0 there is no left bond, so ``C = A``.
    3. Apply the rotation: ``C_rot = U_i @ C`` (along the physical axis).
    4. Marginal probability: ``p(s) = ‖C_rot[s, :]‖²``.  The right-canonical
       form guarantees the right environment is the identity, so no further
       contraction is needed.
    5. Sample *s*, then update: ``left_env = C_rot[s, :] / ‖C_rot[s, :]‖``.

    Cost: O(n · χ²) per shot.
    """
    outcomes = np.empty(n_qubits, dtype=int)
    left_env = np.array([1.0 + 0j])   # trivial open-left boundary, shape (1,)

    for i in range(n_qubits):
        A = _get_site_tensor_obc(canonical_mps, i, n_qubits)

        # Contract accumulated left boundary with left bond of site tensor.
        if i == 0:
            C = A                                              # (d, χ_R) or (d,)
        else:
            C = np.tensordot(left_env, A, axes=([0], [0]))    # (d, ...)

        # Apply local rotation unitary: C_rot[s,...] = Σ_{s'} U[s,s'] C[s',...]
        C_rot = np.tensordot(unitaries[i], C, axes=([1], [0]))

        # Marginal probability from squared norm along right-bond axis.
        if C_rot.ndim == 1:
            probs_raw = np.real(C_rot.conj() * C_rot)
        else:
            probs_raw = np.real(np.einsum("dr,dr->d", C_rot.conj(), C_rot))

        probs_raw = np.maximum(probs_raw, 0.0)
        probs = probs_raw / probs_raw.sum()

        s = int(rng.choice(2, p=probs))
        outcomes[i] = s

        if i < n_qubits - 1:
            raw_norm = np.sqrt(max(probs_raw[s], 1e-300))
            left_env = C_rot[s] / raw_norm   # shape (χ_R,) for next site

    return outcomes


def _precompute_pbc_envs(
    mps: Any,
    n_qubits: int,
) -> Tuple[List[Optional[np.ndarray]], int]:
    """
    Pre-compute double-tensor right environments for PBC MPS sampling.

    Mathematical basis
    ------------------
    Define the double-tensor transfer matrix for site *k*::

        T_k[β,β',γ,γ'] = Σ_s  A_k[s]_{β,γ} · A_k*[s]_{β',γ'}

    where A_k[s] is the bare (unrotated) site matrix.

    **Key identity**: for any local unitary U applied to the physical index,
    the rotated transfer matrix equals the bare one::

        Σ_s M_k[s]_{β,γ} M_k*[s]_{β',γ'}   (M_k[s] = Σ_{s'} U[s,s'] A_k[s'])
            = Σ_{t,t'} (Σ_s U[s,t] U*[s,t']) A_k[t]_{β,γ} A_k*[t']_{β',γ'}
            = Σ_t A_k[t]_{β,γ} A_k*[t]_{β',γ'}   ← since Σ_s U[s,t] U*[s,t'] = δ_{t,t'}
            = T_k[β,β',γ,γ']

    Therefore the right environments::

        R[n][β,β',α,α'] = δ_{β,α} δ_{β',α'}   (identity: closes the periodic trace)
        R[k] = T_k ⊗ R[k+1]  for k = n-1, n-2, …, 1
            i.e. R[k][β,β',α,α'] = Σ_{γ,γ'} T_k[β,β',γ,γ'] · R[k+1][γ,γ',α,α']

    are **independent of the per-shot measurement unitaries** and can be
    pre-computed once before the shot loop.

    Storage
    -------
    ``R_by_site[i]`` stores R[i+1], the right environment used when sampling
    site *i*.  The last site (i = n-1) uses ``|Tr(L·M[s])|²`` directly and
    does not access R_by_site[n-1].

    Parameters
    ----------
    mps:
        PBC quimb ``MatrixProductState`` with *n_qubits* sites.
    n_qubits:
        Number of sites.

    Returns
    -------
    R_by_site : list of length n_qubits
        ``R_by_site[i]`` has shape (χ_L_i, χ_L_i, χ_cyc, χ_cyc) and equals
        R[i+1].  ``R_by_site[n-1]`` is the identity (stored but not used in
        the hot path).
    chi_cyc : int
        Dimension of the periodic bond (= left virtual bond of site 0).
    """
    n = n_qubits

    # Extract bare site matrices: A_phys[i][s] = A_i[:, s, :], shape (χ_L, χ_R)
    A_phys: List[List[np.ndarray]] = []
    for i in range(n):
        A = _get_site_tensor_pbc(mps, i, n)       # (χ_L, d, χ_R)
        A_phys.append([A[:, s, :] for s in range(2)])

    chi_cyc: int = A_phys[0][0].shape[0]           # left bond of site 0 = periodic bond

    # Base case: R[n][β,β',α,α'] = δ_{β,α} δ_{β',α'}
    I_cyc = np.eye(chi_cyc, dtype=complex)
    R_n = np.einsum("ia,jb->ijab", I_cyc, I_cyc)  # (χ_cyc, χ_cyc, χ_cyc, χ_cyc)

    R_by_site: List[Optional[np.ndarray]] = [None] * n
    R_by_site[n - 1] = R_n    # last site handled via |Tr|²; stored for completeness

    R_prev = R_n
    for k in range(n - 1, 0, -1):
        # T_k[β,β',γ,γ'] = Σ_s A_k[s]_{β,γ} · A_k*[s]_{β',γ'}
        T_k = sum(
            np.einsum("ik,jl->ijkl", A_phys[k][s], np.conj(A_phys[k][s]))
            for s in range(2)
        )   # shape: (χ_L_k, χ_L_k, χ_R_k, χ_R_k)

        # R[k] = T_k contracted with R[k+1] over the right-bond indices (γ, γ')
        R_k = np.einsum("ijkl,klmn->ijmn", T_k, R_prev)
        # R_k shape: (χ_L_k, χ_L_k, χ_cyc, χ_cyc) = R[k], used at site i = k-1
        R_by_site[k - 1] = R_k
        R_prev = R_k

    return R_by_site, chi_cyc


def _mps_sample_pbc_shot(
    mps: Any,
    unitaries: List[np.ndarray],
    n_qubits: int,
    rng: np.random.Generator,
    R_by_site: List[Optional[np.ndarray]],
    chi_cyc: int,
) -> np.ndarray:
    """
    Single-shot Born-rule sampling for a PBC (cyclic) MPS.

    Uses pre-computed right environments (see ``_precompute_pbc_envs``).
    The MPS is never mutated.

    Algorithm
    ---------
    Maintain a left matrix *L_i = M_0[b_0]·…·M_{i-1}[b_{i-1}]* (L_0 = I_{χ_cyc}).

    For site *i* (i < n-1), the conditional probability is::

        p(s | b_0,…,b_{i-1}) ∝  f(L_i · M_i[s],  R[i+1])

        f(L, R) = Σ_{a,b,c,d}  L[a,b] · L*[c,d] · R[b,d,a,c]
                = einsum('ab,cd,bdac->', L, conj(L), R)

    Index key: a=α_0∈χ_cyc (row of L), b=β∈χ_R (col of L = left bond of R),
               c=α_0'∈χ_cyc, d=β'∈χ_R.  R stored as R[β,β',α_0,α_0'].

    For the last site (i = n-1), close the periodic trace::

        f(L_n, I) = |Tr(L_n)|²

    *L* is Frobenius-renormalised after each step for numerical stability;
    this leaves all probability ratios unchanged because both p(0) and p(1)
    are evaluated with the same L.

    Cost per shot: O(n·χ²) for L updates + O(n·χ⁴) for f evaluations.

    Parameters
    ----------
    mps:
        PBC MPS (read-only).
    unitaries:
        Per-site 2×2 rotation unitaries for this shot.
    n_qubits:
        Number of sites.
    rng:
        Local numpy RNG.
    R_by_site:
        Pre-computed right environments from ``_precompute_pbc_envs``.
        ``R_by_site[i]`` = R[i+1].
    chi_cyc:
        Periodic bond dimension.

    Returns
    -------
    outcomes : ndarray of int, shape (n_qubits,)
    """
    n = n_qubits

    # Rotated site matrices for this shot.
    # M_phys[i][s] = (U_i @ A_i)[s]  =  Σ_{s'} U_i[s,s'] A_i[:,s',:],  shape (χ_L, χ_R)
    M_phys: List[List[np.ndarray]] = []
    for i in range(n):
        A = _get_site_tensor_pbc(mps, i, n)                   # (χ_L, d, χ_R)
        M_rot = np.einsum("sd,ldr->lsr", unitaries[i], A)     # (χ_L, d, χ_R)
        M_phys.append([M_rot[:, s, :] for s in range(2)])     # 2 × (χ_L, χ_R)

    # L_0 = I_{χ_cyc}  (no sites accumulated yet)
    L = np.eye(chi_cyc, dtype=complex)   # (χ_cyc, χ_cyc)

    outcomes = np.empty(n, dtype=int)

    for i in range(n):
        probs_raw = np.empty(2, dtype=float)

        if i < n - 1:
            # R_by_site[i] = R[i+1], shape (χ_R_i, χ_R_i, χ_cyc, χ_cyc)
            R_next = R_by_site[i]
            for s in range(2):
                L_next = L @ M_phys[i][s]          # (χ_cyc, χ_R_i)
                # f(L_next, R_next) = Σ_{a,b,c,d} L_next[a,b] L_next*[c,d] R_next[b,d,a,c]
                probs_raw[s] = float(np.real(
                    np.einsum("ab,cd,bdac->", L_next, np.conj(L_next), R_next)
                ))
        else:
            # Last site: |Tr(L_n)|² closes the periodic trace exactly.
            for s in range(2):
                L_next = L @ M_phys[i][s]          # (χ_cyc, χ_cyc)
                probs_raw[s] = float(np.abs(np.trace(L_next)) ** 2)

        probs_raw = np.maximum(probs_raw, 0.0)
        prob_sum = probs_raw.sum()
        probs = probs_raw / prob_sum

        s_chosen = int(rng.choice(2, p=probs))
        outcomes[i] = s_chosen

        # Update L and re-normalise (Frobenius) to prevent numerical drift.
        # Normalisation cancels in all probability ratios so correctness is preserved.
        L = L @ M_phys[i][s_chosen]
        L_norm = np.linalg.norm(L)
        if L_norm > 1e-300:
            L /= L_norm

    return outcomes


# ─── Data class ───────────────────────────────────────────────────────────────

@dataclass
class ShadowMeasurement:
    """
    A single classical shadow measurement.

    Attributes:
        basis:   Integer array of shape ``(n_qubits,)`` encoding the per-qubit
                 measurement basis.  Interpretation depends on the mode:

                 - ``"random"`` / ``"pauli"``: Pauli index (0=X, 1=Y, 2=Z).
                 - ``"clifford"``: index into ``SINGLE_QUBIT_CLIFFORDS`` (0–23).
                 - ``"custom"``:  user-defined integer labels.

        outcome: Integer array of shape ``(n_qubits,)`` with values in {0, 1},
                 representing the measurement bit string (big-endian).  A 0
                 (1) outcome corresponds to the +1 (−1) eigenvalue of the
                 measured observable.

        unitaries: Optional complex array of shape ``(n_qubits, 2, 2)``
                   storing the actual per-shot rotation unitaries applied
                   before measurement.  This is primarily used for
                   ``measurement_basis="custom"``, where basis labels alone
                   are insufficient to reconstruct the measurement channel.
                   For built-in modes this is normally ``None``.
    """

    basis: np.ndarray
    outcome: np.ndarray
    unitaries: Optional[np.ndarray] = None


# ─── Main collector class ─────────────────────────────────────────────────────

class ShadowCollector:
    """
    Collect classical shadow measurements from dense quantum state vectors.

    Each measurement consists of:
    1. Sampling a per-qubit measurement basis from the configured mode.
    2. Applying the corresponding single-qubit rotation unitaries to the state.
    3. Sampling an outcome bit string from the resulting Born-rule probabilities.

    See ``ShadowConfig`` for a description of each measurement mode.
    """

    def __init__(self, config: ShadowConfig) -> None:
        """
        Initialise the shadow collector.

        Args:
            config: Shadow configuration.  The config's ``seed`` field seeds a
                    *local* RNG (``numpy.random.default_rng``), which avoids
                    any side-effects on the global numpy random state.
        """
        self.config = config
        self.measurements: List[ShadowMeasurement] = []
        self.shadow_data: Optional[np.ndarray] = None

        # Local RNG — never touches np.random global state.
        self.rng: np.random.Generator = np.random.default_rng(config.seed)

    # ── Public sampling API ───────────────────────────────────────────────────

    def sample_dense(self, state_vector: np.ndarray) -> List[ShadowMeasurement]:
        """
        Collect classical shadows from a dense state vector.

        For each of the ``config.n_shadows`` shots:

        1. Sample a per-qubit measurement basis (and corresponding unitaries).
        2. Rotate the full state: apply the tensor-product unitary in-place.
        3. Compute Born-rule probabilities from the rotated amplitudes.
        4. Sample one outcome bit string from those probabilities.
        5. Record basis + outcome as a ``ShadowMeasurement``.

        Args:
            state_vector: Complex 1D array of length ``2^n_qubits`` representing
                          |ψ⟩.  Normalised automatically (with a UserWarning)
                          if its norm deviates from 1.

        Returns:
            List of ``ShadowMeasurement`` objects.  Also stored in
            ``self.measurements``.
        """
        sv = _validate_state_vector(state_vector, self.config.n_qubits)

        n = self.config.n_qubits
        dim = 2 ** n
        outcome_indices = np.arange(dim, dtype=int)

        measurements: List[ShadowMeasurement] = []

        for _ in tqdm(
            range(self.config.n_shadows),
            desc=f"Sampling shadows ({self.config.measurement_basis})",
            leave=False,
        ):
            # 1. Sample basis + rotation unitaries.
            basis, unitaries = self._sample_basis_with_unitaries()

            # 2. Rotate state into the measurement basis.
            rotated = _apply_local_unitaries(sv, unitaries, n)

            # 3. Born-rule probabilities.
            probs = np.abs(rotated) ** 2
            # Re-normalise to absorb floating-point rounding (sum should be ~1).
            probs /= probs.sum()

            # 4. Sample outcome.
            outcome_int = int(self.rng.choice(outcome_indices, p=probs))
            outcome = _int_to_bits(outcome_int, n)
            stored_unitaries = None
            if self.config.measurement_basis == "custom":
                stored_unitaries = _coerce_shot_unitaries(
                    unitaries,
                    n,
                    context="custom_sampler",
                )

            measurements.append(
                ShadowMeasurement(
                    basis=basis,
                    outcome=outcome,
                    unitaries=stored_unitaries,
                )
            )

        self.measurements = measurements
        return measurements

    def sample_mps(self, mps: Any) -> List[ShadowMeasurement]:
        """
        Collect classical shadows from a quimb ``MatrixProductState``.

        Boundary conditions are detected automatically from the MPS object.

        Open boundary conditions (OBC)
        --------------------------------
        A single right-canonical copy (OC at site 0) is prepared *once* before
        the shot loop, amortising the O(n·χ³) SVD cost.  Each shot sweeps
        left→right in O(n·χ²) via ``_mps_sample_obc_shot``.  The canonical
        copy is never mutated — it is read-only across all shots.

        Periodic boundary conditions (PBC)
        ------------------------------------
        Double-tensor right environments R[k] are pre-computed *once* from
        the bare MPS in O(n·χ⁶) via ``_precompute_pbc_envs``.  The transfer
        matrix is independent of the per-shot measurement unitaries (proven in
        the ``_precompute_pbc_envs`` docstring), so this pre-computation is
        valid for the entire shot loop.  Each shot then sweeps left→right in
        O(n·χ⁴) via ``_mps_sample_pbc_shot``.

        A ``UserWarning`` is emitted when the max bond dimension χ exceeds
        ``_PBC_CHI_WARN_THRESHOLD`` (default 16), because the O(n·χ⁶)
        pre-computation becomes expensive.

        PBC support status: **full and exact** for all bond dimensions and all
        quimb cyclic MPS objects.  There is no approximation.

        Qubit / site ordering
        ---------------------
        quimb site 0 maps to the most significant qubit (qubit 0), matching
        the big-endian convention of ``sample_dense()``.

        Basis encoding
        --------------
        Identical to ``sample_dense()``:

        - ``"random"`` / ``"pauli"`` : 0 = X, 1 = Y, 2 = Z
        - ``"clifford"``             : index 0–23 into ``SINGLE_QUBIT_CLIFFORDS``
        - ``"custom"``               : user-defined integer labels

        Args:
            mps: A ``quimb.tensor.MatrixProductState`` with exactly
                 ``config.n_qubits`` sites.  The object is never mutated.

        Returns:
            List of ``ShadowMeasurement`` objects.  Also stored in
            ``self.measurements``.

        Raises:
            ImportError:  If quimb is not installed.
            TypeError:    If *mps* is not a ``quimb.tensor.MatrixProductState``.
            ValueError:   If the number of MPS sites ≠ ``config.n_qubits``.
        """
        if not QUIMB_AVAILABLE:
            raise ImportError(
                "quimb is required for MPS shadow sampling.\n"
                "Install with:  pip install quimb"
            )

        if not isinstance(mps, _qtn.MatrixProductState):
            raise TypeError(
                f"sample_mps() expects a quimb MatrixProductState, "
                f"got {type(mps).__name__}.\n"
                "For dense state vectors use sample_dense() instead."
            )

        n = self.config.n_qubits
        if mps.L != n:
            raise ValueError(
                f"MPS has {mps.L} sites but config.n_qubits = {n}."
            )

        is_pbc = _is_cyclic_mps(mps, n)
        measurements: List[ShadowMeasurement] = []

        if is_pbc:
            # ── PBC path ───────────────────────────────────────────────────
            # Warn if bond dimension makes pre-computation expensive.
            try:
                chi_max: int = mps.max_bond()
            except Exception:
                chi_max = 0
            if chi_max > _PBC_CHI_WARN_THRESHOLD:
                warnings.warn(
                    f"PBC MPS sampling: max bond dimension χ = {chi_max}.  "
                    f"Pre-computation cost ≈ O(n·χ⁶) = O({n}·{chi_max}⁶ "
                    f"≈ {n * chi_max**6:.2e}) operations.  "
                    f"Per-shot cost ≈ O(n·χ⁴) = O({n * chi_max**4:.2e}).  "
                    "Consider OBC or a smaller bond dimension for better performance.",
                    UserWarning,
                    stacklevel=2,
                )

            # Pre-compute right environments once (O(n·χ⁶), amortised over all shots).
            R_by_site, chi_cyc = _precompute_pbc_envs(mps, n)

            for _ in tqdm(
                range(self.config.n_shadows),
                desc=f"Sampling PBC-MPS shadows ({self.config.measurement_basis})",
                leave=False,
            ):
                basis, unitaries = self._sample_basis_with_unitaries()
                outcome = _mps_sample_pbc_shot(
                    mps, unitaries, n, self.rng, R_by_site, chi_cyc
                )
                stored_unitaries = None
                if self.config.measurement_basis == "custom":
                    stored_unitaries = _coerce_shot_unitaries(
                        unitaries,
                        n,
                        context="custom_sampler",
                    )
                measurements.append(
                    ShadowMeasurement(
                        basis=basis,
                        outcome=outcome,
                        unitaries=stored_unitaries,
                    )
                )

        else:
            # ── OBC path ───────────────────────────────────────────────────
            # Prepare right-canonical form once (O(n·χ³), amortised over all shots).
            canonical = _prepare_canonical_mps(mps)

            for _ in tqdm(
                range(self.config.n_shadows),
                desc=f"Sampling OBC-MPS shadows ({self.config.measurement_basis})",
                leave=False,
            ):
                basis, unitaries = self._sample_basis_with_unitaries()
                outcome = _mps_sample_obc_shot(canonical, unitaries, n, self.rng)
                stored_unitaries = None
                if self.config.measurement_basis == "custom":
                    stored_unitaries = _coerce_shot_unitaries(
                        unitaries,
                        n,
                        context="custom_sampler",
                    )
                measurements.append(
                    ShadowMeasurement(
                        basis=basis,
                        outcome=outcome,
                        unitaries=stored_unitaries,
                    )
                )

        self.measurements = measurements
        return measurements

    # ── Data access ───────────────────────────────────────────────────────────

    def get_shadow_data(self) -> np.ndarray:
        """
        Return all measurements as a numpy integer array.

        Returns:
            Array of shape ``(n_shadows, n_qubits, 2)`` where
            ``arr[i, :, 0]`` is the basis array and ``arr[i, :, 1]`` is the
            outcome array for measurement ``i``.
        """
        if not self.measurements:
            raise ValueError(
                "No measurements collected. "
                "Call sample_dense() or sample_mps() first."
            )

        n = self.config.n_qubits
        shadow_data = np.zeros((len(self.measurements), n, 2), dtype=int)
        for i, m in enumerate(self.measurements):
            shadow_data[i, :, 0] = m.basis
            shadow_data[i, :, 1] = m.outcome

        self.shadow_data = shadow_data
        return shadow_data

    def save_shadows(self, filename: Optional[str] = None) -> str:
        """
        Save shadow measurements to a compressed ``.npz`` file.

        File contents
        -------------
        - ``bases``            — int array, shape ``(n_shadows, n_qubits)``
        - ``outcomes``         — int array, shape ``(n_shadows, n_qubits)``
        - ``n_qubits``         — scalar metadata
        - ``n_shadows``        — scalar metadata (actual number of measurements)
        - ``measurement_basis`` — string metadata (0-d object array)
        - ``has_stored_unitaries`` — bool metadata indicating whether
          per-shot unitary matrices are persisted
        - ``custom_unitaries`` — optional complex array with shape
          ``(n_shadows, n_qubits, 2, 2)`` when per-shot unitaries are stored

        Args:
            filename: Output filename.  Auto-generated as
                      ``shadows_n{n_qubits}_s{n_shadows}.npz`` when ``None``.

        Returns:
            Absolute path to the saved ``.npz`` file.
        """
        if not self.measurements:
            raise ValueError(
                "No measurements to save. "
                "Call sample_dense() or sample_mps() first."
            )

        n_saved = len(self.measurements)
        if filename is None:
            filename = (
                f"shadows_n{self.config.n_qubits}_s{n_saved}.npz"
            )

        os.makedirs(self.config.output_dir, exist_ok=True)
        # np.savez_compressed appends ".npz" if the path does not end in it.
        filepath_stem = os.path.join(self.config.output_dir, filename)
        if filepath_stem.endswith(".npz"):
            filepath_stem = filepath_stem[:-4]

        bases = np.array([m.basis for m in self.measurements], dtype=int)
        outcomes = np.array([m.outcome for m in self.measurements], dtype=int)
        has_unitaries_per_shot = [m.unitaries is not None for m in self.measurements]
        has_stored_unitaries = any(has_unitaries_per_shot)

        if has_stored_unitaries and not all(has_unitaries_per_shot):
            raise ValueError(
                "Cannot save shadow data with mixed unitary presence across shots. "
                "Either all measurements must store per-shot unitaries or none may."
            )
        if self.config.measurement_basis == "custom" and not has_stored_unitaries:
            raise ValueError(
                "Cannot save custom-basis shadows without stored per-shot unitaries."
            )

        save_kwargs = dict(
            bases=bases,
            outcomes=outcomes,
            n_qubits=np.array(self.config.n_qubits),
            n_shadows=np.array(n_saved),
            measurement_basis=np.array(self.config.measurement_basis),
            has_stored_unitaries=np.array(has_stored_unitaries),
        )

        if has_stored_unitaries:
            save_kwargs["custom_unitaries"] = np.stack(
                [
                    _coerce_shot_unitaries(
                        m.unitaries,
                        self.config.n_qubits,
                        context=f"measurement {i}",
                    )
                    for i, m in enumerate(self.measurements)
                ],
                axis=0,
            )

        np.savez_compressed(filepath_stem, **save_kwargs)

        filepath = filepath_stem + ".npz"
        print(f"Saved {n_saved} shadow measurements to {filepath}")
        return filepath

    def load_shadows(self, filepath: str) -> None:
        """
        Load shadow measurements from a ``.npz`` file created by
        ``save_shadows()``.

        Supports both the current format (``bases`` / ``outcomes`` keys) and
        the legacy format (``shadow_data`` key with shape
        ``(n_shadows, n_qubits, 2)``). Current-format files may additionally
        include ``has_stored_unitaries`` and ``custom_unitaries`` for
        ``measurement_basis="custom"`` data.

        Args:
            filepath: Path to the ``.npz`` file.
        """
        data = np.load(filepath, allow_pickle=True)

        if "bases" in data and "outcomes" in data:
            bases = data["bases"].astype(int)
            outcomes = data["outcomes"].astype(int)
        elif "shadow_data" in data:
            # Legacy format written by the old save_shadows().
            sd = data["shadow_data"]
            bases = sd[:, :, 0].astype(int)
            outcomes = sd[:, :, 1].astype(int)
        else:
            raise ValueError(
                f"Unrecognised file format in {filepath!r}. "
                "Expected keys 'bases'+'outcomes' (current) or "
                "'shadow_data' (legacy)."
            )

        measurement_basis = None
        if "measurement_basis" in data:
            measurement_basis = str(np.asarray(data["measurement_basis"]).item())

        if "n_qubits" in data:
            self.config.n_qubits = int(np.asarray(data["n_qubits"]).item())
        self.config.n_shadows = int(bases.shape[0])
        if measurement_basis is not None:
            self.config.measurement_basis = measurement_basis

        has_stored_unitaries = False
        if "has_stored_unitaries" in data:
            has_stored_unitaries = bool(np.asarray(data["has_stored_unitaries"]).item())

        loaded_unitaries: Optional[np.ndarray] = None
        if "custom_unitaries" in data:
            loaded_unitaries = np.asarray(data["custom_unitaries"], dtype=complex)
            expected_shape = (bases.shape[0], bases.shape[1], 2, 2)
            if loaded_unitaries.shape != expected_shape:
                raise ValueError(
                    f"Loaded custom_unitaries with shape {loaded_unitaries.shape}, "
                    f"expected {expected_shape}."
                )
            loaded_unitaries = np.stack(
                [
                    _coerce_shot_unitaries(
                        loaded_unitaries[i],
                        bases.shape[1],
                        context=f"loaded custom_unitaries[{i}]",
                    )
                    for i in range(loaded_unitaries.shape[0])
                ],
                axis=0,
            )
            has_stored_unitaries = True
        elif has_stored_unitaries or measurement_basis == "custom":
            raise ValueError(
                f"Shadow file {filepath!r} declares custom-basis unitary data "
                "but does not contain a valid 'custom_unitaries' array."
            )

        self.measurements = [
            ShadowMeasurement(
                basis=bases[i],
                outcome=outcomes[i],
                unitaries=None if loaded_unitaries is None else loaded_unitaries[i],
            )
            for i in range(bases.shape[0])
        ]

        # Cache the array form.
        n_loaded = len(self.measurements)
        n_qubits = bases.shape[1]
        cached = np.zeros((n_loaded, n_qubits, 2), dtype=int)
        cached[:, :, 0] = bases
        cached[:, :, 1] = outcomes
        self.shadow_data = cached

        print(f"Loaded {n_loaded} shadow measurements from {filepath}")

    def get_statistics(self) -> dict:
        """
        Return summary statistics for the collected measurements.

        Returns:
            Dict with keys: ``n_measurements``, ``n_qubits``,
            ``measurement_basis``, ``basis_value_counts``,
            ``mean_outcome``, ``outcome_std``.
            Returns an empty dict if no measurements have been collected.
        """
        if not self.measurements:
            return {}

        bases = np.array([m.basis for m in self.measurements])    # (n, n_qubits)
        outcomes = np.array([m.outcome for m in self.measurements])  # (n, n_qubits)

        # Count how often each basis value appears (across all qubits).
        unique, counts = np.unique(bases, return_counts=True)
        basis_counts = {int(v): int(c) for v, c in zip(unique, counts)}

        return {
            "n_measurements": len(self.measurements),
            "n_qubits": self.config.n_qubits,
            "measurement_basis": self.config.measurement_basis,
            "basis_value_counts": basis_counts,
            "mean_outcome": float(outcomes.mean()),
            "outcome_std": float(outcomes.std()),
        }

    def __repr__(self) -> str:
        return (
            f"ShadowCollector("
            f"n_qubits={self.config.n_qubits}, "
            f"n_shadows={self.config.n_shadows}, "
            f"basis={self.config.measurement_basis!r})"
        )

    # ── Internal basis sampling ───────────────────────────────────────────────

    def _sample_basis_with_unitaries(
        self,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Sample one measurement basis and return both the integer label array
        and the corresponding per-qubit rotation unitaries.

        Returns:
            basis:     Integer array of shape ``(n_qubits,)``.
            unitaries: List of ``n_qubits`` 2×2 complex unitary matrices.
                       Applying ``unitaries[i]`` to qubit ``i`` and measuring
                       in Z is equivalent to measuring qubit ``i`` in the
                       sampled basis.
        """
        mode = self.config.measurement_basis
        if mode == "random":
            return self._sample_random_pauli()
        elif mode == "pauli":
            return self._sample_weighted_pauli()
        elif mode == "clifford":
            return self._sample_local_clifford()
        elif mode == "custom":
            return self._sample_custom()
        else:
            raise ValueError(f"Unknown measurement_basis: {mode!r}")

    def _sample_random_pauli(
        self,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Sample a uniformly random Pauli (X/Y/Z) independently per qubit.

        Basis encoding: 0 = X, 1 = Y, 2 = Z.
        """
        basis = self.rng.integers(0, 3, size=self.config.n_qubits)
        unitaries = [_PAULI_ROTATIONS[int(b)] for b in basis]
        return basis, unitaries

    def _sample_weighted_pauli(
        self,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Sample a Pauli (X/Y/Z) per qubit according to ``config.pauli_weights``.

        Basis encoding: 0 = X, 1 = Y, 2 = Z.
        """
        weights = np.asarray(self.config.pauli_weights, dtype=float)
        weights = weights / weights.sum()  # defensive renormalisation
        basis = self.rng.choice(3, size=self.config.n_qubits, p=weights)
        unitaries = [_PAULI_ROTATIONS[int(b)] for b in basis]
        return basis, unitaries

    def _sample_local_clifford(
        self,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Sample one of the 24 single-qubit Clifford gates per qubit (uniform).

        This is a *local* random Clifford measurement: each qubit is handled
        independently.  It is strictly more general than random Pauli (which
        uses 3 of the 24 Cliffords) and provides better channel properties for
        estimating k-body observables with bounded locality.

        Basis encoding: integer i ∈ [0, 23] → ``SINGLE_QUBIT_CLIFFORDS[i]``.
        """
        basis = self.rng.integers(0, _N_CLIFFORDS, size=self.config.n_qubits)
        unitaries = [SINGLE_QUBIT_CLIFFORDS[int(b)] for b in basis]
        return basis, unitaries

    def _sample_custom(
        self,
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Delegate basis sampling to the user-supplied ``config.custom_sampler``.

        Validates that the sampler returns the expected shapes before proceeding.
        """
        result = self.config.custom_sampler(self.config.n_qubits, self.rng)

        try:
            basis_raw, unitaries = result
        except (TypeError, ValueError):
            raise ValueError(
                "custom_sampler must return a 2-tuple "
                "(basis_labels: np.ndarray, unitaries: list[np.ndarray])."
            )

        basis = np.asarray(basis_raw, dtype=int)
        n = self.config.n_qubits

        if basis.shape != (n,):
            raise ValueError(
                f"custom_sampler returned basis with shape {basis.shape}, "
                f"expected ({n},)."
            )

        unitary_array = _coerce_shot_unitaries(
            unitaries,
            n,
            context="custom_sampler",
        )

        return basis, [unitary_array[i] for i in range(n)]


# ─── Convenience function ─────────────────────────────────────────────────────

def collect_shadows_from_state(
    state: Union[np.ndarray, Any],
    config: ShadowConfig,
) -> "ShadowCollector":
    """
    Collect classical shadows from a quantum state (convenience wrapper).

    Dispatches to ``sample_dense()`` for ``np.ndarray`` inputs and to
    ``sample_mps()`` for quimb ``MatrixProductState`` objects.

    Args:
        state: Dense state vector (``np.ndarray``) **or** a quimb
               ``MatrixProductState`` (e.g. from ``DMRGSolver.ground_state``).
        config: Shadow configuration.

    Returns:
        ``ShadowCollector`` with collected measurements.
    """
    collector = ShadowCollector(config)
    if isinstance(state, np.ndarray):
        collector.sample_dense(state)
    else:
        collector.sample_mps(state)
    return collector
