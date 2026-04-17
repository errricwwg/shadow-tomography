"""
hamiltonians.py — Hamiltonian abstraction layer for ShadowGPT.

Each Hamiltonian family is represented by a ``HamiltonianSpec`` that bundles:

  * ``dense_matrix``      — exact 2^n × 2^n matrix for exact diagonalisation
                            (ground-state preparation, exact observable targets)
  * ``pauli_hamiltonian`` — PauliPolynomial-like object for shadow-based energy
                            estimation via ShadowProcessor.estimate_energy()

The two representations are kept in sync by ``build_hamiltonian_spec()``.

Currently supported
-------------------
    "tfim"          H = -J Σ Z_i Z_{i+1}  -  h Σ X_i            (OBC)
                    dense_matrix + pauli_hamiltonian both available.

    "ising_general" H = -J Σ ZZ  -  hx Σ X  -  hz Σ Z           (OBC)
                    dense_matrix + pauli_hamiltonian both available.

    "xxz"           H = J Σ (X_i X_{i+1} + Y_i Y_{i+1} + δ Z_i Z_{i+1}) (OBC)
                    dense_matrix + pauli_hamiltonian both available.

    "heisenberg"    H = J Σ (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})   (OBC)
                    dense_matrix + pauli_hamiltonian both available
                    (delegates to XXZ with delta=1).

PauliPolynomial interface
-------------------------
``ShadowProcessor.estimate_energy()`` duck-types its ``hamiltonian`` argument:
it must have a ``.N`` attribute (qubit count) and be iterable over Pauli terms.
Each term must expose:
    .c  — float coefficient
    .p  — int phase power  (effective coeff = c × (1j)^p)
    .g  — np.ndarray of shape (2*N,): flat symplectic bits [x_0,z_0, x_1,z_1, ...]
              x_i=1 → X on qubit i,  z_i=1 → Z on qubit i

When pyclifford is installed, ``build_tfim_pauli_hamiltonian()`` uses
``pyclifford.ham_tf_ising()`` directly.  Otherwise it falls back to the
built-in ``_PauliPolynomial`` class which satisfies the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

# ── Optional pyclifford (only needed for its ham_tf_ising builder) ────────────
try:
    from pyclifford import ham_tf_ising as _pyc_ham_tf_ising   # type: ignore[import]
    _PYCLIFFORD_AVAILABLE = True
except ImportError:
    _pyc_ham_tf_ising = None
    _PYCLIFFORD_AVAILABLE = False

# ── Pauli matrices ─────────────────────────────────────────────────────────────
_I2 = np.eye(2, dtype=complex)
_X  = np.array([[0,   1 ], [1,   0 ]], dtype=complex)
_Y  = np.array([[0, -1j ], [1j,  0 ]], dtype=complex)
_Z  = np.array([[1,   0 ], [0,  -1 ]], dtype=complex)


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# HamiltonianSpec
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

@dataclass
class HamiltonianSpec:
    """
    Bundled specification for a quantum spin-chain Hamiltonian.

    Attributes
    ----------
    family            : str — Hamiltonian family name (e.g., ``"tfim"``).
    n_qubits          : int — number of qubits / lattice sites.
    params            : dict — family-specific parameters (e.g., ``{"J": 1.0, "h": 0.5}``).
    dense_matrix      : np.ndarray | None — exact 2^n × 2^n Hamiltonian matrix.
                        Used for ``np.linalg.eigh()`` (exact diagonalisation).
    pauli_hamiltonian : Any | None — PauliPolynomial-like object for shadow-based
                        energy estimation via ``ShadowProcessor.estimate_energy()``.
                        ``None`` when the family's Pauli builder is not yet implemented.
    """
    family:            str
    n_qubits:          int
    params:            Dict[str, float] = field(default_factory=dict)
    dense_matrix:      Optional[np.ndarray] = None
    pauli_hamiltonian: Optional[Any] = None


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Minimal duck-typed PauliPolynomial (no pyclifford required)
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

class _PauliTerm:
    """
    Single Pauli term compatible with ``ShadowProcessor.estimate_energy()``.

    Attributes
    ----------
    c : float — coefficient (real for a Hermitian Hamiltonian).
    p : int   — phase power; effective coefficient = c * (1j)^p.
    g : np.ndarray — flat symplectic bit vector of shape (2*N,).
                     Encoding: g[2i] = x_i (X bit), g[2i+1] = z_i (Z bit).
                     I → (0,0), X → (1,0), Z → (0,1), Y → (1,1).
    """
    __slots__ = ("c", "p", "g")

    def __init__(self, c: float, p: int, g: np.ndarray) -> None:
        self.c = c
        self.p = p
        self.g = g


class _PauliPolynomial:
    """
    Minimal PauliPolynomial-like container compatible with
    ``ShadowProcessor.estimate_energy()`` duck-typing.

    Satisfies both checks:
        * ``hasattr(ham, "N")``              — qubit count
        * ``hasattr(ham, "__iter__")``       — iterable over _PauliTerm
    """

    def __init__(self, n_qubits: int, terms: List[_PauliTerm]) -> None:
        self.N = n_qubits
        self._terms = list(terms)

    def __iter__(self):
        return iter(self._terms)

    def __len__(self) -> int:
        return len(self._terms)

    def __repr__(self) -> str:
        return f"_PauliPolynomial(N={self.N}, n_terms={len(self._terms)})"


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Internal helpers
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def _kron_op(op: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """Embed single-qubit ``op`` on ``qubit`` into the full 2^n × 2^n space."""
    mats = [_I2] * n_qubits
    mats[qubit] = op
    result = mats[0]
    for m in mats[1:]:
        result = np.kron(result, m)
    return result


def _symplectic_g(qubit_paulis: Dict[int, str], n_qubits: int) -> np.ndarray:
    """
    Build the flat symplectic bit vector g of shape (2*n_qubits,).

    Args:
        qubit_paulis : Mapping from qubit index to Pauli string ('X', 'Y', or 'Z').
                       Qubits absent from the dict are treated as identity.
        n_qubits     : System size.

    Returns:
        np.ndarray of int8, shape (2*n_qubits,).
        Encoding: g[2*i] = x_i (X bit), g[2*i+1] = z_i (Z bit).
            I → (0, 0),  X → (1, 0),  Z → (0, 1),  Y → (1, 1).
    """
    g = np.zeros(2 * n_qubits, dtype=np.int8)
    for i, pauli in qubit_paulis.items():
        if pauli == "X":
            g[2 * i] = 1
        elif pauli == "Z":
            g[2 * i + 1] = 1
        elif pauli == "Y":
            g[2 * i]     = 1
            g[2 * i + 1] = 1
    return g


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Dense matrix builders
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def build_tfim_dense_matrix(n_qubits: int, J: float, h: float) -> np.ndarray:
    """
    Dense 2^n × 2^n matrix for the transverse-field Ising model (OBC):

        H = -J Σ_{i=0}^{n-2} Z_i Z_{i+1}  -  h Σ_{i=0}^{n-1} X_i

    Args:
        n_qubits : System size.
        J        : ZZ coupling (positive = ferromagnetic).
        h        : Transverse-field strength.

    Returns:
        np.ndarray, shape (2**n, 2**n), dtype complex128.
    """
    dim = 2 ** n_qubits
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(n_qubits - 1):                          # ZZ coupling (OBC)
        H -= J * (_kron_op(_Z, i, n_qubits) @ _kron_op(_Z, i + 1, n_qubits))
    for i in range(n_qubits):                              # transverse field
        H -= h * _kron_op(_X, i, n_qubits)
    return H


def build_ising_general_dense_matrix(
    n_qubits: int, J: float, hx: float, hz: float
) -> np.ndarray:
    """
    Dense matrix for the general / longitudinal Ising model (OBC):

        H = -J Σ Z_i Z_{i+1}  -  hx Σ X_i  -  hz Σ Z_i

    Args:
        n_qubits : System size.
        J        : ZZ coupling.
        hx       : Transverse-field (X) strength.
        hz       : Longitudinal-field (Z) strength.

    Returns:
        np.ndarray, shape (2**n, 2**n), dtype complex128.
    """
    dim = 2 ** n_qubits
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(n_qubits - 1):
        H -= J  * (_kron_op(_Z, i, n_qubits) @ _kron_op(_Z, i + 1, n_qubits))
    for i in range(n_qubits):
        H -= hx * _kron_op(_X, i, n_qubits)
        H -= hz * _kron_op(_Z, i, n_qubits)
    return H


def build_xxz_dense_matrix(n_qubits: int, J: float, delta: float) -> np.ndarray:
    """
    Dense matrix for the XXZ model (OBC):

        H = J Σ ( X_i X_{i+1} + Y_i Y_{i+1} + delta * Z_i Z_{i+1} )

    Special cases: delta=0 → XX model; delta=1 → isotropic Heisenberg.

    Args:
        n_qubits : System size.
        J        : Exchange coupling.
        delta    : Anisotropy parameter.

    Returns:
        np.ndarray, shape (2**n, 2**n), dtype complex128.
    """
    dim = 2 ** n_qubits
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(n_qubits - 1):
        H += J * (_kron_op(_X, i, n_qubits) @ _kron_op(_X, i + 1, n_qubits))
        H += J * (_kron_op(_Y, i, n_qubits) @ _kron_op(_Y, i + 1, n_qubits))
        H += J * delta * (_kron_op(_Z, i, n_qubits) @ _kron_op(_Z, i + 1, n_qubits))
    return H


def build_heisenberg_dense_matrix(n_qubits: int, J: float) -> np.ndarray:
    """
    Dense matrix for the isotropic Heisenberg model (OBC, delta=1):

        H = J Σ ( X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1} )

    Args:
        n_qubits : System size.
        J        : Exchange coupling.

    Returns:
        np.ndarray, shape (2**n, 2**n), dtype complex128.
    """
    return build_xxz_dense_matrix(n_qubits, J=J, delta=1.0)


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Consistency check (developer utility)
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def _reconstruct_dense_from_pauli(pauli_ham) -> np.ndarray:
    """
    Reconstruct a dense 2^n × 2^n matrix from a _PauliPolynomial by expanding
    each term's symplectic bits back into a full tensor-product operator.

    Used internally by ``_assert_pauli_matches_dense()`` to verify that the
    Pauli and dense builders encode the same Hamiltonian.

    Each term contributes: c_eff * (σ_0 ⊗ σ_1 ⊗ ... ⊗ σ_{n-1})
    where σ_i ∈ {I, X, Y, Z} is determined by (x_i, z_i):
        (0,0) → I,  (1,0) → X,  (0,1) → Z,  (1,1) → Y.
    """
    n = pauli_ham.N
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)
    # Map symplectic (x,z) bits to the matrix actually used by the estimator:
    #   _pauli_matrix_from_bits(x,z) = X^x Z^z
    # For (1,1) this is XZ = [[0,-1],[1,0]] = -iY, NOT physical Y.
    # Using physical Y here would give a wrong sign for any term with Y operators
    # (harmless for TFIM/ising_general which have no Y terms, but wrong for XXZ).
    _XZ = _X @ _Z   # [[0,-1],[1,0]], matches processor._pauli_matrix_from_bits(1,1)
    _pauli_from_bits = {
        (0, 0): _I2,
        (1, 0): _X,
        (0, 1): _Z,
        (1, 1): _XZ,   # XZ = -iY; phase compensated by c_eff encoding
    }
    for term in pauli_ham:
        c_eff = complex(term.c) * (1j ** int(term.p))
        g = term.g.reshape(-1, 2)    # (n, 2): g[i] = [x_i, z_i]
        mat = _pauli_from_bits[(int(g[0, 0]), int(g[0, 1]))]
        for i in range(1, n):
            mat = np.kron(mat, _pauli_from_bits[(int(g[i, 0]), int(g[i, 1]))])
        H += c_eff * mat
    return H


def _assert_pauli_matches_dense(
    pauli_ham,
    dense_mat: np.ndarray,
    atol: float = 1e-10,
    label: str = "",
) -> None:
    """
    Assert that a _PauliPolynomial and a dense matrix represent the same
    Hamiltonian, up to ``atol``.

    Raises AssertionError with a descriptive message if they differ.
    Intended for developer use and unit tests, not production code paths.

    Example usage::

        spec = build_hamiltonian_spec("ising_general", 3, J=1.0, hx=0.5, hz=0.2)
        _assert_pauli_matches_dense(spec.pauli_hamiltonian, spec.dense_matrix,
                                    label="ising_general n=3")
    """
    H_recon = _reconstruct_dense_from_pauli(pauli_ham)
    if not np.allclose(H_recon, dense_mat, atol=atol):
        max_diff = float(np.max(np.abs(H_recon - dense_mat)))
        raise AssertionError(
            f"Pauli ↔ dense mismatch{' (' + label + ')' if label else ''}: "
            f"max |H_pauli - H_dense| = {max_diff:.3e}  (atol={atol})"
        )


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Pauli Hamiltonian builders
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def build_tfim_pauli_hamiltonian(n_qubits: int, J: float, h: float):
    """
    Build a PauliPolynomial-like object for the TFIM Hamiltonian:

        H = -J Σ_{i=0}^{n-2} Z_i Z_{i+1}  -  h Σ_{i=0}^{n-1} X_i   (OBC)

    The returned object satisfies ``ShadowProcessor.estimate_energy()``'s
    duck-typing interface:
        * ``.N`` (qubit count)
        * iterable over terms with ``.c``, ``.p``, ``.g`` attributes

    Implementation
    --------------
    1. If pyclifford is installed, delegates to ``pyclifford.ham_tf_ising()``.
    2. Otherwise falls back to the built-in ``_PauliPolynomial`` class,
       which constructs the same interface manually.

    Term encoding
    -------------
    ZZ terms at (i, i+1): c = -J, p = 0, g with z_i=1 and z_{i+1}=1.
    X  terms at i:        c = -h, p = 0, g with x_i=1.

    Args:
        n_qubits : System size.
        J        : ZZ coupling.
        h        : Transverse-field strength.

    Returns:
        PauliPolynomial-like object compatible with estimate_energy().
    """
    # 1. Prefer pyclifford (its ham_tf_ising may include phase conventions we lack).
    if _PYCLIFFORD_AVAILABLE:
        try:
            return _pyc_ham_tf_ising(n_qubits, J, h)
        except Exception:
            pass   # fall through to manual construction

    # 2. Manual construction — always works, no external dependency.
    terms: List[_PauliTerm] = []

    for i in range(n_qubits - 1):                        # ZZ coupling terms
        g = _symplectic_g({i: "Z", i + 1: "Z"}, n_qubits)
        terms.append(_PauliTerm(c=-J, p=0, g=g))

    for i in range(n_qubits):                            # transverse-field terms
        g = _symplectic_g({i: "X"}, n_qubits)
        terms.append(_PauliTerm(c=-h, p=0, g=g))

    return _PauliPolynomial(n_qubits, terms)


def build_ising_general_pauli_hamiltonian(
    n_qubits: int, J: float, hx: float, hz: float
):
    """
    Build a PauliPolynomial-like object for the general / longitudinal Ising
    Hamiltonian:

        H = -J Σ_{i=0}^{n-2} Z_i Z_{i+1}  -  hx Σ_{i=0}^{n-1} X_i
             -  hz Σ_{i=0}^{n-1} Z_i                               (OBC)

    The returned object satisfies ``ShadowProcessor.estimate_energy()``'s
    duck-typing interface:
        * ``.N`` (qubit count)
        * iterable over terms with ``.c``, ``.p``, ``.g`` attributes

    This family extends TFIM by adding an extra longitudinal field hz Σ Z_i.
    There is no dedicated pyclifford builder for this family, so the fallback
    ``_PauliPolynomial`` is always used.

    Term encoding
    -------------
    ZZ  terms at (i, i+1): c = -J,  p = 0, g with z_i=1 and z_{i+1}=1.
    X   terms at i:        c = -hx, p = 0, g with x_i=1.
    Z   terms at i:        c = -hz, p = 0, g with z_i=1.

    Consistency with dense builder
    --------------------------------
    ``build_ising_general_dense_matrix(n, J, hx, hz)`` and this function encode
    exactly the same Hamiltonian with OBC:
        dense_matrix: used for np.linalg.eigh() to obtain exact eigenvalues /
                      ground-state vectors.
        pauli_hamiltonian: used for ShadowProcessor.estimate_energy() via the
                           classical-shadow estimator.
    Both must represent the same physical Hamiltonian; otherwise energy
    estimates will not converge to the diagonalisation reference.

    Developer sanity check (n=2):
        H_dense = build_ising_general_dense_matrix(2, J, hx, hz)
        H_recon = sum(term.c * kron_full(g) for term in ham)
        np.allclose(H_dense, H_recon)   # must be True
    See ``_assert_pauli_matches_dense`` below for an automated version.

    Args:
        n_qubits : System size.
        J        : ZZ coupling.
        hx       : Transverse-field (X) strength.
        hz       : Longitudinal-field (Z) strength.

    Returns:
        _PauliPolynomial compatible with estimate_energy().
    """
    terms: List[_PauliTerm] = []

    for i in range(n_qubits - 1):          # ZZ coupling terms (OBC)
        g = _symplectic_g({i: "Z", i + 1: "Z"}, n_qubits)
        terms.append(_PauliTerm(c=-J, p=0, g=g))

    for i in range(n_qubits):              # transverse-field (X) terms
        g = _symplectic_g({i: "X"}, n_qubits)
        terms.append(_PauliTerm(c=-hx, p=0, g=g))

    for i in range(n_qubits):              # longitudinal-field (Z) terms
        g = _symplectic_g({i: "Z"}, n_qubits)
        terms.append(_PauliTerm(c=-hz, p=0, g=g))

    return _PauliPolynomial(n_qubits, terms)


def build_xxz_pauli_hamiltonian(n_qubits: int, J: float, delta: float):
    """
    Build a PauliPolynomial-like object for the XXZ Hamiltonian:

        H = J Σ_{i=0}^{n-2} ( X_i X_{i+1} + Y_i Y_{i+1} + delta * Z_i Z_{i+1} )   (OBC)

    The returned object satisfies ``ShadowProcessor.estimate_energy()``'s
    duck-typing interface:
        * ``.N`` (qubit count)
        * iterable over terms with ``.c``, ``.p``, ``.g`` attributes

    Phase convention for Y terms (important)
    -----------------------------------------
    ``ShadowProcessor._pauli_matrix_from_bits(1, 1)`` returns ``X·Z = -iY``,
    NOT physical Y.  The per-shot estimator evaluates:

        c_eff * ⟨(X·Z)_i (X·Z)_{i+1}⟩

    For a physical Y_i Y_{i+1} term with coefficient J we need:

        c_eff * ⟨(XZ)_i (XZ)_{i+1}⟩  =  J * ⟨Y_i Y_{i+1}⟩
        c_eff * (-i)(-i) * ⟨Y_i Y_{i+1}⟩  =  J * ⟨Y_i Y_{i+1}⟩
        c_eff * (-1)  =  J
        c_eff  =  -J

    So the stored coefficient for YY bonds is ``c = -J`` with ``p = 0``.
    All other terms (XX and delta·ZZ) have the direct coefficient because X and Z
    are their own symplectic operators without a phase correction.

    Term encoding summary
    ---------------------
    XX at (i,i+1)  :  c =  J,       p = 0,  g with x_i=1, x_{i+1}=1
    YY at (i,i+1)  :  c = -J,       p = 0,  g with x_i=z_i=1, x_{i+1}=z_{i+1}=1
    ZZ at (i,i+1)  :  c =  J*delta, p = 0,  g with z_i=1, z_{i+1}=1

    Consistency with dense builder
    --------------------------------
    ``build_xxz_dense_matrix(n, J, delta)`` and this function encode the same
    OBC Hamiltonian.  Verified by ``_assert_pauli_matches_dense`` for n=2,3,4.

    Args:
        n_qubits : System size.
        J        : Exchange coupling.
        delta    : Anisotropy parameter.

    Returns:
        _PauliPolynomial compatible with estimate_energy().
    """
    terms: List[_PauliTerm] = []

    for i in range(n_qubits - 1):
        # XX bond: direct coefficient, no phase correction needed
        g_xx = _symplectic_g({i: "X", i + 1: "X"}, n_qubits)
        terms.append(_PauliTerm(c=J, p=0, g=g_xx))

        # YY bond: coefficient is -J because _pauli_matrix_from_bits(1,1) = XZ = -iY,
        # so (XZ)_i(XZ)_{i+1} = (-iY)_i(-iY)_{i+1} = (-1)*Y_i*Y_{i+1}.
        # To recover physical J*YY we need c_eff = -J.
        g_yy = _symplectic_g({i: "Y", i + 1: "Y"}, n_qubits)
        terms.append(_PauliTerm(c=-J, p=0, g=g_yy))

        # ZZ bond: direct coefficient
        g_zz = _symplectic_g({i: "Z", i + 1: "Z"}, n_qubits)
        terms.append(_PauliTerm(c=J * delta, p=0, g=g_zz))

    return _PauliPolynomial(n_qubits, terms)


def build_heisenberg_pauli_hamiltonian(n_qubits: int, J: float):
    """
    Build a PauliPolynomial-like object for the isotropic Heisenberg model:

        H = J Σ_{i=0}^{n-2} ( X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1} )   (OBC)

    The Heisenberg model is exactly XXZ with delta = 1, so this function
    delegates directly to ``build_xxz_pauli_hamiltonian(n_qubits, J, delta=1.0)``.
    All phase/sign conventions for Y terms (c = -J encoding) are therefore
    inherited without duplication — see ``build_xxz_pauli_hamiltonian`` for the
    detailed phase derivation.

    Consistency with dense builder
    --------------------------------
    ``build_heisenberg_dense_matrix(n, J)`` calls
    ``build_xxz_dense_matrix(n, J, delta=1.0)``, so the same delta=1 substitution
    is applied on both sides.  ``_assert_pauli_matches_dense`` confirms agreement.

    Args:
        n_qubits : System size.
        J        : Exchange coupling.

    Returns:
        _PauliPolynomial compatible with estimate_energy().
    """
    # Heisenberg is the isotropic limit of XXZ: delta = 1 sets ZZ coupling = J.
    return build_xxz_pauli_hamiltonian(n_qubits, J=J, delta=1.0)


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Factory
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

def build_hamiltonian_spec(family: str, n_qubits: int, **params) -> HamiltonianSpec:
    """
    Build a ``HamiltonianSpec`` for the given family and per-state parameters.

    This is the single entry point for constructing Hamiltonians.  It bundles
    the dense matrix (for exact diagonalisation) and the Pauli representation
    (for shadow-based energy estimation) into one object.

    Supported families
    ------------------
    "tfim"
        H = -J Σ Z_i Z_{i+1}  -  h Σ X_i   (OBC)
        Params: J (default 1.0), h (default 0.5).
        pauli_hamiltonian: always set (pyclifford if available, else built-in).

    "ising_general"
        H = -J Σ ZZ  -  hx Σ X  -  hz Σ Z   (OBC)
        Params: J (default 1.0), hx (default 0.5), hz (default 0.0).
        pauli_hamiltonian: always set (built-in _PauliPolynomial fallback).

    "xxz"
        H = J Σ (XX + YY + delta ZZ)   (OBC)
        Params: J (default 1.0), delta (default 1.0).
        pauli_hamiltonian: always set (built-in _PauliPolynomial; Y terms use
        c = -J encoding to compensate XZ = -iY phase).

    "heisenberg"
        H = J Σ (XX + YY + ZZ)   (OBC)
        Params: J (default 1.0).
        pauli_hamiltonian: always set (delegates to XXZ with delta=1).

    Notes
    -----
    * ``dense_matrix`` is always computed.
    * For TFIM, ``pauli_hamiltonian`` is always set (energy estimation works
      out of the box without pyclifford).
    * For other families, ``pauli_hamiltonian=None`` means
      ``ShadowProcessor.estimate_energy()`` will not be called automatically.

    Args:
        family   : Hamiltonian family string.
        n_qubits : System size.
        **params : Family-specific keyword parameters.

    Returns:
        HamiltonianSpec with dense_matrix (and pauli_hamiltonian for TFIM) set.

    Raises:
        ValueError : If family is unknown.
    """
    if family == "tfim":
        J = float(params.get("J", 1.0))
        h = float(params.get("h", 0.5))
        return HamiltonianSpec(
            family=family,
            n_qubits=n_qubits,
            params={"J": J, "h": h},
            dense_matrix=build_tfim_dense_matrix(n_qubits, J=J, h=h),
            pauli_hamiltonian=build_tfim_pauli_hamiltonian(n_qubits, J=J, h=h),
        )

    elif family == "ising_general":
        J  = float(params.get("J",  1.0))
        hx = float(params.get("hx", 0.5))
        hz = float(params.get("hz", 0.0))
        return HamiltonianSpec(
            family=family,
            n_qubits=n_qubits,
            params={"J": J, "hx": hx, "hz": hz},
            dense_matrix=build_ising_general_dense_matrix(n_qubits, J=J, hx=hx, hz=hz),
            pauli_hamiltonian=build_ising_general_pauli_hamiltonian(
                n_qubits, J=J, hx=hx, hz=hz
            ),
        )

    elif family == "xxz":
        J     = float(params.get("J",     1.0))
        delta = float(params.get("delta", 1.0))
        return HamiltonianSpec(
            family=family,
            n_qubits=n_qubits,
            params={"J": J, "delta": delta},
            dense_matrix=build_xxz_dense_matrix(n_qubits, J=J, delta=delta),
            pauli_hamiltonian=build_xxz_pauli_hamiltonian(n_qubits, J=J, delta=delta),
        )

    elif family == "heisenberg":
        J = float(params.get("J", 1.0))
        return HamiltonianSpec(
            family=family,
            n_qubits=n_qubits,
            params={"J": J},
            dense_matrix=build_heisenberg_dense_matrix(n_qubits, J=J),
            pauli_hamiltonian=build_heisenberg_pauli_hamiltonian(n_qubits, J=J),
        )

    else:
        raise ValueError(
            f"Unknown Hamiltonian family {family!r}. "
            "Supported: 'tfim', 'ising_general', 'xxz', 'heisenberg'."
        )


__all__ = [
    "HamiltonianSpec",
    "build_hamiltonian_spec",
    # Dense matrix builders (public for direct use in generate_dataset loops)
    "build_tfim_dense_matrix",
    "build_ising_general_dense_matrix",
    "build_xxz_dense_matrix",
    "build_heisenberg_dense_matrix",
    # Pauli Hamiltonian builders (public for testing / advanced use)
    "build_tfim_pauli_hamiltonian",
    "build_ising_general_pauli_hamiltonian",
    "build_xxz_pauli_hamiltonian",
    "build_heisenberg_pauli_hamiltonian",
    # Developer utilities
    "_assert_pauli_matches_dense",
]
