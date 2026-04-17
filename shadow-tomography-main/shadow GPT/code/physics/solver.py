"""
Quantum many-body ground state solvers.

This module provides EDSolver and DMRGSolver classes for finding ground states
of quantum many-body Hamiltonians using exact diagonalization and DMRG methods.
"""

import numpy as np
from typing import Tuple, Optional, Union, List, Any
from dataclasses import dataclass
from pyclifford import PauliPolynomial
from .operator import Operator

try:
    from quimb.linalg.base_linalg import eigensystem
    from quimb.tensor.tensor_dmrg import DMRG
    from quimb import entropy_subsys
    QUIMB_AVAILABLE = True
except ImportError:
    QUIMB_AVAILABLE = False
    print("Warning: quimb not available. Some solvers will not work.")


@dataclass
class EDConfig:
    """
    Configuration for exact diagonalization solver.
    
    Exact diagonalization finds ground states by diagonalizing the full Hamiltonian matrix.
    It's exact but limited to small systems due to exponential scaling with system size.
    
    Attributes:
        max_dimension: Maximum matrix dimension for ED (2^12 = 4096)
        backend:      Backend solver: 'AUTO' (auto-select), 'NUMPY' (numpy.linalg), 
                     'SCIPY' (scipy.sparse.linalg), 'LOBPCG' (scipy.sparse.linalg.lobpcg), 'SLEPC' (SLEPc)
        k:            Number of eigenvalues/eigenvectors to compute (k=1 for ground state only)
    """
    max_dimension: int = 2**12
    backend: str = 'AUTO'
    k: int = 1
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")


@dataclass
class DMRGConfig:
    """
    Configuration for DMRG (Density Matrix Renormalization Group) solver.
    
    DMRG is a variational method that finds ground states by optimizing a Matrix Product State (MPS)
    representation. It's much more efficient than ED for 1D systems and can handle larger system sizes.
    
    Attributes:
        bond_dims:         Bond dimensions for MPS compression
                          - int: Fixed bond dimension for all bonds
                          - List[int]: Progressive bond dimensions [10, 20, 30, ...]
                          - None: Uses default [10, 20, 30, 40, 50]
        cutoffs:           Singular value cutoff thresholds for compression
                          - float: Single cutoff for all bonds
                          - List[float]: Progressive cutoffs [1e-6, 1e-8, 1e-10, ...]
                          - Lower values = higher accuracy, more computational cost
        bsz:               Block size for DMRG algorithm
                          - 1: 1-site DMRG (faster, less accurate)
                          - 2: 2-site DMRG (slower, more accurate, recommended)
        which:             Which eigenvalues to target
                          - 'SA': Smallest algebraic (ground state)
                          - 'LA': Largest algebraic (highest energy state)
        p0:                Initial state for DMRG
                          - None: Random initial state
                          - MPS: Use provided MPS as starting point
        tol:               Convergence tolerance for energy
                          - DMRG stops when energy change < tol
                          - Lower values = more accurate, more sweeps needed
        max_sweeps:        Maximum number of DMRG sweeps
                          - Each sweep goes left-to-right and right-to-left
                          - More sweeps = better convergence, more time
        verbosity:         Verbosity level for output
                          - 0: Silent (no output)
                          - 1: Basic progress information
                          - 2: Detailed progress and convergence info
        suppress_warnings: Whether to suppress convergence warnings
                          - True: Hide warnings about non-convergence
                          - False: Show all warnings
        sweep_sequence:    Custom sweep sequence pattern
                          - None: Default alternating L-R sweeps
                          - 'RRL': Right-Right-Left pattern
                          - 'LLR': Left-Left-Right pattern
                          - Custom patterns can improve convergence for specific systems
    """
    bond_dims: Union[int, List[int]] = None
    cutoffs: Union[float, List[float]] = 1e-09
    bsz: int = 2
    which: str = 'SA'
    p0: Optional[Any] = None
    tol: float = 0.0001
    max_sweeps: int = 10
    verbosity: int = 0
    suppress_warnings: bool = True
    sweep_sequence: Optional[str] = None
    
    def __post_init__(self):
        """Set default values for mutable fields."""
        if self.bond_dims is None:
            self.bond_dims = [10, 20, 30, 40, 50]
        if isinstance(self.cutoffs, (int, float)) and self.cutoffs == 1e-09:
            # Keep the default single value, but ensure it's properly typed
            pass


class EDSolver:
    """
    Exact diagonalization solver for quantum many-body Hamiltonians.
    
    Uses quimb's groundstate and groundenergy functions to find ground states
    of Hamiltonians using exact diagonalization. Suitable for small systems.
    """
    
    def __init__(self, config: Optional[EDConfig] = None):
        """
        Initialize ED solver.
        
        Args:
            config: Solver configuration (uses defaults if None)
        """
        if not QUIMB_AVAILABLE:
            raise ImportError("quimb is required for ED solver. Install with: pip install quimb")
        
        self.config = config or EDConfig()
        self._ground_state = None
        self._ground_energy = None
        self.ham = None
    
    def solve(self, ham: Operator) -> Tuple[float, np.ndarray]:
        """
        Solve for ground state of the Hamiltonian.
        
        Args:
            ham: Hamiltonian to solve
            
        Returns:
            Tuple of (ground_energy, ground_state_vector)
        """
        self.ham = ham
        
        # Check if system is too large
        dim = 2 ** ham.N
        if dim > self.config.max_dimension:
            raise ValueError(
                f"System too large for ED: {dim} > {self.config.max_dimension}. "
                f"Consider using DMRG solver instead."
            )
        
        # Get dense matrix representation
        H_dense = ham.to_numpy()
        
        # Compute ground state and energy using eigensystem (more efficient)
        # k gets the specified number of smallest eigenvalues, which='SA' for smallest algebraic
        eigenvalues, eigenvectors = eigensystem(
            H_dense, 
            isherm=True, 
            k=self.config.k, 
            which='SA',  # Smallest algebraic (ground state)
            backend=self.config.backend
        )
        
        self._ground_energy = eigenvalues[0]
        self._ground_state = eigenvectors[:, 0]  # First eigenvector
        
        return self._ground_energy, self._ground_state
    
    @property
    def ground_state(self) -> Optional[np.ndarray]:
        """Get the computed ground state vector."""
        if self._ground_state is None:
            raise ValueError("Ground state not computed yet. Call solve() first.")
        return self._ground_state
    
    @property
    def ground_energy(self) -> Optional[float]:
        """Get the computed ground state energy."""
        if self._ground_energy is None:
            raise ValueError("Ground energy not computed yet. Call solve() first.")
        return self._ground_energy

    def expectation_value(self, operator: PauliPolynomial) -> Optional[float]:
        """Get the expectation value of an operator in the ground state."""
        if self._ground_state is None:
            raise ValueError("Ground state not computed yet. Call solve() first.")
        return self._ground_state.H @ operator.embed_qubits(self.ham.N).to_numpy() @ self._ground_state
    
    def entropy(self, i: int, approx_thresh: Optional[int] = None, **approx_opts) -> Optional[float]:
        """
        Calculate the entropy of a subsystem in the ground state.
        
        Args:
            i: The number of sites in the left partition
            approx_thresh: The size of sysa at which to switch to the approx method. 
                          Set to None to never use the approximation.
            **approx_opts: Supplied to entropy_subsys_approx(), if used.
            
        Returns:
            The subsystem entropy.
        """
        if self._ground_state is None:
            raise ValueError("Ground state not computed yet. Call solve() first.")
        
        return entropy_subsys(
            psi_ab=self._ground_state,
            dims=[2] * self.ham.N, # Each qubit has dimension 2
            sysa=list(range(i)),
            approx_thresh=approx_thresh,
            **approx_opts
        )
    
    def __repr__(self) -> str:
        return f"EDSolver(max_dim={self.config.max_dimension}, backend={self.config.backend}, k={self.config.k})"


class DMRGSolver:
    """
    DMRG solver for quantum many-body Hamiltonians.
    
    Uses quimb's DMRG implementation to find ground states of large systems
    where exact diagonalization is not feasible.
    """
    
    def __init__(self, config: Optional[DMRGConfig] = None):
        """
        Initialize DMRG solver.
        
        Args:
            config: Solver configuration (uses defaults if None)
        """
        if not QUIMB_AVAILABLE:
            raise ImportError("quimb is required for DMRG solver. Install with: pip install quimb")
        
        self.config = config or DMRGConfig()
        self.dmrg = None
        self.ham = None
        self._converged = False
    
    def solve(self, ham: Operator) -> Tuple[float, Any]:
        """
        Solve for ground state of the Hamiltonian using DMRG.
        
        Args:
            ham: Hamiltonian to solve
            
        Returns:
            Tuple of (ground_energy, ground_state_MPS)
        """
        self.ham = ham
        
        # Get MPO representation of Hamiltonian
        mpo = ham.to_MPO()
        
        # Create DMRG object with configuration
        self.dmrg = DMRG(
            ham=mpo,
            bond_dims=self.config.bond_dims,
            cutoffs=self.config.cutoffs,
            bsz=self.config.bsz,
            which=self.config.which,
            p0=self.config.p0
        )
        
        # Run DMRG optimization and get convergence status
        self._converged = self.dmrg.solve(
            tol=self.config.tol,
            max_sweeps=self.config.max_sweeps,
            verbosity=self.config.verbosity,
            suppress_warnings=self.config.suppress_warnings,
            sweep_sequence=self.config.sweep_sequence
        )
        
        return self.dmrg.energy, self.dmrg.state
    
    @property
    def ground_state(self) -> Optional[Any]:
        """Get the computed ground state MPS."""
        if self.dmrg is None:
            raise ValueError("Ground state not computed yet. Call solve() first.")
        return self.dmrg.state
    
    @property
    def ground_energy(self) -> Optional[float]:
        """Get the computed ground state energy."""
        if self.dmrg is None:
            raise ValueError("Ground energy not computed yet. Call solve() first.")
        return self.dmrg.energy
    
    @property
    def energies(self) -> Optional[List[float]]:
        """Get the energy after each sweep."""
        if self.dmrg is None:
            raise ValueError("Energies not computed yet. Call solve() first.")
        return self.dmrg.energies
    
    @property
    def local_energies(self) -> Optional[List[List[float]]]:
        """Get the local energies per sweep."""
        if self.dmrg is None:
            raise ValueError("Local energies not computed yet. Call solve() first.")
        return self.dmrg.local_energies
    
    @property
    def total_energies(self) -> Optional[List[List[float]]]:
        """Get the total energies per sweep."""
        if self.dmrg is None:
            raise ValueError("Total energies not computed yet. Call solve() first.")
        return self.dmrg.total_energies
    
    @property
    def converged(self) -> bool:
        """Check if DMRG has converged."""
        if self.dmrg is None:
            raise ValueError("DMRG not initialized. Call solve() first.")
        # Use the convergence flag returned by quimb's solve method
        return getattr(self, '_converged', False)

    def expectation_value(self, operator: Operator) -> Optional[float]:
        """Get the expectation value of an operator in the ground state."""
        if self.dmrg is None:
            raise ValueError("DMRG not initialized. Call solve() first.")
        mpo = operator.embed_qubits(self.ham.N).to_MPO()
        mps = self.ground_state
        mpsH = mps.H
        mps.align_(mpo, mpsH)
        return (mpsH & mpo & mps)^...
    
    def entropy(self, i: int, info: Optional[dict] = None, method: str = 'svd') -> Optional[float]:
        """
        Calculate the entropy of bipartition between the left block of i sites and the rest.
        
        Args:
            i: The number of sites in the left partition
            info: If given, will be used to infer and store various extra information. 
                  Currently the key "cur_orthog" is used to store the current orthogonality center.
            method: Method to use for entropy calculation ('svd')
            
        Returns:
            The entropy of the bipartition.
        """
        if self.dmrg is None:
            raise ValueError("DMRG not initialized. Call solve() first.")
        
        mps = self.ground_state
        return mps.entropy(i, info=info, method=method)
    
    
    def __repr__(self) -> str:
        bond_dims_str = str(self.config.bond_dims) if isinstance(self.config.bond_dims, list) else str(self.config.bond_dims)
        return f"DMRGSolver(bond_dims={bond_dims_str}, bsz={self.config.bsz})"

