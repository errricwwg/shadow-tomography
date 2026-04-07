"""
Operator construction and manipulation for quantum many-body systems.

This module provides an Operator class that extends PyClifford's PauliPolynomial for easy
integration with classical shadow protocols and ground state solvers.
"""

import numpy as np
from pyclifford import Pauli, PauliList, PauliPolynomial, pauli_zero, pauli
from quimb.tensor.tensor_1d import MatrixProductOperator
from quimb.tensor import MPO_zeros, MPO_product_operator
from typing import List

class Operator(PauliPolynomial):
    """
    Extended PauliPolynomial for quantum many-body operators.
    
    Inherits from PyClifford's PauliPolynomial and adds methods to convert to
    quimb MPO format for integration with DMRG solver.
    Can represent any Pauli polynomial including Hamiltonians and observables.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize Operator with given configuration.
        
        Args:
            *args: Arguments passed to PauliPolynomial constructor
            **kwargs: Keyword arguments passed to PauliPolynomial constructor
        """
        # Handle case where we're wrapping an existing PauliPolynomial
        if len(args) == 1:
            if isinstance(args[0], Operator):
                self.__dict__.update(args[0].__dict__)
            else:
                if isinstance(args[0], Pauli):
                    existing = args[0].as_polynomial()
                if isinstance(args[0], PauliPolynomial):
                    existing = args[0]
                existing.__class__ = Operator
                # Copy all attributes to self
                self.__dict__.update(existing.__dict__)                
        else:
            super().__init__(*args, **kwargs)
    
    def to_MPO(self) -> MatrixProductOperator:
        """
        Convert operator to quimb MPO format for DMRG.
        
        Returns:
            quimb MPO representation of the operator
        """
        # Initialize zero MPO
        mpo = MPO_zeros(L=self.N)
        # Add each operator term to the MPO
        for term in self:
            # Extract coefficient
            coeff = term.c * 1j**term.p
            # Skip zero terms
            if abs(coeff) < 1e-15:
                continue
            # Break Pauli string to Pauli matrices
            mats = PauliList(term.g.reshape(-1, 2)).to_numpy()
            # Form operator-product MPO (bond_dim=1)
            op = MPO_product_operator(mats)
            # Add to total MPO
            mpo += coeff * op
            # Compress to keep bond dimensions manageable
            mpo.compress()
        return mpo

def ham_tf_ising(n: int, gs: List[float], bc: str = "open") -> Operator:
    """
    Create a transverse field Ising Hamiltonian.

    Hamiltonian: H = -(1-g) * sum_i Z_{i-1}Z_{i+1} - g * sum_i X_i

    Args:
        n: Number of qubits
        gs: List of three coupling strengths [g1, g2, g3] with g1 + g2 + g3 = 1, gi > 0
        bc: Boundary conditions ("open" or "periodic")
        
    Returns:
        Operator object representing the transverse field Ising Hamiltonian
    """
    try:
        (g,) = gs
    except:
        g = gs
    # create a zero operator
    H = pauli_zero(n)
    # add Ising coupling terms
    if bc == "open":
        for i in range(n-1):
            H += -(1-g) * pauli({i:'Z', i+1:'Z'}, N=n)
    elif bc == "periodic":
        for i in range(n):
            H += -(1-g) * pauli({i:'Z', (i+1)%n:'Z'}, N=n)
    # add transverse field term
    for i in range(n):
        H += -g * pauli({i:'X'}, N=n)
    return Operator(H)


def ham_cluster_ising(n: int, gs: List[float], bc: str = "open") -> Operator:
    """
    Create a Cluster-Ising Hamiltonian with Z2 x Z2 symmetry.
    
    Hamiltonian: H = -g1 * sum_i Z_{i-1}Z_{i+1} - g2 * sum_i X_i - g3 * sum_i Z_{i-1}X_iZ_{i+1}
    
    Args:
        n: Number of qubits
        gs: List of three coupling strengths [g1, g2, g3] with g1 + g2 + g3 = 1, gi > 0
        bc: Boundary conditions ("open" or "periodic")
        
    Returns:
        Operator object representing the Cluster-Ising Hamiltonian
        
    Note:
        The Cluster-Ising model exhibits Z2 x Z2 symmetry and can undergo
        quantum phase transitions between different symmetry-broken phases.
    """
    # Validate parameters
    if len(gs) != 3:
        raise ValueError("gs must have exactly 3 elements")
    if any(g <= 0 for g in gs):
        raise ValueError("All coupling strengths must be positive: gi > 0")
    if abs(sum(gs) - 1.0) > 1e-10:
        raise ValueError("Coupling strengths must sum to 1: g1 + g2 + g3 = 1")
    
    try:
        g1, g2, g3 = gs
    except:
        raise ValueError("gs must be a list of exactly 3 elements [g1, g2, g3]")
    
    # Create zero operator
    H = pauli_zero(n)
    
    if bc == "open":
        # Open boundary conditions
        for i in range(1, n-1):  # Skip first and last sites
            H += -g1 * pauli({i-1:'Z', i+1:'Z'}, N=n)
            H += -g2 * pauli({i:'X'}, N=n)
            H += -g3 * pauli({i-1:'Z', i:'X', i+1:'Z'}, N=n)
        # Add transverse field terms for first and last sites
        H += -g2 * pauli({0:'X'}, N=n)
        H += -g2 * pauli({n-1:'X'}, N=n)
    elif bc == "periodic":
        # Periodic boundary conditions
        for i in range(n):
            H += -g1 * pauli({(i-1)%n:'Z', (i+1)%n:'Z'}, N=n)
            H += -g2 * pauli({i:'X'}, N=n)
            H += -g3 * pauli({(i-1)%n:'Z', i:'X', (i+1)%n:'Z'}, N=n)
    else:
        raise ValueError("bc must be 'open' or 'periodic'")
    
    return Operator(H)
