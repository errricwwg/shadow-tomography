"""
Physics module for quantum many-body systems.

This module provides:
- Hamiltonian construction and manipulation
- Ground state solvers (exact diagonalization and DMRG)
- Integration with classical shadow protocols
"""

from .operator import Operator, ham_tf_ising, ham_cluster_ising
from .solver import EDSolver, DMRGSolver, EDConfig, DMRGConfig

__all__ = ["Operator", "ham_tf_ising", "ham_cluster_ising", "EDSolver", "DMRGSolver", "EDConfig", "DMRGConfig"]
