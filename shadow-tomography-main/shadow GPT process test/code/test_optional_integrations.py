import math
import sys
import unittest
from pathlib import Path

try:
    import pytest
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    raise unittest.SkipTest(f"pytest not installed: {exc}")


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "shadow GPT" / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


def _import_core_shadow_modules():
    np = pytest.importorskip("numpy")
    pytest.importorskip("tqdm")

    from shadows.collector import ShadowCollector
    from shadows.config import ShadowConfig, create_default_config
    from shadows.processor import ShadowProcessor

    return np, ShadowCollector, ShadowConfig, ShadowProcessor, create_default_config


def test_quimb_mps_sampling_round_trip(tmp_path):
    np, ShadowCollector, ShadowConfig, _, create_default_config = _import_core_shadow_modules()
    qtn = pytest.importorskip("quimb.tensor")

    # Small deterministic OBC product state |00>.
    mps = qtn.MPS_computational_state("00")

    config = ShadowConfig(
        n_qubits=2,
        n_shadows=4,
        measurement_basis="pauli",
        pauli_weights=[0.0, 0.0, 1.0],  # always measure in Z
        seed=0,
        output_dir=str(tmp_path),
        renyi_entropy=False,
    )

    collector = ShadowCollector(config)
    measurements = collector.sample_mps(mps)

    assert len(measurements) == 4
    for measurement in measurements:
        assert measurement.basis.shape == (2,)
        assert measurement.outcome.shape == (2,)
        assert np.array_equal(measurement.basis, np.array([2, 2], dtype=int))
        assert np.array_equal(measurement.outcome, np.array([0, 0], dtype=int))

    shadow_data = collector.get_shadow_data()
    assert shadow_data.shape == (4, 2, 2)

    path = collector.save_shadows("quimb_mps_roundtrip.npz")
    loaded = ShadowCollector(create_default_config(n_qubits=1, n_shadows=1))
    loaded.load_shadows(path)

    assert len(loaded.measurements) == 4
    assert loaded.config.measurement_basis == "pauli"
    assert np.array_equal(loaded.get_shadow_data(), shadow_data)


def test_real_hamiltonian_energy_path_runs_end_to_end(tmp_path):
    np, ShadowCollector, ShadowConfig, ShadowProcessor, _ = _import_core_shadow_modules()
    pytest.importorskip("pyclifford")
    pytest.importorskip("quimb")

    from physics.operator import ham_tf_ising

    # Dense |00> state.
    state = np.zeros(4, dtype=complex)
    state[0] = 1.0

    config = ShadowConfig(
        n_qubits=2,
        n_shadows=6,
        measurement_basis="pauli",
        pauli_weights=[0.0, 0.0, 1.0],  # deterministic Z-only measurements
        seed=1,
        output_dir=str(tmp_path),
        median_of_means=False,
        renyi_entropy=False,
        observables=["energy"],
    )

    collector = ShadowCollector(config)
    collector.sample_dense(state)

    # For g=0 and open boundary conditions on 2 qubits:
    # H = - Z0 Z1, so <00|H|00> = -1 exactly.
    hamiltonian = ham_tf_ising(2, 0.0, bc="open")

    processor = ShadowProcessor(config)
    estimate = processor.estimate_energy(collector, hamiltonian)

    assert math.isfinite(estimate.estimate)
    assert math.isfinite(estimate.error)
    assert estimate.n_samples == 6
    # This test is for real-object integration rather than high-precision
    # physical validation: the estimator should accept the real Hamiltonian
    # representation and produce a finite result end to end.
