import math
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "shadow GPT" / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))


try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    np = None
    IMPORT_ERROR = exc
else:
    try:
        from shadows.collector import ShadowCollector, ShadowMeasurement, _PAULI_ROTATIONS
        from shadows.config import (
            ShadowConfig,
            create_clifford_config,
            create_custom_config,
            create_default_config,
            create_pauli_config,
        )
        from shadows.processor import ShadowProcessor
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        IMPORT_ERROR = exc
    else:
        IMPORT_ERROR = None


RUNTIME_AVAILABLE = IMPORT_ERROR is None


def _runtime_message() -> str:
    if IMPORT_ERROR is None:
        return ""
    return f"runtime dependencies unavailable: {IMPORT_ERROR}"


def make_zero_state(n_qubits: int):
    state = np.zeros(2 ** n_qubits, dtype=complex)
    state[0] = 1.0
    return state


def make_fake_hamiltonian(pauli_terms):
    pauli_map = {
        "I": (0, 0),
        "X": (1, 0),
        "Z": (0, 1),
    }

    class FakeTerm:
        def __init__(self, coeff, paulis):
            self.c = coeff
            self.p = 0
            bits = []
            for pauli in paulis:
                bits.extend(pauli_map[pauli])
            self.g = np.array(bits, dtype=int)

    class FakeHamiltonian:
        def __init__(self, terms):
            self.N = len(pauli_terms[0][1])
            self._terms = [FakeTerm(coeff, paulis) for coeff, paulis in terms]

        def __iter__(self):
            return iter(self._terms)

    return FakeHamiltonian(pauli_terms)


@unittest.skipUnless(RUNTIME_AVAILABLE, _runtime_message())
class ShadowConfigTests(unittest.TestCase):
    def test_random_config_works(self):
        config = create_default_config(n_qubits=2, n_shadows=5, measurement_basis="random")
        self.assertEqual(config.measurement_basis, "random")
        self.assertEqual(config.n_qubits, 2)

    def test_pauli_config_works(self):
        config = create_pauli_config(n_qubits=2, n_shadows=5, pauli_weights=[0.2, 0.3, 0.5])
        self.assertEqual(config.measurement_basis, "pauli")
        self.assertEqual(config.pauli_weights, [0.2, 0.3, 0.5])

    def test_clifford_config_works(self):
        config = create_clifford_config(n_qubits=2, n_shadows=5)
        self.assertEqual(config.measurement_basis, "clifford")

    def test_custom_config_requires_sampler(self):
        with self.assertRaises(ValueError):
            ShadowConfig(n_qubits=1, measurement_basis="custom", custom_sampler=None)

    def test_custom_config_works_with_sampler(self):
        def sampler(n_qubits, rng):
            basis = np.arange(n_qubits, dtype=int)
            return basis, [_PAULI_ROTATIONS[2] for _ in range(n_qubits)]

        config = create_custom_config(n_qubits=2, custom_sampler=sampler, n_shadows=5)
        self.assertEqual(config.measurement_basis, "custom")
        self.assertIs(config.custom_sampler, sampler)


@unittest.skipUnless(RUNTIME_AVAILABLE, _runtime_message())
class ShadowCollectorDenseTests(unittest.TestCase):
    def test_dense_collection_shadow_data_save_and_load_random_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ShadowConfig(
                n_qubits=2,
                n_shadows=8,
                measurement_basis="random",
                seed=123,
                output_dir=tmpdir,
                renyi_entropy=False,
            )
            collector = ShadowCollector(config)
            measurements = collector.sample_dense(make_zero_state(2))

            self.assertEqual(len(measurements), 8)
            self.assertTrue(all(m.unitaries is None for m in measurements))

            shadow_data = collector.get_shadow_data()
            self.assertEqual(shadow_data.shape, (8, 2, 2))

            path = collector.save_shadows("random_dense_test.npz")

            loaded = ShadowCollector(create_default_config(n_qubits=1, n_shadows=1))
            loaded.load_shadows(path)
            self.assertEqual(len(loaded.measurements), 8)
            self.assertEqual(loaded.config.measurement_basis, "random")
            self.assertTrue(all(m.unitaries is None for m in loaded.measurements))
            self.assertTrue(np.array_equal(loaded.get_shadow_data(), shadow_data))


@unittest.skipUnless(RUNTIME_AVAILABLE, _runtime_message())
class ShadowCollectorCustomTests(unittest.TestCase):
    def test_custom_dense_collection_stores_and_restores_unitaries(self):
        def custom_sampler(n_qubits, rng):
            basis = rng.integers(10, 20, size=n_qubits)
            unitaries = [_PAULI_ROTATIONS[int(rng.integers(0, 3))] for _ in range(n_qubits)]
            return basis, unitaries

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ShadowConfig(
                n_qubits=2,
                n_shadows=6,
                measurement_basis="custom",
                custom_sampler=custom_sampler,
                seed=7,
                output_dir=tmpdir,
                renyi_entropy=False,
            )
            collector = ShadowCollector(config)
            collector.sample_dense(make_zero_state(2))

            self.assertEqual(len(collector.measurements), 6)
            for measurement in collector.measurements:
                self.assertEqual(measurement.unitaries.shape, (2, 2, 2))

            path = collector.save_shadows("custom_dense_test.npz")

            loaded = ShadowCollector(create_default_config(n_qubits=1, n_shadows=1))
            loaded.load_shadows(path)

            self.assertEqual(loaded.config.measurement_basis, "custom")
            self.assertEqual(len(loaded.measurements), 6)
            for original, restored in zip(collector.measurements, loaded.measurements):
                self.assertTrue(np.array_equal(original.basis, restored.basis))
                self.assertTrue(np.array_equal(original.outcome, restored.outcome))
                self.assertTrue(np.allclose(original.unitaries, restored.unitaries))


@unittest.skipUnless(RUNTIME_AVAILABLE, _runtime_message())
class ShadowProcessorBuiltInTests(unittest.TestCase):
    def test_built_in_modes_run_all_estimators(self):
        state = make_zero_state(2)
        hamiltonian = make_fake_hamiltonian([
            (1.0, ["Z", "I"]),
            (1.0, ["I", "Z"]),
        ])

        configs = [
            ShadowConfig(
                n_qubits=2,
                n_shadows=60,
                measurement_basis="random",
                seed=0,
                median_of_means=False,
                correlation_length=1,
                renyi_entropy=True,
            ),
            ShadowConfig(
                n_qubits=2,
                n_shadows=60,
                measurement_basis="pauli",
                pauli_weights=[0.2, 0.2, 0.6],
                seed=1,
                median_of_means=False,
                correlation_length=1,
                renyi_entropy=True,
            ),
            ShadowConfig(
                n_qubits=2,
                n_shadows=60,
                measurement_basis="clifford",
                seed=2,
                median_of_means=False,
                correlation_length=1,
                renyi_entropy=True,
            ),
        ]

        for config in configs:
            collector = ShadowCollector(config)
            collector.sample_dense(state)
            processor = ShadowProcessor(config)

            magnetization = processor.estimate_magnetization(collector)
            correlations = processor.estimate_correlations(collector)
            energy = processor.estimate_energy(collector, hamiltonian)
            renyi = processor.estimate_renyi_entropy(collector)

            for estimate in (magnetization, correlations, energy, renyi):
                self.assertTrue(math.isfinite(estimate.estimate))
                self.assertTrue(math.isfinite(estimate.error))

    def test_renyi_entropy_single_pair_has_finite_error(self):
        config = ShadowConfig(
            n_qubits=1,
            n_shadows=2,
            measurement_basis="random",
            seed=5,
            median_of_means=False,
            renyi_entropy=True,
        )
        collector = ShadowCollector(config)
        collector.sample_dense(make_zero_state(1))
        processor = ShadowProcessor(config)

        estimate = processor.estimate_renyi_entropy(collector, n_subsystem=1)
        self.assertTrue(math.isfinite(estimate.estimate))
        self.assertTrue(math.isfinite(estimate.error))


@unittest.skipUnless(RUNTIME_AVAILABLE, _runtime_message())
class ShadowProcessorCustomModeTests(unittest.TestCase):
    def test_custom_mode_pipeline_works_end_to_end(self):
        def custom_sampler(n_qubits, rng):
            basis = rng.integers(100, 200, size=n_qubits)
            pauli_choices = rng.integers(0, 3, size=n_qubits)
            unitaries = [_PAULI_ROTATIONS[int(i)] for i in pauli_choices]
            return basis, unitaries

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ShadowConfig(
                n_qubits=2,
                n_shadows=60,
                measurement_basis="custom",
                custom_sampler=custom_sampler,
                seed=9,
                output_dir=tmpdir,
                median_of_means=False,
                correlation_length=1,
                renyi_entropy=True,
            )

            collector = ShadowCollector(config)
            collector.sample_dense(make_zero_state(2))
            path = collector.save_shadows("custom_end_to_end.npz")

            loaded = ShadowCollector(create_default_config(n_qubits=1, n_shadows=1))
            loaded.load_shadows(path)

            processor = ShadowProcessor(loaded.config)
            hamiltonian = make_fake_hamiltonian([
                (1.0, ["Z", "I"]),
                (1.0, ["I", "Z"]),
            ])

            estimates = processor.process_shadows(loaded, hamiltonian=hamiltonian)

            self.assertEqual(set(estimates), {"energy", "magnetization", "correlations", "renyi_entropy"})
            for estimate in estimates.values():
                self.assertTrue(math.isfinite(estimate.estimate))
                self.assertTrue(math.isfinite(estimate.error))

    def test_custom_mode_errors_when_unitaries_are_missing(self):
        config = ShadowConfig(
            n_qubits=1,
            n_shadows=1,
            measurement_basis="custom",
            custom_sampler=lambda n_qubits, rng: (np.zeros(n_qubits, dtype=int), [_PAULI_ROTATIONS[2]]),
            seed=11,
            median_of_means=False,
            renyi_entropy=False,
        )
        collector = ShadowCollector(config)
        collector.measurements = [
            ShadowMeasurement(
                basis=np.array([17], dtype=int),
                outcome=np.array([0], dtype=int),
                unitaries=None,
            )
        ]

        processor = ShadowProcessor(config)
        with self.assertRaises(ValueError):
            processor.estimate_magnetization(collector)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
