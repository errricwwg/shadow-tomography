"""
End-to-end integration tests for the classical-shadow pipeline.

Covers:
  A. Package import / __init__.py exports
  B. Collector sanity (random, pauli, clifford, custom)
  C. Tokenizer × measurement-mode combinations
  D. ShadowDataModule / ShadowDataset sanity
  E. Target handling (scalar & vector)
  F. Edge cases / error handling
  G. Save / load behaviour
  H. End-to-end smoke test (collect → tokenize → dataset → dataloader → batch)

Run from the repo root:
    python "shadow GPT process test/code/test_integration.py"
or:
    python -m pytest "shadow GPT process test/code/test_integration.py" -v
"""

import sys
import tempfile
import unittest
import warnings
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "shadow GPT" / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

# ── dependency detection ──────────────────────────────────────────────────────
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ModuleNotFoundError as _e:
    np = None
    _NUMPY_ERROR = _e
    _NUMPY_AVAILABLE = False
else:
    _NUMPY_ERROR = None

try:
    import torch
    _TORCH_AVAILABLE = True
except ModuleNotFoundError as _e:
    torch = None
    _TORCH_ERROR = _e
    _TORCH_AVAILABLE = False
else:
    _TORCH_ERROR = None

# ── import shadows package once numpy is confirmed available ──────────────────
_SHADOWS_AVAILABLE = False
_SHADOWS_ERROR = None
if _NUMPY_AVAILABLE:
    try:
        from shadows.collector import (
            ShadowCollector,
            ShadowMeasurement,
            SINGLE_QUBIT_CLIFFORDS,
            PAULI_NAMES,
            collect_shadows_from_state,
        )
        from shadows.config import (
            ShadowConfig,
            create_default_config,
            create_pauli_config,
            create_clifford_config,
            create_custom_config,
        )
        from shadows.tokenization import (
            ShadowTokenizer,
            TokenizationConfig,
            create_default_tokenizer,
        )
        from shadows.processor import ShadowProcessor, process_shadow_data
        _SHADOWS_AVAILABLE = True
    except Exception as _e:
        _SHADOWS_ERROR = _e

_DATASET_AVAILABLE = False
_DATASET_ERROR = None
if _TORCH_AVAILABLE and _SHADOWS_AVAILABLE:
    try:
        from shadows.datasets import (
            ShadowDataset,
            ShadowDataModule,
            DatasetConfig,
            create_data_module,
        )
        from shadows.model import (
            ShadowModelConfig,
            ShadowTransformer,
            ShadowTrainer,
            create_model_from_tokenizer,
        )
        _DATASET_AVAILABLE = True
    except Exception as _e:
        _DATASET_ERROR = _e


# ── skip helpers ──────────────────────────────────────────────────────────────
def _need_shadows(tc):
    if not _NUMPY_AVAILABLE:
        tc.skipTest(f"numpy unavailable: {_NUMPY_ERROR}")
    if not _SHADOWS_AVAILABLE:
        tc.skipTest(f"shadows package unavailable: {_SHADOWS_ERROR}")

def _need_dataset(tc):
    _need_shadows(tc)
    if not _TORCH_AVAILABLE:
        tc.skipTest(f"torch unavailable: {_TORCH_ERROR}")
    if not _DATASET_AVAILABLE:
        tc.skipTest(f"shadows.datasets/model unavailable: {_DATASET_ERROR}")


# ── shared fixtures ───────────────────────────────────────────────────────────
def _make_random_state(n_qubits, seed=42):
    """Return a normalised random pure state vector of dimension 2^n."""
    rng = np.random.default_rng(seed)
    sv = rng.standard_normal(2 ** n_qubits) + 1j * rng.standard_normal(2 ** n_qubits)
    return sv / np.linalg.norm(sv)


def _custom_sampler(n_qubits, rng):
    """Minimal custom sampler: uses Pauli rotations, labels 0/1/2."""
    from shadows.collector import _PAULI_ROTATIONS
    basis = rng.integers(0, 3, size=n_qubits)
    unitaries = np.array([_PAULI_ROTATIONS[int(b)] for b in basis])
    return basis, unitaries


def _inject_measurements(collector, n_qubits, n_shots, mode, rng=None):
    """
    Directly inject synthetic measurements into a collector without sampling.
    Used for modes where we only need structural tests (no quantum state needed).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if mode in ("random", "pauli"):
        max_b = 2
    elif mode == "clifford":
        max_b = 23
    else:  # custom
        max_b = 4   # 5 custom bases: 0–4
    measurements = []
    for _ in range(n_shots):
        basis = rng.integers(0, max_b + 1, size=n_qubits)
        outcome = rng.integers(0, 2, size=n_qubits)
        unitaries = None
        if mode == "custom":
            # custom needs per-shot unitaries stored on the measurement
            from shadows.collector import _PAULI_ROTATIONS
            # Use identity matrices for all qubits as valid 2x2 unitaries
            I = np.eye(2, dtype=complex)
            unitaries = np.stack([I for _ in range(n_qubits)])
        measurements.append(ShadowMeasurement(basis=basis, outcome=outcome,
                                               unitaries=unitaries))
    collector.measurements = measurements


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# A. Package import / __init__.py exports
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

class TestPackageImports(unittest.TestCase):
    """Verify the package imports cleanly and __init__.py exports are valid."""

    def setUp(self):
        _need_shadows(self)

    def test_shadows_package_imports(self):
        """Import the top-level shadows package without error."""
        import shadows
        self.assertIsNotNone(shadows)

    def test_core_exports_present(self):
        """__init__.py must export all expected core names."""
        import shadows
        core_names = [
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
        for name in core_names:
            self.assertTrue(
                hasattr(shadows, name),
                f"shadows.{name} missing from __init__.py",
            )

    def test_torch_exports_present_or_none(self):
        """If torch is available, dataset/model exports must be classes, not None."""
        import shadows
        torch_names = [
            "ShadowDataset",
            "ShadowDataModule",
            "DatasetConfig",
            "create_data_module",
            "ShadowModelConfig",
            "ShadowTransformer",
            "ShadowTrainer",
            "create_model_from_tokenizer",
        ]
        for name in torch_names:
            self.assertTrue(
                hasattr(shadows, name),
                f"shadows.{name} missing from __init__.py",
            )
        if _TORCH_AVAILABLE:
            for name in torch_names:
                self.assertIsNotNone(
                    getattr(shadows, name),
                    f"shadows.{name} is None but torch is available",
                )

    def test_torch_flag_is_bool(self):
        import shadows
        self.assertIn(shadows._TORCH_AVAILABLE, (True, False))

    def test_clifford_list_length(self):
        self.assertEqual(len(SINGLE_QUBIT_CLIFFORDS), 24)

    def test_pauli_names(self):
        self.assertEqual(PAULI_NAMES, ["X", "Y", "Z"])


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# B. Collector sanity
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

class TestCollectorSanity(unittest.TestCase):
    """Verify collectors produce structurally valid ShadowMeasurements."""

    def setUp(self):
        _need_shadows(self)
        self.n = 4
        self.sv = _make_random_state(self.n)

    def _check_measurement_structure(self, m, n_qubits, mode):
        """Assert a single ShadowMeasurement has the expected structure."""
        self.assertIsInstance(m.basis, np.ndarray, f"{mode}: basis not ndarray")
        self.assertIsInstance(m.outcome, np.ndarray, f"{mode}: outcome not ndarray")
        self.assertEqual(m.basis.shape, (n_qubits,),
                         f"{mode}: basis shape {m.basis.shape} != ({n_qubits},)")
        self.assertEqual(m.outcome.shape, (n_qubits,),
                         f"{mode}: outcome shape {m.outcome.shape} != ({n_qubits},)")
        self.assertTrue(np.all((m.outcome == 0) | (m.outcome == 1)),
                        f"{mode}: outcome contains values outside {{0,1}}")

    def _check_collector_measurements(self, collector, n_shots, n_qubits, mode):
        self.assertEqual(len(collector.measurements), n_shots,
                         f"{mode}: expected {n_shots} measurements")
        for m in collector.measurements:
            self._check_measurement_structure(m, n_qubits, mode)

    def test_random_mode_structure(self):
        cfg = create_default_config(self.n, n_shadows=5, measurement_basis="random")
        collector = ShadowCollector(cfg)
        collector.sample_dense(self.sv)
        self._check_collector_measurements(collector, 5, self.n, "random")
        # Basis values should be 0, 1, or 2
        for m in collector.measurements:
            self.assertTrue(np.all(m.basis <= 2), "random: basis > 2")
            self.assertTrue(np.all(m.basis >= 0), "random: basis < 0")

    def test_pauli_mode_structure(self):
        cfg = create_pauli_config(self.n, n_shadows=5)
        collector = ShadowCollector(cfg)
        collector.sample_dense(self.sv)
        self._check_collector_measurements(collector, 5, self.n, "pauli")
        for m in collector.measurements:
            self.assertTrue(np.all((m.basis >= 0) & (m.basis <= 2)))

    def test_clifford_mode_structure(self):
        cfg = create_clifford_config(self.n, n_shadows=5)
        collector = ShadowCollector(cfg)
        collector.sample_dense(self.sv)
        self._check_collector_measurements(collector, 5, self.n, "clifford")
        # Clifford basis values should be 0–23
        for m in collector.measurements:
            self.assertTrue(np.all(m.basis >= 0), "clifford: basis < 0")
            self.assertTrue(np.all(m.basis <= 23), "clifford: basis > 23")
        # Clifford measurements should NOT all be in {0,1,2}
        all_bases = np.concatenate([m.basis for m in collector.measurements])
        self.assertTrue(
            np.any(all_bases > 2),
            "clifford mode should produce basis values > 2 in a 5-shot run",
        )

    def test_custom_mode_structure(self):
        cfg = create_custom_config(self.n, custom_sampler=_custom_sampler, n_shadows=5)
        collector = ShadowCollector(cfg)
        collector.sample_dense(self.sv)
        self._check_collector_measurements(collector, 5, self.n, "custom")
        # Custom mode stores per-shot unitaries
        for m in collector.measurements:
            self.assertIsNotNone(m.unitaries, "custom: unitaries should be stored")
            self.assertEqual(m.unitaries.shape, (self.n, 2, 2),
                             "custom: unitaries shape wrong")

    def test_random_mode_no_unitaries(self):
        """Built-in modes (random/pauli/clifford) should not store unitaries."""
        cfg = create_default_config(self.n, n_shadows=3)
        collector = ShadowCollector(cfg)
        collector.sample_dense(self.sv)
        for m in collector.measurements:
            self.assertIsNone(m.unitaries, "random: unitaries should be None")

    def test_shadow_measurement_dataclass_fields(self):
        """ShadowMeasurement must have basis, outcome, and optional unitaries."""
        m = ShadowMeasurement(
            basis=np.array([0, 1, 2]),
            outcome=np.array([0, 1, 0]),
        )
        self.assertTrue(hasattr(m, "basis"))
        self.assertTrue(hasattr(m, "outcome"))
        self.assertTrue(hasattr(m, "unitaries"))
        self.assertIsNone(m.unitaries)


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# C. Tokenizer × measurement-mode combinations
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

class TestTokenizerModes(unittest.TestCase):
    """
    Test tokenize_measurement / tokenize_collector for all supported
    (measurement_mode, token_type) combinations.
    """

    def setUp(self):
        _need_shadows(self)
        self.n = 4
        self.rng = np.random.default_rng(7)

    def _pauli_measurement(self):
        return ShadowMeasurement(
            basis=np.array([0, 1, 2, 0]),
            outcome=np.array([0, 1, 0, 1]),
        )

    def _clifford_measurement(self):
        return ShadowMeasurement(
            basis=np.array([0, 5, 12, 23]),
            outcome=np.array([0, 1, 0, 1]),
        )

    def _custom_measurement(self, max_basis=4):
        return ShadowMeasurement(
            basis=np.array([0, 2, 4, 1]),
            outcome=np.array([1, 0, 1, 0]),
        )

    # ── random/pauli + basis_outcome ─────────────────────────────────────────

    def test_random_basis_outcome(self):
        tok = create_default_tokenizer(self.n, token_type="basis_outcome")
        m = self._pauli_measurement()
        ids = tok.tokenize_measurement(m)
        self.assertEqual(len(ids), self.n)
        self.assertNotIn(tok.special_tokens["UNK"], ids)
        self.assertEqual(tok.get_vocab_size(), 4 + 6)   # 4 special + 6 content

    def test_random_pauli_string(self):
        tok = create_default_tokenizer(self.n, token_type="pauli_string")
        m = self._pauli_measurement()
        ids = tok.tokenize_measurement(m)
        self.assertEqual(len(ids), self.n)
        self.assertNotIn(tok.special_tokens["UNK"], ids)
        self.assertEqual(tok.get_vocab_size(), 4 + 3)   # 4 special + X/Y/Z

    def test_random_binary(self):
        tok = create_default_tokenizer(self.n, token_type="binary")
        m = self._pauli_measurement()
        ids = tok.tokenize_measurement(m)
        self.assertEqual(len(ids), 3 * self.n)   # 2 basis bits + 1 outcome bit
        self.assertNotIn(tok.special_tokens["UNK"], ids)

    # ── clifford + basis_outcome ──────────────────────────────────────────────

    def test_clifford_basis_outcome(self):
        tok = create_default_tokenizer(self.n, token_type="basis_outcome",
                                       measurement_mode="clifford")
        m = self._clifford_measurement()
        ids = tok.tokenize_measurement(m)
        self.assertEqual(len(ids), self.n)
        self.assertNotIn(tok.special_tokens["UNK"], ids)
        # 4 special + 24 bases × 2 outcomes = 52
        self.assertEqual(tok.get_vocab_size(), 52)

    def test_clifford_binary(self):
        tok = create_default_tokenizer(self.n, token_type="binary",
                                       measurement_mode="clifford")
        m = self._clifford_measurement()
        ids = tok.tokenize_measurement(m)
        # 5 basis bits (ceil(log2(24))=5) + 1 outcome bit = 6 per qubit
        self.assertEqual(len(ids), 6 * self.n)
        self.assertNotIn(tok.special_tokens["UNK"], ids)

    # ── custom + basis_outcome ────────────────────────────────────────────────

    def test_custom_basis_outcome(self):
        tok = create_default_tokenizer(self.n, token_type="basis_outcome",
                                       measurement_mode="custom",
                                       max_basis_value=4)
        m = self._custom_measurement()
        ids = tok.tokenize_measurement(m)
        self.assertEqual(len(ids), self.n)
        self.assertNotIn(tok.special_tokens["UNK"], ids)
        # 4 special + 5 bases × 2 outcomes = 14
        self.assertEqual(tok.get_vocab_size(), 14)

    def test_custom_binary(self):
        tok = create_default_tokenizer(self.n, token_type="binary",
                                       measurement_mode="custom",
                                       max_basis_value=4)
        m = self._custom_measurement()
        ids = tok.tokenize_measurement(m)
        # 5 bases → ceil(log2(5))=3 basis bits + 1 outcome bit = 4 per qubit
        self.assertEqual(len(ids), 4 * self.n)
        self.assertNotIn(tok.special_tokens["UNK"], ids)

    # ── pauli_string correctly rejects clifford/custom ────────────────────────

    def test_clifford_rejects_pauli_string_at_init(self):
        with self.assertRaises(ValueError):
            create_default_tokenizer(self.n, token_type="pauli_string",
                                     measurement_mode="clifford")

    def test_custom_rejects_pauli_string_at_init(self):
        with self.assertRaises(ValueError):
            create_default_tokenizer(self.n, token_type="pauli_string",
                                     measurement_mode="custom")

    # ── sequence lengths from factory ────────────────────────────────────────

    def test_factory_seq_length_pauli_basis_outcome(self):
        tok = create_default_tokenizer(self.n, token_type="basis_outcome")
        self.assertEqual(tok.config.max_sequence_length, self.n + 2)

    def test_factory_seq_length_pauli_binary(self):
        tok = create_default_tokenizer(self.n, token_type="binary")
        self.assertEqual(tok.config.max_sequence_length, 3 * self.n + 2)

    def test_factory_seq_length_clifford_binary(self):
        tok = create_default_tokenizer(self.n, token_type="binary",
                                       measurement_mode="clifford")
        self.assertEqual(tok.config.max_sequence_length, 6 * self.n + 2)

    def test_factory_seq_length_custom_binary(self):
        # 5 bases → 3 bits; +1 outcome = 4 per qubit; +2 BOS/EOS
        tok = create_default_tokenizer(self.n, token_type="binary",
                                       measurement_mode="custom",
                                       max_basis_value=4)
        self.assertEqual(tok.config.max_sequence_length, 4 * self.n + 2)

    # ── collector ↔ tokenizer: matching modes work ───────────────────────────

    def test_tokenize_collector_random(self):
        cfg = create_default_config(self.n, n_shadows=3)
        collector = ShadowCollector(cfg)
        sv = _make_random_state(self.n)
        collector.sample_dense(sv)
        tok = create_default_tokenizer(self.n, token_type="basis_outcome")
        seqs = tok.tokenize_collector(collector)
        self.assertEqual(len(seqs), 3)
        for s in seqs:
            self.assertEqual(len(s), self.n)

    def test_tokenize_collector_clifford(self):
        cfg = create_clifford_config(self.n, n_shadows=3)
        collector = ShadowCollector(cfg)
        sv = _make_random_state(self.n)
        collector.sample_dense(sv)
        tok = create_default_tokenizer(self.n, token_type="basis_outcome",
                                       measurement_mode="clifford")
        seqs = tok.tokenize_collector(collector)
        self.assertEqual(len(seqs), 3)
        for s in seqs:
            self.assertEqual(len(s), self.n)

    def test_mismatched_mode_raises(self):
        """A pauli tokenizer (no measurement_mode) must reject clifford data."""
        cfg = create_clifford_config(self.n, n_shadows=2)
        collector = ShadowCollector(cfg)
        sv = _make_random_state(self.n)
        collector.sample_dense(sv)
        tok = create_default_tokenizer(self.n, token_type="basis_outcome")
        with self.assertRaises(ValueError):
            tok.tokenize_collector(collector)

    # ── create_sequences wraps correctly ─────────────────────────────────────

    def test_create_sequences_bos_eos(self):
        tok = create_default_tokenizer(self.n, token_type="basis_outcome")
        raw = tok.tokenize_measurement(self._pauli_measurement())
        seqs = tok.create_sequences([raw], add_special_tokens=True)
        self.assertEqual(seqs[0][0], tok.special_tokens["BOS"])
        self.assertIn(tok.special_tokens["EOS"], seqs[0])

    def test_create_sequences_padded_to_max(self):
        tok = create_default_tokenizer(self.n, token_type="basis_outcome")
        raw = tok.tokenize_measurement(self._pauli_measurement())
        seqs = tok.create_sequences([raw])
        self.assertEqual(len(seqs[0]), tok.config.max_sequence_length)


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# D. Dataset / DataModule sanity
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

class TestDatasetSanity(unittest.TestCase):

    def setUp(self):
        _need_dataset(self)
        self.n = 4
        self.n_shots = 20
        # Build a collector with synthetic measurements
        self.cfg = create_default_config(self.n, n_shadows=self.n_shots)
        self.collector = ShadowCollector(self.cfg)
        sv = _make_random_state(self.n)
        self.collector.sample_dense(sv)
        self.tok = create_default_tokenizer(self.n, token_type="basis_outcome")
        self.targets_scalar = np.random.default_rng(0).standard_normal(self.n_shots)

    def _make_dm(self, targets=None, seed=0):
        dm_cfg = DatasetConfig(
            batch_size=4,
            shuffle=True,
            shuffle_seed=seed,
            train_split=0.7,
            val_split=0.15,
            test_split=0.15,
            pin_memory=False,   # avoid PyTorch warnings on CPU-only machines
        )
        dm = ShadowDataModule(dm_cfg)
        dm.setup(self.collector, self.tok, targets=targets)
        return dm

    def test_setup_creates_three_splits(self):
        dm = self._make_dm(targets=self.targets_scalar)
        self.assertIsNotNone(dm.train_dataset)
        self.assertIsNotNone(dm.val_dataset)
        self.assertIsNotNone(dm.test_dataset)

    def test_split_sizes_sum_to_total(self):
        dm = self._make_dm(targets=self.targets_scalar)
        total = (len(dm.train_dataset) + len(dm.val_dataset)
                 + len(dm.test_dataset))
        self.assertEqual(total, self.n_shots)

    def test_train_split_larger_than_others(self):
        dm = self._make_dm(targets=self.targets_scalar)
        self.assertGreater(len(dm.train_dataset), len(dm.val_dataset))
        self.assertGreater(len(dm.train_dataset), len(dm.test_dataset))

    def test_getitem_keys(self):
        dm = self._make_dm(targets=self.targets_scalar)
        sample = dm.train_dataset[0]
        self.assertIn("input_ids", sample)
        self.assertIn("attention_mask", sample)
        self.assertIn("target", sample)

    def test_getitem_input_ids_shape(self):
        dm = self._make_dm(targets=self.targets_scalar)
        sample = dm.train_dataset[0]
        expected_len = self.tok.config.max_sequence_length
        self.assertEqual(sample["input_ids"].shape, (expected_len,))

    def test_getitem_attention_mask_shape(self):
        dm = self._make_dm(targets=self.targets_scalar)
        sample = dm.train_dataset[0]
        self.assertEqual(sample["attention_mask"].shape,
                         sample["input_ids"].shape)

    def test_getitem_target_shape_scalar(self):
        """Scalar targets → each item target has shape (1,)."""
        dm = self._make_dm(targets=self.targets_scalar)
        sample = dm.train_dataset[0]
        self.assertEqual(sample["target"].shape, (1,))

    def test_attention_mask_binary(self):
        """Attention mask must contain only 0s and 1s."""
        dm = self._make_dm(targets=self.targets_scalar)
        for i in range(min(5, len(dm.train_dataset))):
            mask = dm.train_dataset[i]["attention_mask"]
            unique = set(mask.tolist())
            self.assertTrue(unique.issubset({0, 1}))

    def test_bos_at_start(self):
        """First token of every sequence must be BOS."""
        dm = self._make_dm(targets=self.targets_scalar)
        bos = self.tok.special_tokens["BOS"]
        for i in range(min(5, len(dm.train_dataset))):
            first_tok = dm.train_dataset[i]["input_ids"][0].item()
            self.assertEqual(first_tok, bos, f"sample {i}: first token is not BOS")

    def test_dataloader_batch_keys(self):
        """Train DataLoader must yield batches with required keys."""
        dm = self._make_dm(targets=self.targets_scalar)
        loader = dm.get_train_dataloader()
        batch = next(iter(loader))
        self.assertIn("input_ids", batch)
        self.assertIn("attention_mask", batch)
        self.assertIn("target", batch)

    def test_dataloader_batch_shapes(self):
        """Batch shapes must be (B, L), (B, L), (B, 1) for scalar targets."""
        dm = self._make_dm(targets=self.targets_scalar)
        loader = dm.get_train_dataloader()
        batch = next(iter(loader))
        B = batch["input_ids"].shape[0]
        L = self.tok.config.max_sequence_length
        self.assertEqual(batch["input_ids"].shape, (B, L))
        self.assertEqual(batch["attention_mask"].shape, (B, L))
        self.assertEqual(batch["target"].shape, (B, 1))

    def test_input_ids_dtype(self):
        dm = self._make_dm(targets=self.targets_scalar)
        loader = dm.get_train_dataloader()
        batch = next(iter(loader))
        self.assertEqual(batch["input_ids"].dtype, torch.long)

    def test_target_dtype(self):
        dm = self._make_dm(targets=self.targets_scalar)
        loader = dm.get_train_dataloader()
        batch = next(iter(loader))
        self.assertEqual(batch["target"].dtype, torch.float32)

    def test_no_targets_no_target_key(self):
        """Without targets, batches must NOT contain 'target'."""
        dm = self._make_dm(targets=None)
        sample = dm.train_dataset[0]
        self.assertNotIn("target", sample)

    def test_reproducible_split_with_seed(self):
        """Same seed → same train/val/test split."""
        dm1 = self._make_dm(targets=self.targets_scalar, seed=42)
        dm2 = self._make_dm(targets=self.targets_scalar, seed=42)
        ids1 = [dm1.train_dataset[i]["input_ids"].tolist()
                for i in range(len(dm1.train_dataset))]
        ids2 = [dm2.train_dataset[i]["input_ids"].tolist()
                for i in range(len(dm2.train_dataset))]
        self.assertEqual(ids1, ids2)

    def test_get_dataset_info(self):
        dm = self._make_dm(targets=self.targets_scalar)
        info = dm.get_dataset_info()
        self.assertIn("tokenizer", info)
        self.assertIn("datasets", info)
        self.assertIn("targets", info)
        self.assertTrue(info["targets"]["has_targets"])
        self.assertEqual(info["targets"]["n_targets"], 1)


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# E. Target handling
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

class TestTargetHandling(unittest.TestCase):

    def setUp(self):
        _need_dataset(self)
        self.n = 4
        self.n_shots = 15
        cfg = create_default_config(self.n, n_shadows=self.n_shots)
        self.collector = ShadowCollector(cfg)
        self.collector.sample_dense(_make_random_state(self.n))
        self.tok = create_default_tokenizer(self.n, token_type="basis_outcome")

    def _dm(self, targets):
        dm_cfg = DatasetConfig(
            batch_size=4, shuffle=True, shuffle_seed=0,
            train_split=0.7, val_split=0.15, test_split=0.15,
            pin_memory=False,
        )
        dm = ShadowDataModule(dm_cfg)
        dm.setup(self.collector, self.tok, targets=targets)
        return dm

    def test_scalar_targets_shape(self):
        tgts = np.linspace(0, 1, self.n_shots)
        dm = self._dm(tgts)
        sample = dm.train_dataset[0]
        self.assertEqual(sample["target"].shape, (1,))

    def test_vector_targets_shape(self):
        """2D targets (n_shots, 3) → each item target has shape (3,)."""
        tgts = np.random.default_rng(1).standard_normal((self.n_shots, 3))
        dm = self._dm(tgts)
        sample = dm.train_dataset[0]
        self.assertEqual(sample["target"].shape, (3,))

    def test_vector_targets_batch_shape(self):
        tgts = np.random.default_rng(2).standard_normal((self.n_shots, 3))
        dm = self._dm(tgts)
        loader = dm.get_train_dataloader()
        batch = next(iter(loader))
        B = batch["input_ids"].shape[0]
        self.assertEqual(batch["target"].shape, (B, 3))

    def test_target_values_preserved(self):
        """Target values must be preserved through dataset → batch."""
        tgts = np.arange(self.n_shots, dtype=float)
        dm_cfg = DatasetConfig(
            batch_size=self.n_shots, shuffle=False, shuffle_seed=None,
            train_split=1.0, val_split=0.0, test_split=0.0,
            pin_memory=False,
        )
        dm = ShadowDataModule(dm_cfg)
        dm.setup(self.collector, self.tok, targets=tgts)
        loader = dm.get_train_dataloader()
        batch = next(iter(loader))
        retrieved = batch["target"].squeeze(-1).tolist()
        # Sorted because no guaranteed order in DataLoader
        self.assertEqual(sorted(retrieved), sorted(tgts.tolist()))

    def test_mismatched_targets_raise(self):
        """Target length ≠ n_measurements must raise ValueError."""
        tgts = np.ones(self.n_shots + 5)   # wrong length
        dm_cfg = DatasetConfig(batch_size=4, shuffle=False,
                               train_split=1.0, val_split=0.0, test_split=0.0,
                               pin_memory=False)
        dm = ShadowDataModule(dm_cfg)
        with self.assertRaises(ValueError):
            dm.setup(self.collector, self.tok, targets=tgts)

    def test_n_targets_in_dataset_info(self):
        tgts = np.random.default_rng(3).standard_normal((self.n_shots, 2))
        dm = self._dm(tgts)
        info = dm.get_dataset_info()
        self.assertEqual(info["targets"]["n_targets"], 2)


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# F. Edge cases / error handling
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

class TestEdgeCases(unittest.TestCase):

    def setUp(self):
        _need_shadows(self)
        self.n = 3

    def test_invalid_basis_value_basis_outcome(self):
        tok = create_default_tokenizer(self.n, token_type="basis_outcome")
        m = ShadowMeasurement(basis=np.array([0, 5, 2]), outcome=np.array([0, 1, 0]))
        with self.assertRaises(ValueError):
            tok.tokenize_measurement(m)

    def test_invalid_outcome_value(self):
        tok = create_default_tokenizer(self.n, token_type="basis_outcome")
        m = ShadowMeasurement(basis=np.array([0, 1, 2]), outcome=np.array([0, 2, 0]))
        with self.assertRaises(ValueError):
            tok.tokenize_measurement(m)

    def test_mismatched_basis_outcome_shapes(self):
        tok = create_default_tokenizer(self.n, token_type="basis_outcome")
        m = ShadowMeasurement(basis=np.array([0, 1, 2]), outcome=np.array([0, 1]))
        with self.assertRaises(ValueError):
            tok.tokenize_measurement(m)

    def test_wrong_n_qubits(self):
        tok = create_default_tokenizer(self.n, token_type="basis_outcome")
        m = ShadowMeasurement(basis=np.array([0, 1]), outcome=np.array([0, 1]))
        with self.assertRaises(ValueError) as ctx:
            tok.tokenize_measurement(m)
        self.assertIn("n_qubits", str(ctx.exception))

    def test_empty_collector_raises(self):
        tok = create_default_tokenizer(self.n, token_type="basis_outcome")
        cfg = create_default_config(self.n, n_shadows=5)
        empty_collector = ShadowCollector(cfg)
        with self.assertRaises(ValueError):
            tok.tokenize_collector(empty_collector)

    def test_empty_sequence_list_raises(self):
        _need_dataset(self)
        cfg = DatasetConfig(
            batch_size=4, shuffle=False,
            train_split=0.8, val_split=0.1, test_split=0.1,
            pin_memory=False,
        )
        dm = ShadowDataModule(cfg)
        with self.assertRaises(ValueError):
            dm._split_dataset([], None)

    def test_invalid_split_fractions(self):
        _need_dataset(self)
        with self.assertRaises(ValueError):
            DatasetConfig(train_split=0.9, val_split=0.9, test_split=0.1)

    def test_invalid_split_value_negative(self):
        _need_dataset(self)
        with self.assertRaises(ValueError):
            DatasetConfig(train_split=-0.1, val_split=0.6, test_split=0.5)

    def test_clifford_binary_out_of_range(self):
        tok = create_default_tokenizer(self.n, token_type="binary",
                                       measurement_mode="clifford")
        m = ShadowMeasurement(basis=np.array([0, 24, 0]), outcome=np.array([0, 0, 0]))
        with self.assertRaises(ValueError):
            tok.tokenize_measurement(m)

    def test_custom_above_max_basis_value(self):
        tok = create_default_tokenizer(self.n, token_type="basis_outcome",
                                       measurement_mode="custom",
                                       max_basis_value=4)
        m = ShadowMeasurement(basis=np.array([0, 5, 0]), outcome=np.array([0, 0, 0]))
        with self.assertRaises(ValueError):
            tok.tokenize_measurement(m)

    def test_invalid_token_type(self):
        with self.assertRaises(ValueError):
            create_default_tokenizer(self.n, token_type="nonsense")

    def test_pauli_string_rejects_clifford_mode(self):
        with self.assertRaises(ValueError):
            create_default_tokenizer(self.n, token_type="pauli_string",
                                     measurement_mode="clifford")

    def test_custom_config_without_sampler_raises(self):
        with self.assertRaises(ValueError):
            ShadowConfig(n_qubits=self.n, measurement_basis="custom",
                         custom_sampler=None)

    def test_2d_basis_raises(self):
        tok = create_default_tokenizer(self.n, token_type="basis_outcome")
        m = ShadowMeasurement(
            basis=np.array([[0, 1, 2], [0, 1, 2]]),
            outcome=np.array([[0, 1, 0], [0, 1, 0]]),
        )
        with self.assertRaises(ValueError):
            tok.tokenize_measurement(m)


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# G. Save / load behaviour
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

class TestSaveLoad(unittest.TestCase):

    def setUp(self):
        _need_shadows(self)
        self.n = 4
        self.m = ShadowMeasurement(
            basis=np.array([0, 1, 2, 0]),
            outcome=np.array([1, 0, 1, 0]),
        )

    def _round_trip(self, token_type, measurement_mode=None, max_basis_value=23):
        """Save and reload a tokenizer; return (original, loaded, original_ids)."""
        tok = create_default_tokenizer(self.n, token_type=token_type,
                                       measurement_mode=measurement_mode,
                                       max_basis_value=max_basis_value)
        # Need a measurement that fits the mode
        if measurement_mode == "clifford":
            m = ShadowMeasurement(basis=np.array([0, 5, 12, 23]),
                                  outcome=np.array([0, 1, 0, 1]))
        elif measurement_mode == "custom":
            m = ShadowMeasurement(basis=np.array([0, 1, 2, 1]),
                                  outcome=np.array([1, 0, 1, 0]))
        else:
            m = self.m
        ids_before = tok.tokenize_measurement(m)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        tok.save_tokenizer(path)

        tok2 = create_default_tokenizer(self.n, token_type=token_type,
                                         measurement_mode=measurement_mode,
                                         max_basis_value=max_basis_value)
        tok2.load_tokenizer(path)
        return tok, tok2, ids_before, m

    def test_pauli_basis_outcome_round_trip(self):
        _, tok2, ids_before, m = self._round_trip("basis_outcome")
        self.assertEqual(tok2.tokenize_measurement(m), ids_before)

    def test_pauli_binary_round_trip(self):
        _, tok2, ids_before, m = self._round_trip("binary")
        self.assertEqual(tok2.tokenize_measurement(m), ids_before)

    def test_clifford_basis_outcome_round_trip(self):
        _, tok2, ids_before, m = self._round_trip(
            "basis_outcome", measurement_mode="clifford")
        self.assertEqual(tok2.tokenize_measurement(m), ids_before)

    def test_custom_basis_outcome_round_trip(self):
        _, tok2, ids_before, m = self._round_trip(
            "basis_outcome", measurement_mode="custom", max_basis_value=4)
        self.assertEqual(tok2.tokenize_measurement(m), ids_before)

    def test_measurement_mode_preserved(self):
        _, tok2, _, _ = self._round_trip(
            "basis_outcome", measurement_mode="clifford")
        self.assertEqual(tok2.config.measurement_mode, "clifford")

    def test_max_basis_value_preserved(self):
        _, tok2, _, _ = self._round_trip(
            "basis_outcome", measurement_mode="custom", max_basis_value=7)
        self.assertEqual(tok2.config.max_basis_value, 7)

    def test_vocab_preserved(self):
        tok1, tok2, _, _ = self._round_trip("basis_outcome")
        self.assertEqual(tok1.vocab, tok2.vocab)

    def test_special_tokens_preserved(self):
        tok1, tok2, _, _ = self._round_trip("basis_outcome")
        self.assertEqual(tok1.special_tokens, tok2.special_tokens)

    def test_save_datasets_creates_files(self):
        _need_dataset(self)
        n_shots = 10
        cfg_s = create_default_config(self.n, n_shadows=n_shots)
        collector = ShadowCollector(cfg_s)
        collector.sample_dense(_make_random_state(self.n))
        tok = create_default_tokenizer(self.n, token_type="basis_outcome")
        tgts = np.ones(n_shots)
        dm_cfg = DatasetConfig(
            batch_size=4, shuffle=False,
            train_split=0.8, val_split=0.1, test_split=0.1,
            pin_memory=False,
        )
        dm = ShadowDataModule(dm_cfg)
        dm.setup(collector, tok, targets=tgts)
        with tempfile.TemporaryDirectory() as tmpdir:
            dm.save_datasets(tmpdir)
            import os
            self.assertIn("tokenizer.json", os.listdir(tmpdir))
            self.assertIn("dataset_info.json", os.listdir(tmpdir))

    def test_save_datasets_info_matches(self):
        _need_dataset(self)
        n_shots = 10
        cfg_s = create_default_config(self.n, n_shadows=n_shots)
        collector = ShadowCollector(cfg_s)
        collector.sample_dense(_make_random_state(self.n))
        tok = create_default_tokenizer(self.n, token_type="basis_outcome")
        tgts = np.ones(n_shots)
        dm_cfg = DatasetConfig(
            batch_size=4, shuffle=False,
            train_split=0.8, val_split=0.1, test_split=0.1,
            pin_memory=False,
        )
        dm = ShadowDataModule(dm_cfg)
        dm.setup(collector, tok, targets=tgts)
        with tempfile.TemporaryDirectory() as tmpdir:
            dm.save_datasets(tmpdir)
            import json, os
            with open(os.path.join(tmpdir, "dataset_info.json")) as f:
                info = json.load(f)
            total = (info["datasets"]["train_size"]
                     + info["datasets"]["val_size"]
                     + info["datasets"]["test_size"])
            self.assertEqual(total, n_shots)


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# H. End-to-end smoke tests
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

class TestEndToEnd(unittest.TestCase):
    """
    Full-pipeline smoke tests: collect → tokenize → dataset → dataloader →
    batch, one for each supported measurement mode.
    """

    def setUp(self):
        _need_dataset(self)
        self.n = 4
        self.n_shots = 20
        self.sv = _make_random_state(self.n)
        self._dm_cfg = DatasetConfig(
            batch_size=4, shuffle=True, shuffle_seed=0,
            train_split=0.7, val_split=0.15, test_split=0.15,
            pin_memory=False,
        )

    def _run_pipeline(self, collector, tok, targets):
        """Run setup → train dataloader → one batch."""
        dm = ShadowDataModule(self._dm_cfg)
        dm.setup(collector, tok, targets=targets)
        loader = dm.get_train_dataloader()
        batch = next(iter(loader))
        return batch

    def _check_batch(self, batch, expected_seq_len, expected_n_targets):
        B = batch["input_ids"].shape[0]
        self.assertEqual(batch["input_ids"].shape, (B, expected_seq_len))
        self.assertEqual(batch["attention_mask"].shape, (B, expected_seq_len))
        self.assertEqual(batch["target"].shape, (B, expected_n_targets))
        self.assertEqual(batch["input_ids"].dtype, torch.long)
        self.assertEqual(batch["target"].dtype, torch.float32)

    def test_random_mode_pipeline(self):
        cfg = create_default_config(self.n, n_shadows=self.n_shots)
        collector = ShadowCollector(cfg)
        collector.sample_dense(self.sv)
        tok = create_default_tokenizer(self.n, token_type="basis_outcome")
        targets = np.random.default_rng(0).standard_normal(self.n_shots)
        batch = self._run_pipeline(collector, tok, targets)
        self._check_batch(batch, tok.config.max_sequence_length, 1)

    def test_pauli_mode_pauli_string_pipeline(self):
        cfg = create_pauli_config(self.n, n_shadows=self.n_shots)
        collector = ShadowCollector(cfg)
        collector.sample_dense(self.sv)
        tok = create_default_tokenizer(self.n, token_type="pauli_string")
        targets = np.random.default_rng(1).standard_normal(self.n_shots)
        batch = self._run_pipeline(collector, tok, targets)
        self._check_batch(batch, tok.config.max_sequence_length, 1)

    def test_clifford_mode_basis_outcome_pipeline(self):
        cfg = create_clifford_config(self.n, n_shadows=self.n_shots)
        collector = ShadowCollector(cfg)
        collector.sample_dense(self.sv)
        tok = create_default_tokenizer(self.n, token_type="basis_outcome",
                                       measurement_mode="clifford")
        targets = np.random.default_rng(2).standard_normal(self.n_shots)
        batch = self._run_pipeline(collector, tok, targets)
        self._check_batch(batch, tok.config.max_sequence_length, 1)

    def test_clifford_mode_binary_pipeline(self):
        cfg = create_clifford_config(self.n, n_shadows=self.n_shots)
        collector = ShadowCollector(cfg)
        collector.sample_dense(self.sv)
        tok = create_default_tokenizer(self.n, token_type="binary",
                                       measurement_mode="clifford")
        targets = np.random.default_rng(3).standard_normal(self.n_shots)
        batch = self._run_pipeline(collector, tok, targets)
        self._check_batch(batch, tok.config.max_sequence_length, 1)

    def test_custom_mode_basis_outcome_pipeline(self):
        cfg = create_custom_config(self.n, custom_sampler=_custom_sampler,
                                   n_shadows=self.n_shots)
        collector = ShadowCollector(cfg)
        collector.sample_dense(self.sv)
        # Custom sampler uses Pauli labels 0/1/2, so max_basis_value=2
        tok = create_default_tokenizer(self.n, token_type="basis_outcome",
                                       measurement_mode="custom",
                                       max_basis_value=2)
        targets = np.random.default_rng(4).standard_normal(self.n_shots)
        batch = self._run_pipeline(collector, tok, targets)
        self._check_batch(batch, tok.config.max_sequence_length, 1)

    def test_vector_targets_pipeline(self):
        """Multi-output regression: 3 target values per measurement."""
        cfg = create_default_config(self.n, n_shadows=self.n_shots)
        collector = ShadowCollector(cfg)
        collector.sample_dense(self.sv)
        tok = create_default_tokenizer(self.n, token_type="basis_outcome")
        targets = np.random.default_rng(5).standard_normal((self.n_shots, 3))
        batch = self._run_pipeline(collector, tok, targets)
        self._check_batch(batch, tok.config.max_sequence_length, 3)

    def test_model_forward_pass(self):
        """Model forward pass must produce the right output shape."""
        cfg = create_default_config(self.n, n_shadows=self.n_shots)
        collector = ShadowCollector(cfg)
        collector.sample_dense(self.sv)
        tok = create_default_tokenizer(self.n, token_type="basis_outcome")
        targets = np.random.default_rng(6).standard_normal(self.n_shots)
        dm = ShadowDataModule(self._dm_cfg)
        dm.setup(collector, tok, targets=targets)
        model = create_model_from_tokenizer(tok, n_outputs=1,
                                             d_model=32, n_heads=2,
                                             n_layers=1, d_ff=64)
        loader = dm.get_train_dataloader()
        batch = next(iter(loader))
        with torch.no_grad():
            out = model(batch["input_ids"], batch["attention_mask"])
        B = batch["input_ids"].shape[0]
        self.assertEqual(out.shape, (B, 1))
        self.assertEqual(out.dtype, torch.float32)

    def test_trainer_one_epoch(self):
        """One training epoch must run without error and return a finite loss."""
        cfg = create_default_config(self.n, n_shadows=self.n_shots)
        collector = ShadowCollector(cfg)
        collector.sample_dense(self.sv)
        tok = create_default_tokenizer(self.n, token_type="basis_outcome")
        targets = np.random.default_rng(7).standard_normal(self.n_shots)
        dm = ShadowDataModule(self._dm_cfg)
        dm.setup(collector, tok, targets=targets)
        model = create_model_from_tokenizer(tok, n_outputs=1,
                                             d_model=32, n_heads=2,
                                             n_layers=1, d_ff=64)
        trainer = ShadowTrainer(model)
        train_loss = trainer.train_epoch(dm.get_train_dataloader())
        self.assertIsInstance(train_loss, float)
        self.assertTrue(np.isfinite(train_loss),
                        f"Training loss is not finite: {train_loss}")

    def test_trainer_predict(self):
        """predict() must return a tensor of shape (N_test, n_outputs)."""
        cfg = create_default_config(self.n, n_shadows=self.n_shots)
        collector = ShadowCollector(cfg)
        collector.sample_dense(self.sv)
        tok = create_default_tokenizer(self.n, token_type="basis_outcome")
        targets = np.random.default_rng(8).standard_normal(self.n_shots)
        dm = ShadowDataModule(self._dm_cfg)
        dm.setup(collector, tok, targets=targets)
        model = create_model_from_tokenizer(tok, n_outputs=1,
                                             d_model=32, n_heads=2,
                                             n_layers=1, d_ff=64)
        trainer = ShadowTrainer(model)
        preds = trainer.predict(dm.get_test_dataloader())
        N_test = len(dm.test_dataset)
        self.assertEqual(preds.shape[0], N_test)
        self.assertEqual(preds.shape[1], 1)


# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# Runner
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

if __name__ == "__main__":
    unittest.main(verbosity=2)
