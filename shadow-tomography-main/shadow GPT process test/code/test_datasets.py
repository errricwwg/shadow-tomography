"""
Integration tests for datasets.py.

Verifies compatibility with ShadowCollector, ShadowMeasurement, and
ShadowTokenizer; end-to-end setup; __getitem__ and collate_fn correctness;
split behavior; error handling; and save behavior.

Run from the repo root:
    PYTHONPATH="shadow GPT process test/.local_test_deps:shadow GPT/code" \\
        python3 "shadow GPT process test/code/test_datasets.py"
or with pytest:
    PYTHONPATH="shadow GPT process test/.local_test_deps:shadow GPT/code" \\
        python3 -m pytest "shadow GPT process test/code/test_datasets.py" -v
"""

import sys
import tempfile
import unittest
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "shadow GPT" / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

_IMPORT_ERROR = None

try:
    import numpy as np
    import torch
except ModuleNotFoundError as exc:
    np = None
    torch = None
    _IMPORT_ERROR = exc

if _IMPORT_ERROR is None:
    try:
        from shadows.collector import ShadowCollector, ShadowMeasurement
        from shadows.config import create_default_config
        from shadows.datasets import (
            DatasetConfig,
            ShadowDataModule,
            ShadowDataset,
            create_data_module,
        )
        from shadows.tokenization import (
            ShadowTokenizer,
            TokenizationConfig,
            create_default_tokenizer,
        )
    except ModuleNotFoundError as exc:
        _IMPORT_ERROR = exc

RUNTIME_AVAILABLE = _IMPORT_ERROR is None

# Note: DatasetConfig cannot be imported without torch because datasets.py
# imports torch at module level.  All tests therefore require torch.


def _skip_if_unavailable(tc):
    if not RUNTIME_AVAILABLE:
        tc.skipTest(f"runtime dependencies unavailable: {_IMPORT_ERROR}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_measurement(basis_list, outcome_list):
    return ShadowMeasurement(
        basis=np.array(basis_list, dtype=int),
        outcome=np.array(outcome_list, dtype=int),
    )


def _make_collector(n_qubits, n_shots=20):
    """Return a ShadowCollector pre-populated with n_shots random Pauli measurements."""
    config = create_default_config(n_qubits=n_qubits, n_shadows=n_shots)
    collector = ShadowCollector(config)
    rng = np.random.default_rng(42)
    measurements = []
    for _ in range(n_shots):
        basis = rng.integers(0, 3, size=n_qubits)
        outcome = rng.integers(0, 2, size=n_qubits)
        measurements.append(ShadowMeasurement(basis=basis, outcome=outcome))
    collector.measurements = measurements
    return collector


def _make_tokenizer(n_qubits=3, token_type="basis_outcome"):
    return create_default_tokenizer(n_qubits, token_type=token_type)


def _make_dataset_config(**kwargs):
    defaults = dict(
        batch_size=4,
        shuffle=True,
        shuffle_seed=0,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
    )
    defaults.update(kwargs)
    return DatasetConfig(**defaults)


# ---------------------------------------------------------------------------
# A. End-to-end setup
# ---------------------------------------------------------------------------

class TestEndToEndSetup(unittest.TestCase):

    def setUp(self):
        _skip_if_unavailable(self)
        self.n = 3
        self.collector = _make_collector(self.n, n_shots=20)
        self.tokenizer = _make_tokenizer(self.n)
        self.dc = _make_dataset_config()

    def test_setup_creates_all_three_datasets(self):
        dm = ShadowDataModule(self.dc)
        dm.setup(self.collector, self.tokenizer)
        self.assertIsNotNone(dm.train_dataset)
        self.assertIsNotNone(dm.val_dataset)
        self.assertIsNotNone(dm.test_dataset)

    def test_total_sizes_sum_to_n_shots(self):
        dm = ShadowDataModule(self.dc)
        dm.setup(self.collector, self.tokenizer)
        total = len(dm.train_dataset) + len(dm.val_dataset) + len(dm.test_dataset)
        self.assertEqual(total, 20)

    def test_tokenizer_stored_on_module(self):
        dm = ShadowDataModule(self.dc)
        dm.setup(self.collector, self.tokenizer)
        self.assertIs(dm.tokenizer, self.tokenizer)

    def test_create_data_module_convenience(self):
        dm = create_data_module(self.collector, self.tokenizer, self.dc)
        self.assertIsNotNone(dm.train_dataset)

    def test_setup_without_prior_call_raises_on_dataloader(self):
        dm = ShadowDataModule(self.dc)
        with self.assertRaises(ValueError):
            dm.get_train_dataloader()

    def test_all_three_token_types_setup(self):
        for tt in ("basis_outcome", "pauli_string", "binary"):
            with self.subTest(token_type=tt):
                tok = _make_tokenizer(self.n, token_type=tt)
                dm = ShadowDataModule(self.dc)
                dm.setup(self.collector, tok)
                self.assertIsNotNone(dm.train_dataset)


# ---------------------------------------------------------------------------
# B. Tokenization-to-dataset integration
# ---------------------------------------------------------------------------

class TestTokenizationIntegration(unittest.TestCase):

    def setUp(self):
        _skip_if_unavailable(self)
        self.n = 4
        self.collector = _make_collector(self.n, n_shots=10)
        self.tokenizer = _make_tokenizer(self.n)

    def test_tokenize_collector_feeds_create_sequences(self):
        raw = self.tokenizer.tokenize_collector(self.collector)
        self.assertEqual(len(raw), 10)
        seqs = self.tokenizer.create_sequences(raw, add_special_tokens=True)
        self.assertEqual(len(seqs), 10)

    def test_sequences_are_lists_of_ints(self):
        raw = self.tokenizer.tokenize_collector(self.collector)
        seqs = self.tokenizer.create_sequences(raw)
        for seq in seqs:
            self.assertIsInstance(seq, list)
            for tid in seq:
                self.assertIsInstance(tid, int)

    def test_sequences_can_be_consumed_by_shadow_dataset(self):
        raw = self.tokenizer.tokenize_collector(self.collector)
        seqs = self.tokenizer.create_sequences(raw)
        ds = ShadowDataset(seqs, self.tokenizer)
        self.assertEqual(len(ds), 10)

    def test_dataset_tensors_are_long_dtype(self):
        raw = self.tokenizer.tokenize_collector(self.collector)
        seqs = self.tokenizer.create_sequences(raw)
        ds = ShadowDataset(seqs, self.tokenizer)
        item = ds[0]
        self.assertEqual(item["input_ids"].dtype, torch.long)
        self.assertEqual(item["labels"].dtype, torch.long)


# ---------------------------------------------------------------------------
# C. __getitem__ behavior
# ---------------------------------------------------------------------------

class TestGetItem(unittest.TestCase):

    def setUp(self):
        _skip_if_unavailable(self)
        self.n = 3
        self.tokenizer = _make_tokenizer(self.n, token_type="basis_outcome")

    def _make_dataset_from_raw(self, seq_list):
        """Build a ShadowDataset directly from a list of integer lists."""
        return ShadowDataset(seq_list, self.tokenizer)

    def test_input_is_seq_drop_last(self):
        seq = [10, 11, 12, 13, 14]
        ds = self._make_dataset_from_raw([seq])
        item = ds[0]
        expected_input = torch.tensor([10, 11, 12, 13], dtype=torch.long)
        self.assertTrue(torch.equal(item["input_ids"], expected_input))

    def test_labels_is_seq_drop_first(self):
        seq = [10, 11, 12, 13, 14]
        ds = self._make_dataset_from_raw([seq])
        item = ds[0]
        expected_labels = torch.tensor([11, 12, 13, 14], dtype=torch.long)
        self.assertTrue(torch.equal(item["labels"], expected_labels))

    def test_input_and_labels_same_length(self):
        seq = [0, 4, 5, 6, 1]  # BOS content content content EOS
        ds = self._make_dataset_from_raw([seq])
        item = ds[0]
        self.assertEqual(len(item["input_ids"]), len(item["labels"]))

    def test_causal_shift_explicit(self):
        """input_ids[i+1] must equal labels[i] for all i."""
        seq = list(range(8))
        ds = self._make_dataset_from_raw([seq])
        item = ds[0]
        ids = item["input_ids"].tolist()
        lbls = item["labels"].tolist()
        for i in range(len(ids) - 1):
            self.assertEqual(ids[i + 1], lbls[i])

    def test_bos_is_first_input_token(self):
        """After create_sequences, BOS should be the first input token."""
        m = _make_measurement([0, 1, 2], [0, 1, 0])
        raw_seq = self.tokenizer.tokenize_measurement(m)
        seq = self.tokenizer.create_sequences([raw_seq], add_special_tokens=True)[0]
        ds = self._make_dataset_from_raw([seq])
        item = ds[0]
        self.assertEqual(item["input_ids"][0].item(), self.tokenizer.special_tokens["BOS"])

    def test_eos_is_last_label_token_before_padding(self):
        """After create_sequences, EOS should appear in labels."""
        m = _make_measurement([0, 1, 2], [0, 1, 0])
        raw_seq = self.tokenizer.tokenize_measurement(m)
        seq = self.tokenizer.create_sequences([raw_seq], add_special_tokens=True)[0]
        ds = self._make_dataset_from_raw([seq])
        item = ds[0]
        eos = self.tokenizer.special_tokens["EOS"]
        self.assertIn(eos, item["labels"].tolist())


# ---------------------------------------------------------------------------
# D. collate_fn behavior
# ---------------------------------------------------------------------------

class TestCollateFn(unittest.TestCase):

    def setUp(self):
        _skip_if_unavailable(self)
        # Use padding_strategy="none" so sequences have different lengths,
        # which exercises the collate_fn padding logic.
        config = TokenizationConfig(
            n_qubits=3,
            token_type="basis_outcome",
            max_sequence_length=20,
            padding_strategy="none",
        )
        self.tokenizer = ShadowTokenizer(config)
        self.pad_id = self.tokenizer.special_tokens["PAD"]

    def _make_batch(self, seq_lists):
        ds = ShadowDataset(seq_lists, self.tokenizer)
        batch = [ds[i] for i in range(len(ds))]
        return ds.collate_fn(batch)

    def test_output_keys(self):
        batch = self._make_batch([[4, 5, 6, 7], [4, 5, 6, 7, 8, 9]])
        self.assertIn("input_ids", batch)
        self.assertIn("labels", batch)
        self.assertIn("attention_mask", batch)

    def test_input_ids_padded_to_max_length(self):
        # seq1 input = length 3, seq2 input = length 5; max = 5
        batch = self._make_batch([[4, 5, 6, 7], [4, 5, 6, 7, 8, 9]])
        self.assertEqual(batch["input_ids"].shape, (2, 5))

    def test_labels_shape_matches_input_ids(self):
        batch = self._make_batch([[4, 5, 6, 7], [4, 5, 6, 7, 8, 9]])
        self.assertEqual(batch["labels"].shape, batch["input_ids"].shape)

    def test_attention_mask_shape(self):
        batch = self._make_batch([[4, 5, 6, 7], [4, 5, 6, 7, 8, 9]])
        self.assertEqual(batch["attention_mask"].shape, batch["input_ids"].shape)

    def test_shorter_input_padded_with_pad_token(self):
        # seq1: [4,5,6,7] → input [4,5,6] len=3; seq2: [4,5,6,7,8,9] → input [4,5,6,7,8] len=5
        batch = self._make_batch([[4, 5, 6, 7], [4, 5, 6, 7, 8, 9]])
        short_input = batch["input_ids"][0]
        # last 2 positions should be PAD
        self.assertEqual(short_input[-1].item(), self.pad_id)
        self.assertEqual(short_input[-2].item(), self.pad_id)

    def test_labels_padded_with_minus_100(self):
        batch = self._make_batch([[4, 5, 6, 7], [4, 5, 6, 7, 8, 9]])
        short_labels = batch["labels"][0]
        self.assertEqual(short_labels[-1].item(), -100)
        self.assertEqual(short_labels[-2].item(), -100)

    def test_attention_mask_zero_on_padded_positions(self):
        batch = self._make_batch([[4, 5, 6, 7], [4, 5, 6, 7, 8, 9]])
        short_mask = batch["attention_mask"][0]
        # first 3 positions: real tokens → 1
        self.assertTrue(all(short_mask[:3].tolist()))
        # last 2 positions: padding → 0
        self.assertEqual(short_mask[-1].item(), 0)
        self.assertEqual(short_mask[-2].item(), 0)

    def test_attention_mask_all_ones_for_longest_seq(self):
        batch = self._make_batch([[4, 5, 6, 7], [4, 5, 6, 7, 8, 9]])
        long_mask = batch["attention_mask"][1]
        self.assertTrue(all(v == 1 for v in long_mask.tolist()))

    def test_uniform_length_batch_no_padding(self):
        batch = self._make_batch([[4, 5, 6, 7], [8, 9, 4, 5]])
        # input: both length 3 → no PAD in input_ids
        for i in range(2):
            self.assertNotIn(self.pad_id, batch["input_ids"][i].tolist())

    def test_attention_mask_uses_input_pad_not_label_pad(self):
        """
        Regression: attention_mask must be based on input_ids padding, not
        the (separately computed) labels padding.  With padding_strategy="none"
        both input and labels have the same source length so input_pad ==
        label_pad here, but this checks the variable isolation is correct.
        """
        batch = self._make_batch([[4, 5, 6, 7], [4, 5, 6, 7, 8, 9]])
        # The shorter sequence: input has 3 real tokens, 2 PADs.
        # attention_mask must show 1,1,1,0,0 — not influenced by label padding.
        short_mask = batch["attention_mask"][0].tolist()
        self.assertEqual(short_mask, [1, 1, 1, 0, 0])

    def test_pad_tokens_in_labels_with_pretokenizer_padding(self):
        """
        Known behavior: when the tokenizer uses padding_strategy='right',
        create_sequences pre-pads all sequences to max_sequence_length.
        Those trailing PAD token IDs (not -100) end up in labels because
        collate_fn only adds -100 for sequences shorter than the batch max,
        and all sequences are already the same length.

        This is a training-quality note, not a crash bug.  The model will
        train to predict PAD tokens after EOS, which wastes capacity.
        This test documents the behavior so it cannot regress silently.
        """
        # Build tokenizer with right-padding (the default)
        tok = create_default_tokenizer(3, token_type="basis_outcome")
        m = _make_measurement([0, 1, 2], [0, 1, 0])
        raw = tok.tokenize_measurement(m)
        seq = tok.create_sequences([raw], add_special_tokens=True)[0]

        ds = ShadowDataset([seq, seq], tok)  # two identical sequences
        batch = ds.collate_fn([ds[0], ds[1]])

        pad_id = tok.special_tokens["PAD"]
        # Since both sequences are the same length, label_pad=0 in collate_fn,
        # so labels contain the real PAD token IDs (not -100).
        labels_flat = batch["labels"].flatten().tolist()
        self.assertIn(pad_id, labels_flat,
                      "PAD token IDs appear in labels when tokenizer pre-pads; "
                      "they are NOT masked to -100 by collate_fn in this case.")
        self.assertNotIn(-100, labels_flat,
                         "-100 should not appear when all sequences are equal length.")


# ---------------------------------------------------------------------------
# E. Split behavior
# ---------------------------------------------------------------------------

class TestSplitBehavior(unittest.TestCase):

    def setUp(self):
        _skip_if_unavailable(self)
        self.n = 3
        self.tokenizer = _make_tokenizer(self.n)

    def _setup_dm(self, n_shots=30, **dc_kwargs):
        collector = _make_collector(self.n, n_shots=n_shots)
        dc = _make_dataset_config(**dc_kwargs)
        dm = ShadowDataModule(dc)
        dm.setup(collector, self.tokenizer)
        return dm

    def test_split_sizes_sum_to_total(self):
        dm = self._setup_dm(n_shots=30)
        total = len(dm.train_dataset) + len(dm.val_dataset) + len(dm.test_dataset)
        self.assertEqual(total, 30)

    def test_train_size_approximate(self):
        dm = self._setup_dm(n_shots=100)
        # train_split=0.7 → expect 70 ± 1 (int rounding)
        self.assertAlmostEqual(len(dm.train_dataset), 70, delta=1)

    def test_val_size_approximate(self):
        dm = self._setup_dm(n_shots=100)
        self.assertAlmostEqual(len(dm.val_dataset), 15, delta=1)

    def test_test_is_remainder(self):
        dm = self._setup_dm(n_shots=100)
        n_train = len(dm.train_dataset)
        n_val = len(dm.val_dataset)
        n_test = len(dm.test_dataset)
        self.assertEqual(n_train + n_val + n_test, 100)

    def test_reproducible_with_seed(self):
        collector = _make_collector(self.n, n_shots=30)
        tok = self.tokenizer

        dc1 = _make_dataset_config(shuffle_seed=99)
        dm1 = ShadowDataModule(dc1)
        dm1.setup(collector, tok)

        dc2 = _make_dataset_config(shuffle_seed=99)
        dm2 = ShadowDataModule(dc2)
        dm2.setup(collector, tok)

        # Same seed → same train set
        train1 = [dm1.train_dataset.token_sequences[i] for i in range(len(dm1.train_dataset))]
        train2 = [dm2.train_dataset.token_sequences[i] for i in range(len(dm2.train_dataset))]
        self.assertEqual(train1, train2)

    def test_different_seeds_give_different_splits(self):
        collector = _make_collector(self.n, n_shots=50)
        tok = self.tokenizer

        dc1 = _make_dataset_config(shuffle_seed=1)
        dm1 = ShadowDataModule(dc1)
        dm1.setup(collector, tok)

        dc2 = _make_dataset_config(shuffle_seed=2)
        dm2 = ShadowDataModule(dc2)
        dm2.setup(collector, tok)

        train1 = dm1.train_dataset.token_sequences
        train2 = dm2.train_dataset.token_sequences
        self.assertNotEqual(train1, train2)

    def test_no_shuffle_preserves_order(self):
        collector = _make_collector(self.n, n_shots=20)
        tok = self.tokenizer
        dc = _make_dataset_config(shuffle=False)
        dm = ShadowDataModule(dc)
        dm.setup(collector, tok)

        # With no shuffle, train sequences should be the first n_train raw sequences.
        raw = tok.tokenize_collector(collector)
        seqs = tok.create_sequences(raw, add_special_tokens=True)
        n_train = int(20 * 0.7)
        self.assertEqual(dm.train_dataset.token_sequences, seqs[:n_train])


# ---------------------------------------------------------------------------
# F-1. DatasetConfig validation (numpy + shadows only, no torch required)
# ---------------------------------------------------------------------------

class TestDatasetConfigValidation(unittest.TestCase):
    """DatasetConfig validation — requires torch because datasets.py imports it at module level."""

    def setUp(self):
        _skip_if_unavailable(self)

    def test_invalid_train_split_negative(self):
        with self.assertRaises(ValueError):
            DatasetConfig(train_split=-0.1, val_split=0.5, test_split=0.6)

    def test_invalid_val_split_over_one(self):
        with self.assertRaises(ValueError):
            DatasetConfig(train_split=0.5, val_split=1.5, test_split=0.0)

    def test_splits_not_summing_to_one_raises(self):
        with self.assertRaises(ValueError):
            DatasetConfig(train_split=0.5, val_split=0.1, test_split=0.1)

    def test_splits_summing_to_one_ok(self):
        # Should not raise
        dc = DatasetConfig(train_split=0.8, val_split=0.1, test_split=0.1)
        self.assertAlmostEqual(dc.train_split + dc.val_split + dc.test_split, 1.0, places=5)

    def test_all_zeros_raises(self):
        with self.assertRaises(ValueError):
            DatasetConfig(train_split=0.0, val_split=0.0, test_split=0.0)

    def test_shuffle_seed_default_none(self):
        dc = DatasetConfig()
        self.assertIsNone(dc.shuffle_seed)

    def test_shuffle_seed_accepted(self):
        dc = DatasetConfig(shuffle_seed=42)
        self.assertEqual(dc.shuffle_seed, 42)


# ---------------------------------------------------------------------------
# F-2. Runtime error / edge cases (torch required)
# ---------------------------------------------------------------------------

class TestErrorCases(unittest.TestCase):

    def setUp(self):
        _skip_if_unavailable(self)
        self.n = 3
        self.tokenizer = _make_tokenizer(self.n)

    def test_empty_collector_raises_in_tokenizer(self):
        config = create_default_config(n_qubits=self.n)
        collector = ShadowCollector(config)
        dc = _make_dataset_config()
        dm = ShadowDataModule(dc)
        with self.assertRaises(ValueError):
            dm.setup(collector, self.tokenizer)

    def test_empty_sequence_list_raises(self):
        dc = _make_dataset_config()
        dm = ShadowDataModule(dc)
        dm.tokenizer = self.tokenizer
        with self.assertRaises(ValueError):
            dm._split_dataset([])

    def test_small_dataset_warns_on_empty_splits(self):
        """2 sequences with 70/15/15 split → val and test will be empty."""
        collector = _make_collector(self.n, n_shots=2)
        dc = _make_dataset_config(shuffle=False)
        dm = ShadowDataModule(dc)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dm.setup(collector, self.tokenizer)
        warning_messages = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        self.assertTrue(
            any("split is empty" in msg for msg in warning_messages),
            f"Expected an empty-split UserWarning; got: {warning_messages}"
        )

    def test_save_datasets_before_setup_raises(self):
        dc = _make_dataset_config()
        dm = ShadowDataModule(dc)
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                dm.save_datasets(tmpdir)

    def test_get_dataloader_before_setup_raises(self):
        dc = _make_dataset_config()
        dm = ShadowDataModule(dc)
        with self.assertRaises(ValueError):
            dm.get_val_dataloader()
        with self.assertRaises(ValueError):
            dm.get_test_dataloader()

    def test_dataset_info_before_setup_returns_zeros(self):
        dc = _make_dataset_config()
        dm = ShadowDataModule(dc)
        info = dm.get_dataset_info()
        self.assertEqual(info["datasets"]["train_size"], 0)
        self.assertEqual(info["datasets"]["val_size"], 0)
        self.assertIsNone(info["tokenizer"]["vocab_size"])


# ---------------------------------------------------------------------------
# G. Save behavior
# ---------------------------------------------------------------------------

class TestSaveBehavior(unittest.TestCase):

    def setUp(self):
        _skip_if_unavailable(self)
        self.n = 3
        self.collector = _make_collector(self.n, n_shots=20)
        self.tokenizer = _make_tokenizer(self.n)
        self.dc = _make_dataset_config()

    def test_tokenizer_json_created(self):
        dm = ShadowDataModule(self.dc)
        dm.setup(self.collector, self.tokenizer)
        with tempfile.TemporaryDirectory() as tmpdir:
            dm.save_datasets(tmpdir)
            self.assertTrue((Path(tmpdir) / "tokenizer.json").exists())

    def test_dataset_info_json_created(self):
        dm = ShadowDataModule(self.dc)
        dm.setup(self.collector, self.tokenizer)
        with tempfile.TemporaryDirectory() as tmpdir:
            dm.save_datasets(tmpdir)
            self.assertTrue((Path(tmpdir) / "dataset_info.json").exists())

    def test_no_tensor_files_saved(self):
        """
        save_datasets() only saves tokenizer.json and dataset_info.json.
        No PyTorch tensor data (.pt files) should be written — the docstring
        explicitly states this limitation.
        """
        import json as _json
        dm = ShadowDataModule(self.dc)
        dm.setup(self.collector, self.tokenizer)
        with tempfile.TemporaryDirectory() as tmpdir:
            dm.save_datasets(tmpdir)
            pt_files = list(Path(tmpdir).glob("*.pt"))
            self.assertEqual(pt_files, [],
                             "save_datasets() must not write .pt tensor files.")
            saved = sorted(p.name for p in Path(tmpdir).iterdir())
            self.assertEqual(saved, ["dataset_info.json", "tokenizer.json"])

    def test_dataset_info_json_is_valid_json(self):
        import json as _json
        dm = ShadowDataModule(self.dc)
        dm.setup(self.collector, self.tokenizer)
        with tempfile.TemporaryDirectory() as tmpdir:
            dm.save_datasets(tmpdir)
            with open(Path(tmpdir) / "dataset_info.json") as f:
                info = _json.load(f)
        self.assertIn("tokenizer", info)
        self.assertIn("datasets", info)
        self.assertIn("config", info)

    def test_dataset_sizes_in_info_match_actual(self):
        import json as _json
        dm = ShadowDataModule(self.dc)
        dm.setup(self.collector, self.tokenizer)
        with tempfile.TemporaryDirectory() as tmpdir:
            dm.save_datasets(tmpdir)
            with open(Path(tmpdir) / "dataset_info.json") as f:
                info = _json.load(f)
        self.assertEqual(info["datasets"]["train_size"], len(dm.train_dataset))
        self.assertEqual(info["datasets"]["val_size"], len(dm.val_dataset))
        self.assertEqual(info["datasets"]["test_size"], len(dm.test_dataset))

    def test_tokenizer_json_loadable_back(self):
        dm = ShadowDataModule(self.dc)
        dm.setup(self.collector, self.tokenizer)
        with tempfile.TemporaryDirectory() as tmpdir:
            dm.save_datasets(tmpdir)
            tok2 = create_default_tokenizer(self.n)
            tok2.load_tokenizer(str(Path(tmpdir) / "tokenizer.json"))
        self.assertEqual(tok2.vocab, self.tokenizer.vocab)
        self.assertEqual(tok2.special_tokens, self.tokenizer.special_tokens)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
