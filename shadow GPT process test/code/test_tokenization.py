"""
Integration tests for tokenization.py.

Verifies compatibility with ShadowMeasurement and ShadowCollector, correctness
of all three tokenization modes, sequence creation, validation/error handling,
and save/load round-trip.

Run from the repo root:
    python "shadow GPT process test/code/test_tokenization.py"
or:
    python -m pytest "shadow GPT process test/code/test_tokenization.py" -v
"""

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
except ModuleNotFoundError as exc:
    np = None
    _IMPORT_ERROR = exc
else:
    try:
        from shadows.collector import ShadowCollector, ShadowMeasurement
        from shadows.config import ShadowConfig, create_default_config
        from shadows.tokenization import (
            ShadowTokenizer,
            TokenizationConfig,
            create_default_tokenizer,
        )
    except ModuleNotFoundError as exc:
        _IMPORT_ERROR = exc
    else:
        _IMPORT_ERROR = None

RUNTIME_AVAILABLE = _IMPORT_ERROR is None


def _skip_if_unavailable(test_case):
    if not RUNTIME_AVAILABLE:
        test_case.skipTest(f"runtime dependencies unavailable: {_IMPORT_ERROR}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_measurement(basis_list, outcome_list):
    """Build a ShadowMeasurement from plain Python lists."""
    return ShadowMeasurement(
        basis=np.array(basis_list, dtype=int),
        outcome=np.array(outcome_list, dtype=int),
    )


def _make_collector_with_measurements(measurements):
    """
    Build a minimal ShadowCollector whose .measurements list is pre-populated.
    Uses a 1-qubit config just to satisfy __init__; measurements override it.
    """
    config = create_default_config(n_qubits=len(measurements[0].basis))
    collector = ShadowCollector(config)
    collector.measurements = list(measurements)
    return collector


# ---------------------------------------------------------------------------
# A. Single-measurement tokenization
# ---------------------------------------------------------------------------

class TestSingleMeasurementTokenization(unittest.TestCase):

    def setUp(self):
        _skip_if_unavailable(self)
        # 4-qubit measurement: bases X,Y,Z,X with outcomes 0,1,0,1
        self.n = 4
        self.m = _make_measurement([0, 1, 2, 0], [0, 1, 0, 1])

    # ── basis_outcome ─────────────────────────────────────────────────────────

    def test_basis_outcome_length(self):
        tok = create_default_tokenizer(self.n, token_type="basis_outcome")
        ids = tok.tokenize_measurement(self.m)
        self.assertEqual(len(ids), self.n)

    def test_basis_outcome_all_ints(self):
        tok = create_default_tokenizer(self.n, token_type="basis_outcome")
        ids = tok.tokenize_measurement(self.m)
        for tid in ids:
            self.assertIsInstance(tid, int)

    def test_basis_outcome_no_unk(self):
        tok = create_default_tokenizer(self.n, token_type="basis_outcome")
        unk = tok.special_tokens["UNK"]
        ids = tok.tokenize_measurement(self.m)
        self.assertNotIn(unk, ids, "UNK should not appear for valid Pauli data")

    def test_basis_outcome_token_values(self):
        """Spot-check: basis=0 outcome=0 → 'B0O0', basis=1 outcome=1 → 'B1O1'."""
        tok = create_default_tokenizer(self.n, token_type="basis_outcome")
        ids = tok.tokenize_measurement(self.m)
        self.assertEqual(ids[0], tok.vocab["B0O0"])  # qubit 0: basis=0, outcome=0
        self.assertEqual(ids[1], tok.vocab["B1O1"])  # qubit 1: basis=1, outcome=1
        self.assertEqual(ids[2], tok.vocab["B2O0"])  # qubit 2: basis=2, outcome=0
        self.assertEqual(ids[3], tok.vocab["B0O1"])  # qubit 3: basis=0, outcome=1

    def test_basis_outcome_detokenize_round_trip(self):
        tok = create_default_tokenizer(self.n, token_type="basis_outcome")
        ids = tok.tokenize_measurement(self.m)
        text = tok.detokenize(ids)
        self.assertIn("B0O0", text)
        self.assertIn("B1O1", text)

    def test_basis_outcome_vocab_size(self):
        # 4 special + 6 content = 10
        tok = create_default_tokenizer(self.n, token_type="basis_outcome")
        self.assertEqual(tok.get_vocab_size(), 10)

    # ── pauli_string ──────────────────────────────────────────────────────────

    def test_pauli_string_length(self):
        tok = create_default_tokenizer(self.n, token_type="pauli_string")
        ids = tok.tokenize_measurement(self.m)
        self.assertEqual(len(ids), self.n)

    def test_pauli_string_no_unk(self):
        tok = create_default_tokenizer(self.n, token_type="pauli_string")
        unk = tok.special_tokens["UNK"]
        ids = tok.tokenize_measurement(self.m)
        self.assertNotIn(unk, ids)

    def test_pauli_string_token_values(self):
        """Bases [0,1,2,0] → ['X','Y','Z','X']."""
        tok = create_default_tokenizer(self.n, token_type="pauli_string")
        ids = tok.tokenize_measurement(self.m)
        self.assertEqual(ids[0], tok.vocab["X"])
        self.assertEqual(ids[1], tok.vocab["Y"])
        self.assertEqual(ids[2], tok.vocab["Z"])
        self.assertEqual(ids[3], tok.vocab["X"])

    def test_pauli_string_outcome_discarded(self):
        """pauli_string mode ignores outcome; two measurements same basis but
        different outcome must produce identical token sequences."""
        tok = create_default_tokenizer(self.n, token_type="pauli_string")
        m2 = _make_measurement([0, 1, 2, 0], [1, 0, 1, 0])  # flipped outcomes
        self.assertEqual(
            tok.tokenize_measurement(self.m),
            tok.tokenize_measurement(m2),
        )

    def test_pauli_string_vocab_size(self):
        # 4 special + 3 content = 7
        tok = create_default_tokenizer(self.n, token_type="pauli_string")
        self.assertEqual(tok.get_vocab_size(), 7)

    # ── binary ────────────────────────────────────────────────────────────────

    def test_binary_length(self):
        """3 bits per qubit (2 for basis, 1 for outcome)."""
        tok = create_default_tokenizer(self.n, token_type="binary")
        ids = tok.tokenize_measurement(self.m)
        self.assertEqual(len(ids), 3 * self.n)

    def test_binary_no_unk(self):
        tok = create_default_tokenizer(self.n, token_type="binary")
        unk = tok.special_tokens["UNK"]
        ids = tok.tokenize_measurement(self.m)
        self.assertNotIn(unk, ids)

    def test_binary_only_zero_and_one(self):
        """All tokens must map back to '0' or '1'."""
        tok = create_default_tokenizer(self.n, token_type="binary")
        ids = tok.tokenize_measurement(self.m)
        chars = {tok.reverse_vocab[tid] for tid in ids}
        self.assertEqual(chars, {"0", "1"})

    def test_binary_encoding_spot_check(self):
        """
        basis=0 (00), outcome=0 (0) → bits '000'
        basis=1 (01), outcome=1 (1) → bits '011'
        """
        tok = create_default_tokenizer(self.n, token_type="binary")
        ids = tok.tokenize_measurement(self.m)
        v0, v1 = tok.vocab["0"], tok.vocab["1"]
        # qubit 0: basis=0 → 00, outcome=0 → 0 → '000'
        self.assertEqual(ids[0:3], [v0, v0, v0])
        # qubit 1: basis=1 → 01, outcome=1 → 1 → '011'
        self.assertEqual(ids[3:6], [v0, v1, v1])

    def test_binary_vocab_size(self):
        # 4 special + 2 content = 6
        tok = create_default_tokenizer(self.n, token_type="binary")
        self.assertEqual(tok.get_vocab_size(), 6)


# ---------------------------------------------------------------------------
# B. Collector tokenization
# ---------------------------------------------------------------------------

class TestCollectorTokenization(unittest.TestCase):

    def setUp(self):
        _skip_if_unavailable(self)
        self.n = 3
        self.measurements = [
            _make_measurement([0, 1, 2], [0, 0, 1]),
            _make_measurement([2, 2, 0], [1, 0, 0]),
            _make_measurement([1, 0, 1], [1, 1, 0]),
        ]

    def test_result_count_matches_measurements(self):
        tok = create_default_tokenizer(self.n, token_type="basis_outcome")
        collector = _make_collector_with_measurements(self.measurements)
        result = tok.tokenize_collector(collector)
        self.assertEqual(len(result), len(self.measurements))

    def test_each_sequence_is_list_of_ints(self):
        tok = create_default_tokenizer(self.n)
        collector = _make_collector_with_measurements(self.measurements)
        for seq in tok.tokenize_collector(collector):
            self.assertIsInstance(seq, list)
            for tid in seq:
                self.assertIsInstance(tid, int)

    def test_each_sequence_length(self):
        tok = create_default_tokenizer(self.n, token_type="basis_outcome")
        collector = _make_collector_with_measurements(self.measurements)
        for seq in tok.tokenize_collector(collector):
            self.assertEqual(len(seq), self.n)

    def test_all_modes_with_collector(self):
        for mode in ("basis_outcome", "pauli_string", "binary"):
            with self.subTest(mode=mode):
                tok = create_default_tokenizer(self.n, token_type=mode)
                collector = _make_collector_with_measurements(self.measurements)
                result = tok.tokenize_collector(collector)
                self.assertEqual(len(result), 3)


# ---------------------------------------------------------------------------
# C. Sequence creation (BOS/EOS, padding, truncation)
# ---------------------------------------------------------------------------

class TestCreateSequences(unittest.TestCase):

    def setUp(self):
        _skip_if_unavailable(self)
        self.n = 3
        self.tok = create_default_tokenizer(self.n, token_type="basis_outcome")
        self.m = _make_measurement([0, 1, 2], [0, 1, 0])

    def test_bos_eos_added(self):
        raw = self.tok.tokenize_measurement(self.m)
        seqs = self.tok.create_sequences([raw], add_special_tokens=True)
        seq = seqs[0]
        self.assertEqual(seq[0], self.tok.special_tokens["BOS"])
        # EOS may be at end before padding, or exactly at index n+1 if no padding
        self.assertIn(self.tok.special_tokens["EOS"], seq)

    def test_no_special_tokens_when_disabled(self):
        raw = self.tok.tokenize_measurement(self.m)
        seqs = self.tok.create_sequences([raw], add_special_tokens=False)
        seq = seqs[0]
        self.assertNotEqual(seq[0], self.tok.special_tokens["BOS"])

    def test_right_padding_fills_to_max_length(self):
        raw = self.tok.tokenize_measurement(self.m)
        seqs = self.tok.create_sequences([raw])
        seq = seqs[0]
        self.assertEqual(len(seq), self.tok.config.max_sequence_length)

    def test_padding_token_is_pad(self):
        raw = self.tok.tokenize_measurement(self.m)
        seqs = self.tok.create_sequences([raw])
        seq = seqs[0]
        pad = self.tok.special_tokens["PAD"]
        # Sequence shorter than max_length → trailing PADs
        for tid in seq[self.n + 2:]:  # after content + BOS + EOS
            self.assertEqual(tid, pad)

    def test_left_padding(self):
        config = TokenizationConfig(
            n_qubits=self.n,
            token_type="basis_outcome",
            max_sequence_length=10,
            padding_strategy="left",
        )
        tok = ShadowTokenizer(config)
        raw = tok.tokenize_measurement(self.m)
        seqs = tok.create_sequences([raw])
        self.assertEqual(seqs[0][0], tok.special_tokens["PAD"])

    def test_right_truncation(self):
        config = TokenizationConfig(
            n_qubits=self.n,
            token_type="basis_outcome",
            max_sequence_length=3,   # shorter than BOS + n_qubits + EOS
            padding_strategy="none",
            truncation_strategy="right",
        )
        tok = ShadowTokenizer(config)
        raw = tok.tokenize_measurement(self.m)
        seqs = tok.create_sequences([raw])
        self.assertEqual(len(seqs[0]), 3)

    def test_left_truncation_keeps_tail(self):
        config = TokenizationConfig(
            n_qubits=self.n,
            token_type="basis_outcome",
            max_sequence_length=3,
            padding_strategy="none",
            truncation_strategy="left",
        )
        tok = ShadowTokenizer(config)
        raw = tok.tokenize_measurement(self.m)
        full = [tok.special_tokens["BOS"]] + raw + [tok.special_tokens["EOS"]]
        seqs = tok.create_sequences([raw])
        self.assertEqual(seqs[0], full[-3:])

    def test_no_truncation_preserves_full_length(self):
        config = TokenizationConfig(
            n_qubits=self.n,
            token_type="basis_outcome",
            max_sequence_length=2,   # smaller than actual
            padding_strategy="none",
            truncation_strategy="none",
        )
        tok = ShadowTokenizer(config)
        raw = tok.tokenize_measurement(self.m)
        seqs = tok.create_sequences([raw])
        # BOS + 3 content + EOS = 5 tokens, no truncation → length stays 5
        self.assertEqual(len(seqs[0]), self.n + 2)

    def test_handle_sequence_length_alias(self):
        """_handle_sequence_length must exist and behave identically to
        _apply_length_constraints (backward-compatibility alias)."""
        raw = self.tok.tokenize_measurement(self.m)
        result_new = self.tok._apply_length_constraints(list(raw))
        result_old = self.tok._handle_sequence_length(list(raw))
        self.assertEqual(result_new, result_old)


# ---------------------------------------------------------------------------
# D. Validation / error handling
# ---------------------------------------------------------------------------

class TestValidation(unittest.TestCase):

    def setUp(self):
        _skip_if_unavailable(self)
        self.n = 3
        self.tok = create_default_tokenizer(self.n, token_type="basis_outcome")

    def test_mismatched_basis_outcome_lengths(self):
        m = ShadowMeasurement(
            basis=np.array([0, 1, 2]),
            outcome=np.array([0, 1]),
        )
        with self.assertRaises(ValueError):
            self.tok.tokenize_measurement(m)

    def test_invalid_basis_value_basis_outcome(self):
        m = _make_measurement([0, 1, 5], [0, 1, 0])  # 5 is invalid
        with self.assertRaises(ValueError):
            self.tok.tokenize_measurement(m)

    def test_invalid_outcome_value(self):
        m = _make_measurement([0, 1, 2], [0, 1, 2])  # outcome 2 is invalid
        with self.assertRaises(ValueError):
            self.tok.tokenize_measurement(m)

    def test_wrong_n_qubits_length(self):
        # tokenizer configured for n=3, measurement has 5 qubits
        m = _make_measurement([0, 1, 2, 0, 1], [0, 1, 0, 1, 0])
        with self.assertRaises(ValueError) as ctx:
            self.tok.tokenize_measurement(m)
        self.assertIn("n_qubits", str(ctx.exception))

    def test_empty_collector_raises(self):
        config = create_default_config(n_qubits=self.n)
        collector = ShadowCollector(config)
        # measurements list is empty by default
        with self.assertRaises(ValueError):
            self.tok.tokenize_collector(collector)

    def test_invalid_token_type_raises(self):
        config = TokenizationConfig(token_type="nonsense")
        with self.assertRaises(ValueError):
            ShadowTokenizer(config)

    def test_create_default_tokenizer_invalid_type(self):
        with self.assertRaises(ValueError):
            create_default_tokenizer(self.n, token_type="nonsense")

    def test_binary_mode_rejects_clifford_basis(self):
        """Clifford basis values (0-23) must be rejected in binary mode."""
        tok = create_default_tokenizer(self.n, token_type="binary")
        m = _make_measurement([0, 5, 2], [0, 0, 1])  # 5 is a Clifford index
        with self.assertRaises(ValueError):
            tok.tokenize_measurement(m)

    def test_pauli_string_mode_rejects_clifford_basis(self):
        tok = create_default_tokenizer(self.n, token_type="pauli_string")
        m = _make_measurement([0, 23, 2], [0, 0, 1])  # 23 is a Clifford index
        with self.assertRaises(ValueError):
            tok.tokenize_measurement(m)

    def test_clifford_collector_raises_clearly(self):
        """
        Known limitation: clifford-mode data (basis values 0-23) is not
        supported by the tokenizer.  Verify the error is a clear ValueError,
        not a silent UNK or crash.
        """
        from shadows.config import create_clifford_config
        config = create_clifford_config(n_qubits=self.n, n_shadows=2)
        collector = ShadowCollector(config)
        # Build a fake clifford measurement with a non-Pauli basis value
        collector.measurements = [
            ShadowMeasurement(
                basis=np.array([3, 7, 15]),   # Clifford indices 0-23
                outcome=np.array([0, 1, 0]),
            )
        ]
        tok = create_default_tokenizer(self.n, token_type="basis_outcome")
        with self.assertRaises(ValueError):
            tok.tokenize_collector(collector)

    def test_2d_basis_raises(self):
        m = ShadowMeasurement(
            basis=np.array([[0, 1], [2, 0]]),
            outcome=np.array([[0, 1], [1, 0]]),
        )
        with self.assertRaises(ValueError):
            self.tok.tokenize_measurement(m)


# ---------------------------------------------------------------------------
# E. Save / load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad(unittest.TestCase):

    def setUp(self):
        _skip_if_unavailable(self)
        self.n = 4
        self.m = _make_measurement([0, 1, 2, 0], [1, 0, 1, 0])

    def _round_trip(self, token_type):
        tok_orig = create_default_tokenizer(self.n, token_type=token_type)
        ids_before = tok_orig.tokenize_measurement(self.m)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        tok_orig.save_tokenizer(path)

        tok_loaded = create_default_tokenizer(self.n, token_type=token_type)
        tok_loaded.load_tokenizer(path)

        return tok_orig, tok_loaded, ids_before

    def test_basis_outcome_round_trip(self):
        orig, loaded, ids_before = self._round_trip("basis_outcome")
        self.assertEqual(loaded.tokenize_measurement(self.m), ids_before)

    def test_pauli_string_round_trip(self):
        orig, loaded, ids_before = self._round_trip("pauli_string")
        self.assertEqual(loaded.tokenize_measurement(self.m), ids_before)

    def test_binary_round_trip(self):
        orig, loaded, ids_before = self._round_trip("binary")
        self.assertEqual(loaded.tokenize_measurement(self.m), ids_before)

    def test_vocab_preserved(self):
        orig, loaded, _ = self._round_trip("basis_outcome")
        self.assertEqual(orig.vocab, loaded.vocab)

    def test_reverse_vocab_preserved(self):
        orig, loaded, _ = self._round_trip("basis_outcome")
        self.assertEqual(orig.reverse_vocab, loaded.reverse_vocab)

    def test_special_tokens_preserved(self):
        orig, loaded, _ = self._round_trip("basis_outcome")
        self.assertEqual(orig.special_tokens, loaded.special_tokens)

    def test_config_token_type_preserved(self):
        orig, loaded, _ = self._round_trip("pauli_string")
        self.assertEqual(loaded.config.token_type, "pauli_string")

    def test_config_n_qubits_preserved(self):
        orig, loaded, _ = self._round_trip("basis_outcome")
        self.assertEqual(loaded.config.n_qubits, self.n)

    def test_create_sequences_identical_after_load(self):
        orig, loaded, ids_before = self._round_trip("basis_outcome")
        seqs_orig = orig.create_sequences([ids_before])
        seqs_loaded = loaded.create_sequences([ids_before])
        self.assertEqual(seqs_orig, seqs_loaded)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
