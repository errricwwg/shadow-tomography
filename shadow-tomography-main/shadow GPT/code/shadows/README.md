# ShadowGPT Natural-Language Hamiltonian Analysis Demo

This project includes a command-line demo that lets users describe a quantum spin-chain Hamiltonian in natural language, automatically parse the Hamiltonian family and parameters, run a physics backend, and return a human-readable explanation of the resulting physical properties.

The goal is to turn the ShadowGPT pipeline into an interactive Hamiltonian analysis tool.

## One-sentence project summary

This project builds a ShadowGPT-based generative classical-shadow pipeline and wraps it with a natural-language interface, allowing users to describe quantum spin-chain Hamiltonians in plain English and receive exact or model-generated estimates of physical properties such as energy, magnetization, correlations, and Rényi-2 entropy.

---

## What the demo does

The demo supports the following end-to-end pipeline:

```text
natural-language Hamiltonian
→ structured Hamiltonian parser
→ exact or ShadowGPT backend
→ physical property estimation
→ natural-language explanation
```

For example, a user can input:

```text
4-qubit TFIM with J=1 and h=0.8
```

The system parses it as:

```text
family: tfim
n_qubits: 4
params: J=1, h=0.8
boundary: obc
```

Then it computes physical observables such as:

- ground-state energy
- average Z magnetization
- nearest-neighbor ZZ correlations
- Rényi-2 entropy

Finally, it explains the result in readable language.

---

## Supported Hamiltonian families

The natural-language interface currently supports four Hamiltonian families:

| Family | Example input | Parameters |
|---|---|---|
| `tfim` | `4-qubit TFIM with J=1 and h=0.8` | `J`, `h` |
| `ising_general` | `4-qubit general Ising model with J=1, hx=0.7, hz=0.3` | `J`, `hx`, `hz` |
| `xxz` | `4-qubit XXZ with J=1 and delta=1.5` | `J`, `delta` |
| `heisenberg` | `4-qubit Heisenberg with J=1` | `J` |

At the moment, the parser assumes open boundary conditions by default. Periodic boundary conditions are detected but not currently supported by the inference pipeline.

---

## Running the CLI demo

From the project directory, run:

```bash
python demo_cli.py --text "4-qubit TFIM with J=1 and h=0.8" --backend exact
```

To show detailed backend notes:

```bash
python demo_cli.py --text "4-qubit TFIM with J=1 and h=0.8" --backend exact --show-notes
```

To run XXZ:

```bash
python demo_cli.py --text "4-qubit XXZ with J=1 and delta=1.5" --backend exact --show-notes
```

To run Heisenberg:

```bash
python demo_cli.py --text "4-qubit Heisenberg with J=1" --backend exact --show-notes
```

---

## Example output structure

The CLI prints four main sections:

```text
=== Parsed Hamiltonian ===
family: tfim
n_qubits: 4
boundary: obc
params: J=1, h=0.8
supported: True
confidence: 0.95

=== Physics Result ===
energy: ...
magnetization: ...
correlations: ...
renyi2_entropy: ...

=== Backend Notes ===
...

=== Explanation ===
Short summary:
...

Detailed summary:
...

Warnings:
...
```

This makes the demo transparent: the user can see how the input was parsed, what the backend computed, and how the result was explained.

---

## Exact backend

The exact backend uses dense exact diagonalization. It constructs the Hamiltonian through the existing `build_hamiltonian_spec(...)` interface and computes the exact ground state.

The exact backend reports:

- total ground-state energy
- average Z magnetization
- average nearest-neighbor ZZ correlation
- half-chain Rényi-2 entropy

This backend is best for small systems such as 4–8 qubits.

---

## ShadowGPT backend

The learned backend uses the trained ShadowGPT generative model.

Instead of directly predicting physical properties, it follows the original ShadowGPT pipeline:

```text
Hamiltonian condition + measurement basis
→ generate synthetic classical shadows
→ estimate physical observables from generated shadows
```

A ShadowGPT run can be launched with:

```bash
python demo_cli.py \
  --text "4-qubit TFIM with J=1 and h=0.8" \
  --backend shadowgpt \
  --checkpoint-dir /path/to/checkpoints \
  --n-shadows 300 \
  --seed 42
```

The expected checkpoint layout is:

```text
/path/to/checkpoints/
  tfim/
    best_gpt.pt
    tokenizer_gpt.json
  ising_general/
    best_gpt.pt
    tokenizer_gpt.json
  xxz/
    best_gpt.pt
    tokenizer_gpt.json
  heisenberg/
    best_gpt.pt
    tokenizer_gpt.json
```

If no checkpoint is found, the CLI fails gracefully and prints a clear error message.

---

## Optional LLM-assisted interface

The project also includes an optional LLM-assisted layer.

The LLM is only allowed to help with:

1. parsing freer-form Hamiltonian descriptions into the structured schema
2. rewriting the final explanation into more natural language

The LLM does **not** compute physics directly. All physical quantities still come from the exact backend or the ShadowGPT backend.

To enable LLM-assisted parsing:

```bash
python demo_cli.py \
  --text "a four-site transverse Ising chain with J=1 and h=0.6" \
  --backend exact \
  --use-llm-parse
```

To enable LLM-assisted explanation rewriting:

```bash
python demo_cli.py \
  --text "4-qubit TFIM with J=1 and h=0.8" \
  --backend exact \
  --use-llm-rewrite
```

If the LLM package or API key is not available, the system falls back to the deterministic rule-based parser and template-based report generator.

---

## Interactive mode

The CLI also supports interactive mode:

```bash
python demo_cli.py --backend exact --interactive
```

Then enter Hamiltonian descriptions one at a time:

```text
> 4-qubit TFIM with J=1 and h=0.8
> 4-qubit XXZ with J=1 and delta=1.5
> quit
```

---

## Current validation status

The exact CLI demo has been tested on:

| Input | Status |
|---|---|
| `4-qubit TFIM with J=1 and h=0.8` | pass |
| `4-qubit XXZ with J=1 and delta=1.5` | pass |
| `4-qubit Heisenberg with J=1` | pass |
| `unknown weird model with alpha=2` | graceful failure |

The failure-path test confirms that unsupported inputs do not crash the program. Instead, the CLI prints a clear error message and exits with a nonzero status.

---

## Known limitations

Current limitations:

- Only four Hamiltonian families are supported.
- Only open boundary conditions are currently supported.
- Exact diagonalization is practical only for small systems.
- The ShadowGPT backend requires trained family-specific checkpoints.
- The LLM layer is optional and does not replace the deterministic parser or physics backend.
- The natural-language parser is currently strongest on semi-structured descriptions with explicit parameter names.

---

## Summary

This demo turns the project into a working natural-language Hamiltonian analysis pipeline.

It supports:

```text
human-language Hamiltonian input
→ automatic family and parameter parsing
→ exact or ShadowGPT-based physical analysis
→ readable natural-language explanation
```

This makes the project usable not only as a research pipeline, but also as an interactive prototype for explaining quantum spin-chain Hamiltonians.
