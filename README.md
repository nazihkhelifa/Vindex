# Vindex — Python inference

Small set of **Python tools** around an existing **Vindex** directory (for example produced with **LarQL** / `larql extract`). There is **no** Hugging Face extractor here: you are expected to have a folder with at least `index.json`, the expected binaries (`embeddings.bin`, `gate_vectors.bin`, etc.), and for a full forward pass the attention / FFN weights according to the extract level.

## Repository layout

| File | Purpose |
|------|---------|
| `vindex_infer_python.py` | Python port of Rust-style `vindex-infer`: loads the Vindex in **f16** (mmap), **attention + FFN** forward or `--forward ffn-only` ablation, logits via the **tied LM head** on `embeddings.bin`. |
| `vindex_infer_ffn_att.py` | Same engine as above, plus **`--attn-meta`** to print a short summary of **Option C** semantic labels (`attn_meta.bin`). |
| `build_attn_semantic_meta.py` | Builds `attn_meta.bin` (and optionally `attn_meta_scores.bin`) from `attn_weights.bin` + `embeddings.bin`, and updates `index.json` (`attention_metadata`). Requires **PyTorch** (optional GPU acceleration). |

## Prerequisites

- **Python 3.10+** recommended.
- A valid **Vindex** directory (extract level must include attention weights if you want the full path without skipping attention).

## Installation

```bash
pip install -r requirements.txt
```

**Minimal profile** (without building `attn_meta`): `numpy` + `tokenizers` are enough to run inference with `--prompt` if `tokenizer.json` sits inside the Vindex. `build_attn_semantic_meta.py` additionally needs **PyTorch** (see `requirements.txt`).

## Usage examples

Inference (explicit token IDs or text prompt):

```bash
python vindex_infer_python.py --vindex ./path/to/model.vindex --token-ids 818,5279,529,7001,563
python vindex_infer_python.py --vindex ./path/to/model.vindex -p "The capital of France is"
```

Same with an **attn_meta** summary after the forward:

```bash
python vindex_infer_ffn_att.py --vindex ./path/to/model.vindex -p "Hello" --attn-meta --attn-meta-layer 20
```

Build attention metadata (“Option C”):

```bash
python build_attn_semantic_meta.py --vindex ./path/to/model.vindex --top-k 20
python build_attn_semantic_meta.py --vindex ./path/to/model.vindex --cpu
```

Logging: **INFO** on stderr by default; `-v` = DEBUG, `-q` = WARNING.

## Windows / PowerShell

The commands above work as-is. For environment variables, use PowerShell syntax (`$env:NAME="value"`) instead of Unix `NAME=value command`.

## License

Apache-2.0 — consistent with the LarQL / Vindex ecosystem where applicable.
