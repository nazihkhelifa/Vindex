# Vindex — Python tools for LarQL Vindexes (Gemma 3)

This repository is a **Python companion** to a **LarQL Vindex**: the on-disk layout produced by the LarQL / `larql-vindex` pipeline (FFN blocks, embeddings, optional full attention weights, norms, etc.). LarQL (Rust) remains the reference for **extracting** a dense model into that binary format. Here we focus on what you can do **in Python today** on top of an existing Vindex folder—especially for **Gemma 3**-style trees that already contain both **FFN** and **attention** tensors.

## What this work covers

### FFN and full forward inference (LarQL Vindex)

**`vindex_infer_python.py`** implements a faithful **numpy** forward over the mmap’d binaries that LarQL writes at inference-capable extract levels: **self-attention** (Q/K/V/O, RoPE, GQA where applicable) and the **SwiGLU FFN** (`gate` / `up` / `down` weights), then **final norm** and **next-token logits** via the **tied LM head** (`embeddings.bin`). So this repo **can run inference** on a LarQL Vindex for the **FFN path and the rest of the transformer block**, not a separate ad-hoc graph walk—provided your `index.json` and weight blobs match what the loader expects (see script docstrings for shapes and dtypes, typically **float16** on disk).

**`vindex_infer_ffn_att.py`** uses the **same engine** and adds **`--attn-meta`**: after the forward, it can print a compact view of **Option C** semantic labels read from **`attn_meta.bin`** (see below), layer by layer and head by head.

### Attention Vindex: Python over mmap’d attention weights

For the **attention** side of the Vindex, this repo does **not** replace LarQL’s Rust extractor. Instead, **`build_attn_semantic_meta.py`** assumes **`attn_weights.bin`** is already present (memory-mapped **float16**, layout aligned with LarQL’s attention pack: Q, K, V, O, plus Q/K norms when applicable). It **reads** those mmap slices, projects through the embedding table, and writes:

- **`attn_meta.bin`** — top‑*k* token ids per layer and query head (uint32),
- optionally **`attn_meta_scores.bin`**,
- and merges **`attention_metadata`** into **`index.json`**.

That is the Python path for **deriving extra attention-side metadata** from the **mmap’d attention Vindex** files, without loading the original Hugging Face PyTorch model.

### Gemma 3 in practice

The inference scripts are exercised against **Gemma 3**-class Vindexes that include **both** FFN and attention weights (e.g. the public snapshot below). Use **`--forward full`** (default) for attention + FFN; **`--forward ffn-only`** for an ablation without the attention sublayer.

## Repository layout

| File | Purpose |
|------|---------|
| `vindex_infer_python.py` | Full **attention + FFN** LarQL Vindex inference in Python (mmap **f16**), tied LM head on embeddings. |
| `vindex_infer_ffn_att.py` | Same as above + optional **`--attn-meta`** summary from `attn_meta.bin`. |
| `build_attn_semantic_meta.py` | Builds **`attn_meta.bin`** from **`attn_weights.bin`** + **`embeddings.bin`** (PyTorch for matmuls). |
| `download_vindex_from_hf.py` | Downloads a **ready-made Vindex** from the Hugging Face Hub if you do not have it locally. |

## Getting a Vindex (no local copy)

Example: **~7.8 GB** Gemma 3 4B IT Vindex published as a Hub repo. After `pip install -r requirements.txt`:

```bash
python download_vindex_from_hf.py
```

Defaults: `--repo-id cronos3k/gemma-3-4b-it-vindex`, `--local-dir gemma3-4b.vindex`. Override as needed:

```bash
python download_vindex_from_hf.py --repo-id cronos3k/gemma-3-4b-it-vindex --local-dir ./my.vindex
```

For gated or private repos, log in with the Hugging Face CLI or set **`HF_TOKEN`**, or pass **`--token`**.

Equivalent one-off in Python:

```python
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id = "cronos3k/gemma-3-4b-it-vindex"
local_dir = "gemma3-4b.vindex"

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    resume_download=True,
)

print(f"Files are in: {Path(local_dir).resolve()}")
```

(`huggingface_hub` may ignore deprecated kwargs such as `local_dir_use_symlinks`; the CLI script sticks to the supported surface.)

## Prerequisites

- **Python 3.10+** recommended.
- A **LarQL-compatible Vindex** directory (from LarQL extract or from a Hub snapshot like the one above), with the weight files required for the code path you use (e.g. **`attn_weights.bin`** for full attention forward and for **`build_attn_semantic_meta.py`**).

## Installation

```bash
pip install -r requirements.txt
```

**Lighter install** (inference only, no `attn_meta` build, no Hub download): `numpy` and `tokenizers` are the minimum for `--prompt` when `tokenizer.json` is inside the Vindex. **`build_attn_semantic_meta.py`** needs **PyTorch**; **`download_vindex_from_hf.py`** needs **`huggingface_hub`**.

## Usage examples

Inference:

```bash
python vindex_infer_python.py --vindex ./gemma3-4b.vindex --token-ids 818,5279,529,7001,563
python vindex_infer_python.py --vindex ./gemma3-4b.vindex -p "The capital of France is"
```

Inference + **attn_meta** summary after the forward:

```bash
python vindex_infer_ffn_att.py --vindex ./gemma3-4b.vindex -p "Hello" --attn-meta --attn-meta-layer 20
```

Build attention semantic metadata (“Option C”):

```bash
python build_attn_semantic_meta.py --vindex ./gemma3-4b.vindex --top-k 20
python build_attn_semantic_meta.py --vindex ./gemma3-4b.vindex --cpu
```

Logging: **INFO** on stderr by default; `-v` = DEBUG, `-q` = WARNING.

## Windows / PowerShell

Commands above work as written. For environment variables, use **`$env:NAME="value"`** instead of Unix **`NAME=value command`**.

## License

Apache-2.0 — consistent with the LarQL / Vindex ecosystem where applicable.
