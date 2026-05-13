# Vindex (Python Toolkit)

> The model **IS** the database. A pure-Python toolkit that decompiles a
> Hugging Face transformer into a queryable **vindex** — and then runs
> inference **straight from the vindex files**, reproducing the model's
> forward pass with **vAttention + vFFN** instead of the original
> Hugging Face model.

## Project goal

**One sentence:** at inference time, never re-open the original model.
Read a user prompt, tokenise it, and produce the next-token distribution
**only from vindex bytes**, via two re-implemented pieces:

| Original model component | Vindex replacement (what runs at inference) | Files read |
|---|---|---|
| `self_attn.*` — multi-head causal attention over the prompt | **vAttention** — numpy RMSNorm + RoPE + GQA softmax + `o_proj`, layer by layer over the full token sequence (no HF, no PyTorch model) | `vindex_attn_ctx_*.{json,bin}` + `attn_k_proj_weights.bin` |
| `mlp.gate_proj` / `up_proj` / `down_proj` — FFN forward | **vFFN walk** — KNN over `gate_vectors` rows + weighted vote from `down_meta` top-K tokens (the LARQL FFN-as-graph idea) | `gate_vectors.bin` + `down_meta.bin` |
| `embed_tokens` + `lm_head` (approximated) | `embeddings.bin` lookup for input; the `down_meta` top-K stands in for `lm_head` at the vote stage | `embeddings.bin` + `down_meta.bin` |

`vindex-infer.py` / `vindex-infernce-next.py` / `avindex-infer.py` /
`avindex-infer-next.py` are wired to do exactly that and **import
neither `transformers` nor a HF `AutoModel`**. The only two scripts
that ever touch the original model are the **extract** scripts
(`vindex.py` and `avindex_attention.py extract`), which decompile its
weights into the vindex once.

This is the same trick LARQL uses on the **FFN** ("the model *is* the
database"), now extended to the **attention** layer.

### Inference pipeline (vindex-only)

```
                     ┌──────────────────────────────────────────────┐
                     │   USER PROMPT  →  tokenizer.json  →  ids[]   │
                     └──────────────────────────────────────────────┘
                                          │
                ids                       ▼
                ┌──────────────  embeddings.bin  ──────────────┐
                │ h₀ = embed[ids] (× embed_scale if Gemma)     │
                └──────────────────────────────────────────────┘
                                          │
                                          ▼
   ┌──────────────────────  vAttention (vindex_attn_context.py)  ──────────────────────┐
   │  for layer L in 0..N-1:                                                            │
   │     x = RMSNorm(h, input_layernorm[L])                                             │
   │     q = x @ q_proj[L].T                                                            │
   │     k = x @ k_proj[L].T          ← read from attn_k_proj_weights.bin               │
   │     v = x @ v_proj[L].T                                                            │
   │     (q,k) = apply_rope(q,k, rope_theta)   (+ optional q_norm/k_norm on Qwen)       │
   │     attn = softmax( causal_mask( q @ kᵀ / √d ) ) @ v                               │
   │     h = h + attn @ o_proj[L].T                                                     │
   │  return h[-1]              ← residual at the last position, full prompt mixed in   │
   └────────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
   ┌─────────────────────────  vFFN walk (vindex-infer.py)  ────────────────────────────┐
   │  e = h[-1]                                                                         │
   │  votes = zeros(vocab)                                                              │
   │  for layer L in walk_layers:                                                       │
   │     scores = gate_vectors[L] @ e                ← gate_vectors.bin                 │
   │     for feature f in top-K(scores) (positive):                                     │
   │        for (tok_id, logit) in down_meta[L][f]:  ← down_meta.bin                    │
   │            votes[tok_id] += scores[f] * logit                                      │
   │  return softmax(votes)  →  top-k next tokens                                       │
   └────────────────────────────────────────────────────────────────────────────────────┘
```

Two boxes, two `.bin` files each, **zero PyTorch model load**. The
"original model" only lives on disk in the form of those files. If you
delete the Hugging Face cache after `vindex.py` + `avindex_attention.py
extract`, the inference scripts keep working.

---

This repo is the "notebook flavour" of the LARQL extraction pipeline: a
handful of Python scripts that take a Hugging Face model (Qwen, Llama,
Gemma, Mistral, …) and produce a browsable, labelled, indexable
`*.vindex/` directory. The project goes one step beyond LARQL on the
**browse** side by adding an experimental sidecar — the **A-Vindex** —
which applies the same idea to the **attention** layer, and the
**attention-context** files that make `vindex-infer` use that attention
end-to-end at runtime.

After extraction, **every inference script reads only the vindex files**:
embeddings, gate vectors, down-projection metadata, attention centroids,
the stored `k_proj` slices, and the Q/V/O / norm blob. The original
Hugging Face checkpoint is never re-loaded.

```
# FFN  (vindex.py)            A-Vindex / Attention (avindex_attention.py)
─────────────────────         ──────────────────────────────────────────
gate_proj  → KNN of edges     k_proj × embed → centroids per KV head
down_proj  → edge labels      k_proj slice stored alongside centroids
embed      → token lookup     (probe + scan are HF-free)
```

The goal: look inside a model, label its neurons (FFN features and
attention heads), query its knowledge, and — eventually — edit that
knowledge without fine-tuning. All of it fits in a free Colab for
small models (Qwen3 0.6B, etc.).

---

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# (optional) Native LARQL bindings for the fast inference path
# See crates/larql-python/README.md in the larql-main repo
```

```bash
# 1) Build a browse vindex from Hugging Face  (the ONLY HF-touching step)
python vindex.py --model Qwen/Qwen3-0.6B -o ./.larql_colab/vindex_out

# 2) Single INFER step (top-k next token) — vindex-only walk
python vindex-infer.py --vindex ./.larql_colab/vindex_out \
    -p "The capital of France is" --top-k 5

# 3) Multi-token chat / generation — vindex-only walk
python vindex-infernce-next.py --vindex ./.larql_colab/vindex_out --chat

# 4) (Bonus) Build the A-Vindex attention sidecar (one more HF-touching step)
python avindex_attention.py extract --vindex ./.larql_colab/vindex_out

# 5) Centroid probe + vindex walk top-k — fully vindex-only
python avindex-infer.py --vindex ./.larql_colab/vindex_out -p "Hello" --walk-topk 10
```

> **Windows / PowerShell users:** the Unix-style `VAR=value cmd` syntax used
> in some examples below does not work in PowerShell. Set environment
> variables first, then run the script:
> ```powershell
> $env:VINDEX_MODEL="Qwen/Qwen3-0.6B"
> $env:AVINDEX_EXTRACT="1"
> python avindex_attention.py
> ```
> Or use the explicit subcommand form: `python avindex_attention.py extract …`.

Everything also works **without `argv`** in Colab mode: just set the
relevant environment variables (`VINDEX_MODEL`, `VINDEX_DIR`, …) and
run the cell. See the **Environment variables** section below.

> **Two scripts touch Hugging Face, ever.** `vindex.py` downloads the
> source model to build the browse vindex. `avindex_attention.py extract`
> reads `k_proj` once to build the attention sidecar. Every other script
> — `vindex-infer.py`, `vindex-infernce-next.py`, `avindex-infer.py`,
> `avindex-infer-next.py` — only reads files inside the vindex directory.

---

## What is a vindex?

A **vindex** is a directory that holds a model's weights **reorganised
for queryability**. Instead of an opaque pile of tensors, you get
something that looks like a graph database:

```
vindex_out/
├── index.json                  # config, arch, layer bands, provenance
├── gate_vectors.bin            # W_gate rows stacked per layer  (≃ KNN index)
├── embeddings.bin              # W_embed matrix                 (≃ token lookup)
├── down_meta.bin               # top-K vocab per feature        (≃ edge labels)
├── tokenizer.json              # tokenizer of the source model
├── attn_centroids.bin          # (A-Vindex) MiniBatchKMeans centroids
├── attn_k_proj_weights.bin     # (A-Vindex) stored k_proj per layer × kv_head
├── attn_index.json             # (A-Vindex) offsets, shapes
├── vindex_attn_ctx_index.json  # (walk) attention weights layout for prompt mixing
└── vindex_attn_ctx_weights.bin # (walk) Q/V/O + input LN (+ Qwen q_norm/k_norm)
```

Each weight bridges the "neural net" world and the "edge database" world:

| Model tensor | Role in the vindex | File |
|---|---|---|
| `mlp.gate_proj.weight` (per layer) | Query vectors of a KNN index — each row is the "fingerprint" of an FFN feature | `gate_vectors.bin` |
| `embed_tokens.weight` | Token → residual vector lookup | `embeddings.bin` |
| `mlp.down_proj.weight @ embed.T` (per-feature logits) | Top-K tokens that fire when this feature activates → **label** | `down_meta.bin` |
| `self_attn.k_proj.weight` (per layer × KV head) | Stored verbatim for centroid probe + attention forward | `attn_k_proj_weights.bin` |
| KMeans clusters of `k_proj × embed` | Compact "where vocab embeddings land under K" summary | `attn_centroids.bin` |
| `self_attn.{q,v,o}_proj` + `input_layernorm` (+ optional `q_norm`/`k_norm`) | **Prompt-wide causal attention** (numpy): mix all prompt tokens, then FFN walk reads the last residual | `vindex_attn_ctx_*.bin/json` |

Rows 1–3 are produced by `vindex.py`. Rows 4–5 are produced by
`avindex_attention.py extract` (centroid pass). Rows 6–7 are written **at the
end of the same extract** (no second HF download) and power ``vindex-infer``'s
attention-context path before the gate KNN.

On the code side, `vindex.py` loads the safetensors, writes
`gate_vectors.bin` layer by layer, and projects every column of
`down_proj` through the embedding matrix to recover the `top_k` tokens
and their logits. `avindex_attention.py extract` adds the attention
sidecar files. After that the source model is no longer needed.

> **Browse level only** in this repo: the inference stack at
> `--level inference` (full down/up weights, layer norms, attention
> matrices) is the territory of native LARQL. This toolkit gives you
> enough of the model to (a) browse and label features, (b) run a
> coarse **walk INFER** in pure numpy, or (c) hand off to native
> `larql` for the precise forward pass when you have an
> `inference`-level vindex.

---

## How it works — the **FFN** side

In a modern transformer (Qwen, Llama, Gemma, Mistral, …), every FFN
block has the form:

```
y = down_proj( act(gate_proj(x)) * up_proj(x) )
```

where `act` is `silu` (SwiGLU: Llama, Qwen, Mistral, Phi) or `gelu`
(GeGLU: Gemma 2/3). The core LARQL insight — replicated here in Python
— is that **every row of `gate_proj`** behaves like a **key** in an
index: when input `x` is close (in the dot-product sense) to a given
row, the corresponding feature activates. The columns of `down_proj` are then
the **values**: what gets injected into the residual stream when that
feature fires.

The `vindex.py` pipeline industrialises this view:

1. **`gate_vectors.bin`** — we flatten all `gate_proj` rows across all
   layers into a single mmap-friendly file. At query time, this is
   exactly the array you'd multiply against `x` to run a per-layer
   KNN.
2. **`down_meta.bin`** — for every feature we compute
   `logits = embed @ down_proj[:, feature]` (shape `[vocab]`), then
   keep the top-K. Those tokens give a **human label** to the feature:
   this is what you read in a LARQL `DESCRIBE`.
3. **`embeddings.bin`** — the token table, used both for graph-node
   lookup and for the "unembedding" step in `down_meta`.

### The vindex walk (vindex-only INFER)

`vindex-infer.py` runs a pure-numpy **walk INFER** on ``gate_vectors.bin`` +
``down_meta.bin``. The vector ``e`` that gets dotted into the gates is either:

* **Attention context (default when `vindex_attn_ctx_*` exists)** — after
  ``avindex_attention.py extract``, the vindex also stores Q/V/O projections,
  input RMSNorm, optional Qwen ``q_norm``/``k_norm``, and reuses the stored
  ``k_proj`` slices. For each transformer layer we run **one causal
  self-attention block** over the full prompt (no FFN between layers), then
  take ``e = h[-1]`` (the last position residual after all attention layers).
  That is exactly *"what the model sees at the final position after mixing
  France + capital + of + is through attention"*, up to f32 numpy fidelity.
* **Last-token fallback** — if those files are missing or you set
  ``VINDEX_NO_ATTN_CTX=1`` / ``--no-attention-context``, ``e`` is just
  ``embed[last_token]`` (the old behaviour).

Then for each FFN walk layer ``L``:

```text
  scores = gate_vectors[L] @ e
  top_features = argpartition(scores)[:K]     # K = features_per_layer
  for f in top_features (positive scores only):
    for (token_id, logit) in down_meta[L][f].top_k:
      votes[token_id] += scores[f] × logit

softmax(votes)  →  top-k next tokens
```

This is still **not** a full forward pass (no FFN up/down between layers, no
final model norm, no lm_head logits) — but the attention half of your question
is now real. For exact next-token parity use native ``larql`` on an
``inference``-level vindex.

When the native `larql` bindings are installed AND the vindex was
built at `--level inference`, `vindex-infer.py` automatically delegates
to `larql.infer(...)` instead. That path also reads only vindex files,
just with the full inference machinery.

```python
# Example — using the runner directly
from pathlib import Path
import importlib.util, sys

spec = importlib.util.spec_from_file_location("vix_inf", "vindex-infer.py")
vif = importlib.util.module_from_spec(spec); sys.modules["vix_inf"] = vif
spec.loader.exec_module(vif)

runner = vif.VindexWalkRunner(Path("./.larql_colab/vindex_out"))
print(runner.infer_topk("The capital of France is", top_k=5))
print(runner.generate("Einstein is known for", max_new_tokens=20))
```

---

## How it works — the **Attention** side (A-Vindex)

LARQL traditionally indexes the FFN because that's where most factual
knowledge lives. The **A-Vindex** is an experimental extension that
asks the same question for **attention**:

> Where do my vocabulary embeddings land after passing through
> `k_proj`, layer by layer, head by head?

`avindex_attention.py` answers like this:

1. For every layer, load `self_attn.k_proj.weight` and split it into
   **`num_key_value_heads`** blocks (GQA-aware).
2. Sample `vocab_sample` rows (8,000 by default) from `embed_tokens`
   and project them through each KV head:
   `projected[token] = embedding[token] @ K_h.T`.
3. Cluster those projections with **MiniBatchKMeans** (256 centroids
   by default).
4. **Store both the centroids AND the `k_proj` slice** so the probe
   path is fully self-contained.

```
vindex_out/
├── …  (FFN / browse — see above)
├── attn_centroids.bin           # f32 centroids, concat layers × kv_heads
├── attn_k_proj_weights.bin      # f32 k_proj slices, same ordering as centroids
└── attn_index.json              # offsets, shapes, head_dim_kv, num centroids
```

At inference time, `avindex-infer.py` (1) tokenises the prompt with
the vindex `tokenizer.json`, (2) reads the last token's embedding from
`embeddings.bin`, (3) projects it through the stored `k_proj` slice,
(4) asks the `AvindexAttentionReader` which centroid is the closest in
**cosine similarity**. **No source model is loaded.**

```python
# Example — vindex-only centroid probe
from pathlib import Path
import importlib.util, sys

spec = importlib.util.spec_from_file_location("av", "avindex_attention.py")
av = importlib.util.module_from_spec(spec); sys.modules["av"] = av
spec.loader.exec_module(av)

out = av.probe_last_token_k_space(
    Path("./.larql_colab/vindex_out"),
    prompt="The capital of France is",
    layer=13,
    kv_head=0,
)
# {'centroid_id': 42, 'cosine': 0.71, 'last_token_piece': ' is', ...}
```

> ⚠️ **This is not a runtime attention replacement.** No RoPE, no
> softmax over positions, no Q. It is an **offline summary** of K-space
> useful for: (a) interpretability ("which heads look at which token
> groups"), (b) conditional routing, and (c) acting as a prelude to a
> low-rank KV cache. The actual prediction in `avindex-infer.py` and
> `avindex-infer-next.py` comes from the FFN walk on `gate_vectors.bin`
> + `down_meta.bin` — the centroid probe is shown alongside as a
> diagnostic.

---

## Script inventory

| Script (invocation) | Role | Touches HF? |
|---|---|---|
| `vindex.py` | Extracts a **browse vindex** (FFN + embed + down_meta) from HF | yes — extract only |
| `avindex_attention.py extract` | Builds the **A-Vindex sidecar** (centroids + k_proj) | yes — extract only |
| `avindex_attention.py probe` | Stand-alone single-layer **centroid probe** (vindex-only) | no |
| `avindex_attention.py scan` | Centroid hits for **every** (layer, kv_head) — diagnostic | no |
| `vindex-infer.py` | Single **INFER top-k** — native larql or pure-numpy walk | **no** |
| `vindex-infernce-next.py` | **Multi-token chat** (walk loop or larql greedy) | **no** |
| `avindex-infer.py` | **Centroid probe** + vindex walk top-k + optional scan | **no** |
| `avindex-infer-next.py` | Chat with optional per-turn centroid probe annotation | **no** |
| `vindex_attn_context.py` | NumPy attention forward + extract helper (imported by extract / infer) | **no** |

All of them follow the same Colab convention: the scripts detect an
empty argv (Jupyter cell), strip the parasitic `-f kernel.json`, and
fall back to an "environment variables" mode so you don't have to touch
the `argparse` surface.

---

## Environment variables (Colab / cell mode)

```bash
# ── Extraction (vindex.py) ───────────────────────────────────────────
VINDEX_MODEL=Qwen/Qwen3-0.6B      # HF id or local path
VINDEX_DOWN_TOP_K=10              # K of the top-K in down_meta.bin
VINDEX_ZIP=1                      # output .zip under .larql_colab/
VINDEX_SYNC_TO_DRIVE=1            # copy to /content/drive/MyDrive
VINDEX_DRIVE_DIR=/path/to/dest    # explicit destination

# ── One-shot INFER (vindex-infer.py) ─────────────────────────────────
VINDEX_DIR=./.larql_colab/vindex_out
VINDEX_PROMPT="The capital of France is"
VINDEX_TOP_K=5
VINDEX_WALK_FEATURES=32           # features kept per layer
VINDEX_WALK_LAYERS=knowledge      # band name or comma list, e.g. "12,13,14"
VINDEX_NO_ATTN_CTX=1              # skip attention forward; use last-token embed only

# ── Chat / multi-token (vindex-infernce-next.py) ─────────────────────
VNEXT_BUILD=1                     # extract before chat (HF download)
VNEXT_CHAT=1                      # REPL
VNEXT_PROMPT="Hello"              # one-shot user text
VNEXT_MAX_TOKENS=256
VNEXT_STREAM=1

# ── A-Vindex extract (avindex_attention.py) ──────────────────────────
AVINDEX_EXTRACT=1                 # build the attention sidecar (HF download)
AVINDEX_CENTROIDS=256
AVINDEX_VOCAB_SAMPLE=8000

# ── A-Vindex probe / inference (vindex-only, no HF) ──────────────────
AVINDEX_PROBE=1                   # run the centroid probe
AVINDEX_PROBE_LAYER=13            # else middle of knowledge band
AVINDEX_PROBE_KV_HEAD=0
AVINDEX_WALK_TOPK=10              # 0 to skip the FFN walk block
AVINDEX_SCAN=1                    # all (layer, kv_head) centroid hits
AVINDEX_SCAN_LAYERS=13            # comma-separated list

# ── A-Vindex chat (avindex-infer-next.py) ────────────────────────────
ANEXT_BUILD=1                     # extract before chat (HF download)
ANEXT_CHAT=1                      # REPL
ANEXT_PROBE=1                     # annotate each turn with centroid probe
ANEXT_PROMPT="Hello"
ANEXT_MAX_TOKENS=256
ANEXT_STREAM=1

# ── Colab plumbing ───────────────────────────────────────────────────
VINDEX_PY=/content/vindex.py      # if `import vindex` fails
AVINDEX_PY=/content/avindex_attention.py
VINDEX_INFER_PY=/content/vindex-infer.py
HF_TOKEN=hf_xxx                   # for gated models (extract only)
```

---

## Typical Colab cell

```python
# Cell 1 — install + drop the scripts in cwd
!pip install -q -r requirements.txt

# Cell 2 — extract the browse vindex (HF download, ONLY here)
import os
os.environ["VINDEX_MODEL"] = "Qwen/Qwen3-0.6B"
os.environ["VINDEX_ZIP"]   = "1"
%run vindex.py

# Cell 3 — vindex-only INFER top-5
os.environ["VINDEX_PROMPT"] = "Einstein is known for"
os.environ["VINDEX_TOP_K"]  = "5"
%run vindex-infer.py

# Cell 4 — A-Vindex sidecar (HF download, ONLY here)
os.environ["AVINDEX_EXTRACT"] = "1"
%run avindex_attention.py

# Cell 5 — vindex-only probe + walk top-k
os.environ["AVINDEX_WALK_TOPK"] = "10"
%run avindex-infer.py
```

After Cell 4 the source model is no longer needed; you can wipe the HF
cache and Cells 3 and 5 will keep working as long as the vindex
directory survives.

---

## Supported models

Any "dense" Hugging Face model using the standard
`model.layers.{i}.mlp.*` layout with `gate_proj` / `up_proj` /
`down_proj` and `embed_tokens.weight`:

- **Qwen** 2 / 2.5 / 3 (tested on Qwen3-0.6B)
- **Llama** 2 / 3 (7B and up, RAM permitting)
- **Mistral** 7B
- **Gemma** 2 / 3 (dense variants; Per-Layer Embeddings not handled here)
- **Phi** 2 / 3

**Not supported** by this repo (use the Rust `larql` instead):

- MoE (Mixtral 8×7B, Gemma 4 26B-A4B, DeepSeek V2/V3, GPT-OSS) — the
  script raises an explicit `SystemExit` if `num_experts > 0` in the
  config.
- GGUF / Q4_K quantisation — here we work in `float32` in memory and
  write raw `f32` to disk.
- GPT-2 (fused `attn_qkv` layout not handled).

---

## Known limitations

1. **RAM at extract** — `vindex.py` loads the entire safetensors into
   memory before casting to `f32`. For Qwen3-0.6B that's ~2 GB, for
   Llama-3-8B plan ~32 GB. After extract, the inference scripts are
   modest (mmap'd embeddings + gates, in-memory down_meta).
2. **Walk INFER is approximate, on purpose** — at inference time the
   two re-implemented blocks are **vAttention** (faithful: numpy
   RMSNorm + RoPE + GQA softmax + `o_proj`, full sequence, every
   layer) and **vFFN walk** (approximate: top-K gate KNN + `down_meta`
   token votes, not a real `up_proj`/`down_proj` matmul). What is *not*
   reproduced from the original model:
    - the **FFN is not interleaved per layer**. The current pipeline
      runs the *whole* attention stack first, then runs the FFN-walk
      *once* on the final residual. The original model would alternate
      `h += attn_L(h)` and `h += ffn_L(h)` per layer.
    - `post_attention_layernorm` and the final `model.norm` are skipped.
    - the `lm_head` is approximated by the `down_meta` top-K votes
      rather than a full `embed.T` projection.
   Native ``larql`` on an ``inference``-level vindex is the supported
   path when you need exact next-token parity.
3. **A-Vindex centroid probe ≠ runtime attention** — the centroid probe is a
   geometric summary of K-space (clustering), not the causal softmax path.
   The **attention-context** walk, by contrast, does run real RoPE + causal
   softmax attention using stored Q/K/V/O weights.
4. **Attention-context disk** — ``vindex_attn_ctx_weights.bin`` is ~10 MiB
   per layer on a 1k-hidden / 16-head model (~280 MiB for Qwen3-0.6B's 28
   layers). It is written automatically at the end of ``avindex_attention.py
   extract``. Older vindexes without these files keep the last-token fallback.
5. **No real chat template** — `tokenizers.Tokenizer.from_file` reads
   the BPE/Unigram graph from `tokenizer.json` but not the Jinja2 chat
   template embedded in `tokenizer_config.json`. The chat scripts use a
   generic `USER:/ASSISTANT:` framing. The standalone `tokenizers`
   library is enough for everything in this repo — `transformers` is
   never imported by the inference path.
6. **f32 on disk** — `gate_vectors.bin`, `embeddings.bin`,
   `attn_k_proj_weights.bin`, and `vindex_attn_ctx_weights.bin` are all
   `float32`. Native LARQL can write f16 (size ÷ 2); not here.

---

## Compatibility with LARQL

The files produced honour the LARQL binary format:

- `index.json` v2 with the expected fields (`extract_level: "browse"`,
  `dtype: "f32"`, `layers[]` with offsets/lengths, `layer_bands` in
  thirds `syntax` / `knowledge` / `output`).
- `down_meta.bin` in the **DMET v1** format (`magic = 0x444D4554`,
  header `[u32 magic, u32 version, u32 num_layers, u32 top_k]`, then
  per layer `u32 num_features` + records `[u32 top_id, f32 c_score,
  top_k × (u32 token_id, f32 logit)]`). The Rust decoder lives in
  [`crates/larql-vindex/src/format/down_meta.rs`](../larql-main/crates/larql-vindex/src/format/down_meta.rs).
- The A-Vindex files (`attn_centroids.bin`, `attn_k_proj_weights.bin`,
  `attn_index.json` with `avindex_version: 3`) and the attention-context
  files (`vindex_attn_ctx_*.json/bin`) are specific to this toolkit —
  LARQL doesn't read them today.

Practical consequence: a vindex produced here can be read by
`larql repl` or by the `larql._native` bindings *in browse mode*
(`DESCRIBE`, `WALK`, `SELECT`). Native `INFER` still requires the
`inference`-level files this repo does not write (`down_weights.bin`,
`up_weights.bin`, norms, etc.) — use the Rust `larql extract` for
those.

---

## Further reading

- [`../larql-main/README.md`](../larql-main/README.md) — the full
  LARQL ecosystem (Rust, LQL, HTTP/gRPC server, MoE, Metal GPU).
- [`../larql-main/docs/ffn-graph-layer.md`](../larql-main/docs/ffn-graph-layer.md)
  — why the FFN walk beats the dense path, and how the native `walk`
  reads exactly the files we produce here.
- [`../larql-main/docs/specs/vindex-format-spec.md`](../larql-main/docs/specs/vindex-format-spec.md)
  — the official vindex format spec (v0.3).
- [`../larql-main/docs/inference-engine.md`](../larql-main/docs/inference-engine.md)
  — inference engine: BLAS-fused attention, Metal GPU.

---

## License

Apache-2.0 — aligned with LARQL.
