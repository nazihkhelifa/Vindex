#!/usr/bin/env python3
"""
**Vindex-only INFER** — top-K next-token predictions read straight from the
vindex files. **No Hugging Face model is ever loaded.**

Two execution paths, both vindex-native:

1. **Native LARQL** (when ``import larql`` works AND the vindex was built with
   ``--level inference`` or ``all``): delegate to ``larql.load(...).infer(...)``
   for the accurate Walk-FFN forward pass.
2. **Pure-numpy walk** (always works, including browse-only vindex from
   ``vindex.py``): run **multi-head causal attention** over the full prompt
   (when ``vindex_attn_ctx_*.bin`` are present — produced by the latest
   ``avindex_attention.py extract`` or ``extract-ctx``), take the
   **last-position residual**, then for each FFN layer score
   ``gate_vectors @ residual`` and vote via ``down_meta``. If the sidecar is
   missing, falls back to a **causal-decay mean-pool** of the prompt
   embeddings (every prompt token contributes; still not real attention).

The tokenizer is read from ``<vindex>/tokenizer.json`` directly via the
``tokenizers`` package — no ``AutoTokenizer.from_pretrained`` call, no HF
download, no transformers import.

Environment (cell mode, no CLI):
  VINDEX_DIR             path to vindex directory (default: ./.larql_colab/vindex_out)
  VINDEX_PROMPT          input text (default: "The capital of France is")
  VINDEX_TOP_K           int (default: 5)
  VINDEX_WALK_LAYERS     comma list, e.g. "12,13,14" or band name "knowledge"
  VINDEX_WALK_FEATURES   features kept per layer (default: 32)
  VINDEX_NO_ATTN_CTX=1   force last-token embedding only (skip attention forward)

CLI:
  python vindex-infer.py --vindex ./.larql_colab/vindex_out -p "Hello" --top-k 5
  python vindex-infer.py --vindex ./.larql_colab/vindex_out -p "Hi" --layers knowledge
  python vindex-infer.py --vindex ./.larql_colab/vindex_out -p "Hi" --no-attention-context

Dependencies:
  pip install numpy tokenizers
  (Optional native fast path: `larql` wheel — see crates/larql-python README.)
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


def _log(msg: str) -> None:
    print(msg, flush=True)


def _strip_jupyter_kernel_argv(rest: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(rest):
        if rest[i] == "-f" and i + 1 < len(rest):
            i += 2
            continue
        out.append(rest[i])
        i += 1
    return out


def user_argv() -> list[str]:
    return _strip_jupyter_kernel_argv(sys.argv[1:])


def work_base() -> Path:
    p = Path.cwd().resolve() / ".larql_colab"
    p.mkdir(parents=True, exist_ok=True)
    return p


def default_vindex_dir() -> Path:
    return Path(os.environ.get("VINDEX_DIR", str(work_base() / "vindex_out"))).expanduser().resolve()


def load_index(vindex_dir: Path) -> dict[str, Any]:
    p = vindex_dir / "index.json"
    if not p.is_file():
        raise FileNotFoundError(f"Not a vindex directory (missing index.json): {vindex_dir}")
    return json.loads(p.read_text(encoding="utf-8"))


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


# ── DMET v1 parser (down_meta.bin) ────────────────────────────────────────

DMET_MAGIC = 0x444D4554  # "DMET"
DMET_VERSION = 1


def _parse_down_meta(buf: bytes) -> tuple[list[list[list[tuple[int, float]] | None]], int]:
    """Return ``(per_layer_features, top_k)``.

    ``per_layer_features[layer][feature]`` is either ``None`` (padding feature)
    or a list of ``(token_id, logit)`` pairs read from disk (length ≤ top_k).
    """
    pos = 0
    magic, version, num_layers, top_k = struct.unpack_from("<IIII", buf, pos)
    pos += 16
    if magic != DMET_MAGIC:
        raise ValueError(f"Bad DMET magic: 0x{magic:08x} (expected 0x{DMET_MAGIC:08x})")
    if version != DMET_VERSION:
        raise ValueError(f"Unsupported DMET version: {version} (expected {DMET_VERSION})")

    per_layer: list[list[list[tuple[int, float]] | None]] = []
    for _layer in range(num_layers):
        (num_feats,) = struct.unpack_from("<I", buf, pos)
        pos += 4
        feats: list[list[tuple[int, float]] | None] = []
        for _feat in range(num_feats):
            top_id, c_score = struct.unpack_from("<If", buf, pos)
            pos += 8
            topk_pairs: list[tuple[int, float]] = []
            for _ in range(top_k):
                tid, lg = struct.unpack_from("<If", buf, pos)
                pos += 8
                if not (tid == 0 and lg == 0.0):
                    topk_pairs.append((int(tid), float(lg)))
            if top_id == 0 and c_score == 0.0 and not topk_pairs:
                feats.append(None)
            else:
                feats.append(topk_pairs if topk_pairs else None)
        per_layer.append(feats)
    return per_layer, int(top_k)


# ── Pure-numpy walk runner ────────────────────────────────────────────────


class VindexWalkRunner:
    """
    Vindex-only INFER. Reads ``embeddings.bin``, ``gate_vectors.bin``,
    ``down_meta.bin`` and ``tokenizer.json``. Optionally loads
    ``vindex_attn_ctx_*.bin`` (from ``avindex_attention.py extract``) to run a
    **full causal self-attention stack** over the prompt and feed the **last
    position residual** into the FFN walk — mixing all prompt tokens through
    attention before the gate KNN.

    Algorithm (per prediction step):
      1. Tokenise the prompt; build ``h`` from token embeddings.
      2. If attention-context sidecar is loaded: for each transformer layer,
         apply pre-norm RMSNorm → Q/K/V projections (K read from stored
         ``k_proj``) → optional Qwen ``q_norm``/``k_norm`` → RoPE → causal
         softmax attention → ``o_proj``; ``h += attn_out`` (no FFN between
         layers). Take ``e = h[-1]``.
         Else: causal-decay mean-pool of all prompt embeddings (every token
         contributes, last token weighted most).
      3. For each FFN walk layer, score ``gate_vectors[layer] @ e``, keep top
         features, accumulate ``down_meta`` votes, softmax → top-k.
    """

    def __init__(self, vindex_dir: Path, *, use_attention_context: bool | None = None) -> None:
        self.vindex_dir = Path(vindex_dir).resolve()
        self.idx = load_index(self.vindex_dir)
        self.hidden = int(self.idx["hidden_size"])
        self.inter = int(self.idx["intermediate_size"])
        self.vocab = int(self.idx["vocab_size"])
        self.num_layers = int(self.idx["num_layers"])
        self.embed_scale = float(self.idx.get("embed_scale", 1.0))
        self.layer_bands: dict[str, list[int]] = dict(self.idx.get("layer_bands") or {})
        self.layer_meta = list(self.idx["layers"])

        # mmap embeddings: [vocab, hidden]
        embed_path = self.vindex_dir / "embeddings.bin"
        if not embed_path.is_file():
            raise FileNotFoundError(f"Missing {embed_path}")
        self.embed = np.memmap(
            embed_path, dtype="<f4", mode="r", shape=(self.vocab, self.hidden)
        )

        # mmap gate_vectors.bin as a flat f32 array; per-layer slicing is a view
        gate_path = self.vindex_dir / "gate_vectors.bin"
        if not gate_path.is_file():
            raise FileNotFoundError(f"Missing {gate_path}")
        gate_floats = gate_path.stat().st_size // 4
        self._gate_mm = np.memmap(gate_path, dtype="<f4", mode="r", shape=(gate_floats,))

        # full down_meta in memory (small: ~num_layers × inter × top_k × 8 bytes)
        dm_path = self.vindex_dir / "down_meta.bin"
        if not dm_path.is_file():
            raise FileNotFoundError(f"Missing {dm_path}")
        self.down_meta, self.down_top_k = _parse_down_meta(dm_path.read_bytes())

        # tokenizer from vindex (no HF model id required)
        self.tok = self._load_tokenizer()

        self._attn_ctx: Any | None = None
        ctx_idx = self.vindex_dir / "vindex_attn_ctx_index.json"
        ctx_w = self.vindex_dir / "vindex_attn_ctx_weights.bin"
        files_ok = ctx_idx.is_file() and ctx_w.is_file()
        forced_off = _env_truthy("VINDEX_NO_ATTN_CTX") or use_attention_context is False
        want = (use_attention_context is True) or (files_ok and not forced_off)

        if want:
            try:
                from vindex_attn_context import AttnContextForward

                self._attn_ctx = AttnContextForward(self.vindex_dir)
            except Exception as e:  # noqa: BLE001
                _log(f"[vindex-infer] attention-context unavailable ({e}); using prompt mean-pool fallback")
                self._attn_ctx = None
        elif not files_ok and not forced_off:
            _log(
                "[vindex-infer] WARNING: vindex_attn_ctx_* missing — full-prompt attention is NOT being run.\n"
                "                Falling back to prompt **mean-pool of token embeddings** (passes every prompt\n"
                "                token, but no real RoPE/softmax attention). Re-extract to enable real attention:\n"
                "                  python avindex_attention.py extract-ctx --vindex "
                f"{self.vindex_dir}\n"
                "                (or env-mode: AVINDEX_EXTRACT_CTX=1 VINDEX_DIR=... python avindex_attention.py)"
            )

    def _load_tokenizer(self) -> Any:
        p = self.vindex_dir / "tokenizer.json"
        if not p.is_file():
            raise FileNotFoundError(
                f"{p} missing — vindex.py copies tokenizer.json next to embeddings.bin."
            )
        try:
            from tokenizers import Tokenizer  # type: ignore
        except ImportError as e:
            raise SystemExit("Install the `tokenizers` package: pip install tokenizers") from e
        return Tokenizer.from_file(str(p))

    def embedding_for_ffn_walk(self, token_ids: list[int]) -> np.ndarray:
        """Last-position vector fed into the FFN gate KNN.

        * Best path — ``vindex_attn_ctx_*`` loaded: real causal-attention stack
          over the **entire prompt**, return ``h[-1]``.
        * Fallback path — sidecar missing: **causal-decay mean-pool** of the
          token embeddings (every prompt token contributes, with an
          exponentially decaying weight; bias toward the last token). This is
          NOT real attention but at least uses the whole prompt instead of
          ``embed[last_token]`` alone.
        """
        if self._attn_ctx is not None:
            return self._attn_ctx.last_residual_after_attention(
                self.embed, self.embed_scale, token_ids
            )
        ids = np.asarray([int(t) for t in token_ids], dtype=np.int64)
        if ids.size == 0:
            raise ValueError("empty token_ids")
        embs = np.asarray(self.embed[ids], dtype=np.float32) * float(self.embed_scale)
        n = int(embs.shape[0])
        # decay = 0.85: last token weight 1.0, previous 0.85, then 0.7225, ...
        decay = 0.85
        w = np.array([decay ** (n - 1 - i) for i in range(n)], dtype=np.float32)
        w /= float(w.sum())
        return (embs * w[:, None]).sum(axis=0)

    # ── public helpers ────────────────────────────────────────────────────

    def gates_layer(self, layer: int) -> np.ndarray:
        meta = self.layer_meta[layer]
        offset_floats = int(meta["offset"]) // 4
        n_feats = int(meta["num_features"])
        end = offset_floats + n_feats * self.hidden
        return self._gate_mm[offset_floats:end].reshape(n_feats, self.hidden)

    def encode(self, prompt: str) -> list[int]:
        ids = list(self.tok.encode(prompt).ids)
        if not ids:
            raise ValueError("empty tokenization")
        return ids

    def decode_one(self, token_id: int) -> str:
        try:
            return self.tok.decode([int(token_id)])
        except Exception:  # noqa: BLE001
            return f"<id {token_id}>"

    def decode(self, ids: Iterable[int]) -> str:
        try:
            return self.tok.decode(list(int(i) for i in ids), skip_special_tokens=True)
        except Exception:  # noqa: BLE001
            return "".join(self.decode_one(i) for i in ids)

    def resolve_layers(self, layers: list[int] | str | None) -> list[int]:
        if layers is None or layers == "" or (isinstance(layers, list) and not layers):
            return list(range(self.num_layers))
        if isinstance(layers, str):
            s = layers.strip()
            if s in self.layer_bands:
                lo, hi = self.layer_bands[s]
                return list(range(int(lo), int(hi) + 1))
            parts = [p for p in s.split(",") if p.strip()]
            return [int(p) for p in parts]
        return [int(L) for L in layers]

    # ── core step ─────────────────────────────────────────────────────────

    def _predict_step_ids(
        self,
        token_ids: list[int],
        *,
        top_k: int,
        features_per_layer: int,
        layers: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Internal: returns ``(top_token_ids, top_probs)`` as numpy arrays."""
        e = self.embedding_for_ffn_walk(token_ids)
        votes = np.zeros(self.vocab, dtype=np.float32)
        for layer in layers:
            gates = self.gates_layer(layer)
            scores = gates @ e
            n_feats = gates.shape[0]
            kf = min(int(features_per_layer), n_feats)
            if kf <= 0:
                continue
            sel = np.argpartition(-scores, kf - 1)[:kf]
            mask = scores[sel] > 0
            sel = sel[mask]
            if sel.size == 0:
                continue
            sel_scores = scores[sel]
            feat_meta = self.down_meta[layer]
            for fi, fs in zip(sel.tolist(), sel_scores.tolist()):
                pairs = feat_meta[fi]
                if not pairs:
                    continue
                f_scalar = float(fs)
                for tid, lg in pairs:
                    votes[tid] += f_scalar * lg

        if not np.any(votes):
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

        s = votes - votes.max()
        p = np.exp(s, dtype=np.float32)
        p_sum = float(p.sum())
        if p_sum <= 0.0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        p = p / p_sum

        k = min(int(top_k), self.vocab)
        idx = np.argpartition(-p, k - 1)[:k]
        idx = idx[np.argsort(-p[idx])]
        return idx.astype(np.int64), p[idx].astype(np.float32)

    def predict_step(
        self,
        token_ids: list[int],
        *,
        top_k: int = 5,
        features_per_layer: int = 32,
        layers: list[int] | str | None = None,
    ) -> list[tuple[int, str, float]]:
        """Top-``top_k`` next-token predictions from a full prompt token id list."""
        layers_resolved = self.resolve_layers(layers)
        ids, probs = self._predict_step_ids(
            token_ids,
            top_k=top_k,
            features_per_layer=features_per_layer,
            layers=layers_resolved,
        )
        out: list[tuple[int, str, float]] = []
        for tid, pr in zip(ids.tolist(), probs.tolist()):
            out.append((int(tid), self.decode_one(int(tid)), float(pr)))
        return out

    def infer_topk(
        self,
        prompt: str,
        *,
        top_k: int = 5,
        features_per_layer: int = 32,
        layers: list[int] | str | None = None,
    ) -> list[tuple[str, float]]:
        ids = self.encode(prompt)
        preds = self.predict_step(
            ids,
            top_k=top_k,
            features_per_layer=features_per_layer,
            layers=layers,
        )
        return [(piece, prob) for _tid, piece, prob in preds]

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        features_per_layer: int = 32,
        layers: list[int] | str | None = None,
        stop_token_ids: list[int] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        """Greedy multi-token generation. Each step re-runs attention over the
        full prefix when the attention-context sidecar is loaded, then the FFN
        walk for the next token."""
        ids = self.encode(prompt)
        layers_resolved = self.resolve_layers(layers)
        stop = set(int(x) for x in (stop_token_ids or []))
        out_ids: list[int] = []
        for _ in range(int(max_new_tokens)):
            top_ids, _ = self._predict_step_ids(
                ids,
                top_k=1,
                features_per_layer=features_per_layer,
                layers=layers_resolved,
            )
            if top_ids.size == 0:
                break
            chosen = int(top_ids[0])
            if chosen in stop:
                break
            out_ids.append(chosen)
            ids.append(chosen)
            if on_token:
                on_token(self.decode_one(chosen))
        return self.decode(out_ids)


# ── Optional native LARQL fast path (still vindex-only) ───────────────────


def try_larql_infer(
    vindex_dir: Path,
    prompt: str,
    top_k: int,
) -> list[tuple[str, float]] | None:
    """Use ``larql.load(...).infer(...)`` when the bindings are installed AND
    the vindex was built at ``--level inference`` or higher. Returns ``None``
    in every other case so the caller can fall back to the pure-numpy walk."""
    try:
        import larql  # type: ignore
    except ImportError:
        return None

    idx = load_index(vindex_dir)
    level = str(idx.get("extract_level", "browse"))
    if level == "browse":
        return None

    t0 = time.perf_counter()
    try:
        v = larql.load(str(vindex_dir))
    except Exception as e:  # noqa: BLE001
        _log(f"[vindex-infer] native: larql.load failed: {e}")
        return None
    _log(f"[vindex-infer] native: larql.load OK in {time.perf_counter() - t0:.2f}s")

    t1 = time.perf_counter()
    try:
        preds = v.infer(prompt, top_k_predictions=top_k)
    except Exception as e:  # noqa: BLE001
        _log(f"[vindex-infer] native: v.infer() failed: {e}")
        return None
    _log(f"[vindex-infer] native: INFER done in {time.perf_counter() - t1:.2f}s")
    return list(preds)


# ── orchestration ─────────────────────────────────────────────────────────


def run_infer(
    vindex_dir: Path,
    prompt: str,
    top_k: int,
    *,
    features_per_layer: int = 32,
    layers: list[int] | str | None = None,
    runner: VindexWalkRunner | None = None,
    use_attention_context: bool | None = None,
) -> list[tuple[str, float]]:
    vindex_dir = vindex_dir.resolve()
    _log("=" * 60)
    _log(f"[vindex-infer] vindex_dir={vindex_dir}")
    _log(f"[vindex-infer] prompt={prompt!r}  top_k={top_k}")

    preds = try_larql_infer(vindex_dir, prompt, top_k)
    if preds is not None:
        _log("[vindex-infer] path: **native LARQL INFER** (Walk FFN + mmap weights)")
        for i, (tok, pr) in enumerate(preds, 1):
            _log(f"  {i:2}. p={pr:.4f}  {tok!r}")
        _log("=" * 60)
        return preds

    own_runner = runner is None
    if own_runner:
        t0 = time.perf_counter()
        runner = VindexWalkRunner(vindex_dir, use_attention_context=use_attention_context)
        _log(f"[vindex-infer] runner ready in {time.perf_counter() - t0:.2f}s")
    assert runner is not None

    t1 = time.perf_counter()
    out = runner.infer_topk(
        prompt, top_k=top_k, features_per_layer=features_per_layer, layers=layers
    )
    ctx_note = (
        "FULL-PROMPT attention→last residual + "
        if runner._attn_ctx is not None
        else "prompt mean-pool fallback + "
    )
    _log(
        "[vindex-infer] path: **vindex walk (pure numpy)** — "
        f"{ctx_note}FFN gates×down_meta — "
        f"layers={runner.resolve_layers(layers)[:1]}…{runner.resolve_layers(layers)[-1:]} "
        f"step={time.perf_counter() - t1:.2f}s"
    )
    for i, (tok, pr) in enumerate(out, 1):
        _log(f"  {i:2}. p={pr:.4e}  {tok!r}")
    _log("=" * 60)
    return out


# ── env / CLI ─────────────────────────────────────────────────────────────


def _parse_layers_env(value: str | None) -> list[int] | str | None:
    if value is None:
        return None
    s = value.strip()
    if not s:
        return None
    # band name passes through as a string for VindexWalkRunner.resolve_layers
    if s.isalpha():
        return s
    return [int(p.strip()) for p in s.split(",") if p.strip()]


def run_default_cell() -> None:
    vd = default_vindex_dir()
    prompt = os.environ.get("VINDEX_PROMPT", "The capital of France is").strip()
    top_k = int(os.environ.get("VINDEX_TOP_K", "5"))
    features = int(os.environ.get("VINDEX_WALK_FEATURES", "32"))
    layers = _parse_layers_env(os.environ.get("VINDEX_WALK_LAYERS"))
    use_ctx = False if _env_truthy("VINDEX_NO_ATTN_CTX") else None
    run_infer(vd, prompt, top_k, features_per_layer=features, layers=layers, use_attention_context=use_ctx)


def main_cli() -> None:
    ap = argparse.ArgumentParser(description="Vindex-only INFER (no HF model load)")
    ap.add_argument("--vindex", type=Path, default=default_vindex_dir(), help="path to vindex directory")
    ap.add_argument("-p", "--prompt", default=os.environ.get("VINDEX_PROMPT", "The capital of France is"))
    ap.add_argument("--top-k", type=int, default=int(os.environ.get("VINDEX_TOP_K", "5")))
    ap.add_argument(
        "--features-per-layer",
        type=int,
        default=int(os.environ.get("VINDEX_WALK_FEATURES", "32")),
        help="how many top features to keep per layer in the walk (default: 32)",
    )
    ap.add_argument(
        "--layers",
        default=os.environ.get("VINDEX_WALK_LAYERS", ""),
        help='comma list (e.g. "12,13,14") or band name ("syntax", "knowledge", "output"); default = all',
    )
    ap.add_argument(
        "--no-attention-context",
        action="store_true",
        help="use last-token embedding only (ignore vindex_attn_ctx_* if present)",
    )
    args, rest = ap.parse_known_args(user_argv())
    if rest:
        _log(f"[vindex-infer] warning: ignored argv: {rest}")
    layers = _parse_layers_env(args.layers) if args.layers else None
    use_ctx = False if args.no_attention_context else None
    run_infer(
        args.vindex.resolve(),
        args.prompt,
        args.top_k,
        features_per_layer=args.features_per_layer,
        layers=layers,
        use_attention_context=use_ctx,
    )


if __name__ == "__main__":
    if not user_argv():
        _log("[vindex-infer] entry: Colab / notebook default (empty argv)")
        run_default_cell()
    else:
        _log(f"[vindex-infer] entry: CLI  argv={user_argv()!r}")
        main_cli()
