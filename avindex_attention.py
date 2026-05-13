#!/usr/bin/env python3
"""
**A-Vindex attention sidecar** — vindex-only build + probe.

What this module does
---------------------
* **Build** (one-shot, needs the HF safetensors once): for each layer and each
  KV head, projects a vocabulary sample through ``self_attn.k_proj`` and writes
  ``attn_centroids.bin`` (MiniBatchKMeans clusters). It also writes
  ``k_proj_weights.bin`` so the probe and scan paths never need the source
  model again.
* **Probe** (vindex-only): tokenises the prompt with ``tokenizer.json`` already
  inside the vindex, looks the last-token embedding up in ``embeddings.bin``,
  applies the stored ``k_proj`` slice, and returns the closest centroid in
  cosine similarity. **No Hugging Face model load, no transformers, no
  torchvision.**

Layout (alongside an existing browse vindex directory)
------------------------------------------------------
  <vindex_dir>/attn_centroids.bin       # MiniBatchKMeans centroids, f32
  <vindex_dir>/attn_k_proj_weights.bin  # full k_proj per layer, f32
  <vindex_dir>/attn_index.json          # offsets, shapes, k_proj layout
  <vindex_dir>/vindex_attn_ctx_weights.bin   # Q/V/O + input LN (+ optional Qwen q_norm/k_norm)
  <vindex_dir>/vindex_attn_ctx_index.json    # byte layout for the above

Dependencies
------------
  pip install torch safetensors numpy scikit-learn tokenizers
  Needs ``vindex.py`` on disk: same folder as this file, current working
  directory, or set ``VINDEX_PY`` to the full path of ``vindex.py``.

Env (notebook, empty argv)
--------------------------
  AVINDEX_EXTRACT=1     run extract into VINDEX_DIR (default ./.larql_colab/vindex_out)
  VINDEX_MODEL          HF id or local path (same as vindex.py)
  VINDEX_DIR            output vindex directory
  AVINDEX_CENTROIDS=256
  AVINDEX_VOCAB_SAMPLE=8000
  AVINDEX_PROBE=1       run the vindex-only probe after extract (no HF model needed)
  AVINDEX_PROBE_LAYER   optional explicit layer (else middle of knowledge band)
  AVINDEX_PROBE_KV_HEAD default 0
  VINDEX_PY             optional absolute path to vindex.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ── logging / argv (match vindex.py style) ───────────────────────────────


def _log(msg: str) -> None:
    print(f"[avindex-attn] {msg}", flush=True)


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


def _env_truthy(name: str, *, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _scripts_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


def _candidate_vindex_py_paths() -> list[Path]:
    raw: list[Path] = []
    ev = os.environ.get("VINDEX_PY", "").strip()
    if ev:
        p = Path(ev).expanduser().resolve()
        raw.append((p / "vindex.py") if p.is_dir() else p)
    raw.append(_scripts_dir() / "vindex.py")
    raw.append(Path.cwd().resolve() / "vindex.py")
    out: list[Path] = []
    seen: set[str] = set()
    for p in raw:
        try:
            rp = p.resolve()
        except OSError:
            continue
        key = str(rp)
        if key not in seen:
            seen.add(key)
            out.append(rp)
    return out


def _load_vindex_from_path(py_path: Path) -> Any:
    name = "vindex_colab_avindex"
    spec = importlib.util.spec_from_file_location(name, py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {py_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _import_vindex() -> Any:
    try:
        import vindex as vx  # type: ignore

        return vx
    except ImportError:
        pass
    for py in _candidate_vindex_py_paths():
        if py.is_file():
            try:
                return _load_vindex_from_path(py)
            except Exception as e:  # noqa: BLE001
                _log(f"warning: failed to load {py}: {e}")
                continue
    tried = ", ".join(str(p) for p in _candidate_vindex_py_paths())
    raise RuntimeError(
        "Could not import `vindex`. Options:\n"
        "  • Run from the repo folder where `vindex.py` lives, or paste `vindex.py` in cwd.\n"
        "  • Set VINDEX_PY to the full path of vindex.py.\n"
        f"  Tried: {tried}"
    )


def _work_base() -> Path:
    try:
        return _import_vindex().work_base()
    except Exception:
        p = Path.cwd().resolve() / ".larql_colab"
        p.mkdir(parents=True, exist_ok=True)
        return p


# ── extract ───────────────────────────────────────────────────────────────


def _reshape_k_proj(
    wk: Any,
    *,
    n_kv: int,
    hidden: int,
) -> tuple[Any, int]:
    """Return (wk_kv, head_dim_kv) with wk_kv shape [n_kv, head_dim_kv, hidden]."""
    if wk.shape[1] == hidden and wk.shape[0] % n_kv == 0:
        pass
    elif wk.shape[0] == hidden and wk.shape[1] % n_kv == 0:
        wk = wk.T.contiguous()
    else:
        raise ValueError(f"Unexpected k_proj shape {tuple(wk.shape)} for hidden={hidden}, n_kv={n_kv}")

    r0 = int(wk.shape[0])
    if r0 % n_kv != 0:
        raise ValueError(f"k_proj out_features {r0} not divisible by num_key_value_heads={n_kv}")
    head_dim_kv = r0 // n_kv
    wk_kv = wk.reshape(n_kv, head_dim_kv, hidden)
    return wk_kv, head_dim_kv


def extract_attention_avindex(
    model_dir: Path,
    out_dir: Path,
    *,
    num_centroids: int = 256,
    vocab_sample: int = 8000,
    random_state: int = 42,
    force: bool = False,
) -> Path:
    """
    Build ``attn_centroids.bin`` + ``attn_k_proj_weights.bin`` + ``attn_index.json``
    under ``out_dir`` (an existing browse vindex directory is fine). Also calls
    ``extract_attn_context_sidecar`` so ``vindex_attn_ctx_*`` are written in the
    same run.

    Idempotent: when ``out_dir`` already contains every centroid / k_proj /
    attention-context file and ``force=False``, returns without loading any HF
    safetensors.
    """
    out_dir = Path(out_dir).resolve()
    if not force and avindex_centroids_complete(out_dir) and attn_context_sidecar_complete(out_dir):
        _log("=" * 60)
        _log(f"SKIP extract — A-Vindex already complete at {out_dir} (no HF load)")
        _log(f"  centroids+k_proj: {', '.join(AVINDEX_REQUIRED_FILES)}")
        _log(f"  attn-context:     {', '.join(ATTN_CTX_REQUIRED_FILES)}")
        _log("  set force=True (env AVINDEX_FORCE=1, CLI --force) to rebuild")
        return out_dir

    from sklearn.cluster import MiniBatchKMeans

    vx = _import_vindex()
    import torch

    model_dir = model_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    num_layers = int(cfg["num_hidden_layers"])
    hidden = int(cfg["hidden_size"])
    vocab = int(cfg["vocab_size"])
    n_heads = int(cfg.get("num_attention_heads", cfg.get("num_heads", 1)))
    n_kv = int(cfg.get("num_key_value_heads", n_heads))

    _log("=" * 60)
    _log(f"EXTRACT  model_dir={model_dir}")
    _log(f"         out_dir={out_dir}")
    _log(f"         layers={num_layers}  q_heads={n_heads}  kv_heads={n_kv}  hidden={hidden}")
    _log(
        f"         vocab_sample≤{vocab_sample}  k_means_clusters={num_centroids} "
        f"(actual vocab sample logged after draw)"
    )

    _log("loading safetensors (full load — same as vindex.py; RAM heavy on large models) …")
    t0 = time.perf_counter()
    sd = vx.load_safetensors_dir(model_dir)
    prefix = vx.detect_prefix(list(sd.keys()))
    _log(f"  prefix={prefix!r}  ({time.perf_counter() - t0:.1f}s)")

    ek = vx.find_key(
        sd,
        f"{prefix}embed_tokens.weight",
        "model.embed_tokens.weight",
        "embed_tokens.weight",
    )
    emb = sd[ek].numpy().astype(np.float32, copy=False)
    if emb.shape != (vocab, hidden):
        if emb.shape == (hidden, vocab):
            emb = emb.T
        else:
            raise ValueError(f"Unexpected embed shape {emb.shape}")

    n_sample = min(int(vocab_sample), vocab)
    rng = np.random.default_rng(random_state)
    if n_sample < vocab:
        sample_idx = np.sort(rng.choice(vocab, size=n_sample, replace=False))
    else:
        sample_idx = np.arange(vocab, dtype=np.int64)
    emb_s = emb[sample_idx]
    _log(f"         drawn vocab_sample={n_sample} rows for clustering")

    attn_index: dict[str, Any] = {
        "avindex_version": 3,  # bumped: now includes k_proj_weights
        "kind": "k_proj_vocab_projection_centroids",
        "num_layers": num_layers,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv,
        "hidden_size": hidden,
        "vocab_sample_size": int(n_sample),
        "num_centroids": int(num_centroids),
        "dtype": "f32",
        "layers": [],
        "k_proj_layers": [],  # offsets into attn_k_proj_weights.bin
    }

    centroid_path = out_dir / "attn_centroids.bin"
    kproj_path = out_dir / "attn_k_proj_weights.bin"
    t_all = time.perf_counter()

    with open(centroid_path, "wb") as f_bin, open(kproj_path, "wb") as f_k:
        for layer in range(num_layers):
            t_l = time.perf_counter()
            k_key = vx.find_key(
                sd,
                f"{prefix}layers.{layer}.self_attn.k_proj.weight",
                f"model.layers.{layer}.self_attn.k_proj.weight",
            )
            wk = sd[k_key].to(torch.float32)
            wk_kv, head_dim_kv = _reshape_k_proj(wk, n_kv=n_kv, hidden=hidden)

            layer_entry: dict[str, Any] = {
                "layer": layer,
                "head_dim_kv": head_dim_kv,
                "kv_heads": [],
            }
            kproj_entry: dict[str, Any] = {
                "layer": layer,
                "head_dim_kv": head_dim_kv,
                "kv_heads": [],
            }

            for kv_h in range(n_kv):
                head_k = wk_kv[kv_h].contiguous()
                head_k_np = np.ascontiguousarray(head_k.numpy(), dtype=np.float32)

                # 1) Save the raw k_proj slice so the probe can run vindex-only.
                k_offset = f_k.tell()
                f_k.write(head_k_np.tobytes(order="C"))
                kproj_entry["kv_heads"].append(
                    {
                        "kv_head": kv_h,
                        "offset": int(k_offset),
                        "size_bytes": int(head_k_np.nbytes),
                        "shape": [int(head_k_np.shape[0]), int(head_k_np.shape[1])],
                    }
                )

                # 2) Cluster the projected vocabulary sample.
                projected = emb_s @ head_k_np.T
                k_use = min(int(num_centroids), int(projected.shape[0]))
                km = MiniBatchKMeans(
                    n_clusters=k_use,
                    batch_size=min(1024, projected.shape[0]),
                    n_init=3,
                    random_state=random_state,
                )
                km.fit(projected)
                centroids = np.ascontiguousarray(km.cluster_centers_, dtype=np.float32)
                c_offset = f_bin.tell()
                f_bin.write(centroids.tobytes(order="C"))
                layer_entry["kv_heads"].append(
                    {
                        "kv_head": kv_h,
                        "offset": int(c_offset),
                        "size_bytes": int(centroids.nbytes),
                        "shape": [int(centroids.shape[0]), int(centroids.shape[1])],
                    }
                )
            attn_index["layers"].append(layer_entry)
            attn_index["k_proj_layers"].append(kproj_entry)
            _log(f"  layer {layer + 1}/{num_layers}  kv_heads={n_kv}  ({time.perf_counter() - t_l:.1f}s)")

    attn_index["head_dim_kv"] = attn_index["layers"][0]["head_dim_kv"] if num_layers else None
    (out_dir / "attn_index.json").write_text(json.dumps(attn_index, indent=2), encoding="utf-8")
    sz_mb = centroid_path.stat().st_size / (1024 * 1024)
    sz_k_mb = kproj_path.stat().st_size / (1024 * 1024)
    _log("=" * 60)
    _log(
        f"DONE  centroids={sz_mb:.2f} MiB  k_proj={sz_k_mb:.2f} MiB  "
        f"total {time.perf_counter() - t_all:.1f}s"
    )
    if not force and attn_context_sidecar_complete(out_dir):
        _log("attention-context sidecar already present — skipping sidecar pass")
    else:
        try:
            from vindex_attn_context import extract_attn_context_sidecar

            extract_attn_context_sidecar(out_dir, sd=sd, prefix=prefix, cfg=cfg, vx=vx)
        except Exception as e:  # noqa: BLE001
            _log(f"warning: attention-context sidecar skipped: {e}")
    return out_dir


# ── read / score ──────────────────────────────────────────────────────────


class AvindexAttentionReader:
    """
    Reads ``attn_centroids.bin`` and ``attn_k_proj_weights.bin`` from a vindex
    directory. Sequential file access via ``seek+read``; cheap per-call cache.
    """

    def __init__(self, vindex_dir: Path) -> None:
        self.vindex_dir = vindex_dir.resolve()
        p_meta = self.vindex_dir / "attn_index.json"
        p_bin = self.vindex_dir / "attn_centroids.bin"
        p_k = self.vindex_dir / "attn_k_proj_weights.bin"
        if not p_meta.is_file():
            raise FileNotFoundError(f"Missing {p_meta} — run extract_attention_avindex first.")
        if not p_bin.is_file():
            raise FileNotFoundError(f"Missing {p_bin}")
        self.meta: dict[str, Any] = json.loads(p_meta.read_text(encoding="utf-8"))
        self._normalize_meta_layers()
        self._bin_path = p_bin
        self._fp = open(p_bin, "rb")
        # k_proj is optional for old (v2) bundles; for v3+ we expect it.
        self._k_path = p_k
        self._fp_k: Any | None = open(p_k, "rb") if p_k.is_file() else None
        self._cache_centroids: dict[tuple[int, int], np.ndarray] = {}
        self._cache_kproj: dict[tuple[int, int], np.ndarray] = {}

    def _normalize_meta_layers(self) -> None:
        raw = self.meta.get("layers")
        if raw is None:
            raise KeyError("attn_index.json missing top-level 'layers'")
        if isinstance(raw, dict):
            try:
                sorted_items = sorted(((int(str(k)), v) for k, v in raw.items()), key=lambda t: t[0])
            except (TypeError, ValueError) as e:
                raise ValueError(
                    "attn_index.json: when 'layers' is an object, keys must be numeric layer ids (e.g. \"0\")."
                ) from e
            self.meta["layers"] = [v for _, v in sorted_items]
        elif isinstance(raw, list):
            self.meta["layers"] = raw
        else:
            raise TypeError(f"attn_index.json 'layers' must be list or dict, got {type(raw)!r}")

        for i, entry in enumerate(self.meta["layers"]):
            if not isinstance(entry, dict):
                raise TypeError(f"layers[{i}] must be object, got {type(entry)!r}")
            if "kv_heads" not in entry:
                if "heads" in entry:
                    entry["kv_heads"] = entry["heads"]
                elif "head" in entry and isinstance(entry["head"], list):
                    entry["kv_heads"] = entry["head"]

    @staticmethod
    def _heads_array(layer_entry: dict[str, Any]) -> list[dict[str, Any]]:
        if "kv_heads" in layer_entry:
            return list(layer_entry["kv_heads"])
        if "heads" in layer_entry:
            return list(layer_entry["heads"])
        raise KeyError(
            "attn_index.json layer entry has no 'kv_heads' or 'heads'. "
            f"Keys: {sorted(layer_entry.keys())}. Re-run extract_attention_avindex()."
        )

    def _head_dim_for_layer(self, layer_entry: dict[str, Any], hm: dict[str, Any]) -> int:
        sh = hm.get("shape")
        if isinstance(sh, list) and len(sh) == 2:
            return int(sh[1])
        d = layer_entry.get("head_dim_kv") or self.meta.get("head_dim_kv") or self.meta.get("head_dim")
        if d is not None and int(d) > 0:
            return int(d)
        n_cent = int(self.meta.get("num_centroids", 256))
        nbytes = int(hm["size_bytes"])
        if nbytes > 0 and nbytes % (4 * n_cent) == 0:
            return nbytes // (4 * n_cent)
        raise ValueError(
            "Cannot infer head_dim from attn_index.json; re-run extract with current avindex_attention.py."
        )

    def close(self) -> None:
        self._fp.close()
        if self._fp_k is not None:
            self._fp_k.close()

    def __enter__(self) -> AvindexAttentionReader:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @property
    def num_layers(self) -> int:
        return len(self.meta["layers"])

    @property
    def hidden_size(self) -> int:
        return int(self.meta.get("hidden_size", 0))

    def n_kv_for_layer(self, layer: int) -> int:
        return len(self._heads_array(self.meta["layers"][layer]))

    def centroids(self, layer: int, kv_head: int) -> np.ndarray:
        key = (layer, kv_head)
        if key in self._cache_centroids:
            return self._cache_centroids[key]
        layer_entry = self.meta["layers"][layer]
        heads = self._heads_array(layer_entry)
        hm = heads[kv_head]
        self._fp.seek(int(hm["offset"]))
        raw = self._fp.read(int(hm["size_bytes"]))
        d = self._head_dim_for_layer(layer_entry, hm)
        sh = hm.get("shape")
        if isinstance(sh, list) and len(sh) == 2:
            arr = np.frombuffer(raw, dtype=np.float32).reshape(int(sh[0]), int(sh[1]))
        else:
            nbytes = len(raw)
            if nbytes % (4 * d) != 0:
                raise ValueError(
                    f"centroid blob size {nbytes} not divisible by 4*head_dim={4 * d}; "
                    "re-run extract or fix attn_index.json."
                )
            k = nbytes // (4 * d)
            arr = np.frombuffer(raw, dtype=np.float32).reshape(int(k), int(d))
        self._cache_centroids[key] = arr
        return arr

    def k_slice(self, layer: int, kv_head: int) -> np.ndarray:
        """
        Return the stored ``k_proj`` slice for a given (layer, kv_head),
        shape ``[head_dim_kv, hidden]``. Reads ``attn_k_proj_weights.bin``;
        raises ``FileNotFoundError`` for legacy bundles without the file.
        """
        if self._fp_k is None:
            raise FileNotFoundError(
                f"{self._k_path} is missing — this vindex was built with an older "
                "avindex_attention.py (avindex_version<3). Re-run extract to bake "
                "k_proj into the vindex."
            )
        key = (layer, kv_head)
        if key in self._cache_kproj:
            return self._cache_kproj[key]
        kproj_layers = self.meta.get("k_proj_layers")
        if not kproj_layers:
            raise KeyError("attn_index.json missing 'k_proj_layers' (re-run extract).")
        entry = kproj_layers[layer]
        heads = entry.get("kv_heads") or []
        hm = heads[kv_head]
        self._fp_k.seek(int(hm["offset"]))
        raw = self._fp_k.read(int(hm["size_bytes"]))
        sh = hm.get("shape")
        if not (isinstance(sh, list) and len(sh) == 2):
            raise ValueError("k_proj entry missing 'shape' — re-run extract.")
        arr = np.frombuffer(raw, dtype=np.float32).reshape(int(sh[0]), int(sh[1]))
        self._cache_kproj[key] = arr
        return arr

    def nearest_cosine(self, layer: int, kv_head: int, q: np.ndarray) -> tuple[int, float]:
        """Unit-norm cosine similarity → best centroid index and score in [-1, 1]."""
        q = np.asarray(q, dtype=np.float32).reshape(-1)
        c = self.centroids(layer, kv_head)
        if q.shape[0] != c.shape[1]:
            raise ValueError(f"query dim {q.shape[0]} != head_dim_kv {c.shape[1]}")
        qn = q / (np.linalg.norm(q) + 1e-8)
        cn = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-8)
        sims = cn @ qn
        j = int(np.argmax(sims))
        return j, float(sims[j])


def _attn_layers_normalized(attn: dict[str, Any]) -> list[dict[str, Any]]:
    raw = attn.get("layers") or []
    if isinstance(raw, dict):
        return [v for _, v in sorted(((int(str(k)), v) for k, v in raw.items()), key=lambda t: t[0])]
    if isinstance(raw, list):
        return list(raw)
    return []


def load_embeddings_f32(vindex_dir: Path, vocab: int, hidden: int) -> np.ndarray:
    p = vindex_dir / "embeddings.bin"
    if not p.is_file():
        raise FileNotFoundError(f"Missing {p} — run vindex.py extract first.")
    raw = np.fromfile(str(p), dtype="<f4")
    if raw.size != vocab * hidden:
        raise ValueError(f"embeddings.bin has {raw.size} floats, expected {vocab * hidden}")
    return raw.reshape(vocab, hidden)


# ── vindex-only tokenizer ─────────────────────────────────────────────────


def load_vindex_tokenizer(vindex_dir: Path) -> Any:
    """Load ``tokenizer.json`` straight from the vindex (no HF model id required)."""
    p = Path(vindex_dir).resolve() / "tokenizer.json"
    if not p.is_file():
        raise FileNotFoundError(
            f"{p} missing — vindex.py copies tokenizer.json next to embeddings.bin."
        )
    try:
        from tokenizers import Tokenizer  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "Install the `tokenizers` package: pip install tokenizers"
        ) from e
    return Tokenizer.from_file(str(p))


def _encode_prompt(tok: Any, prompt: str) -> list[int]:
    enc = tok.encode(prompt)
    ids = list(enc.ids)
    if not ids:
        raise ValueError("empty tokenization")
    return ids


def _decode_one(tok: Any, token_id: int) -> str:
    try:
        return tok.decode([int(token_id)])
    except Exception:  # noqa: BLE001
        return f"<id {token_id}>"


# ── probe / scan (vindex-only) ────────────────────────────────────────────


def probe_last_token_k_space(
    vindex_dir: Path,
    *,
    prompt: str,
    layer: int,
    kv_head: int = 0,
    reader: AvindexAttentionReader | None = None,
) -> dict[str, Any]:
    """
    Honest probe in the **same space as extract** — but vindex-only:
    last-token embedding row × stored ``k_proj`` slice → nearest centroid (cosine).

    Requires only files inside ``vindex_dir`` (no HF model, no transformers).
    """
    vindex_dir = Path(vindex_dir).resolve()
    main = json.loads((vindex_dir / "index.json").read_text(encoding="utf-8"))
    vocab = int(main["vocab_size"])
    hidden = int(main["hidden_size"])

    tok = load_vindex_tokenizer(vindex_dir)
    ids = _encode_prompt(tok, prompt)
    last = int(ids[-1])

    emb = load_embeddings_f32(vindex_dir, vocab, hidden)
    e = emb[last].astype(np.float32)

    own = reader is None
    r = reader or AvindexAttentionReader(vindex_dir)
    try:
        kh = r.k_slice(layer, kv_head)
        q = e @ kh.T
        cid, score = r.nearest_cosine(layer, kv_head, q)
    finally:
        if own:
            r.close()

    return {
        "prompt": prompt,
        "last_token_id": last,
        "last_token_piece": _decode_one(tok, last),
        "layer": int(layer),
        "kv_head": int(kv_head),
        "centroid_id": int(cid),
        "cosine": float(score),
    }


def scan_centroid_path(
    vindex_dir: Path,
    *,
    prompt: str,
    layers_only: list[int] | None = None,
    reader: AvindexAttentionReader | None = None,
) -> dict[str, Any]:
    """
    For **each** (layer, kv_head): project last-token embedding through the
    stored ``k_proj`` slice, then nearest centroid (cosine). Diagnostic only.
    """
    vindex_dir = Path(vindex_dir).resolve()
    main = json.loads((vindex_dir / "index.json").read_text(encoding="utf-8"))
    vocab = int(main["vocab_size"])
    hidden = int(main["hidden_size"])

    tok = load_vindex_tokenizer(vindex_dir)
    ids = _encode_prompt(tok, prompt)
    last = int(ids[-1])
    emb = load_embeddings_f32(vindex_dir, vocab, hidden)
    e = emb[last].astype(np.float32)

    own = reader is None
    r = reader or AvindexAttentionReader(vindex_dir)
    steps: list[dict[str, Any]] = []
    try:
        n_layers = r.num_layers
        layer_iter = layers_only if layers_only is not None else range(n_layers)
        for layer in layer_iter:
            if layer < 0 or layer >= n_layers:
                continue
            n_kv = r.n_kv_for_layer(layer)
            for kv_h in range(n_kv):
                kh = r.k_slice(layer, kv_h)
                q = e @ kh.T
                cid, score = r.nearest_cosine(layer, kv_h, q)
                steps.append(
                    {
                        "layer": int(layer),
                        "kv_head": int(kv_h),
                        "centroid_id": int(cid),
                        "cosine": float(score),
                    }
                )
    finally:
        if own:
            r.close()
    return {
        "kind": "a_vindex_centroid_hit_per_kv_head",
        "prompt": prompt,
        "last_token_id": last,
        "last_token_piece": _decode_one(tok, last),
        "steps": steps,
        "num_steps": len(steps),
    }


AVINDEX_REQUIRED_FILES: tuple[str, ...] = (
    "attn_index.json",
    "attn_centroids.bin",
    "attn_k_proj_weights.bin",
)
ATTN_CTX_REQUIRED_FILES: tuple[str, ...] = (
    "vindex_attn_ctx_index.json",
    "vindex_attn_ctx_weights.bin",
)


def avindex_centroids_complete(out_dir: Path) -> bool:
    """``True`` if the centroid / k_proj files of the A-Vindex are all present."""
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return False
    for name in AVINDEX_REQUIRED_FILES:
        p = out_dir / name
        if not p.is_file() or p.stat().st_size == 0:
            return False
    return True


def attn_context_sidecar_complete(out_dir: Path) -> bool:
    """``True`` if both ``vindex_attn_ctx_*`` files are present and non-empty."""
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return False
    for name in ATTN_CTX_REQUIRED_FILES:
        p = out_dir / name
        if not p.is_file() or p.stat().st_size == 0:
            return False
    return True


def extract_attn_context_only(
    model_dir: Path,
    out_dir: Path,
    *,
    force: bool = False,
) -> Path:
    """
    Standalone re-extract of just the **attention-context sidecar**
    (``vindex_attn_ctx_*``). Useful when an older A-Vindex was built without it
    — runs through the HF safetensors only to read Q/V/O + input LN (+ optional
    Qwen q_norm/k_norm) and writes the sidecar next to ``attn_index.json``.

    Idempotent: returns immediately when both sidecar files are already there
    and ``force=False`` (no HF safetensors load).
    """
    vx = _import_vindex()
    out_dir = Path(out_dir).resolve()
    model_dir = Path(model_dir).resolve()
    if not (out_dir / "attn_index.json").is_file():
        raise FileNotFoundError(
            f"{out_dir}/attn_index.json missing — run `extract` first so that "
            "attn_centroids.bin + attn_k_proj_weights.bin exist alongside the sidecar."
        )

    if not force and attn_context_sidecar_complete(out_dir):
        _log("=" * 60)
        _log(f"SKIP extract-ctx — sidecar already present at {out_dir} (no HF load)")
        _log(f"  files: {', '.join(ATTN_CTX_REQUIRED_FILES)}")
        return out_dir

    _log("=" * 60)
    _log(f"EXTRACT-CTX  model_dir={model_dir}")
    _log(f"             out_dir={out_dir}")
    _log("loading safetensors (full load — same RAM cost as `extract`) …")
    t0 = time.perf_counter()
    sd = vx.load_safetensors_dir(model_dir)
    prefix = vx.detect_prefix(list(sd.keys()))
    _log(f"  prefix={prefix!r}  ({time.perf_counter() - t0:.1f}s)")
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))

    from vindex_attn_context import extract_attn_context_sidecar

    extract_attn_context_sidecar(out_dir, sd=sd, prefix=prefix, cfg=cfg, vx=vx)
    _log("OK — vindex_attn_ctx_index.json + vindex_attn_ctx_weights.bin written")
    return out_dir


def _default_probe_layer(main_index: dict[str, Any]) -> int:
    bands = main_index.get("layer_bands") or {}
    kn = bands.get("knowledge")
    if isinstance(kn, list) and len(kn) == 2:
        a, b = int(kn[0]), int(kn[1])
        return (a + b) // 2
    nl = int(main_index.get("num_layers", 1))
    return max(0, nl // 2)


# ── entry points (extract is the only HF-touching path) ───────────────────


def run_default_cell() -> None:
    vx = _import_vindex()
    base = vx.work_base()
    vdir = Path(os.environ.get("VINDEX_DIR", str(base / "vindex_out"))).expanduser().resolve()

    force = _env_truthy("AVINDEX_FORCE")

    if _env_truthy("AVINDEX_EXTRACT"):
        if not force and avindex_centroids_complete(vdir) and attn_context_sidecar_complete(vdir):
            _log(f"SKIP extract — A-Vindex already complete at {vdir}  (set AVINDEX_FORCE=1 to rebuild)")
        else:
            model_id = os.environ.get("VINDEX_MODEL", "Qwen/Qwen3-0.6B").strip()
            mdir = vx.resolve_model_dir(model_id, base / "hf_models")
            k = int(os.environ.get("AVINDEX_CENTROIDS", "256"))
            vs = int(os.environ.get("AVINDEX_VOCAB_SAMPLE", "8000"))
            extract_attention_avindex(mdir, vdir, num_centroids=k, vocab_sample=vs, force=force)

    if _env_truthy("AVINDEX_EXTRACT_CTX"):
        if not force and attn_context_sidecar_complete(vdir):
            _log(f"SKIP extract-ctx — sidecar already present at {vdir}  (set AVINDEX_FORCE=1 to rebuild)")
        else:
            model_id = os.environ.get("VINDEX_MODEL", "Qwen/Qwen3-0.6B").strip()
            mdir = vx.resolve_model_dir(model_id, base / "hf_models")
            extract_attn_context_only(mdir, vdir, force=force)

    if _env_truthy("AVINDEX_PROBE"):
        main = json.loads((vdir / "index.json").read_text(encoding="utf-8"))
        layer = int(os.environ.get("AVINDEX_PROBE_LAYER", str(_default_probe_layer(main))))
        prompt = os.environ.get("AVINDEX_PROMPT", "The capital of France is").strip()
        kv = int(os.environ.get("AVINDEX_PROBE_KV_HEAD", "0"))
        out = probe_last_token_k_space(vdir, prompt=prompt, layer=layer, kv_head=kv)
        _log("PROBE (vindex-only: embedding × stored K_h^T vs centroids, last token only)")
        for k, v in out.items():
            _log(f"  {k}: {v!r}")
        return

    if not _env_truthy("AVINDEX_EXTRACT"):
        _log("nothing to do — set AVINDEX_EXTRACT=1 and/or AVINDEX_PROBE=1")


def main_cli() -> None:
    vx = _import_vindex()
    ap = argparse.ArgumentParser(description="A-Vindex attention sidecar (vindex-only probe)")
    ap.add_argument(
        "command",
        choices=["extract", "extract-ctx", "probe", "scan"],
        help="action to perform (`extract-ctx` only writes vindex_attn_ctx_*)",
    )
    ap.add_argument("--model-dir", type=Path, default=None, help="(extract only) HF safetensors folder")
    ap.add_argument("--vindex", type=Path, default=None, help="vindex directory")
    ap.add_argument("--centroids", type=int, default=256)
    ap.add_argument("--vocab-sample", type=int, default=8000)
    ap.add_argument("--prompt", "-p", default="The capital of France is")
    ap.add_argument("--layer", type=int, default=-1, help="-1 = middle/knowledge band from index.json")
    ap.add_argument("--kv-head", type=int, default=0)
    ap.add_argument(
        "--scan-layers",
        default="",
        help='comma-separated layer ids (default: all when scan)',
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if A-Vindex / sidecar files already exist (default: skip).",
    )
    args, rest = ap.parse_known_args(user_argv())
    if rest:
        _log(f"warning: ignored argv: {rest}")

    base = vx.work_base()
    vdir = (args.vindex or Path(os.environ.get("VINDEX_DIR", str(base / "vindex_out")))).resolve()

    force = args.force or _env_truthy("AVINDEX_FORCE")

    if args.command == "extract":
        if not force and avindex_centroids_complete(vdir) and attn_context_sidecar_complete(vdir):
            _log(f"SKIP extract — A-Vindex already complete at {vdir}")
            _log("  use --force or AVINDEX_FORCE=1 to rebuild from scratch")
            return
        mid = os.environ.get("VINDEX_MODEL", "Qwen/Qwen3-0.6B").strip()
        mdir = (args.model_dir or vx.resolve_model_dir(mid, base / "hf_models")).resolve()
        extract_attention_avindex(
            mdir,
            vdir,
            num_centroids=args.centroids,
            vocab_sample=args.vocab_sample,
            force=force,
        )
        return

    if args.command == "extract-ctx":
        if not force and attn_context_sidecar_complete(vdir):
            _log(f"SKIP extract-ctx — sidecar already present at {vdir}")
            _log("  use --force or AVINDEX_FORCE=1 to rebuild from scratch")
            return
        mid = os.environ.get("VINDEX_MODEL", "Qwen/Qwen3-0.6B").strip()
        mdir = (args.model_dir or vx.resolve_model_dir(mid, base / "hf_models")).resolve()
        extract_attn_context_only(mdir, vdir, force=force)
        return

    main = json.loads((vdir / "index.json").read_text(encoding="utf-8"))

    if args.command == "probe":
        layer = args.layer if args.layer >= 0 else _default_probe_layer(main)
        out = probe_last_token_k_space(
            vdir, prompt=args.prompt, layer=layer, kv_head=args.kv_head
        )
        _log("PROBE result:")
        print(json.dumps(out, indent=2))
        return

    # scan
    sl = args.scan_layers.strip()
    layers_only = [int(x.strip()) for x in sl.split(",") if x.strip()] if sl else None
    out = scan_centroid_path(vdir, prompt=args.prompt, layers_only=layers_only)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    if not user_argv():
        _log("entry: notebook default (empty argv)")
        run_default_cell()
    else:
        _log(f"entry: CLI argv={user_argv()!r}")
        main_cli()
