#!/usr/bin/env python3
"""
Experimental **Attention sidecar** (A‑Vindex sketch) for the Colab layout used by `vindex.py`.

What it does (offline)
-----------------------
For each layer and each **KV head** (GQA‑aware), clusters a sample of vocabulary rows
projected through that head’s **k_proj** slice:

    projected[token] = embedding[token] @ K_h.T    # shape (head_dim_kv,)

Then writes **MiniBatchKMeans** centroids to `attn_centroids.bin` plus `attn_index.json`.

This is **not** a replacement for runtime attention (no RoPE, no Q, no softmax over
sequence positions). It is a **searchable summary** of “where token embeddings land
under K” for analysis / routing experiments.

Layout (alongside an existing browse vindex directory)
------------------------------------------------------
  <vindex_dir>/attn_centroids.bin
  <vindex_dir>/attn_index.json

Dependencies
------------
  pip install torch safetensors numpy scikit-learn
  Needs `vindex.py` on disk: same folder as this file, current working directory,
  or set **VINDEX_PY** to the full path of `vindex.py` (Colab-friendly when pasting one cell).

Env (notebook, empty argv)
--------------------------
  AVINDEX_EXTRACT=1     run extract into VINDEX_DIR (default ./.larql_colab/vindex_out)
  VINDEX_MODEL          HF id or local path (same as vindex.py)
  VINDEX_DIR            output vindex directory
  AVINDEX_CENTROIDS=256
  AVINDEX_VOCAB_SAMPLE=8000
  AVINDEX_PROBE=1       run an honest probe (needs model weights for k_proj)
  AVINDEX_PROBE_KV_HEAD  default 0
  VINDEX_PY              optional absolute path to vindex.py (when not importable as `vindex`)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
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
    """Search order for vindex.py when `import vindex` fails (single-cell Colab)."""
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
    """Load vindex.py by path; register sys.modules so @dataclass in vindex works."""
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
        "  • Set VINDEX_PY to the full path of vindex.py (file or directory containing it).\n"
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
    wk: "torch.Tensor",
    *,
    n_kv: int,
    hidden: int,
) -> tuple[Any, int]:
    """Return (wk_kv, head_dim_kv) with wk_kv shape [n_kv, head_dim_kv, hidden]."""
    import torch

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
) -> Path:
    """
    Build attn_centroids.bin + attn_index.json under ``out_dir`` (existing vindex folder OK).
    """
    from sklearn.cluster import MiniBatchKMeans

    vx = _import_vindex()
    import torch

    model_dir = model_dir.resolve()
    out_dir = out_dir.resolve()
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
        "avindex_version": 2,
        "kind": "k_proj_vocab_projection_centroids",
        "num_layers": num_layers,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv,
        "hidden_size": hidden,
        "vocab_sample_size": int(n_sample),
        "num_centroids": int(num_centroids),
        "dtype": "f32",
        "layers": [],
    }

    centroid_path = out_dir / "attn_centroids.bin"
    t_all = time.perf_counter()

    with open(centroid_path, "wb") as f_bin:
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

            for kv_h in range(n_kv):
                head_k = wk_kv[kv_h]
                projected = emb_s @ head_k.numpy().T
                k_use = min(int(num_centroids), int(projected.shape[0]))
                km = MiniBatchKMeans(
                    n_clusters=k_use,
                    batch_size=min(1024, projected.shape[0]),
                    n_init=3,
                    random_state=random_state,
                )
                km.fit(projected)
                centroids = np.ascontiguousarray(km.cluster_centers_, dtype=np.float32)
                offset = f_bin.tell()
                f_bin.write(centroids.tobytes(order="C"))
                layer_entry["kv_heads"].append(
                    {
                        "kv_head": kv_h,
                        "offset": int(offset),
                        "size_bytes": int(centroids.nbytes),
                        "shape": [int(centroids.shape[0]), int(centroids.shape[1])],
                    }
                )
            attn_index["layers"].append(layer_entry)
            _log(f"  layer {layer + 1}/{num_layers}  kv_heads={n_kv}  ({time.perf_counter() - t_l:.1f}s)")

    attn_index["head_dim_kv"] = attn_index["layers"][0]["head_dim_kv"] if num_layers else None
    (out_dir / "attn_index.json").write_text(json.dumps(attn_index, indent=2), encoding="utf-8")
    sz_mb = centroid_path.stat().st_size / (1024 * 1024)
    _log("=" * 60)
    _log(f"DONE  {centroid_path}  ({sz_mb:.2f} MiB)  total {time.perf_counter() - t_all:.1f}s")
    return out_dir


# ── read / score ──────────────────────────────────────────────────────────


class AvindexAttentionReader:
    """
    mmap-friendly sequential file; uses seek+read. Call ``close()`` or use as context manager.
    """

    def __init__(self, vindex_dir: Path) -> None:
        self.vindex_dir = vindex_dir.resolve()
        p_meta = self.vindex_dir / "attn_index.json"
        p_bin = self.vindex_dir / "attn_centroids.bin"
        if not p_meta.is_file():
            raise FileNotFoundError(f"Missing {p_meta} — run extract_attention_avindex first.")
        if not p_bin.is_file():
            raise FileNotFoundError(f"Missing {p_bin}")
        self.meta: dict[str, Any] = json.loads(p_meta.read_text(encoding="utf-8"))
        self._normalize_meta_layers()
        self._bin_path = p_bin
        self._fp = open(p_bin, "rb")
        self._cache: dict[tuple[int, int], np.ndarray] = {}

    def _normalize_meta_layers(self) -> None:
        """Coerce ``layers`` to a list[dict] and alias legacy ``heads`` / ``head`` → ``kv_heads``."""
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
        """Return per-KV-head centroid metadata (after ``_normalize_meta_layers``)."""
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

    def __enter__(self) -> AvindexAttentionReader:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def centroids(self, layer: int, kv_head: int) -> np.ndarray:
        key = (layer, kv_head)
        if key in self._cache:
            return self._cache[key]
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
        self._cache[key] = arr
        return arr

    def nearest_cosine(self, layer: int, kv_head: int, q: np.ndarray) -> tuple[int, float]:
        """Unit-norm cosine similarity → best centroid index and score in [-1, 1]."""
        q = np.asarray(q, dtype=np.float32).reshape(-1)
        C = self.centroids(layer, kv_head)
        if q.shape[0] != C.shape[1]:
            raise ValueError(f"query dim {q.shape[0]} != head_dim_kv {C.shape[1]}")
        qn = q / (np.linalg.norm(q) + 1e-8)
        Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)
        sims = Cn @ qn
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


def load_k_slice_torch(
    model_dir: Path,
    layer: int,
    kv_head: int,
    *,
    n_kv: int,
    hidden: int,
) -> Any:
    """Load k_proj once per call; prefer :class:`KProjWeightCache` in hot loops."""
    return KProjWeightCache(model_dir).k_slice(layer, kv_head, n_kv=n_kv, hidden=hidden)


class KProjWeightCache:
    """Single ``load_safetensors_dir``; cheap per-(layer, kv_head) k_proj slices."""

    def __init__(self, model_dir: Path) -> None:
        vx = _import_vindex()
        self._vx = vx
        self.model_dir = Path(model_dir).resolve()
        _log(f"KProjWeightCache: loading safetensors from {self.model_dir} …")
        t0 = time.perf_counter()
        self.sd = vx.load_safetensors_dir(self.model_dir)
        self.prefix = vx.detect_prefix(list(self.sd.keys()))
        _log(f"KProjWeightCache: ready in {time.perf_counter() - t0:.2f}s ({len(self.sd)} tensors)")

    def k_slice(self, layer: int, kv_head: int, *, n_kv: int, hidden: int) -> Any:
        import torch

        k_key = self._vx.find_key(
            self.sd,
            f"{self.prefix}layers.{layer}.self_attn.k_proj.weight",
            f"model.layers.{layer}.self_attn.k_proj.weight",
        )
        wk = self.sd[k_key].to(torch.float32)
        wk_kv, _ = _reshape_k_proj(wk, n_kv=n_kv, hidden=hidden)
        return wk_kv[kv_head].contiguous()


def _kv_heads_list_raw(layer_entry: dict[str, Any]) -> list[Any]:
    return list(layer_entry.get("kv_heads") or layer_entry.get("heads") or [])


def _n_kv_for_probe(attn: dict[str, Any], main: dict[str, Any], layer: int) -> int:
    """
    KV head count for k_proj reshape: prefer on-disk attn layout (matches centroids),
    then attn_index / index.json metadata.
    """
    layers = _attn_layers_normalized(attn)
    if layer < 0 or layer >= len(layers):
        raise IndexError(f"attn layer {layer} out of range (0..{len(layers) - 1})")
    n_from_blocks = len(_kv_heads_list_raw(layers[layer]))
    n_meta = int(attn.get("num_key_value_heads", 0) or attn.get("num_kv_heads", 0) or 0)
    mc = main.get("model_config") or {}
    n_index = int(mc.get("num_kv_heads", mc.get("num_key_value_heads", 0)) or 0)
    if n_from_blocks > 0:
        if n_meta > 0 and n_meta != n_from_blocks:
            _log(
                f"warning: attn_index num_key_value_heads={n_meta} != len(kv_heads)={n_from_blocks} "
                f"— using len(kv_heads) so probe matches centroids."
            )
        if n_index > 0 and n_index != n_from_blocks:
            _log(
                f"warning: index.json num_kv_heads={n_index} != attn len(kv_heads)={n_from_blocks} "
                f"— using attn layout."
            )
        return n_from_blocks
    if n_meta > 0:
        return n_meta
    if n_index > 0:
        return n_index
    return 1


def probe_last_token_k_space(
    vindex_dir: Path,
    model_dir: Path,
    *,
    prompt: str,
    layer: int,
    kv_head: int = 0,
    weight_cache: KProjWeightCache | None = None,
) -> dict[str, Any]:
    """
    Honest probe in the **same space as extract**: last prompt token embedding row,
    multiplied by K_h^T, then nearest centroid (cosine).

    Requires:
      - attn_index.json + attn_centroids.bin from this script
      - embeddings.bin + index.json from vindex.py
      - model_dir safetensors for k_proj (same checkpoint as extract)
    """
    from transformers import AutoTokenizer

    vindex_dir = vindex_dir.resolve()
    model_dir = model_dir.resolve()
    main = json.loads((vindex_dir / "index.json").read_text(encoding="utf-8"))
    attn = json.loads((vindex_dir / "attn_index.json").read_text(encoding="utf-8"))
    model_id = str(main.get("model", "")).strip() or None
    vocab = int(main["vocab_size"])
    hidden = int(main["hidden_size"])
    n_kv = _n_kv_for_probe(attn, main, layer)

    tok = AutoTokenizer.from_pretrained(model_id or str(model_dir), trust_remote_code=True)
    ids = tok.encode(prompt, add_special_tokens=False)
    if not ids:
        raise ValueError("empty tokenization")
    last = int(ids[-1])

    emb = load_embeddings_f32(vindex_dir, vocab, hidden)
    e = emb[last].astype(np.float32)

    import torch

    cache = weight_cache or KProjWeightCache(model_dir)
    Kh = cache.k_slice(layer, kv_head, n_kv=n_kv, hidden=hidden)
    q = (torch.from_numpy(e) @ Kh.T).numpy()

    with AvindexAttentionReader(vindex_dir) as r:
        cid, score = r.nearest_cosine(layer, kv_head, q)

    return {
        "prompt": prompt,
        "last_token_id": last,
        "last_token_piece": tok.decode([last]),
        "layer": layer,
        "kv_head": kv_head,
        "n_kv_used": n_kv,
        "centroid_id": cid,
        "cosine": score,
    }


def scan_centroid_path(
    vindex_dir: Path,
    model_dir: Path,
    *,
    prompt: str,
    weight_cache: KProjWeightCache | None = None,
    layers_only: list[int] | None = None,
) -> dict[str, Any]:
    """
    For **each** (layer, kv_head): project last-token embedding through ``k_proj`` slice,
    then nearest centroid (cosine). Produces a linear ``steps`` list you can print or
    filter (this is diagnostic geometry, not full attention).
    """
    from transformers import AutoTokenizer

    import torch

    vindex_dir = vindex_dir.resolve()
    model_dir = model_dir.resolve()
    main = json.loads((vindex_dir / "index.json").read_text(encoding="utf-8"))
    attn = json.loads((vindex_dir / "attn_index.json").read_text(encoding="utf-8"))
    model_id = str(main.get("model", "")).strip() or None
    vocab = int(main["vocab_size"])
    hidden = int(main["hidden_size"])
    layers_arr = _attn_layers_normalized(attn)
    n_layers = len(layers_arr)

    tok = AutoTokenizer.from_pretrained(model_id or str(model_dir), trust_remote_code=True)
    ids = tok.encode(prompt, add_special_tokens=False)
    if not ids:
        raise ValueError("empty tokenization")
    last = int(ids[-1])
    emb = load_embeddings_f32(vindex_dir, vocab, hidden)
    e = emb[last].astype(np.float32)

    cache = weight_cache or KProjWeightCache(model_dir)
    steps: list[dict[str, Any]] = []
    layer_iter = layers_only if layers_only is not None else range(n_layers)
    with AvindexAttentionReader(vindex_dir) as r:
        for layer in layer_iter:
            if layer < 0 or layer >= n_layers:
                continue
            n_kv = _n_kv_for_probe(attn, main, layer)
            for kv_h in range(n_kv):
                Kh = cache.k_slice(layer, kv_h, n_kv=n_kv, hidden=hidden)
                q = (torch.from_numpy(e) @ Kh.T).numpy()
                cid, score = r.nearest_cosine(layer, kv_h, q)
                steps.append(
                    {
                        "layer": int(layer),
                        "kv_head": int(kv_h),
                        "centroid_id": int(cid),
                        "cosine": float(score),
                    }
                )
    return {
        "kind": "a_vindex_centroid_hit_per_kv_head",
        "prompt": prompt,
        "last_token_id": last,
        "last_token_piece": tok.decode([last]),
        "steps": steps,
        "num_steps": len(steps),
    }


def _transformers_skip_torchvision_for_text_models() -> None:
    """
    Qwen3 causal LM import can transitively import ``torchvision`` (vision utils).
    On Colab, mismatched torch/torchvision CUDA wheels raise before the model loads.
    For **text-only** use we force torchvision to appear unavailable so that path is skipped.

    Set ``AVINDEX_ALLOW_TORCHVISION=1`` to skip this patch (use real torchvision checks).
    """
    if os.environ.get("AVINDEX_ALLOW_TORCHVISION", "").strip().lower() in ("1", "true", "yes", "on"):
        return
    try:
        import transformers.utils.import_utils as iu

        if hasattr(iu, "is_torchvision_available"):
            iu.is_torchvision_available = lambda: False  # type: ignore[assignment]
    except Exception:
        return
    try:
        import transformers.utils as tu

        if hasattr(tu, "is_torchvision_available"):
            tu.is_torchvision_available = lambda: False  # type: ignore[assignment]
    except Exception:
        pass


def hf_next_token_topk(
    model_ref: str,
    prompt: str,
    *,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """
    One forward pass, softmax at last position → top-``k`` **token ids** (same spirit
    as ``vindex-infer`` HF path). This is true next-token **LM** ranking, not A‑Vindex
    centroid geometry.
    """
    _transformers_skip_torchvision_for_text_models()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise RuntimeError("pip install transformers torch accelerate") from e

    use_cpu = os.environ.get("AVINDEX_HF_CPU", "").strip().lower() in ("1", "true", "yes", "on")
    device_map: str | dict[str, str] = "cpu" if use_cpu else "auto"
    dtype = torch.float32 if use_cpu else torch.float16

    _log(f"hf_next_token_topk: loading model {model_ref!r} (device_map={device_map!r}) …")
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    _log(f"hf_next_token_topk: weights ready in {time.perf_counter() - t0:.1f}s")

    inputs = tok(prompt, return_tensors="pt")
    dev = next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    t1 = time.perf_counter()
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits[0, -1, :].float()
    probs = torch.softmax(logits, dim=-1)
    k = min(int(top_k), probs.numel())
    vals, inds = probs.topk(k)
    rows: list[dict[str, Any]] = []
    for rank, (p, tid) in enumerate(zip(vals.tolist(), inds.tolist()), start=1):
        tid = int(tid)
        s = tok.decode([tid], skip_special_tokens=True)
        rows.append(
            {
                "rank": rank,
                "token_id": tid,
                "token": s if s.strip() else f"<id {tid}>",
                "prob": float(p),
            }
        )
    _log(f"hf_next_token_topk: forward+topk in {time.perf_counter() - t1:.2f}s")
    return rows


def _default_probe_layer(main_index: dict[str, Any]) -> int:
    bands = main_index.get("layer_bands") or {}
    kn = bands.get("knowledge")
    if isinstance(kn, list) and len(kn) == 2:
        a, b = int(kn[0]), int(kn[1])
        return (a + b) // 2
    nl = int(main_index.get("num_layers", 1))
    return max(0, nl // 2)


def run_default_cell() -> None:
    vx = _import_vindex()
    base = vx.work_base()
    model_id = os.environ.get("VINDEX_MODEL", "Qwen/Qwen3-0.6B").strip()
    vdir = Path(os.environ.get("VINDEX_DIR", str(base / "vindex_out"))).expanduser().resolve()
    mdir = vx.resolve_model_dir(model_id, base / "hf_models")

    if not _env_truthy("AVINDEX_EXTRACT") and not _env_truthy("AVINDEX_PROBE"):
        _log("nothing to do — set AVINDEX_EXTRACT=1 and/or AVINDEX_PROBE=1")
        return

    if _env_truthy("AVINDEX_EXTRACT"):
        k = int(os.environ.get("AVINDEX_CENTROIDS", "256"))
        vs = int(os.environ.get("AVINDEX_VOCAB_SAMPLE", "8000"))
        extract_attention_avindex(mdir, vdir, num_centroids=k, vocab_sample=vs)

    if _env_truthy("AVINDEX_PROBE"):
        main = json.loads((vdir / "index.json").read_text(encoding="utf-8"))
        layer = int(os.environ.get("AVINDEX_PROBE_LAYER", str(_default_probe_layer(main))))
        prompt = os.environ.get("AVINDEX_PROMPT", "The capital of France is").strip()
        kv = int(os.environ.get("AVINDEX_PROBE_KV_HEAD", "0"))
        cache = KProjWeightCache(mdir)
        out = probe_last_token_k_space(vdir, mdir, prompt=prompt, layer=layer, kv_head=kv, weight_cache=cache)
        _log("PROBE (embedding × K_h^T vs centroids, last token only)")
        for k, v in out.items():
            _log(f"  {k}: {v!r}")


def main_cli() -> None:
    vx = _import_vindex()
    ap = argparse.ArgumentParser(description="Attention centroid sidecar for Colab vindex layout")
    ap.add_argument("command", choices=["extract", "probe"], help="extract centroids or run probe")
    ap.add_argument("--model-dir", type=Path, default=None, help="HF model folder (safetensors + config)")
    ap.add_argument("--vindex", type=Path, default=None, help="vindex directory (attn_* + embeddings.bin)")
    ap.add_argument("--centroids", type=int, default=256)
    ap.add_argument("--vocab-sample", type=int, default=8000)
    ap.add_argument("--prompt", "-p", default="The capital of France is")
    ap.add_argument("--layer", type=int, default=-1, help="-1 = middle/knowledge band from index.json")
    ap.add_argument("--kv-head", type=int, default=0)
    args, rest = ap.parse_known_args(user_argv())
    if rest:
        _log(f"warning: ignored argv: {rest}")

    base = vx.work_base()
    vdir = (args.vindex or Path(os.environ.get("VINDEX_DIR", str(base / "vindex_out")))).resolve()

    if args.command == "extract":
        mid = os.environ.get("VINDEX_MODEL", "Qwen/Qwen3-0.6B").strip()
        mdir = (args.model_dir or vx.resolve_model_dir(mid, base / "hf_models")).resolve()
        extract_attention_avindex(
            mdir,
            vdir,
            num_centroids=args.centroids,
            vocab_sample=args.vocab_sample,
        )
        return

    # probe
    mid = os.environ.get("VINDEX_MODEL", "Qwen/Qwen3-0.6B").strip()
    mdir = (args.model_dir or vx.resolve_model_dir(mid, base / "hf_models")).resolve()
    main = json.loads((vdir / "index.json").read_text(encoding="utf-8"))
    layer = args.layer if args.layer >= 0 else _default_probe_layer(main)
    cache = KProjWeightCache(mdir)
    out = probe_last_token_k_space(
        vdir, mdir, prompt=args.prompt, layer=layer, kv_head=args.kv_head, weight_cache=cache
    )
    _log("PROBE result:")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    if not user_argv():
        _log("entry: notebook default (empty argv)")
        run_default_cell()
    else:
        _log(f"entry: CLI argv={user_argv()!r}")
        main_cli()
