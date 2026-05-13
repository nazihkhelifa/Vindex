#!/usr/bin/env python3
"""
Attention-context sidecar for **prompt-wide** residual vectors.

Writes ``vindex_attn_ctx_index.json`` + ``vindex_attn_ctx_weights.bin`` next to
the browse vindex. ``vindex-infer.VindexWalkRunner`` loads these files (when
present) and replaces the naive ``embed[last_token]`` query with the **last
position residual after a full multi-head causal attention stack** (still no
FFN between layers — pre-norm residual += attn only).

This answers: *what does the model see at the final position after mixing
France / capital / of / is through attention?* — up to the fidelity of a
numpy attention forward with stored f32 weights (no MoE, no sliding-window
special cases beyond a simple causal mask).

``k_proj`` slices are **not** duplicated here; they are read from the existing
``attn_k_proj_weights.bin`` + ``attn_index.json`` produced by
``avindex_attention.extract_attention_avindex``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

CTX_INDEX = "vindex_attn_ctx_index.json"
CTX_WEIGHTS = "vindex_attn_ctx_weights.bin"


def _rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """x: [..., hidden], weight: [hidden] → RMSNorm (Llama / Qwen style)."""
    x = x.astype(np.float32, copy=False)
    w = weight.astype(np.float32, copy=False)
    var = np.mean(x * x, axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    return (x * inv) * w


def _rotate_half(x: np.ndarray) -> np.ndarray:
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return np.concatenate([-x2, x1], axis=-1)


def _apply_rope(
    q: np.ndarray,
    k: np.ndarray,
    positions: np.ndarray,
    *,
    rope_theta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    RoPE on the last dimension. q: [n_heads, T, head_dim], k: [n_kv, T, head_dim].
    positions: [T] int.
    """
    d = int(q.shape[-1])
    half = d // 2
    t = int(positions.shape[0])
    inv = 1.0 / (rope_theta ** (np.arange(0, d, 2, dtype=np.float64) / d))
    freqs = np.outer(positions.astype(np.float64), inv)  # [T, half]
    cos = np.cos(freqs).astype(np.float32)
    sin = np.sin(freqs).astype(np.float32)
    cos_f = np.repeat(cos, 2, axis=-1)  # [T, d]
    sin_f = np.repeat(sin, 2, axis=-1)

    def rope(x: np.ndarray) -> np.ndarray:
        # x [nh, T, d]
        c = cos_f[None, :, :]
        s = sin_f[None, :, :]
        return x * c + _rotate_half(x) * s

    return rope(q), rope(k)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x, dtype=np.float32)
    return ex / (np.sum(ex, axis=axis, keepdims=True) + 1e-30)


class KProjReader:
    """Minimal reader for ``attn_k_proj_weights.bin`` using ``attn_index.json``."""

    def __init__(self, vindex_dir: Path) -> None:
        self.vdir = Path(vindex_dir).resolve()
        meta = json.loads((self.vdir / "attn_index.json").read_text(encoding="utf-8"))
        raw = meta.get("k_proj_layers")
        if not raw:
            raise FileNotFoundError("attn_index.json missing 'k_proj_layers' — re-run avindex extract")
        self.layers_meta = list(raw)
        self._fp = open(self.vdir / "attn_k_proj_weights.bin", "rb")
        self._cache: dict[tuple[int, int], np.ndarray] = {}

    def close(self) -> None:
        self._fp.close()

    def k_slice(self, layer: int, kv_head: int) -> np.ndarray:
        key = (layer, kv_head)
        if key in self._cache:
            return self._cache[key]
        hm = self.layers_meta[layer]["kv_heads"][kv_head]
        self._fp.seek(int(hm["offset"]))
        raw = self._fp.read(int(hm["size_bytes"]))
        sh = hm["shape"]
        arr = np.frombuffer(raw, dtype=np.float32).reshape(int(sh[0]), int(sh[1]))
        self._cache[key] = arr
        return arr


class AttnContextForward:
    """
    Loads attention-context weights + k_proj reader, runs ``L`` layers of
    pre-norm self-attention (no FFN), returns ``h`` shape ``[T, hidden]``.
    """

    def __init__(self, vindex_dir: Path) -> None:
        self.vdir = Path(vindex_dir).resolve()
        p_idx = self.vdir / CTX_INDEX
        p_w = self.vdir / CTX_WEIGHTS
        if not p_idx.is_file() or not p_w.is_file():
            raise FileNotFoundError(f"Missing {CTX_INDEX} or {CTX_WEIGHTS}")
        self.meta: dict[str, Any] = json.loads(p_idx.read_text(encoding="utf-8"))
        self._fp = open(p_w, "rb")
        self.hidden = int(self.meta["hidden_size"])
        self.n_heads = int(self.meta["num_attention_heads"])
        self.n_kv = int(self.meta["num_key_value_heads"])
        self.head_dim = int(self.meta["head_dim"])
        self.rope_theta = float(self.meta.get("rope_theta", 10_000.0))
        self.rms_eps = float(self.meta.get("rms_eps", 1e-6))
        self.n_rep = self.n_heads // self.n_kv
        if self.n_heads % self.n_kv != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.layers = list(self.meta["layers"])
        self._kproj = KProjReader(self.vdir)

    def close(self) -> None:
        self._fp.close()
        self._kproj.close()

    def _read_f32(self, offset: int, n_floats: int) -> np.ndarray:
        self._fp.seek(offset)
        raw = self._fp.read(n_floats * 4)
        return np.frombuffer(raw, dtype=np.float32)

    def last_residual_after_attention(
        self,
        embed: np.ndarray,
        embed_scale: float,
        token_ids: list[int],
    ) -> np.ndarray:
        """
        ``embed``: ``[vocab, hidden]`` mmap or ndarray.
        Returns ``h[-1]`` after ``num_layers`` attention sublayers (no FFN).
        """
        ids = [int(t) for t in token_ids]
        t_len = len(ids)
        if t_len == 0:
            raise ValueError("empty token_ids")

        ids_arr = np.array(ids, dtype=np.int64)
        h = np.asarray(embed[ids_arr], dtype=np.float32) * float(embed_scale)
        positions = np.arange(t_len, dtype=np.int32)

        for li, entry in enumerate(self.layers):
            layer = int(entry["layer"])
            off = int(entry["weights_offset"])
            nbytes = int(entry["weights_size_bytes"])
            blob = self._read_f32(off, nbytes // 4)

            pos = 0
            hd = self.head_dim
            nh = self.n_heads
            nkv = self.n_kv
            hsz = self.hidden

            def take(n: int) -> np.ndarray:
                nonlocal pos
                arr = blob[pos : pos + n].reshape(-1)
                pos += n
                return arr

            input_ln = take(hsz)
            q_w = take(nh * hd * hsz).reshape(nh * hd, hsz)
            v_w = take(nkv * hd * hsz).reshape(nkv * hd, hsz)
            o_w = take(hsz * nh * hd).reshape(hsz, nh * hd)
            has_qk = bool(entry.get("has_qk_norm", False))
            qn = take(hd) if has_qk else None
            kn = take(hd) if has_qk else None
            if pos != len(blob):
                raise ValueError(f"layer {layer}: weight blob parse mismatch {pos} vs {len(blob)}")

            x = _rms_norm(h, input_ln, self.rms_eps)

            q_flat = x @ q_w.T
            k_flat = np.zeros((t_len, nkv * hd), dtype=np.float32)
            v_flat = x @ v_w.T
            for kv in range(nkv):
                Kh = self._kproj.k_slice(layer, kv)
                k_flat[:, kv * hd : (kv + 1) * hd] = x @ Kh.T

            q = q_flat.reshape(t_len, nh, hd).transpose(1, 0, 2)
            k = k_flat.reshape(t_len, nkv, hd).transpose(1, 0, 2)
            v = v_flat.reshape(t_len, nkv, hd).transpose(1, 0, 2)

            if qn is not None and kn is not None:
                for hi in range(nh):
                    q[hi] = _rms_norm(q[hi], qn, self.rms_eps)
                for ki in range(nkv):
                    k[ki] = _rms_norm(k[ki], kn, self.rms_eps)

            q, k = _apply_rope(q, k, positions, rope_theta=self.rope_theta)

            k_exp = np.repeat(k, self.n_rep, axis=0)
            v_exp = np.repeat(v, self.n_rep, axis=0)

            out_heads: list[np.ndarray] = []
            scale = 1.0 / np.sqrt(hd)
            for hi in range(nh):
                qh = q[hi]
                kh = k_exp[hi]
                vh = v_exp[hi]
                scores = (qh @ kh.transpose(1, 0)) * scale
                mask = np.triu(np.ones((t_len, t_len), dtype=np.float32), k=1) * (-1e9)
                scores = scores + mask
                attn = _softmax(scores, axis=-1)
                oh = attn @ vh
                out_heads.append(oh)
            merged = np.concatenate(out_heads, axis=-1)
            attn_out = merged @ o_w.T.astype(np.float32)
            h = h + attn_out

        return h[-1].astype(np.float32, copy=False)


def extract_attn_context_sidecar(
    out_dir: Path,
    *,
    sd: dict[str, Any],
    prefix: str,
    cfg: dict[str, Any],
    vx: Any,
) -> Path:
    """
    Append ``vindex_attn_ctx_*`` files under ``out_dir``. Requires existing
    ``attn_index.json`` + ``attn_k_proj_weights.bin`` (same extract run).
    """
    import torch

    out_dir = Path(out_dir).resolve()
    if not (out_dir / "attn_index.json").is_file():
        raise FileNotFoundError("attn_index.json missing — run centroid+k_proj extract first")

    num_layers = int(cfg["num_hidden_layers"])
    hidden = int(cfg["hidden_size"])
    n_heads = int(cfg.get("num_attention_heads", cfg.get("num_heads", 1)))
    n_kv = int(cfg.get("num_key_value_heads", n_heads))
    head_dim = hidden // n_heads
    rope_theta = float(cfg.get("rope_theta", cfg.get("rope_embedding_base", 10_000.0)))
    rms_eps = float(cfg.get("rms_norm_eps", 1e-6))

    layers_json: list[dict[str, Any]] = []
    weights_path = out_dir / CTX_WEIGHTS
    import time as _time

    t0 = _time.perf_counter()
    with open(weights_path, "wb") as f_out:
        for layer in range(num_layers):
            def get_w(short: str) -> Any:
                k = vx.find_key(
                    sd,
                    f"{prefix}layers.{layer}.self_attn.{short}",
                    f"model.layers.{layer}.self_attn.{short}",
                )
                return vx.get_weight(sd, k)

            ln_k = vx.find_key(
                sd,
                f"{prefix}layers.{layer}.input_layernorm.weight",
                f"model.layers.{layer}.input_layernorm.weight",
            )
            w_ln = vx.get_weight(sd, ln_k).to(torch.float32).numpy().astype(np.float32, copy=False)

            wq = get_w("q_proj.weight").to(torch.float32).numpy().astype(np.float32, copy=False)
            wv = get_w("v_proj.weight").to(torch.float32).numpy().astype(np.float32, copy=False)
            wo = get_w("o_proj.weight").to(torch.float32).numpy().astype(np.float32, copy=False)

            if wq.shape != (n_heads * head_dim, hidden):
                raise ValueError(f"L{layer} q_proj {wq.shape} expect ({n_heads * head_dim},{hidden})")
            if wv.shape != (n_kv * head_dim, hidden):
                raise ValueError(f"L{layer} v_proj {wv.shape}")
            if wo.shape != (hidden, n_heads * head_dim):
                raise ValueError(f"L{layer} o_proj {wo.shape}")

            w_qn = w_kn = None
            try:
                qnk = vx.find_key(
                    sd,
                    f"{prefix}layers.{layer}.self_attn.q_norm.weight",
                    f"model.layers.{layer}.self_attn.q_norm.weight",
                )
                knk = vx.find_key(
                    sd,
                    f"{prefix}layers.{layer}.self_attn.k_norm.weight",
                    f"model.layers.{layer}.self_attn.k_norm.weight",
                )
                w_qn = vx.get_weight(sd, qnk)
                w_kn = vx.get_weight(sd, knk)
            except KeyError:
                pass
            has_qk = w_qn is not None and w_kn is not None
            if has_qk:
                qn = w_qn.to(torch.float32).numpy().astype(np.float32, copy=False).reshape(-1)
                kn = w_kn.to(torch.float32).numpy().astype(np.float32, copy=False).reshape(-1)
                if qn.size != head_dim or kn.size != head_dim:
                    raise ValueError(f"L{layer} q_norm/k_norm shape mismatch")

            blobs: list[np.ndarray] = [
                w_ln.reshape(-1),
                wq.reshape(-1),
                wv.reshape(-1),
                wo.reshape(-1),
            ]
            if has_qk:
                blobs.extend([qn, kn])
            chunk = np.concatenate([b.astype(np.float32, copy=False) for b in blobs])
            off = f_out.tell()
            f_out.write(chunk.tobytes(order="C"))
            nbytes = int(chunk.nbytes)
            layers_json.append(
                {
                    "layer": layer,
                    "weights_offset": int(off),
                    "weights_size_bytes": int(nbytes),
                    "has_qk_norm": bool(has_qk),
                }
            )
    index: dict[str, Any] = {
        "version": 1,
        "hidden_size": hidden,
        "num_layers": num_layers,
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv,
        "head_dim": head_dim,
        "rope_theta": rope_theta,
        "rms_eps": rms_eps,
        "layers": layers_json,
    }
    (out_dir / CTX_INDEX).write_text(json.dumps(index, indent=2), encoding="utf-8")
    sz_mb = weights_path.stat().st_size / (1024 * 1024)
    print(
        f"[attn-ctx] wrote {CTX_INDEX} + {CTX_WEIGHTS} ({sz_mb:.1f} MiB) in "
        f"{_time.perf_counter() - t0:.1f}s",
        flush=True,
    )
    return out_dir
