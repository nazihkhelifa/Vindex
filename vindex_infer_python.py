#!/usr/bin/env python3
"""
Inférence Vindex en Python — port fidèle de vindex-infer (Rust) :
  vindex.rs + inference.rs

Dépendances : numpy
Optionnel : tokenizers (pip install tokenizers) pour --prompt

Exemple :
  python vindex_infer_python.py --vindex ./gemma3-4b.vindex --token-ids 818,5279,529,7001,563
  python vindex_infer_python.py --vindex ./gemma3-4b.vindex -p "The capital of France is"

Progression : logs sur stderr (INFO par defaut).  -v = DEBUG,  -q = WARNING.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config / chargement Vindex (équivalent vindex.rs)
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    head_dim: Optional[int] = None
    num_q_heads: Optional[int] = None
    num_kv_heads: Optional[int] = None
    rope_base: Optional[float] = None
    sliding_window: Optional[int] = None


@dataclass
class VindexConfig:
    version: int
    model: str
    family: str
    num_layers: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    embed_scale: float
    dtype: str
    model_config: Optional[ModelConfig] = None

    @staticmethod
    def from_json(obj: dict[str, Any]) -> "VindexConfig":
        mc = obj.get("model_config")
        mcfg: Optional[ModelConfig] = None
        if isinstance(mc, dict):
            mcfg = ModelConfig(
                head_dim=mc.get("head_dim"),
                num_q_heads=mc.get("num_q_heads"),
                num_kv_heads=mc.get("num_kv_heads"),
                rope_base=mc.get("rope_base"),
                sliding_window=mc.get("sliding_window"),
            )
        return VindexConfig(
            version=int(obj["version"]),
            model=str(obj.get("model", "")),
            family=str(obj.get("family", "")),
            num_layers=int(obj["num_layers"]),
            hidden_size=int(obj["hidden_size"]),
            intermediate_size=int(obj["intermediate_size"]),
            vocab_size=int(obj["vocab_size"]),
            embed_scale=float(obj["embed_scale"]),
            dtype=str(obj.get("dtype", "f16")),
            model_config=mcfg,
        )


class Vindex:
    """Charge index.json + binaires f16 en mémoire mappée (comme memmap2 en Rust)."""

    def __init__(self, root: Path, cfg: VindexConfig) -> None:
        self.root = root
        self.config = cfg
        h = cfg.hidden_size
        nl = cfg.num_layers
        inter = cfg.intermediate_size
        v = cfg.vocab_size

        def open_mm(name: str, shape: Tuple[int, ...]) -> np.memmap:
            p = root / name
            return np.memmap(p, dtype=np.float16, mode="r", shape=shape)

        self.embed = open_mm("embeddings.bin", (v, h))
        self.gate = open_mm("gate_vectors.bin", (nl, inter, h))
        up_path = root / "up_weights.bin"
        down_path = root / "down_weights.bin"
        attn_path = root / "attn_weights.bin"
        norms_path = root / "norms.bin"
        self.up: Optional[np.memmap] = (
            np.memmap(up_path, dtype=np.float16, mode="r", shape=(nl, inter, h))
            if up_path.is_file()
            else None
        )
        self.down: Optional[np.memmap] = (
            np.memmap(down_path, dtype=np.float16, mode="r", shape=(nl, h, inter))
            if down_path.is_file()
            else None
        )
        nq, nkv, hd = self.attn_dims()
        q_sz = nq * hd * h
        k_sz = nkv * hd * h
        v_sz = k_sz
        o_sz = h * nq * hd
        self._attn_layer_floats = q_sz + k_sz + v_sz + o_sz + hd + hd
        self.attn: Optional[np.memmap] = None
        if attn_path.is_file():
            self.attn = np.memmap(attn_path, dtype=np.float16, mode="r")
        self.norms: Optional[np.memmap] = None
        if norms_path.is_file():
            # 4 normes par couche + 1 finale : (nl * 4 + 1) * h
            self.norms = np.memmap(norms_path, dtype=np.float16, mode="r", shape=(nl * 4 + 1, h))

        log.debug(
            "Mmap: embed=%s gate=%s up=%s down=%s attn=%s norms=%s",
            _mm_bytes(self.embed),
            _mm_bytes(self.gate),
            _mm_bytes(self.up) if self.up is not None else "absent",
            _mm_bytes(self.down) if self.down is not None else "absent",
            _mm_bytes(self.attn) if self.attn is not None else "absent",
            _mm_bytes(self.norms) if self.norms is not None else "absent",
        )

    @classmethod
    def load(cls, path: str | Path) -> "Vindex":
        root = Path(path)
        log.info("Chargement Vindex depuis %s", root.resolve())
        raw = json.loads((root / "index.json").read_text(encoding="utf-8"))
        v = cls(root, VindexConfig.from_json(raw))
        log.info(
            "Vindex pret: family=%s dtype=%s attn_dims nq=%d nkv=%d hd=%d",
            v.config.family,
            v.config.dtype,
            *v.attn_dims(),
        )
        return v

    def attn_dims(self) -> Tuple[int, int, int]:
        mc = self.config.model_config
        nq = mc.num_q_heads if mc and mc.num_q_heads is not None else 8
        nkv = mc.num_kv_heads if mc and mc.num_kv_heads is not None else 4
        hd = mc.head_dim if mc and mc.head_dim is not None else 256
        return int(nq), int(nkv), int(hd)

    def embedding_f32(self, tid: int) -> np.ndarray:
        return self.embed[tid].astype(np.float32, copy=False)

    def gate_matvec(self, layer: int, x: np.ndarray) -> np.ndarray:
        # (inter, h) @ (h,) -> (inter,)
        w = self.gate[layer].astype(np.float32, copy=False)
        return w @ x.astype(np.float32, copy=False)

    def up_matvec(self, layer: int, x: np.ndarray) -> np.ndarray:
        if self.up is None:
            return np.zeros(self.config.intermediate_size, dtype=np.float32)
        w = self.up[layer].astype(np.float32, copy=False)
        return w @ x.astype(np.float32, copy=False)

    def down_matvec(self, layer: int, act: np.ndarray) -> np.ndarray:
        if self.down is None:
            return np.zeros(self.config.hidden_size, dtype=np.float32)
        w = self.down[layer].astype(np.float32, copy=False)
        return w @ act.astype(np.float32, copy=False)

    def attn_layer_views(self, layer: int) -> Optional[dict[str, np.ndarray]]:
        if self.attn is None:
            return None
        h = self.config.hidden_size
        nq, nkv, hd = self.attn_dims()
        q_sz = nq * hd * h
        k_sz = nkv * hd * h
        v_sz = k_sz
        o_sz = h * nq * hd
        off = layer * self._attn_layer_floats
        end = off + self._attn_layer_floats
        if end > self.attn.shape[0]:
            return None
        sl = self.attn[off:end].astype(np.float32)
        i = 0
        w_q = sl[i : i + q_sz].reshape(nq * hd, h)
        i += q_sz
        w_k = sl[i : i + k_sz].reshape(nkv * hd, h)
        i += k_sz
        w_v = sl[i : i + v_sz].reshape(nkv * hd, h)
        i += v_sz
        w_o = sl[i : i + o_sz].reshape(h, nq * hd)
        i += o_sz
        q_norm = sl[i : i + hd]
        i += hd
        k_norm = sl[i : i + hd]
        return {"w_q": w_q, "w_k": w_k, "w_v": w_v, "w_o": w_o, "q_norm": q_norm, "k_norm": k_norm}

    def norm_weights(self, layer: int, idx: int) -> np.ndarray:
        if self.norms is None:
            return np.zeros(self.config.hidden_size, dtype=np.float32)
        return self.norms[layer * 4 + idx].astype(np.float32, copy=False)

    def final_norm(self) -> np.ndarray:
        if self.norms is None:
            return np.zeros(self.config.hidden_size, dtype=np.float32)
        return self.norms[self.config.num_layers * 4].astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Petits noyaux numériques (équivalent inference.rs)
# ---------------------------------------------------------------------------


def rms_norm_1(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    g = g.astype(np.float32, copy=False)
    n = x.shape[0]
    rms = math.sqrt(float(np.sum(x.astype(np.float64) ** 2)) / n + 1e-6)
    return (x / rms) * (1.0 + g)


def rms_norm_qk(h: np.ndarray, g: np.ndarray) -> None:
    n = h.shape[0]
    rms = math.sqrt(float(np.sum(h.astype(np.float64) ** 2)) / n + 1e-6)
    h[:] = (h / rms) * (1.0 + g)


def apply_rope_hf(h: np.ndarray, pos: float, base: float, hd: int) -> None:
    half = hd // 2
    for i in range(half):
        f = pos / (base ** (2.0 * i / hd))
        c, s = math.cos(f), math.sin(f)
        x1, x2 = float(h[i]), float(h[i + half])
        h[i] = x1 * c - x2 * s
        h[i + half] = x2 * c + x1 * s


def gelu_tanh(x: float) -> float:
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))


def gelu_tanh_vec(v: np.ndarray) -> np.ndarray:
    x = v.astype(np.float64)
    return (0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))).astype(np.float32)


# ---------------------------------------------------------------------------
# Résultat + inférence
# ---------------------------------------------------------------------------


@dataclass
class InferenceResult:
    top_k: List[Tuple[int, float]]
    total_ms: float
    layer_ms: float
    logits_ms: float


def _mm_bytes(a: Union[np.memmap, np.ndarray]) -> str:
    return f"{a.nbytes / 1e9:.2f} Go"


def infer(
    vindex: Vindex,
    token_ids: Sequence[int],
    top_k: int = 10,
    *,
    skip_attention: bool = False,
    logits_chunk: int = 4096,
) -> InferenceResult:
    """Forward complet ou FFN-only (skip_attention=True), aligné sur inference.rs."""
    t0 = time.perf_counter()
    cfg = vindex.config
    h = cfg.hidden_size
    nl = cfg.num_layers
    nq, nkv, hd = vindex.attn_dims()
    groups = nq // nkv
    attn_scale = 1.0 / math.sqrt(hd)
    scale = cfg.embed_scale

    log.info(
        "Forward: %d token(s), %d couches, hidden=%d, attention=%s",
        len(token_ids),
        nl,
        h,
        "desactivee (ffn-only)" if skip_attention else "activee",
    )

    t_emb = time.perf_counter()
    residuals: List[np.ndarray] = [
        (vindex.embedding_f32(int(tid)) * scale).astype(np.float32) for tid in token_ids
    ]
    nt = len(residuals)
    log.info(
        "Embeddings initiaux (echelle %.4f) en %.2f ms",
        scale,
        (time.perf_counter() - t_emb) * 1000.0,
    )

    t_layer = time.perf_counter()
    for layer in range(nl):
        pfln = vindex.norm_weights(layer, 2)
        pfnl = vindex.norm_weights(layer, 3)
        is_global = (layer + 1) % 6 == 0
        rb, rf = (1e6, 8.0) if is_global else (1e4, 1.0)
        t_l0 = time.perf_counter()

        if not skip_attention:
            iln = vindex.norm_weights(layer, 0)
            paln = vindex.norm_weights(layer, 1)
            attn = vindex.attn_layer_views(layer)
            if attn is None:
                log.warning(
                    "Couche %d/%d: poids attention absents ou tronques — couche entiere sautee (comme Rust)",
                    layer + 1,
                    nl,
                )
                # Comme Rust : continue la boucle couche (pas de FFN cette couche)
                continue

            w_q, w_k, w_v, w_o = attn["w_q"], attn["w_k"], attn["w_v"], attn["w_o"]
            q_norm, k_norm = attn["q_norm"], attn["k_norm"]

            normed = [rms_norm_1(residuals[tok], iln) for tok in range(nt)]
            all_q: List[np.ndarray] = []
            all_k: List[np.ndarray] = []
            all_v: List[np.ndarray] = []
            for tok in range(nt):
                x = normed[tok].astype(np.float32, copy=False)
                q = (w_q @ x).copy()
                k = (w_k @ x).copy()
                vv = w_v @ x
                for hi in range(nq):
                    sl = slice(hi * hd, (hi + 1) * hd)
                    rms_norm_qk(q[sl], q_norm)
                for hi in range(nkv):
                    sl = slice(hi * hd, (hi + 1) * hd)
                    rms_norm_qk(k[sl], k_norm)
                pos = tok / rf
                for hi in range(nq):
                    sl = slice(hi * hd, (hi + 1) * hd)
                    apply_rope_hf(q[sl], pos, rb, hd)
                for hi in range(nkv):
                    sl = slice(hi * hd, (hi + 1) * hd)
                    apply_rope_hf(k[sl], pos, rb, hd)
                all_q.append(q)
                all_k.append(k)
                all_v.append(vv)

            for tok in range(nt):
                ho = np.zeros(nq * hd, dtype=np.float32)
                for hi in range(nq):
                    kv_hi = hi // groups
                    qs = hi * hd
                    ks = kv_hi * hd
                    qrow = all_q[tok]
                    scores = np.empty(tok + 1, dtype=np.float32)
                    for j in range(tok + 1):
                        scores[j] = float(
                            np.sum(qrow[qs : qs + hd] * all_k[j][ks : ks + hd])
                        ) * attn_scale
                    max_s = float(np.max(scores))
                    exp_s = np.exp(scores - max_s)
                    sum_e = max(float(np.sum(exp_s)), 1e-10)
                    for j in range(tok + 1):
                        w = exp_s[j] / sum_e
                        ho[qs : qs + hd] += w * all_v[j][ks : ks + hd]
                attn_out = w_o @ ho
                na = rms_norm_1(attn_out, paln)
                residuals[tok] = residuals[tok] + na
            log.debug(
                "Couche %d/%d: attention terminee (RoPE %s, %.0f ms)",
                layer + 1,
                nl,
                "global" if is_global else "local",
                (time.perf_counter() - t_l0) * 1000.0,
            )

        for tok in range(nt):
            x = rms_norm_1(residuals[tok], pfln)
            gs = vindex.gate_matvec(layer, x)
            us = vindex.up_matvec(layer, x)
            act = gelu_tanh_vec(gs) * us
            delta = vindex.down_matvec(layer, act)
            nf = rms_norm_1(delta, pfnl)
            residuals[tok] = residuals[tok] + nf

        if (layer + 1) % 10 == 0 or layer == nl - 1:
            norm = float(np.linalg.norm(residuals[nt - 1]))
            log.info(
                "Couche %d/%d terminee en %.0f ms (cumul couches %.0f ms) — ||residu dernier token||=%.1f%s",
                layer + 1,
                nl,
                (time.perf_counter() - t_l0) * 1000.0,
                (time.perf_counter() - t_layer) * 1000.0,
                norm,
                " [ffn-only]" if skip_attention else "",
            )
        else:
            rope = "n/a (ffn-only)" if skip_attention else ("global" if is_global else "local")
            log.info(
                "Couche %d/%d terminee en %.0f ms (RoPE %s)",
                layer + 1,
                nl,
                (time.perf_counter() - t_l0) * 1000.0,
                rope,
            )

    layer_ms = (time.perf_counter() - t_layer) * 1000.0
    log.info("Toutes les couches: %.0f ms", layer_ms)

    t_logits = time.perf_counter()
    final_n = vindex.final_norm()
    last = rms_norm_1(residuals[nt - 1], final_n).astype(np.float32, copy=False)
    vocab = cfg.vocab_size

    # Logits = table d'embedding @ dernier état (tied LM head), par blocs pour limiter la RAM pic
    log.info(
        "Logits: vocab=%d, blocs de %d (~%d passes)",
        vocab,
        logits_chunk,
        (vocab + logits_chunk - 1) // logits_chunk,
    )
    logits_full = np.empty(vocab, dtype=np.float32)
    next_log_pct = 0.0
    for start in range(0, vocab, logits_chunk):
        end = min(start + logits_chunk, vocab)
        t_b = time.perf_counter()
        block = vindex.embed[start:end].astype(np.float32, copy=False)
        logits_full[start:end] = block @ last
        pct = 100.0 * end / vocab
        if start == 0 or end >= vocab or pct >= next_log_pct:
            log.info(
                "  logits [%d, %d) %.1f %% vocab — %.0f ms (bloc)",
                start,
                end,
                pct,
                (time.perf_counter() - t_b) * 1000.0,
            )
            next_log_pct += 25.0

    k = min(top_k, vocab)
    part = np.argpartition(-logits_full, k - 1)[:k]
    order = np.argsort(-logits_full[part])
    idx = part[order]
    pairs = [(int(i), float(logits_full[i])) for i in idx]
    logits_ms = (time.perf_counter() - t_logits) * 1000.0
    total_ms = (time.perf_counter() - t0) * 1000.0
    log.info("Top-%d selectionne en %.2f ms — total forward %.0f ms", k, logits_ms, total_ms)
    return InferenceResult(top_k=pairs, total_ms=total_ms, layer_ms=layer_ms, logits_ms=logits_ms)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def load_tokenizer(path: Path):
    try:
        from tokenizers import Tokenizer
    except ImportError as e:
        raise SystemExit(
            "Le paquet 'tokenizers' est requis pour --prompt. "
            "Installez-le avec : pip install tokenizers"
        ) from e
    return Tokenizer.from_file(str(path))


def main() -> None:
    ap = argparse.ArgumentParser(description="Inférence Vindex (port Python de vindex-infer)")
    ap.add_argument("--vindex", type=Path, required=True, help="Répertoire du modèle .vindex")
    ap.add_argument("-p", "--prompt", type=str, default=None)
    ap.add_argument("--token-ids", type=str, default=None, help="Liste d'IDs séparés par des virgules")
    ap.add_argument("--tokenizer", type=Path, default=None)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument(
        "--forward",
        choices=("full", "ffn-only"),
        default="full",
        help="full = attention+FFN ; ffn-only = sans attention (ablation)",
    )
    ap.add_argument("--logits-chunk", type=int, default=4096, help="Taille de bloc pour les logits (mémoire)")
    ap.add_argument("-v", "--verbose", action="store_true", help="Logs DEBUG (detail par couche / mmap)")
    ap.add_argument("-q", "--quiet", action="store_true", help="Moins de logs (WARNING)")
    args = ap.parse_args()

    if args.quiet and args.verbose:
        print("Ne pas combiner --quiet et --verbose", file=sys.stderr)
        raise SystemExit(2)
    level = logging.DEBUG if args.verbose else (logging.WARNING if args.quiet else logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    vix = Vindex.load(args.vindex)
    print(
        f"Modèle: {vix.config.model} ({vix.config.num_layers} couches, "
        f"h={vix.config.hidden_size}, vocab={vix.config.vocab_size})"
    )

    tok_path = args.tokenizer or (args.vindex / "tokenizer.json")
    tokenizer = None
    if tok_path.is_file():
        try:
            tokenizer = load_tokenizer(tok_path)
        except SystemExit:
            if args.tokenizer is not None:
                raise
    elif args.tokenizer is not None:
        raise SystemExit(f"Tokenizer introuvable: {tok_path}")

    if args.token_ids:
        token_ids = [int(x.strip()) for x in args.token_ids.split(",") if x.strip()]
    elif args.prompt:
        if tokenizer is None:
            raise SystemExit(
                f"Un tokenizer est nécessaire pour --prompt. "
                f"Placez tokenizer.json dans {args.vindex} ou passez --tokenizer."
            )
        enc = tokenizer.encode(args.prompt, add_special_tokens=False)
        token_ids = enc.ids
    else:
        token_ids = [818, 5279, 529, 7001, 563]

    if token_ids[0] != 2:
        token_ids.insert(0, 2)

    if tokenizer is not None:
        try:
            print("Prompt (décodé):", tokenizer.decode(token_ids, skip_special_tokens=True))
        except Exception:
            print("Prompt: <erreur decode>")
    print("Token IDs:", token_ids)
    print("Forward:", args.forward)
    print()

    skip = args.forward == "ffn-only"
    res = infer(vix, token_ids, top_k=args.top_k, skip_attention=skip, logits_chunk=args.logits_chunk)

    print("Prédictions (top token suivant):")
    for i, (tid, score) in enumerate(res.top_k):
        piece = f"<{tid}>"
        if tokenizer is not None:
            try:
                piece = tokenizer.decode([tid], skip_special_tokens=False)
                piece = piece.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
            except Exception:
                pass
        print(f"  {i+1:2}. {piece!r:24}  id {tid:6}  score {score:+.4f}")

    print()
    print(
        f"Temps: {res.total_ms:.1f} ms total "
        f"({res.layer_ms:.0f} ms couches, {res.logits_ms:.0f} ms logits)"
    )


if __name__ == "__main__":
    main()
