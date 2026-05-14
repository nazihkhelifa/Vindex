#!/usr/bin/env python3
"""
Option C — Semantic probing for attention (Vindex « X-Ray »).

Même idée que down_meta.bin pour le FFN : pour chaque couche L et chaque tête Q H,
on isole la tranche de W_o qui réécrit la tête dans le résidu, on en déduit une
direction dans R^hidden (moyenne des colonnes de la tranche), puis on projette
sur la table d'embeddings E pour obtenir les top-k tokens « lexicaux » associés
à cette tête.

Layout attn_weights.bin (aligné sur vindex-infer / larql-vindex, f16) par couche :
  W_q | W_k | W_v | W_o | q_norm | k_norm
avec tailles en *éléments* float16 :
  q_sz = num_q_heads * head_dim * hidden
  k_sz = num_kv_heads * head_dim * hidden
  v_sz = k_sz
  o_sz = hidden * num_q_heads * head_dim
  + head_dim (q_norm) + head_dim (k_norm)

Sorties :
  attn_meta.bin          — uint32 little-endian, shape (num_layers, num_q_heads, top_k) token ids
  attn_meta_scores.bin   — float32, même shape, logits top-k (optionnel avec --no-scores)

index.json reçoit une clé attention_metadata (fusion sans écraser le reste).

Usage :
  python build_attn_semantic_meta.py --vindex ./gemma3-4b.vindex --top-k 20
  python build_attn_semantic_meta.py --vindex ./gemma3-4b.vindex --cpu   # sans CUDA
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError as e:
    raise SystemExit("PyTorch requis : pip install torch") from e

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x  # type: ignore


def attn_layer_floats(num_q: int, num_kv: int, head_dim: int, hidden: int) -> tuple[int, int, int, int, int]:
    q_sz = num_q * head_dim * hidden
    k_sz = num_kv * head_dim * hidden
    v_sz = k_sz
    o_sz = hidden * num_q * head_dim
    tail_norms = 2 * head_dim
    layer_total = q_sz + k_sz + v_sz + o_sz + tail_norms
    return q_sz, k_sz, v_sz, o_sz, layer_total


def main() -> None:
    ap = argparse.ArgumentParser(description="Construire attn_meta (Option C) pour un dossier .vindex")
    ap.add_argument("--vindex", type=Path, required=True, help="Répertoire du vindex (index.json + attn_weights.bin + embeddings.bin)")
    ap.add_argument("--top-k", type=int, default=10, dest="top_k")
    ap.add_argument("--cpu", action="store_true", help="Forcer CPU même si CUDA est disponible")
    ap.add_argument("--no-scores", action="store_true", help="Ne pas écrire attn_meta_scores.bin")
    args = ap.parse_args()

    vdir: Path = args.vindex
    index_path = vdir / "index.json"
    attn_path = vdir / "attn_weights.bin"
    embed_path = vdir / "embeddings.bin"

    if not index_path.is_file():
        raise SystemExit(f"Manquant : {index_path}")
    if not attn_path.is_file():
        raise SystemExit(f"Manquant : {attn_path} (niveau extract >= attention)")
    if not embed_path.is_file():
        raise SystemExit(f"Manquant : {embed_path}")

    meta = json.loads(index_path.read_text(encoding="utf-8"))
    num_layers = int(meta["num_layers"])
    hidden_size = int(meta["hidden_size"])
    vocab_size = int(meta["vocab_size"])
    mc = meta.get("model_config") or {}
    num_q_heads = int(mc.get("num_q_heads", 8))
    num_kv_heads = int(mc.get("num_kv_heads", 4))
    head_dim = int(mc.get("head_dim", 256))

    q_sz, k_sz, v_sz, o_sz, layer_total = attn_layer_floats(
        num_q_heads, num_kv_heads, head_dim, hidden_size
    )

    file_size = attn_path.stat().st_size
    elem_size = 2  # float16
    n_elem = file_size // elem_size
    expected = layer_total * num_layers
    print(f"attn_weights.bin : {file_size} octets ({n_elem} f16) | attendu {expected * elem_size} octets ({expected} f16/layer × {num_layers})")
    if n_elem < layer_total * num_layers:
        raise SystemExit(
            f"attn_weights.bin trop court : {n_elem} < {layer_total * num_layers} éléments f16 attendus"
        )
    if n_elem != expected:
        print(
            "Avertissement : taille fichier != layout nominal "
            f"({n_elem} vs {expected} éléments). Lecture des {num_layers} premières couches."
        )

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    embeddings = np.memmap(embed_path, dtype=np.float16, mode="r", shape=(vocab_size, hidden_size))
    emb_tensor = torch.from_numpy(np.asarray(embeddings, dtype=np.float32)).to(device)

    full_attn = np.memmap(attn_path, dtype=np.float16, mode="r", shape=(n_elem,))

    out_ids = np.zeros((num_layers, num_q_heads, args.top_k), dtype=np.uint32)
    out_scores = (
        None
        if args.no_scores
        else np.zeros((num_layers, num_q_heads, args.top_k), dtype=np.float32)
    )

    print(f"Périphérique : {device} | couches={num_layers} têtes Q={num_q_heads} top_k={args.top_k}")

    for layer_idx in tqdm(range(num_layers), desc="Layers"):
        layer_start = layer_idx * layer_total
        o_offset = layer_start + q_sz + k_sz + v_sz
        o_raw = np.asarray(full_attn[o_offset : o_offset + o_sz], dtype=np.float32).reshape(
            hidden_size, num_q_heads * head_dim
        )
        o_proj = torch.from_numpy(o_raw).to(device)

        for head_idx in range(num_q_heads):
            c0 = head_idx * head_dim
            c1 = (head_idx + 1) * head_dim
            # Direction agrégée : moyenne des colonnes W_o[:, slice]  ->  (hidden,)
            head_direction = o_proj[:, c0:c1].mean(dim=1)
            logits = torch.matmul(emb_tensor, head_direction)
            scores, indices = torch.topk(logits, args.top_k)
            out_ids[layer_idx, head_idx] = indices.to(torch.int32).cpu().numpy().astype(np.uint32)
            if out_scores is not None:
                out_scores[layer_idx, head_idx] = scores.cpu().numpy().astype(np.float32)

    ids_path = vdir / "attn_meta.bin"
    out_ids.tofile(ids_path)
    print(f"Écrit : {ids_path}  shape={out_ids.shape} dtype=uint32 (C-order layer, head, top_k)")

    meta_out: dict = {
        "ids_path": "attn_meta.bin",
        "top_k": args.top_k,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "hidden_size": hidden_size,
        "layout": "row-major: layer * (num_q_heads * top_k) + head * top_k",
        "probe": "mean_columns_W_o_then_dot_E_row",
    }

    if out_scores is not None:
        scores_path = vdir / "attn_meta_scores.bin"
        out_scores.tofile(scores_path)
        print(f"Écrit : {scores_path}  shape={out_scores.shape} dtype=float32 (logits top-k)")
        meta_out["scores_path"] = "attn_meta_scores.bin"
        meta_out["scores_dtype"] = "float32"

    meta["attention_metadata"] = meta_out
    index_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Mis à jour : {index_path} (clé attention_metadata)")


if __name__ == "__main__":
    main()
