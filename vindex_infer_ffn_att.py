#!/usr/bin/env python3
"""
Inférence Vindex : FFN + attention (forward complet), même moteur que vindex_infer_python.py.

- Par défaut : **attention + FFN** (équivalent Rust `infer`, `--forward full`).
- Option `--forward ffn-only` : ablation sans sous-couche attention.
- Option `--attn-meta` : après l'inférence, affiche les libellés sémantiques Option C
  (`attn_meta.bin` produit par build_attn_semantic_meta.py), pour une couche donnée.

Dépendances : numpy ; optionnel tokenizers (--prompt / décodage).

Exemples :
  python vindex_infer_ffn_att.py --vindex ./gemma3-4b.vindex --token-ids 818,5279,529,7001,563
  python vindex_infer_ffn_att.py --vindex ./gemma3-4b.vindex -p "Hello" --attn-meta --attn-meta-layer 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Import du moteur partagé (les deux scripts sont dans le même dossier)
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import vindex_infer_python as vip


log = logging.getLogger(__name__)


def load_attn_meta(root: Path) -> tuple[Optional[np.ndarray], Optional[dict[str, Any]]]:
    """Charge attn_meta.bin selon index.json → attention_metadata."""
    index_path = root / "index.json"
    if not index_path.is_file():
        return None, None
    meta = json.loads(index_path.read_text(encoding="utf-8"))
    am = meta.get("attention_metadata")
    if not am or not isinstance(am, dict):
        return None, None
    rel = am.get("ids_path", "attn_meta.bin")
    bin_path = root / str(rel)
    if not bin_path.is_file():
        log.warning("attention_metadata present mais fichier absent : %s", bin_path)
        return None, am
    n_layers = int(meta["num_layers"])
    n_heads = int(am.get("num_q_heads", meta.get("model_config", {}).get("num_q_heads", 8)))
    top_k = int(am.get("top_k", 10))
    raw = np.fromfile(bin_path, dtype=np.uint32)
    need = n_layers * n_heads * top_k
    if raw.size != need:
        log.warning(
            "attn_meta.bin taille inattendue : %d uint32, attendu %d (L=%d H=%d K=%d)",
            raw.size,
            need,
            n_layers,
            n_heads,
            top_k,
        )
        return None, am
    return raw.reshape(n_layers, n_heads, top_k), am


def print_attn_meta_summary(
    root: Path,
    tokenizer: Any,
    *,
    layer: int,
    max_heads: int = 8,
) -> None:
    arr, am = load_attn_meta(root)
    if arr is None:
        print("Aucun attn_meta.bin utilisable (--attn-meta ignore ou fichier manquant).")
        print("  Générez-le avec : python build_attn_semantic_meta.py --vindex <dir>")
        return
    nl, nh, topk = arr.shape
    if layer < 0:
        layer = nl - 1
    if layer >= nl:
        print(f"--attn-meta-layer {layer} hors plage (0..{nl - 1})")
        return
    print()
    print(f"--- attn_meta (Option C) : couche L={layer}, top-{topk} tokens par tête Q ---")
    if am:
        print(f"    (probe indexé : {am.get('probe', '?')})")
    for h in range(min(nh, max_heads)):
        ids = arr[layer, h].tolist()
        parts = []
        for tid in ids[: min(5, len(ids))]:
            try:
                parts.append(
                    repr(tokenizer.decode([int(tid)], skip_special_tokens=False).replace("\n", "\\n"))
                )
            except Exception:
                parts.append(f"<{tid}>")
        tail = ", …" if topk > 5 else ""
        print(f"  head {h:2}: {', '.join(parts)}{tail}")
    if nh > max_heads:
        print(f"  … ({nh - max_heads} autres têtes non affichées ; augmentez --attn-meta-max-heads)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Inférence Vindex : FFN + attention (forward complet) + option attn_meta"
    )
    ap.add_argument("--vindex", type=Path, required=True, help="Répertoire du modèle .vindex")
    ap.add_argument("-p", "--prompt", type=str, default=None)
    ap.add_argument("--token-ids", type=str, default=None)
    ap.add_argument("--tokenizer", type=Path, default=None)
    ap.add_argument("--top-k", type=int, default=10, help="Top-k logits sortie LM")
    ap.add_argument(
        "--forward",
        choices=("full", "ffn-only"),
        default="full",
        help="full = attention + FFN (défaut) ; ffn-only = sans attention",
    )
    ap.add_argument("--logits-chunk", type=int, default=4096)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("-q", "--quiet", action="store_true")
    ap.add_argument(
        "--attn-meta",
        action="store_true",
        help="Afficher un extrait de attn_meta.bin après l'inférence (Option C)",
    )
    ap.add_argument(
        "--attn-meta-layer",
        type=int,
        default=-1,
        help="Couche pour --attn-meta (-1 = dernière couche)",
    )
    ap.add_argument(
        "--attn-meta-max-heads",
        type=int,
        default=8,
        help="Nombre max de têtes Q listées pour --attn-meta",
    )
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

    vix = vip.Vindex.load(args.vindex)
    print(
        f"Modèle: {vix.config.model} ({vix.config.num_layers} couches, "
        f"h={vix.config.hidden_size}, vocab={vix.config.vocab_size})"
    )
    print("Pipeline: FFN + attention (Python) — même logique que vindex_infer_python --forward full")

    tok_path = args.tokenizer or (args.vindex / "tokenizer.json")
    tokenizer = None
    if tok_path.is_file():
        try:
            tokenizer = vip.load_tokenizer(tok_path)
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
                f"Ajoutez tokenizer.json sous {args.vindex} ou --tokenizer."
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
    if args.forward == "full" and vix.attn is None:
        print(
            "Avertissement: mode full mais attn_weights.bin absent — "
            "le moteur sautera les blocs attention (comme Rust).",
            file=sys.stderr,
        )
    print()

    skip = args.forward == "ffn-only"
    res = vip.infer(
        vix,
        token_ids,
        top_k=args.top_k,
        skip_attention=skip,
        logits_chunk=args.logits_chunk,
    )

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

    if args.attn_meta:
        if tokenizer is None:
            print("\n--attn-meta nécessite tokenizer.json pour décoder les token ids.")
        else:
            print_attn_meta_summary(
                args.vindex,
                tokenizer,
                layer=args.attn_meta_layer,
                max_heads=args.attn_meta_max_heads,
            )


if __name__ == "__main__":
    main()
