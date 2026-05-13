#!/usr/bin/env python3
"""
**A-Vindex INFER** — vindex-only diagnostic + next-token prediction.

This script never loads a Hugging Face model. Every piece of data it reads
lives inside the vindex directory:

1. **Centroid probe** (default): last-token embedding × stored ``k_proj`` slice
   → nearest centroid for one (layer, kv_head). Reads
   ``embeddings.bin`` + ``attn_centroids.bin`` + ``attn_k_proj_weights.bin``.

2. **Walk top-k** (default ``AVINDEX_WALK_TOPK=10``): pure-numpy vindex walk
   over ``gate_vectors.bin`` + ``down_meta.bin``. Same primitive as
   ``vindex-infer.py`` — this is the actual next-token prediction.

3. **Full A-Vindex scan** (optional ``AVINDEX_SCAN=1``): every (layer, kv_head)
   centroid hit for the same last-token projection. Diagnostic only.

Notebook (empty argv)
---------------------
  VINDEX_DIR, AVINDEX_PY, VINDEX_PY — same conventions as before.

  AVINDEX_WALK_TOPK    int, default 10; set 0 to skip the FFN walk block.
  AVINDEX_SCAN=1       emit full ``steps`` list (layer, kv_head, centroid_id, cosine).
  AVINDEX_SCAN_LAYERS  comma list, e.g. "13" or "0,5,10" (default: all layers).

CLI
---
  python avindex-infer.py --vindex ./.larql_colab/vindex_out -p "Hello" --walk-topk 10
  python avindex-infer.py --vindex ./.larql_colab/vindex_out --scan --scan-layers 13
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


def _log(msg: str) -> None:
    print(f"[avindex-infer] {msg}", flush=True)


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


def _scripts_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


# ── sibling module loaders (path-based, Colab-safe) ───────────────────────


def _candidate_paths(env_var: str, filename: str) -> list[Path]:
    raw: list[Path] = []
    ev = os.environ.get(env_var, "").strip()
    if ev:
        p = Path(ev).expanduser().resolve()
        raw.append((p / filename) if p.is_dir() else p)
    raw.append(_scripts_dir() / filename)
    raw.append(Path.cwd().resolve() / filename)
    out: list[Path] = []
    seen: set[str] = set()
    for p in raw:
        try:
            rp = p.resolve()
        except OSError:
            continue
        if str(rp) not in seen:
            seen.add(str(rp))
            out.append(rp)
    return out


def _load_module(module_name: str, py: Path) -> Any:
    if module_name in sys.modules:
        del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {py}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_avindex_attention() -> Any:
    for py in _candidate_paths("AVINDEX_PY", "avindex_attention.py"):
        if py.is_file():
            try:
                return _load_module("avindex_attention_colab_infer", py)
            except Exception as e:  # noqa: BLE001
                _log(f"warning: could not load {py}: {e}")
                continue
    try:
        import avindex_attention as av  # type: ignore

        return av
    except ImportError:
        pass
    tried = ", ".join(str(p) for p in _candidate_paths("AVINDEX_PY", "avindex_attention.py"))
    raise RuntimeError(
        "Could not load avindex_attention.py. Set AVINDEX_PY or place the file next to "
        f"this script / cwd. Tried: {tried}"
    )


def load_vindex_infer() -> Any:
    for py in _candidate_paths("VINDEX_INFER_PY", "vindex-infer.py"):
        if py.is_file():
            try:
                return _load_module("vindex_infer_colab_avindex", py)
            except Exception as e:  # noqa: BLE001
                _log(f"warning: could not load {py}: {e}")
                continue
    tried = ", ".join(str(p) for p in _candidate_paths("VINDEX_INFER_PY", "vindex-infer.py"))
    raise RuntimeError(
        "Could not load vindex-infer.py. Set VINDEX_INFER_PY or place the file next to "
        f"this script. Tried: {tried}"
    )


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _validate_vindex_for_probe(vdir: Path) -> None:
    for name in (
        "index.json",
        "embeddings.bin",
        "attn_index.json",
        "attn_centroids.bin",
        "attn_k_proj_weights.bin",
    ):
        p = vdir / name
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing {p} — run vindex.py then avindex_attention.py extract first."
            )


def _parse_scan_layers_env() -> list[int] | None:
    sl = os.environ.get("AVINDEX_SCAN_LAYERS", "").strip()
    if not sl:
        return None
    return [int(x.strip()) for x in sl.split(",") if x.strip()]


# ── pipelines ─────────────────────────────────────────────────────────────


def run_probe(
    vindex_dir: Path,
    *,
    prompt: str,
    layer: int | None,
    kv_head: int,
    reader: Any = None,
) -> dict[str, Any]:
    av = load_avindex_attention()
    vdir = vindex_dir.resolve()
    _validate_vindex_for_probe(vdir)
    main = json.loads((vdir / "index.json").read_text(encoding="utf-8"))
    L = int(layer) if layer is not None else av._default_probe_layer(main)
    return av.probe_last_token_k_space(
        vdir, prompt=prompt, layer=L, kv_head=kv_head, reader=reader
    )


def run_scan(
    vindex_dir: Path,
    *,
    prompt: str,
    layers_only: list[int] | None,
    reader: Any = None,
) -> dict[str, Any]:
    av = load_avindex_attention()
    vdir = vindex_dir.resolve()
    _validate_vindex_for_probe(vdir)
    return av.scan_centroid_path(
        vdir, prompt=prompt, layers_only=layers_only, reader=reader
    )


def run_walk_topk(
    vindex_dir: Path,
    *,
    prompt: str,
    top_k: int,
    features_per_layer: int = 32,
    layers: list[int] | str | None = None,
) -> list[dict[str, Any]]:
    """Pure-numpy FFN walk top-k prediction (same as ``vindex-infer.py``)."""
    inf = load_vindex_infer()
    runner = inf.VindexWalkRunner(
        vindex_dir.resolve(),
        use_attention_context=(
            False
            if os.environ.get("VINDEX_NO_ATTN_CTX", "").strip().lower()
            in ("1", "true", "yes", "on")
            else None
        ),
    )
    preds = runner.infer_topk(
        prompt, top_k=top_k, features_per_layer=features_per_layer, layers=layers
    )
    out: list[dict[str, Any]] = []
    for rank, (piece, prob) in enumerate(preds, start=1):
        out.append({"rank": rank, "token": piece, "prob": float(prob)})
    return out


# ── env / CLI entry points ────────────────────────────────────────────────


def run_default_cell() -> None:
    vdir = default_vindex_dir()
    if not (vdir / "attn_index.json").is_file() or not (vdir / "attn_centroids.bin").is_file():
        _log(f"No A-Vindex under {vdir}")
        _log("  1) vindex.py browse extract  2) avindex_attention AVINDEX_EXTRACT=1")
        _log("  3) AVINDEX_PY=/content/avindex_attention.py if imports are wrong")
        return

    prompt = os.environ.get(
        "AVINDEX_PROMPT", os.environ.get("VINDEX_PROMPT", "The capital of France is")
    ).strip()
    kv = int(os.environ.get("AVINDEX_PROBE_KV_HEAD", "0"))
    layer_env = os.environ.get("AVINDEX_PROBE_LAYER", "").strip()
    layer = int(layer_env) if layer_env else None

    av = load_avindex_attention()

    _log(f"run  vindex_dir={vdir}")
    _log(f"     prompt={prompt!r}")
    t0 = time.perf_counter()

    with av.AvindexAttentionReader(vdir) as reader:
        _log("— (1) single centroid probe (vindex-only) —")
        out = run_probe(vdir, prompt=prompt, layer=layer, kv_head=kv, reader=reader)
        print(json.dumps({"centroid_probe": out}, indent=2), flush=True)

        walk_k = int(os.environ.get("AVINDEX_WALK_TOPK", "10"))
        if walk_k > 0:
            _log(f"— (2) vindex FFN walk top-{walk_k} (gates × down_meta votes, pure numpy) —")
            try:
                rows = run_walk_topk(
                    vdir, prompt=prompt, top_k=walk_k,
                    features_per_layer=int(os.environ.get("VINDEX_WALK_FEATURES", "32")),
                    layers=os.environ.get("VINDEX_WALK_LAYERS") or None,
                )
                print(json.dumps({"vindex_walk_topk": rows}, indent=2), flush=True)
            except Exception as e:  # noqa: BLE001
                _log(f"walk block failed (centroid probe above is still valid): {e}")
                print(json.dumps({"vindex_walk_topk_error": str(e)}, indent=2), flush=True)

        if _env_truthy("AVINDEX_SCAN"):
            _log("— (3) A-Vindex scan: all layer × kv_head centroid hits —")
            scan = run_scan(
                vdir, prompt=prompt, layers_only=_parse_scan_layers_env(), reader=reader
            )
            print(json.dumps(scan, indent=2), flush=True)

    _log(f"done in {time.perf_counter() - t0:.2f}s")


def main_cli() -> None:
    ap = argparse.ArgumentParser(description="A-Vindex probe + vindex walk top-k + optional scan")
    ap.add_argument("--vindex", type=Path, default=default_vindex_dir())
    ap.add_argument("-p", "--prompt", default=os.environ.get("AVINDEX_PROMPT", "The capital of France is"))
    ap.add_argument("--layer", type=int, default=-1, help="-1 = auto from index.json layer_bands")
    ap.add_argument("--kv-head", type=int, default=0)
    ap.add_argument("--walk-topk", type=int, default=10, help="0 = skip the FFN walk block")
    ap.add_argument(
        "--features-per-layer",
        type=int,
        default=int(os.environ.get("VINDEX_WALK_FEATURES", "32")),
    )
    ap.add_argument(
        "--walk-layers",
        default=os.environ.get("VINDEX_WALK_LAYERS", ""),
        help='comma list or band name ("syntax", "knowledge", "output")',
    )
    ap.add_argument("--scan", action="store_true", help="emit full layer×kv_head centroid steps")
    ap.add_argument(
        "--scan-layers",
        default="",
        help='comma-separated layer ids, e.g. "13" or "0,5,10" (default: all when --scan)',
    )
    args, rest = ap.parse_known_args(user_argv())
    if rest:
        _log(f"warning: ignored argv: {rest}")

    av = load_avindex_attention()
    vdir = args.vindex.resolve()
    layer = None if args.layer < 0 else args.layer

    with av.AvindexAttentionReader(vdir) as reader:
        out = run_probe(vdir, prompt=args.prompt, layer=layer, kv_head=args.kv_head, reader=reader)
        print(json.dumps({"centroid_probe": out}, indent=2))

        if args.walk_topk > 0:
            try:
                layers_in = args.walk_layers or None
                if isinstance(layers_in, str) and layers_in and not layers_in.isalpha():
                    layers_in = [int(p.strip()) for p in layers_in.split(",") if p.strip()]
                rows = run_walk_topk(
                    vdir,
                    prompt=args.prompt,
                    top_k=args.walk_topk,
                    features_per_layer=args.features_per_layer,
                    layers=layers_in,
                )
                print(json.dumps({"vindex_walk_topk": rows}, indent=2))
            except Exception as e:  # noqa: BLE001
                _log(f"walk top-k failed: {e}")
                print(json.dumps({"vindex_walk_topk_error": str(e)}, indent=2))

        if args.scan:
            sl = args.scan_layers.strip()
            layers_only = [int(x.strip()) for x in sl.split(",") if x.strip()] if sl else None
            scan = run_scan(vdir, prompt=args.prompt, layers_only=layers_only, reader=reader)
            print(json.dumps(scan, indent=2))


if __name__ == "__main__":
    if not user_argv():
        _log("entry: notebook default (empty argv)")
        run_default_cell()
    else:
        _log(f"entry: CLI argv={user_argv()!r}")
        main_cli()
