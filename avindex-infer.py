#!/usr/bin/env python3
"""
Colab-friendly **A‑Vindex inference** helper.

1. **Centroid probe** (default): last-token embedding × ``k_proj`` → nearest centroid
   for one (layer, kv_head), via ``avindex_attention.probe_last_token_k_space``.

2. **HF next-token top-k** (default ``AVINDEX_HF_TOPK=10``): one full-model forward pass,
   softmax at last position → top token strings (**same idea as** ``vindex-infer``).
   This is **not** read from A‑Vindex centroids; it is the real LM ranking.

3. **Full A‑Vindex scan** (optional ``AVINDEX_SCAN=1``): every (layer, kv_head) centroid hit
   for the same last-token projection (can be large JSON — narrow with ``AVINDEX_SCAN_LAYERS``).

Notebook (empty argv)
-----------------------
  VINDEX_DIR, VINDEX_MODEL, AVINDEX_PY, VINDEX_PY — as before.

  AVINDEX_HF_TOPK     int, default 10; set ``0`` to skip HF next-token block.
  AVINDEX_HF_CPU=1    force ``device_map=\"cpu\"`` (slower, avoids some GPU wheel issues).
  AVINDEX_ALLOW_TORCHVISION=1  disable the torchvision-skip patch (only if you need vision deps).

  AVINDEX_SCAN=1      emit full ``steps`` list (layer, kv_head, centroid_id, cosine).
  AVINDEX_SCAN_LAYERS comma list, e.g. ``13`` or ``0,5,10`` (default: all layers).

CLI
---
  python avindex-infer.py --vindex ./.larql_colab/vindex_out -p "Hello" --hf-topk 10
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


def _candidate_avindex_attention_paths() -> list[Path]:
    raw: list[Path] = []
    ev = os.environ.get("AVINDEX_PY", "").strip()
    if ev:
        p = Path(ev).expanduser().resolve()
        raw.append((p / "avindex_attention.py") if p.is_dir() else p)
    raw.append(_scripts_dir() / "avindex_attention.py")
    raw.append(Path.cwd().resolve() / "avindex_attention.py")
    out: list[Path] = []
    seen: set[str] = set()
    for p in raw:
        try:
            rp = p.resolve()
        except OSError:
            continue
        k = str(rp)
        if k not in seen:
            seen.add(k)
            out.append(rp)
    return out


_MODULE_NAME = "avindex_attention_colab_infer"


def _load_avindex_from_path(py: Path) -> Any:
    if _MODULE_NAME in sys.modules:
        del sys.modules[_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {py}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


def load_avindex_attention() -> Any:
    for py in _candidate_avindex_attention_paths():
        if py.is_file():
            try:
                return _load_avindex_from_path(py)
            except Exception as e:  # noqa: BLE001
                _log(f"warning: could not load {py}: {e}")
                continue
    try:
        import avindex_attention as av  # type: ignore

        return av
    except ImportError:
        pass
    tried = ", ".join(str(p) for p in _candidate_avindex_attention_paths())
    raise RuntimeError(
        "Could not load avindex_attention.py. Set AVINDEX_PY to its path, or place the file "
        f"next to this script / cwd. Tried: {tried}"
    )


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _validate_vindex_for_probe(vdir: Path) -> None:
    for name in ("index.json", "embeddings.bin", "attn_index.json", "attn_centroids.bin"):
        p = vdir / name
        if not p.is_file():
            raise FileNotFoundError(f"Missing {p}")


def _parse_scan_layers_env() -> list[int] | None:
    sl = os.environ.get("AVINDEX_SCAN_LAYERS", "").strip()
    if not sl:
        return None
    return [int(x.strip()) for x in sl.split(",") if x.strip()]


def run_probe(
    vindex_dir: Path,
    *,
    prompt: str,
    layer: int | None,
    kv_head: int,
    model_dir: Path | None,
    weight_cache: Any = None,
) -> dict[str, Any]:
    _validate_vindex_for_probe(vindex_dir.resolve())
    av = load_avindex_attention()
    vx = av._import_vindex()
    vindex_dir = vindex_dir.resolve()
    main = json.loads((vindex_dir / "index.json").read_text(encoding="utf-8"))
    mid = os.environ.get("VINDEX_MODEL", "").strip() or str(main.get("model", "")).strip()
    if not mid:
        raise RuntimeError(
            'Set VINDEX_MODEL or ensure index.json has a non-empty "model" field for HF weights.'
        )
    mdir = (model_dir or vx.resolve_model_dir(mid, vx.work_base() / "hf_models")).resolve()
    L = int(layer) if layer is not None else av._default_probe_layer(main)
    return av.probe_last_token_k_space(
        vindex_dir, mdir, prompt=prompt, layer=L, kv_head=kv_head, weight_cache=weight_cache
    )


def run_default_cell() -> None:
    if _env_truthy("AVINDEX_OFF"):
        _log("AVINDEX_OFF=1 — skipping.")
        return

    vdir = default_vindex_dir()
    if not (vdir / "attn_index.json").is_file() or not (vdir / "attn_centroids.bin").is_file():
        _log(f"No A-Vindex under {vdir}")
        _log("  1) vindex.py browse extract  2) avindex_attention AVINDEX_EXTRACT=1")
        _log("  3) AVINDEX_PY=/content/avindex_attention.py if imports are wrong")
        return

    prompt = os.environ.get(
        "AVINDEX_PROMPT",
        os.environ.get("VINDEX_PROMPT", "The capital of France is"),
    ).strip()
    kv = int(os.environ.get("AVINDEX_PROBE_KV_HEAD", "0"))
    layer_env = os.environ.get("AVINDEX_PROBE_LAYER", "").strip()
    layer = int(layer_env) if layer_env else None

    av = load_avindex_attention()
    vx = av._import_vindex()
    main = json.loads((vdir / "index.json").read_text(encoding="utf-8"))
    mid = os.environ.get("VINDEX_MODEL", "").strip() or str(main.get("model", "")).strip()
    if not mid:
        raise RuntimeError('Set VINDEX_MODEL or index.json "model" for HF weights.')
    mdir = vx.resolve_model_dir(mid, vx.work_base() / "hf_models").resolve()

    _log(f"run  vindex_dir={vdir}")
    _log(f"     prompt={prompt!r}")
    t0 = time.perf_counter()
    cache = av.KProjWeightCache(mdir)

    _log("— (1) single centroid probe —")
    out = run_probe(vdir, prompt=prompt, layer=layer, kv_head=kv, model_dir=mdir, weight_cache=cache)
    print(json.dumps({"centroid_probe": out}, indent=2), flush=True)

    hf_k = int(os.environ.get("AVINDEX_HF_TOPK", "10"))
    if hf_k > 0:
        _log(f"— (2) HF next-token top-{hf_k} (full softmax; same spirit as vindex-infer) —")
        try:
            tops = av.hf_next_token_topk(str(mdir), prompt, top_k=hf_k)
            print(json.dumps({"hf_next_token_topk": tops}, indent=2), flush=True)
        except Exception as e:  # noqa: BLE001
            _log(f"HF next-token block failed (centroid probe above is still valid): {e}")
            _log("  Hints: AVINDEX_HF_CPU=1 for CPU; pip install torchvision matching your torch; AVINDEX_HF_TOPK=0 to skip.")
            print(json.dumps({"hf_next_token_topk_error": str(e)}, indent=2), flush=True)

    if _env_truthy("AVINDEX_SCAN"):
        _log("— (3) A-Vindex scan: all layer × kv_head centroid hits —")
        layers_only = _parse_scan_layers_env()
        scan = av.scan_centroid_path(
            vdir, mdir, prompt=prompt, weight_cache=cache, layers_only=layers_only
        )
        print(json.dumps(scan, indent=2), flush=True)

    _log(f"done in {time.perf_counter() - t0:.2f}s")


def main_cli() -> None:
    ap = argparse.ArgumentParser(description="A-Vindex probe + optional HF top-k + scan")
    ap.add_argument("--vindex", type=Path, default=default_vindex_dir())
    ap.add_argument("-p", "--prompt", default=os.environ.get("AVINDEX_PROMPT", "The capital of France is"))
    ap.add_argument("--layer", type=int, default=-1, help="-1 = auto from index.json layer_bands")
    ap.add_argument("--kv-head", type=int, default=0)
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--hf-topk", type=int, default=10, help="0 = skip HF next-token block")
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
    vx = av._import_vindex()
    vdir = args.vindex.resolve()
    main = json.loads((vdir / "index.json").read_text(encoding="utf-8"))
    mid = os.environ.get("VINDEX_MODEL", "").strip() or str(main.get("model", "")).strip()
    if not mid:
        raise RuntimeError('Set VINDEX_MODEL or index.json "model".')
    mdir = (args.model_dir or vx.resolve_model_dir(mid, vx.work_base() / "hf_models")).resolve()
    layer = None if args.layer < 0 else args.layer

    cache = av.KProjWeightCache(mdir)
    out = run_probe(
        vdir,
        prompt=args.prompt,
        layer=layer,
        kv_head=args.kv_head,
        model_dir=mdir,
        weight_cache=cache,
    )
    print(json.dumps({"centroid_probe": out}, indent=2))

    if args.hf_topk > 0:
        try:
            tops = av.hf_next_token_topk(str(mdir), args.prompt, top_k=args.hf_topk)
            print(json.dumps({"hf_next_token_topk": tops}, indent=2))
        except Exception as e:  # noqa: BLE001
            _log(f"HF next-token failed: {e}")
            print(json.dumps({"hf_next_token_topk_error": str(e)}, indent=2))

    if args.scan:
        sl = args.scan_layers.strip()
        layers_only = [int(x.strip()) for x in sl.split(",") if x.strip()] if sl else None
        scan = av.scan_centroid_path(
            vdir, mdir, prompt=args.prompt, weight_cache=cache, layers_only=layers_only
        )
        print(json.dumps(scan, indent=2))


if __name__ == "__main__":
    if not user_argv():
        _log("entry: notebook default (empty argv)")
        run_default_cell()
    else:
        _log(f"entry: CLI argv={user_argv()!r}")
        main_cli()
