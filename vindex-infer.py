#!/usr/bin/env python3
"""
Colab-friendly **INFER** helper (separate from `vindex.py` extract).

INFER (LARQL): next-token predictions via a full forward pass with **Walk FFN**
when the native `larql` Python bindings are installed and the vindex was built
with **`--level inference` or `all`** (Rust `larql extract`). Browse-only
vindexes (e.g. from pure-Python `vindex.py`) do **not** include FFN weights —
this script then falls back to **Hugging Face Transformers** on the same
`model` id from `index.json` (next-token top-k, not identical to Walk FFN).

Environment (cell mode, no CLI):
  VINDEX_DIR       — path to .vindex directory (default: ./.larql_colab/vindex_out)
  VINDEX_PROMPT    — input text (default: a short demo prompt)
  VINDEX_TOP_K     — int (default: 5)
  VINDEX_HF_MODEL  — optional HF id override for Transformers fallback

CLI:
  python vindex-infer.py --vindex ./.larql_colab/vindex_out -p "Hello" --top-k 5

Dependencies:
  - Native INFER: `larql` wheel built with PyO3 (see crates/larql-python README).
  - Fallback: `pip install transformers torch accelerate`
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


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


def try_larql_infer(
    vindex_dir: Path,
    prompt: str,
    top_k: int,
) -> list[tuple[str, float]] | None:
    """Return predictions if `larql` + mmap weights work; else None."""
    try:
        import larql  # type: ignore
    except ImportError:
        _log("[vindex-infer] native: `import larql` failed — install PyO3 bindings (see crates/larql-python).")
        return None

    idx = load_index(vindex_dir)
    level = str(idx.get("extract_level", "browse"))
    _log(f"[vindex-infer] native: opening vindex extract_level={level!r} …")
    if level == "browse":
        _log(
            "[vindex-infer] native: browse-only vindex has no FFN weights for Walk INFER — "
            "use `larql extract --level inference` or use the Transformers fallback."
        )
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


def transformers_next_token(
    model_id: str,
    prompt: str,
    top_k: int,
) -> list[tuple[str, float]]:
    """HF fallback: top-k next-token predictions (single step)."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise SystemExit(
            "Transformers fallback needs: pip install transformers torch accelerate"
        ) from e

    _log(f"[vindex-infer] fallback: loading {model_id!r} (Transformers) …")
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    _log(f"[vindex-infer]   model loaded in {time.perf_counter() - t0:.1f}s")

    inputs = tok(prompt, return_tensors="pt")
    if hasattr(model, "device"):
        dev = model.device
    else:
        dev = next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    t1 = time.perf_counter()
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits[0, -1, :].float()
    probs = torch.softmax(logits, dim=-1)
    k = min(top_k, probs.numel())
    vals, inds = probs.topk(k)
    rows: list[tuple[str, float]] = []
    for p, tid in zip(vals.tolist(), inds.tolist()):
        s = tok.decode([tid], skip_special_tokens=True)
        rows.append((s if s.strip() else f"<id {tid}>", float(p)))
    _log(f"[vindex-infer] fallback: forward pass {time.perf_counter() - t1:.2f}s")
    return rows


def run_infer(
    vindex_dir: Path,
    prompt: str,
    top_k: int,
    hf_model_override: str | None,
) -> list[tuple[str, float]]:
    vindex_dir = vindex_dir.resolve()
    _log("=" * 60)
    _log(f"[vindex-infer] vindex_dir={vindex_dir}")
    _log(f"[vindex-infer] prompt={prompt!r}  top_k={top_k}")

    preds = try_larql_infer(vindex_dir, prompt, top_k)
    if preds is not None:
        _log("[vindex-infer] path: **native LARQL INFER** (Walk FFN + mmap weights)")
        for i, (tok, pr) in enumerate(preds):
            _log(f"  {i + 1:2}. p={pr:.4f}  {tok!r}")
        _log("=" * 60)
        return preds

    meta = load_index(vindex_dir)
    model_id = (hf_model_override or "").strip() or str(meta.get("model", "")).strip()
    if not model_id:
        raise SystemExit(
            "Cannot run fallback: no HF model id. Set VINDEX_HF_MODEL or ensure index.json has \"model\"."
        )

    _log("[vindex-infer] path: **Transformers fallback** (single-step top-k; not Walk-FFN parity)")
    preds = transformers_next_token(model_id, prompt, top_k)
    for i, (tok, pr) in enumerate(preds):
        _log(f"  {i + 1:2}. p={pr:.4f}  {tok!r}")
    _log("=" * 60)
    return preds


def run_default_cell() -> None:
    vd = default_vindex_dir()
    prompt = os.environ.get(
        "VINDEX_PROMPT",
        "The capital of France is",
    ).strip()
    top_k = int(os.environ.get("VINDEX_TOP_K", "5"))
    hf_ov = os.environ.get("VINDEX_HF_MODEL", "").strip() or None
    run_infer(vd, prompt, top_k, hf_ov)


def main_cli() -> None:
    ap = argparse.ArgumentParser(description="INFER: LARQL native or HF Transformers fallback")
    ap.add_argument("--vindex", type=Path, default=default_vindex_dir(), help="path to vindex directory")
    ap.add_argument("-p", "--prompt", default=os.environ.get("VINDEX_PROMPT", "The capital of France is"))
    ap.add_argument("--top-k", type=int, default=int(os.environ.get("VINDEX_TOP_K", "5")))
    ap.add_argument(
        "--hf-model",
        default=None,
        help="HF model id for Transformers fallback (else VINDEX_HF_MODEL env or index.json model)",
    )
    args, rest = ap.parse_known_args(user_argv())
    if rest:
        _log(f"[vindex-infer] warning: ignored argv: {rest}")
    hf = (args.hf_model or os.environ.get("VINDEX_HF_MODEL", "").strip() or None)
    run_infer(args.vindex.resolve(), args.prompt, args.top_k, hf)


if __name__ == "__main__":
    if not user_argv():
        _log("[vindex-infer] entry: Colab / notebook default (empty argv)")
        run_default_cell()
    else:
        _log(f"[vindex-infer] entry: CLI  argv={user_argv()!r}")
        main_cli()
