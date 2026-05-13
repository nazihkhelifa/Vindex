#!/usr/bin/env python3
"""
Pure-Python **browse** vindex builder (no Rust, no `larql` binary).

Produces a directory compatible with LARQL’s on-disk **browse** layout:
  `index.json`, `gate_vectors.bin`, `embeddings.bin`, `down_meta.bin`, `tokenizer.json`

If you run the cell **without any CLI arguments** (typical Colab kernel argv is only
`-f …kernel.json`), the script auto-runs with defaults — no `-o` required.

Typical cell (no CLI paths — uses ./.larql_colab/ under the notebook cwd):

  import os
  os.environ["VINDEX_MODEL"] = "Qwen/Qwen3-0.6B"   # optional
  # Optional: persist after you mounted Drive in another cell:
  #   from google.colab import drive
  #   drive.mount("/content/drive")
  # os.environ["VINDEX_SYNC_TO_DRIVE"] = "1"   # → MyDrive/larql_colab_vindexes/<model>/
  # os.environ["VINDEX_DRIVE_DIR"] = "/content/drive/MyDrive/backups/my_vindex"  # exact folder
  # os.environ["VINDEX_ZIP"] = "1"             # → .larql_colab/<model>__browse_vindex.zip
  # paste this entire file, then it runs automatically when __name__ == "__main__"
  #
  # INFER (next-token) lives in a separate cell script: `vindex-infer.py` (native larql or HF fallback).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _log(msg: str) -> None:
    """Line-buffered progress for Colab / notebooks."""
    print(msg, flush=True)


# ── Colab / Jupyter argv noise ───────────────────────────────────────────


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


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _model_slug(model_name: str) -> str:
    return model_name.replace("/", "__").replace(":", "_")[:180]


def maybe_persist_vindex(
    out_dir: Path,
    model_name: str,
    *,
    save_zip: bool = False,
    copy_to_dir: Path | None = None,
    sync_colab_drive: bool = False,
) -> None:
    """
    Optional copies for Google Colab:
    - ZIP under ./.larql_colab/ (easy “Download” from the file browser)
    - Full directory copy to Google Drive when mounted
    - Explicit destination directory

    Env (used when CLI flags not passed): VINDEX_ZIP, VINDEX_DRIVE_DIR, VINDEX_SYNC_TO_DRIVE
    """
    out_dir = out_dir.resolve()
    slug = _model_slug(model_name)
    base = work_base()

    want_zip = save_zip or _env_truthy("VINDEX_ZIP")
    explicit_drive = copy_to_dir or (
        Path(os.environ["VINDEX_DRIVE_DIR"]).expanduser()
        if os.environ.get("VINDEX_DRIVE_DIR", "").strip()
        else None
    )
    want_sync = sync_colab_drive or _env_truthy("VINDEX_SYNC_TO_DRIVE")

    if want_zip:
        _log("[vindex] persist: creating ZIP archive …")
        t0 = time.perf_counter()
        zip_base = base / f"{slug}_browse_vindex"
        # make_archive adds .zip
        archive = shutil.make_archive(
            str(zip_base),
            "zip",
            root_dir=str(out_dir.parent),
            base_dir=out_dir.name,
        )
        zp = Path(archive)
        _log(f"[vindex]   ZIP → {zp} ({zp.stat().st_size / (1024 * 1024):.2f} MiB) in {time.perf_counter() - t0:.1f}s")

    dest_dir: Path | None = None
    if explicit_drive is not None:
        dest_dir = explicit_drive.expanduser().resolve()
    elif want_sync:
        gdrive = Path("/content/drive/MyDrive")
        if gdrive.is_dir():
            dest_dir = gdrive / "larql_colab_vindexes" / slug / out_dir.name
        else:
            _log(
                "[vindex] persist: VINDEX_SYNC_TO_DRIVE set but "
                "/content/drive/MyDrive not found — mount Drive first, then re-run persist or extract"
            )

    if dest_dir is not None:
        if dest_dir.resolve() == out_dir.resolve():
            _log("[vindex] persist: skip directory copy (destination is the build output itself)")
        else:
            _log(f"[vindex] persist: copying vindex directory → {dest_dir}")
            t0 = time.perf_counter()
            dest_dir.parent.mkdir(parents=True, exist_ok=True)
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            shutil.copytree(out_dir, dest_dir)
            _log(f"[vindex]   copy done in {time.perf_counter() - t0:.1f}s ({dest_dir})")


# ── down_meta.bin (must match crates/larql-vindex/src/format/down_meta.rs) ─

DMET_MAGIC = 0x444D4554  # "DMET"
DMET_VERSION = 1


def write_down_meta_bin(
    out_dir: Path,
    per_layer: list[list[FeatureMeta | None]],
    top_k: int,
) -> None:
    """Binary down_meta v1: header + per-layer u32 num_features + fixed records."""
    _log("[vindex] step: packing down_meta.bin …")
    buf = bytearray()
    buf += struct.pack("<IIII", DMET_MAGIC, DMET_VERSION, len(per_layer), top_k)
    for layer_feats in per_layer:
        buf += struct.pack("<I", len(layer_feats))
        rec_body = top_k * 8
        for meta in layer_feats:
            if meta is None:
                buf += struct.pack("<If", 0, 0.0)
                buf += b"\x00" * rec_body
                continue
            buf += struct.pack("<If", meta.top_token_id, meta.c_score)
            tk = meta.top_k[:top_k]
            pad = top_k - len(tk)
            for e in tk:
                buf += struct.pack("<If", e.token_id, e.logit)
            for _ in range(pad):
                buf += struct.pack("<If", 0, 0.0)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "down_meta.bin"
    path.write_bytes(buf)
    _log(f"[vindex]   → {path.name} ({len(buf) / (1024 * 1024):.2f} MiB)")


@dataclass
class TopKEntry:
    token_id: int
    logit: float


@dataclass
class FeatureMeta:
    top_token_id: int
    c_score: float
    top_k: list[TopKEntry]


# ── Safetensors + HF ────────────────────────────────────────────────────


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise SystemExit("Install torch:  pip install torch") from e


def load_safetensors_dir(model_dir: Path) -> dict[str, Any]:
    _require_torch()
    from safetensors.torch import load_file
    import torch

    merged: dict[str, Any] = {}
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No *.safetensors under {model_dir}")
    _log(f"[vindex] step: loading {len(files)} safetensors shard(s) …")
    for i, fp in enumerate(files, 1):
        t0 = time.perf_counter()
        part = load_file(str(fp), device="cpu")
        merged.update(part)
        _log(f"[vindex]   shard {i}/{len(files)} {fp.name}: +{len(part)} tensors ({time.perf_counter() - t0:.1f}s)")
    _log("[vindex] step: casting weights to float32 (RAM spike possible) …")
    t0 = time.perf_counter()
    out = {k: v.to(torch.float32) for k, v in merged.items()}
    _log(f"[vindex]   → {len(out)} tensors in {time.perf_counter() - t0:.1f}s")
    return out


def snapshot_hf(model_id: str, cache: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SystemExit("pip install huggingface_hub") from e

    dest = cache / model_id.replace("/", "__")
    dest.mkdir(parents=True, exist_ok=True)
    _log(f"[vindex] step: Hugging Face snapshot_download → {dest}")
    t0 = time.perf_counter()
    snapshot_download(
        model_id,
        local_dir=str(dest),
        local_dir_use_symlinks=False,
        token=os.environ.get("HF_TOKEN"),
    )
    _log(f"[vindex]   snapshot done in {time.perf_counter() - t0:.1f}s")
    return dest


def resolve_model_dir(model: str, cache: Path) -> Path:
    p = Path(model).expanduser()
    if p.is_dir():
        _log(f"[vindex] step: using local model directory {p.resolve()}")
        return p.resolve()
    _log(f"[vindex] step: resolving HF model id {model!r} (not a local path)")
    return snapshot_hf(model, cache).resolve()


def detect_prefix(keys: list[str]) -> str:
    for k in keys:
        if k.endswith("embed_tokens.weight"):
            return k[: -len("embed_tokens.weight")]
    raise KeyError("Could not find *embed_tokens.weight in safetensors keys")


def find_key(sd: dict[str, Any], *candidates: str) -> str:
    for c in candidates:
        if c in sd:
            return c
    raise KeyError(f"None of {candidates} in weights (sample keys: {list(sd)[:8]})")


def layer_mlp_key(prefix: str, layer: int, name: str) -> str:
    return f"{prefix}layers.{layer}.mlp.{name}"


def get_weight(sd: dict[str, Any], key: str) -> Any:
    if key in sd:
        return sd[key]
    alts = [key.replace("model.", "", 1), f"model.{key}"]
    for a in alts:
        if a in sd:
            return sd[a]
    raise KeyError(f"Missing weight {key!r} (also tried {alts})")


# ── Core extract ──────────────────────────────────────────────────────────


def _layer_bands_thirds(n: int) -> dict[str, list[int]]:
    a = n // 3
    b = 2 * n // 3
    return {
        "syntax": [0, max(0, a - 1)],
        "knowledge": [min(a, n - 1), max(a, min(b - 1, n - 1))],
        "output": [min(b, n - 1), n - 1],
    }


BROWSE_REQUIRED_FILES: tuple[str, ...] = (
    "index.json",
    "embeddings.bin",
    "gate_vectors.bin",
    "down_meta.bin",
    "tokenizer.json",
)


def browse_vindex_is_complete(out_dir: Path) -> bool:
    """Return ``True`` when every browse vindex file exists and is non-empty."""
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return False
    for name in BROWSE_REQUIRED_FILES:
        p = out_dir / name
        if not p.is_file() or p.stat().st_size == 0:
            return False
    return True


def extract_browse(
    model_dir: Path,
    out_dir: Path,
    *,
    model_name: str,
    down_top_k: int = 10,
    feature_batch: int = 64,
    save_zip: bool = False,
    copy_to_dir: Path | None = None,
    sync_colab_drive: bool = False,
    force: bool = False,
) -> Path:
    """
    Dense transformer (Qwen2/3-style HF layout): gate_proj, down_proj, embed_tokens.
    Does not support MoE in this minimal script.

    Idempotent: if ``out_dir`` already contains every browse file and ``force``
    is ``False``, returns without re-downloading anything from Hugging Face.
    """
    out_dir = out_dir.resolve()
    if not force and browse_vindex_is_complete(out_dir):
        _log("=" * 60)
        _log(f"[vindex] SKIP extract — vindex already complete at {out_dir}")
        _log(f"[vindex]   files: {', '.join(BROWSE_REQUIRED_FILES)} all present (use force=True to rebuild)")
        return out_dir

    _log("=" * 60)
    _log("[vindex] EXTRACT BROWSE (pure Python) — start")
    _log(f"[vindex]   model_dir={model_dir}")
    _log(f"[vindex]   out_dir={out_dir}")
    _log(f"[vindex]   down_top_k={down_top_k}  feature_batch={feature_batch}")

    model_dir = model_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _log("[vindex] STEP 1/9 — read config.json …")
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    num_layers = int(cfg["num_hidden_layers"])
    hidden = int(cfg["hidden_size"])
    inter = int(cfg["intermediate_size"])
    vocab = int(cfg["vocab_size"])
    model_type = str(cfg.get("model_type", "unknown"))
    n_heads = int(cfg.get("num_attention_heads", cfg.get("num_heads", 1)))
    n_kv = int(cfg.get("num_key_value_heads", n_heads))
    rope_base = float(cfg.get("rope_theta", cfg.get("rope_embedding_base", 10_000.0)))
    head_dim = hidden // n_heads
    embed_scale = float(cfg.get("embedding_multiplier", 1.0))
    _log(
        f"[vindex]   arch: type={model_type}  L={num_layers}  "
        f"hidden={hidden}  inter={inter}  vocab={vocab}  heads={n_heads}/{n_kv}"
    )

    _log("[vindex] STEP 2/9 — MoE / layout guard …")
    num_experts = int(cfg.get("num_experts") or 0)
    if num_experts > 0:
        raise SystemExit(
            "This pure-Python extractor supports **dense** FFN only (no MoE). "
            "Use a dense checkpoint or the Rust `larql extract` for MoE."
        )
    _log("[vindex]   OK (dense FFN)")

    _log("[vindex] STEP 3/9 — load safetensors …")
    sd = load_safetensors_dir(model_dir)
    keys = list(sd.keys())
    _log("[vindex] STEP 4/9 — detect tensor prefix …")
    prefix = detect_prefix(keys)
    _log(f"[vindex]   prefix={prefix!r}  (from embed_tokens.weight key)")

    _log("[vindex] STEP 5/9 — build embeddings.bin …")
    ek = find_key(
        sd,
        f"{prefix}embed_tokens.weight",
        "model.embed_tokens.weight",
        "embed_tokens.weight",
    )
    _log(f"[vindex]   embed key: {ek}")
    emb = sd[ek].numpy().astype(np.float32, copy=False)
    if emb.shape != (vocab, hidden):
        # some models transpose
        if emb.shape == (hidden, vocab):
            emb = emb.T
            _log(f"[vindex]   transposed embed to ({vocab}, {hidden})")
        else:
            raise ValueError(f"Unexpected embed shape {emb.shape}, expect ({vocab},{hidden})")

    emb_path = out_dir / "embeddings.bin"
    emb.astype("<f4").tofile(emb_path)
    _log(f"[vindex]   wrote {emb_path.name} ({emb_path.stat().st_size / (1024 * 1024):.2f} MiB)")

    _log("[vindex] STEP 6/9 — build gate_vectors.bin …")
    gate_path = out_dir / "gate_vectors.bin"
    layers_json: list[dict[str, Any]] = []
    offset = 0
    t_gate = time.perf_counter()
    with open(gate_path, "wb") as gw:
        for layer in range(num_layers):
            gk = layer_mlp_key(prefix, layer, "gate_proj.weight")
            g = get_weight(sd, gk).numpy().astype(np.float32, copy=False)
            # HF Linear: [out_features, in_features] = [inter, hidden]
            if g.shape != (inter, hidden):
                raise ValueError(f"L{layer} gate_proj shape {g.shape}, expect ({inter},{hidden})")
            raw = np.ascontiguousarray(g).tobytes(order="C")
            gw.write(raw)
            nbytes = len(raw)
            layers_json.append(
                {
                    "layer": layer,
                    "num_features": inter,
                    "offset": offset,
                    "length": nbytes,
                }
            )
            offset += nbytes
            _log(f"[vindex]   gate layer {layer + 1}/{num_layers}  bytes+={nbytes // 1024} KiB")
    _log(
        f"[vindex]   gate_vectors done in {time.perf_counter() - t_gate:.1f}s → "
        f"{gate_path.stat().st_size / (1024 * 1024):.2f} MiB"
    )

    _log("[vindex] STEP 7/9 — compute down_meta (matmul vocab×hidden × batches) …")
    per_layer_meta: list[list[FeatureMeta | None]] = []
    for layer in range(num_layers):
        t_l = time.perf_counter()
        _log(f"[vindex]   down_proj layer {layer + 1}/{num_layers} …")
        dk = layer_mlp_key(prefix, layer, "down_proj.weight")
        w_down = get_weight(sd, dk).numpy().astype(np.float32, copy=False)
        # [hidden, intermediate]
        if w_down.shape != (hidden, inter):
            if w_down.shape == (inter, hidden):
                w_down = w_down.T
            else:
                raise ValueError(f"L{layer} down_proj shape {w_down.shape}")
        feats: list[FeatureMeta | None] = []
        n_batches = (inter + feature_batch - 1) // feature_batch
        for bi, j0 in enumerate(range(0, inter, feature_batch)):
            j1 = min(inter, j0 + feature_batch)
            chunk = w_down[:, j0:j1]
            logits = emb @ chunk
            for b in range(j1 - j0):
                col = logits[:, b]
                k = min(down_top_k, vocab)
                if k <= 0:
                    feats.append(None)
                    continue
                if k >= vocab:
                    idx = np.argsort(-col)
                else:
                    part = np.argpartition(-col, k - 1)[:k]
                    idx = part[np.argsort(-col[part])]
                scores = col[idx].astype(np.float64)
                top_ids = idx.astype(np.uint32)
                tk_entries = [
                    TopKEntry(int(top_ids[i]), float(scores[i])) for i in range(len(top_ids))
                ]
                feats.append(
                    FeatureMeta(
                        top_token_id=int(top_ids[0]),
                        c_score=float(scores[0]),
                        top_k=tk_entries,
                    )
                )
            if n_batches > 1 and (bi + 1) % max(1, n_batches // 4) == 0:
                _log(f"[vindex]     batch {bi + 1}/{n_batches} (features {j1}/{inter})")
        per_layer_meta.append(feats)
        _log(f"[vindex]     layer {layer + 1} finished in {time.perf_counter() - t_l:.1f}s")

    write_down_meta_bin(out_dir, per_layer_meta, down_top_k)

    _log("[vindex] STEP 8/9 — copy tokenizer files …")
    copied = 0
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        src = model_dir / name
        if src.is_file():
            shutil.copy2(src, out_dir / name)
            _log(f"[vindex]   copied {name}")
            copied += 1
    if copied == 0:
        _log("[vindex]   warning: no tokenizer.json sidecars found")

    _log("[vindex] STEP 9/9 — write index.json …")
    mc: dict[str, Any] = {
        "model_type": model_type,
        "head_dim": head_dim,
        "num_q_heads": n_heads,
        "num_kv_heads": n_kv,
        "rope_base": rope_base,
        "attention_k_eq_v": False,
    }
    if cfg.get("sliding_window") is not None:
        mc["sliding_window"] = int(cfg["sliding_window"])

    index: dict[str, Any] = {
        "version": 2,
        "model": model_name,
        "family": model_type,
        "num_layers": num_layers,
        "hidden_size": hidden,
        "intermediate_size": inter,
        "vocab_size": vocab,
        "embed_scale": embed_scale,
        "extract_level": "browse",
        "dtype": "f32",
        "quant": "none",
        "down_top_k": down_top_k,
        "has_model_weights": False,
        "layers": layers_json,
        "layer_bands": _layer_bands_thirds(num_layers),
        "model_config": mc,
        "source": {
            "huggingface_repo": model_name,
            "huggingface_revision": None,
            "safetensors_sha256": None,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "larql_version": "python-vindex",
        },
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    _log(f"[vindex]   wrote index.json ({(out_dir / 'index.json').stat().st_size // 1024} KiB)")
    _log("[vindex] STEP persist — optional ZIP / Google Drive copy …")
    maybe_persist_vindex(
        out_dir,
        model_name,
        save_zip=save_zip,
        copy_to_dir=copy_to_dir,
        sync_colab_drive=sync_colab_drive,
    )
    _log("=" * 60)
    _log(f"[vindex] DONE — browse vindex → {out_dir}")
    return out_dir


def run_default_colab() -> Path:
    _log("=" * 60)
    _log("[vindex] MODE: Colab / notebook default (no CLI args)")
    _log(f"[vindex]   cwd={Path.cwd().resolve()}")
    base = work_base()
    _log(f"[vindex]   work_base={base}")
    model = os.environ.get("VINDEX_MODEL", "Qwen/Qwen3-0.6B").strip()
    out = (base / "vindex_out").resolve()
    cache = base / "hf_models"
    topk = int(os.environ.get("VINDEX_DOWN_TOP_K", "10"))
    force = _env_truthy("VINDEX_FORCE")
    _log(f"[vindex]   VINDEX_MODEL={model!r}  VINDEX_DOWN_TOP_K={topk}  VINDEX_FORCE={force}")
    _log(f"[vindex]   cache_dir={cache}  output_dir={out}")
    _log(
        "[vindex]   persist: set VINDEX_ZIP=1 and/or VINDEX_SYNC_TO_DRIVE=1 "
        "and/or VINDEX_DRIVE_DIR=/path/to/dest (after mounting Drive if needed)"
    )
    if not force and browse_vindex_is_complete(out):
        _log(f"[vindex] SKIP — vindex already complete at {out} (no HF download)")
        _log("[vindex]   set VINDEX_FORCE=1 to rebuild from scratch")
        return out
    mdir = resolve_model_dir(model, cache)
    return extract_browse(mdir, out, model_name=model, down_top_k=topk, force=force)


def main_cli() -> None:
    base = work_base()
    ap = argparse.ArgumentParser(description="Pure-Python browse vindex extract")
    ap.add_argument("--model", default=os.environ.get("VINDEX_MODEL", "Qwen/Qwen3-0.6B"))
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"default: {base / 'vindex_out'}",
    )
    ap.add_argument("--down-top-k", type=int, default=10)
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=base / "hf_models",
    )
    ap.add_argument(
        "--zip",
        action="store_true",
        help="Also write a .zip under .larql_colab/ (same as VINDEX_ZIP=1)",
    )
    ap.add_argument(
        "--sync-drive",
        action="store_true",
        help="Copy vindex to Drive if /content/drive/MyDrive exists (same as VINDEX_SYNC_TO_DRIVE=1)",
    )
    ap.add_argument(
        "--copy-to",
        type=Path,
        default=None,
        help="Copy vindex directory to this path (overrides VINDEX_DRIVE_DIR env)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if the browse vindex files already exist (default: skip).",
    )
    args, rest = ap.parse_known_args(user_argv())
    if rest:
        _log(f"[vindex] warning: ignored CLI tokens: {rest}")
    out = (args.output or (base / "vindex_out")).resolve()
    force = args.force or _env_truthy("VINDEX_FORCE")
    _log("=" * 60)
    _log("[vindex] MODE: CLI")
    _log(
        f"[vindex]   model={args.model!r}  out={out}  "
        f"cache={args.cache_dir.resolve()}  down_top_k={args.down_top_k}  force={force}"
    )
    if not force and browse_vindex_is_complete(out):
        _log(f"[vindex] SKIP — vindex already complete at {out} (no HF download)")
        _log("[vindex]   use --force or VINDEX_FORCE=1 to rebuild from scratch")
        return
    mdir = resolve_model_dir(args.model, args.cache_dir)
    extract_browse(
        mdir,
        out,
        model_name=args.model,
        down_top_k=args.down_top_k,
        save_zip=args.zip,
        copy_to_dir=args.copy_to,
        sync_colab_drive=args.sync_drive,
        force=force,
    )


if __name__ == "__main__":
    if not user_argv():
        _log("[vindex] entry: __main__ → Colab default path (empty argv after kernel strip)")
        run_default_colab()
    else:
        _log(f"[vindex] entry: __main__ → CLI path  argv={user_argv()!r}")
        main_cli()
