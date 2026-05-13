#!/usr/bin/env python3
"""
**Multi-token chat with A-Vindex annotation — vindex-only.**

Mirror of ``vindex-infernce-next.py``, but with an optional **centroid probe**
shown before each generation step so users see what the attention side of the
vindex "sees" at the same time as the FFN walk produces a reply.

Everything reads from the vindex directory:
* tokenizer            → ``<vindex>/tokenizer.json``
* embeddings           → ``embeddings.bin``
* FFN walk             → ``gate_vectors.bin`` + ``down_meta.bin``
* attention sidecar    → ``attn_centroids.bin`` + ``attn_k_proj_weights.bin``

**No Hugging Face model is loaded for inference.** The only HF-touching path is
``vindex.extract_browse`` (triggered by ``ANEXT_BUILD=1`` / ``--build``), which
downloads the source model once to build the vindex.

Notebook (empty argv)
---------------------
  VINDEX_DIR           vindex path (default: ./.larql_colab/vindex_out)
  VINDEX_MODEL         only used by ANEXT_BUILD=1 (HF id for extract)
  ANEXT_BUILD=1        run ``vindex.extract_browse`` first  (HF download)
  ANEXT_CHAT=1         conversational REPL
  ANEXT_PROMPT         one-shot user text
  ANEXT_MAX_TOKENS     default 256
  ANEXT_STREAM=1       default on
  ANEXT_PROBE=1        show the centroid probe before each generation
  AVINDEX_PROBE_LAYER  optional explicit layer (else middle/knowledge band)
  AVINDEX_PROBE_KV_HEAD default 0
  VINDEX_WALK_FEATURES features kept per layer (default: 32)
  VINDEX_WALK_LAYERS   comma list or band name (syntax/knowledge/output)

CLI
---
  python avindex-infer-next.py --vindex ./.larql_colab/vindex_out -p "Hello" --max-tokens 128
  python avindex-infer-next.py --vindex ./.larql_colab/vindex_out --chat --probe
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable


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


def _log(msg: str) -> None:
    print(f"[anxt] {msg}", flush=True)


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


def _load_module_from_file(module_name: str, file_path: Path) -> Any:
    if not file_path.is_file():
        raise FileNotFoundError(f"Required script missing: {file_path}")
    if module_name in sys.modules:
        del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_vnxt() -> Any:
    """Load ``vindex-infernce-next.py`` (session, chat loop, build helper)."""
    return _load_module_from_file("vindex_next_colab_anxt", _scripts_dir() / "vindex-infernce-next.py")


def load_avindex_attention() -> Any:
    return _load_module_from_file("avindex_attention_colab_anxt", _scripts_dir() / "avindex_attention.py")


# ── A-Vindex probe wrapper ────────────────────────────────────────────────


def _default_probe_layer(main_index: dict[str, Any]) -> int:
    av = load_avindex_attention()
    return av._default_probe_layer(main_index)


def run_centroid_probe(
    vindex_dir: Path,
    *,
    prompt: str,
    layer: int | None,
    kv_head: int,
    reader: Any = None,
) -> dict[str, Any]:
    av = load_avindex_attention()
    main = json.loads((vindex_dir / "index.json").read_text(encoding="utf-8"))
    L = int(layer) if layer is not None else _default_probe_layer(main)
    return av.probe_last_token_k_space(
        vindex_dir.resolve(), prompt=prompt, layer=L, kv_head=kv_head, reader=reader
    )


# ── Annotated session: walk + optional centroid probe ────────────────────


class AvindexConversationalSession:
    """
    Thin wrapper around ``VindexConversationalSession`` that optionally prints
    a centroid probe before each generation. Reuses one ``VindexWalkRunner``
    and one ``AvindexAttentionReader`` for the whole session.
    """

    def __init__(
        self,
        vindex_dir: Path,
        *,
        probe: bool = False,
        probe_layer: int | None = None,
        probe_kv_head: int = 0,
        system: str | None = None,
    ) -> None:
        self.vindex_dir = vindex_dir.resolve()
        self.probe = bool(probe)
        self.probe_layer = probe_layer
        self.probe_kv_head = int(probe_kv_head)
        self._vnxt = load_vnxt()
        self.session = self._vnxt.open_conversational_session(self.vindex_dir, system=system)
        self._reader: Any | None = None
        if self.probe:
            av = load_avindex_attention()
            try:
                self._reader = av.AvindexAttentionReader(self.vindex_dir)
            except FileNotFoundError as e:
                _log(f"probe disabled — {e}")
                self.probe = False

    def messages_to_prompt(self, messages: list[dict[str, str]]) -> str:
        return self.session.messages_to_prompt(messages)

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def __enter__(self) -> AvindexConversationalSession:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _emit_probe(self, prompt: str) -> None:
        if not (self.probe and self._reader is not None):
            return
        try:
            out = run_centroid_probe(
                self.vindex_dir,
                prompt=prompt,
                layer=self.probe_layer,
                kv_head=self.probe_kv_head,
                reader=self._reader,
            )
        except Exception as e:  # noqa: BLE001
            _log(f"probe failed (continuing without it): {e}")
            return
        _log("centroid probe (vindex-only attention summary)")
        _log(
            f"  layer={out['layer']} kv_head={out['kv_head']} "
            f"centroid_id={out['centroid_id']} cosine={out['cosine']:.4f} "
            f"last_token={out['last_token_piece']!r}"
        )

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        features_per_layer: int = 32,
        layers: list[int] | str | None = None,
        stream: bool = False,
        on_chunk: Callable[[str], None] | None = None,
    ) -> str:
        self._emit_probe(prompt)
        return self.session.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            features_per_layer=features_per_layer,
            layers=layers,
            stream=stream,
            on_chunk=on_chunk,
        )


def chat_loop(
    session: AvindexConversationalSession,
    *,
    max_new_tokens: int = 256,
    features_per_layer: int = 32,
    layers: list[int] | str | None = None,
    stream: bool = True,
) -> None:
    _log("chat REPL — `/quit` or empty line to exit")
    history: list[dict[str, str]] = []
    while True:
        try:
            line = input("You> ").strip()
        except EOFError:
            break
        if line in ("/quit", "/exit", ""):
            break
        history.append({"role": "user", "content": line})
        prompt = session.messages_to_prompt(history)
        print("Assistant> ", end="", flush=True)
        reply = session.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            features_per_layer=features_per_layer,
            layers=layers,
            stream=stream,
            on_chunk=(lambda c: print(c, end="", flush=True)) if stream else None,
        )
        if stream:
            print(flush=True)
        else:
            print(reply, flush=True)
        history.append({"role": "assistant", "content": reply})


# ── env / CLI ─────────────────────────────────────────────────────────────


def _parse_layers(value: str | None) -> list[int] | str | None:
    if value is None:
        return None
    s = value.strip()
    if not s:
        return None
    if s.isalpha():
        return s
    return [int(p.strip()) for p in s.split(",") if p.strip()]


def run_default_cell() -> None:
    vnxt = load_vnxt()
    base = vnxt.load_vindex_scripts()[0].work_base()
    extract_model = os.environ.get("VINDEX_MODEL", "Qwen/Qwen3-0.6B").strip()
    if _env_truthy("ANEXT_BUILD"):
        if not extract_model:
            raise SystemExit("ANEXT_BUILD=1 requires VINDEX_MODEL (HF id for extract).")
        _log("ANEXT_BUILD=1 → building browse vindex …")
        vnxt.build_browse_vindex(extract_model)

    vdir = Path(os.environ.get("VINDEX_DIR", str(base / "vindex_out"))).expanduser().resolve()
    probe_kv = int(os.environ.get("AVINDEX_PROBE_KV_HEAD", "0"))
    probe_layer_env = os.environ.get("AVINDEX_PROBE_LAYER", "").strip()
    probe_layer = int(probe_layer_env) if probe_layer_env else None
    features = int(os.environ.get("VINDEX_WALK_FEATURES", "32"))
    layers = _parse_layers(os.environ.get("VINDEX_WALK_LAYERS"))
    probe = _env_truthy("ANEXT_PROBE", default=False)

    with AvindexConversationalSession(
        vdir, probe=probe, probe_layer=probe_layer, probe_kv_head=probe_kv
    ) as session:
        if _env_truthy("ANEXT_CHAT"):
            chat_loop(
                session,
                max_new_tokens=int(os.environ.get("ANEXT_MAX_TOKENS", "256")),
                features_per_layer=features,
                layers=layers,
                stream=_env_truthy("ANEXT_STREAM", default=True),
            )
            return

        prompt = os.environ.get("ANEXT_PROMPT", "Hello! Briefly introduce yourself.").strip()
        mx = int(os.environ.get("ANEXT_MAX_TOKENS", "256"))
        st = _env_truthy("ANEXT_STREAM", default=True)
        _log(f"one-shot vindex={vdir}  max_new_tokens={mx}  probe={probe}")
        _log("--- generation ---")
        out = session.generate(
            prompt,
            max_new_tokens=mx,
            features_per_layer=features,
            layers=layers,
            stream=st,
            on_chunk=None,
        )
        if not st:
            print(out, flush=True)
        _log("--- done ---")


def main_cli() -> None:
    vnxt = load_vnxt()
    ap = argparse.ArgumentParser(
        description="Multi-token vindex chat with optional A-Vindex centroid probe (no HF model load)."
    )
    ap.add_argument(
        "--vindex",
        type=Path,
        default=vnxt.load_vindex_scripts()[1].default_vindex_dir(),
        help="Vindex directory (default: VINDEX_DIR or ./.larql_colab/vindex_out)",
    )
    ap.add_argument(
        "--model",
        default=os.environ.get("VINDEX_MODEL", ""),
        help="Only with --build: HF id for extract_browse (else VINDEX_MODEL)",
    )
    ap.add_argument("--build", action="store_true", help="Run vindex.extract_browse first (--model required)")
    ap.add_argument(
        "--probe",
        action="store_true",
        help="Show the A-Vindex centroid probe before each generation",
    )
    ap.add_argument("--probe-layer", type=int, default=-1, help="-1 = auto from layer_bands")
    ap.add_argument("--probe-kv-head", type=int, default=0)
    ap.add_argument("--prompt", "-p", default="Hello! Reply in one short sentence.")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument(
        "--features-per-layer",
        type=int,
        default=int(os.environ.get("VINDEX_WALK_FEATURES", "32")),
    )
    ap.add_argument(
        "--layers",
        default=os.environ.get("VINDEX_WALK_LAYERS", ""),
        help='comma list or band name ("syntax","knowledge","output")',
    )
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--chat", action="store_true", help="Interactive REPL")
    args, rest = ap.parse_known_args(user_argv())
    if rest:
        _log(f"warning: ignored argv: {rest}")

    vdir = args.vindex.resolve()

    if args.build:
        em = (args.model or "").strip()
        if not em:
            raise SystemExit("--build requires --model or VINDEX_MODEL (HF id for extract).")
        vnxt.build_browse_vindex(em)

    probe_layer = None if args.probe_layer < 0 else args.probe_layer
    layers = _parse_layers(args.layers) if args.layers else None

    with AvindexConversationalSession(
        vdir,
        probe=args.probe,
        probe_layer=probe_layer,
        probe_kv_head=args.probe_kv_head,
    ) as session:
        if args.chat:
            chat_loop(
                session,
                max_new_tokens=args.max_tokens,
                features_per_layer=args.features_per_layer,
                layers=layers,
                stream=not args.no_stream,
            )
            return

        out = session.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            features_per_layer=args.features_per_layer,
            layers=layers,
            stream=not args.no_stream,
            on_chunk=None,
        )
        if args.no_stream:
            print(out, flush=True)


if __name__ == "__main__":
    if not user_argv():
        _log("entry: notebook default (empty argv)")
        run_default_cell()
    else:
        _log(f"entry: CLI argv={user_argv()!r}")
        main_cli()
