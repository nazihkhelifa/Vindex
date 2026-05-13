#!/usr/bin/env python3
"""
**Multi-token vindex chat / completion — vindex-only.**

Same idea as ``vindex-infer.py`` but extended to a full generation loop:
greedy multi-token decoding driven by repeated walk-INFER top-1 calls. No
Hugging Face model is loaded; the tokenizer comes from
``<vindex>/tokenizer.json``.

Paths
-----
* **Native `larql`** (if the bindings are installed AND the vindex is at
  ``--level inference`` or higher): repeat ``v.infer(text, top_k_predictions=1)``.
* **Vindex walk** (always works on browse vindex): repeat
  ``VindexWalkRunner.predict_step`` with ``top_k=1``.

Optional browse-vindex build via ``vindex.extract_browse`` (the only HF-touching
path in this whole repo — extraction needs to download the source model once).

Dependencies
------------
  pip install numpy tokenizers
  (Optional native fast path: `larql` wheel — see crates/larql-python README.)

Environment (cell, empty argv)
-------------------------------
  VINDEX_DIR         vindex path (default: ./.larql_colab/vindex_out)
  VNEXT_BUILD=1      run `vindex.extract_browse` first  (HF download)
  VINDEX_MODEL       HF id (only used by VNEXT_BUILD=1)
  VNEXT_CHAT=1       REPL
  VNEXT_PROMPT       one-shot user text
  VNEXT_MAX_TOKENS   default 256
  VNEXT_STREAM=1     default on
  VINDEX_WALK_FEATURES  features kept per layer (default: 32)
  VINDEX_WALK_LAYERS    comma list or band name (syntax/knowledge/output)

CLI examples
------------
  python vindex-infernce-next.py --vindex ./.larql_colab/vindex_out -p "Hello" --max-tokens 128
  python vindex-infernce-next.py --vindex ./.larql_colab/vindex_out --chat
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
    print(msg, flush=True)


def _log_tokens_per_sec(phase: str, n_tokens: int, elapsed_s: float) -> None:
    if elapsed_s <= 0:
        return
    if n_tokens <= 0:
        _log(f"[vnxt] {phase}: 0 new tokens in {elapsed_s:.2f}s")
        return
    tps = n_tokens / elapsed_s
    _log(f"[vnxt] {phase}: {n_tokens} new tokens in {elapsed_s:.2f}s → {tps:.2f} tok/s")


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


def load_vindex_scripts() -> tuple[Any, Any]:
    """Return (vindex_py_module, vindex_infer_py_module)."""
    root = _scripts_dir()
    vix = _load_module_from_file("vindex_colab", root / "vindex.py")
    inf = _load_module_from_file("vindex_infer_colab", root / "vindex-infer.py")
    return vix, inf


# ── Public API ────────────────────────────────────────────────────────────


def build_browse_vindex(
    model_id: str,
    *,
    out_dir: Path | None = None,
    down_top_k: int = 10,
    **persist: Any,
) -> Path:
    """
    Run the pure-Python browse extract from ``vindex.py``. **This is the only
    HF-touching path in the repo** — once the vindex is built, every other
    script reads vindex files only.
    """
    vix, _ = load_vindex_scripts()
    base = vix.work_base()
    out = (out_dir or (base / "vindex_out")).resolve()
    cache = base / "hf_models"
    mdir = vix.resolve_model_dir(model_id, cache)
    return vix.extract_browse(
        mdir,
        out,
        model_name=model_id,
        down_top_k=down_top_k,
        save_zip=bool(persist.get("save_zip", False)),
        copy_to_dir=persist.get("copy_to_dir"),
        sync_colab_drive=bool(persist.get("sync_colab_drive", False)),
    )


def single_step_infer(
    vindex_dir: Path,
    prompt: str,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Delegate to ``vindex-infer.run_infer`` (native larql or vindex walk)."""
    _, inf = load_vindex_scripts()
    return inf.run_infer(vindex_dir.resolve(), prompt, top_k)


def _try_larql_vindex_handle(vindex_dir: Path) -> Any | None:
    """Return a loaded ``larql`` handle if available + the vindex has inference
    weights; ``None`` otherwise so we fall back to the walk."""
    _, inf = load_vindex_scripts()
    try:
        import larql  # type: ignore
    except ImportError:
        return None
    idx = inf.load_index(vindex_dir.resolve())
    if str(idx.get("extract_level", "browse")) == "browse":
        return None
    try:
        return larql.load(str(vindex_dir.resolve()))
    except Exception:  # noqa: BLE001
        return None


# ── Conversation: single load, repeated generation ───────────────────────


_DEFAULT_SYS_PROMPT = "You are a helpful assistant. Answer concisely."


def _format_chat_plain(messages: list[dict[str, str]], system: str | None = None) -> str:
    """Generic 'USER: ... ASSISTANT: ...' template. We deliberately stay
    away from model-specific special tokens (the vindex doesn't carry the
    Jinja2 chat template metadata)."""
    parts: list[str] = []
    if system:
        parts.append(f"SYSTEM: {system}")
    for m in messages:
        role = str(m.get("role", "user")).upper()
        content = str(m.get("content", ""))
        parts.append(f"{role}: {content}")
    parts.append("ASSISTANT:")
    return "\n".join(parts)


class VindexConversationalSession:
    """
    One ``VindexWalkRunner`` (or native ``larql`` handle) reused for every turn.

    No Hugging Face model is loaded. Tokenizer comes from the vindex.
    """

    def __init__(self, vindex_dir: Path, *, system: str | None = None) -> None:
        _, self._inf = load_vindex_scripts()
        self.vindex_dir = vindex_dir.resolve()
        self.system = system or _DEFAULT_SYS_PROMPT
        self._larql = _try_larql_vindex_handle(self.vindex_dir)
        if self._larql is not None:
            _log("[vnxt] path: **native larql** — greedy multi-step INFER on inference vindex")
            self._runner: Any | None = None
            return

        _log("[vnxt] path: **vindex walk (pure numpy)** — greedy multi-step on browse vindex")
        t0 = time.perf_counter()
        use_ctx: bool | None = None
        if os.environ.get("VINDEX_NO_ATTN_CTX", "").strip().lower() in ("1", "true", "yes", "on"):
            use_ctx = False
        self._runner = self._inf.VindexWalkRunner(self.vindex_dir, use_attention_context=use_ctx)
        _log(f"[vnxt] runner ready in {time.perf_counter() - t0:.2f}s")

    def messages_to_prompt(self, messages: list[dict[str, str]]) -> str:
        return _format_chat_plain(messages, system=self.system)

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
        if self._larql is not None:
            return self._generate_larql_greedy(
                prompt,
                max_new_tokens=max_new_tokens,
                stream=stream,
                on_chunk=on_chunk,
            )
        return self._generate_walk(
            prompt,
            max_new_tokens=max_new_tokens,
            features_per_layer=features_per_layer,
            layers=layers,
            stream=stream,
            on_chunk=on_chunk,
        )

    def _generate_larql_greedy(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        stream: bool,
        on_chunk: Callable[[str], None] | None,
    ) -> str:
        text = prompt
        t0 = time.perf_counter()
        out_chunks: list[str] = []
        for _ in range(int(max_new_tokens)):
            preds = self._larql.infer(text, top_k_predictions=1)
            if not preds:
                break
            piece = preds[0][0]
            if not piece:
                break
            out_chunks.append(piece)
            text += piece
            if stream:
                if on_chunk:
                    on_chunk(piece)
                else:
                    print(piece, end="", flush=True)
        if stream and on_chunk is None:
            print(flush=True)
        elapsed = time.perf_counter() - t0
        n_tok = len(out_chunks)
        _log(f"[vnxt] larql greedy finished in {elapsed:.1f}s ({n_tok} infer steps)")
        _log_tokens_per_sec("larql greedy (1 step ≈ 1 token)", n_tok, elapsed)
        return "".join(out_chunks)

    def _generate_walk(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        features_per_layer: int,
        layers: list[int] | str | None,
        stream: bool,
        on_chunk: Callable[[str], None] | None,
    ) -> str:
        assert self._runner is not None
        t0 = time.perf_counter()
        emitted: list[str] = []

        def _on_token(piece: str) -> None:
            emitted.append(piece)
            if stream:
                if on_chunk:
                    on_chunk(piece)
                else:
                    print(piece, end="", flush=True)

        text = self._runner.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            features_per_layer=features_per_layer,
            layers=layers,
            on_token=_on_token if stream else None,
        )
        if stream and on_chunk is None:
            print(flush=True)

        elapsed = time.perf_counter() - t0
        n_tok = len(emitted) if stream else max(1, len(text.split()))
        _log(f"[vnxt] walk generation finished in {elapsed:.1f}s ({len(text)} chars)")
        _log_tokens_per_sec("vindex walk (1 step ≈ 1 token)", n_tok, elapsed)
        return text


def open_conversational_session(
    vindex_dir: Path,
    *,
    system: str | None = None,
) -> VindexConversationalSession:
    return VindexConversationalSession(vindex_dir, system=system)


def chat_loop(
    session: VindexConversationalSession,
    *,
    max_new_tokens: int = 256,
    features_per_layer: int = 32,
    layers: list[int] | str | None = None,
    stream: bool = True,
) -> None:
    """REPL using the same loaded session for every turn (no reload)."""
    _log("[vnxt] chat REPL — `/quit` or empty line to exit")
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
    vix, _ = load_vindex_scripts()
    base = vix.work_base()
    extract_model = os.environ.get("VINDEX_MODEL", "Qwen/Qwen3-0.6B").strip()
    if _env_truthy("VNEXT_BUILD"):
        if not extract_model:
            raise SystemExit("VNEXT_BUILD=1 requires VINDEX_MODEL (HF id for extract).")
        _log("[vnxt] VNEXT_BUILD=1 → building browse vindex …")
        build_browse_vindex(extract_model)

    vdir = Path(os.environ.get("VINDEX_DIR", str(base / "vindex_out"))).expanduser().resolve()
    session = open_conversational_session(vdir)

    features = int(os.environ.get("VINDEX_WALK_FEATURES", "32"))
    layers = _parse_layers(os.environ.get("VINDEX_WALK_LAYERS"))

    if _env_truthy("VNEXT_CHAT"):
        chat_loop(
            session,
            max_new_tokens=int(os.environ.get("VNEXT_MAX_TOKENS", "256")),
            features_per_layer=features,
            layers=layers,
            stream=_env_truthy("VNEXT_STREAM", default=True),
        )
        return

    prompt = os.environ.get("VNEXT_PROMPT", "Hello! Briefly introduce yourself.").strip()
    mx = int(os.environ.get("VNEXT_MAX_TOKENS", "256"))
    st = _env_truthy("VNEXT_STREAM", default=True)
    _log(f"[vnxt] one-shot vindex={vdir}  max_new_tokens={mx}")
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
    _, inf = load_vindex_scripts()
    ap = argparse.ArgumentParser(
        description="Vindex-only conversational chat (no HF model load)"
    )
    ap.add_argument(
        "--vindex",
        type=Path,
        default=inf.default_vindex_dir(),
        help="Vindex directory (default: VINDEX_DIR or ./.larql_colab/vindex_out)",
    )
    ap.add_argument(
        "--model",
        default=os.environ.get("VINDEX_MODEL", ""),
        help="Only for --build: HF id to extract into a browse vindex (else VINDEX_MODEL)",
    )
    ap.add_argument("--build", action="store_true", help="Run vindex.extract_browse first (--model required)")
    ap.add_argument(
        "--infer-step",
        action="store_true",
        help="Run single-step INFER via vindex-infer.py then exit",
    )
    ap.add_argument("--prompt", "-p", default="Hello! Reply in one short sentence.")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument(
        "--features-per-layer",
        type=int,
        default=int(os.environ.get("VINDEX_WALK_FEATURES", "32")),
        help="how many top features per layer (default: 32)",
    )
    ap.add_argument(
        "--layers",
        default=os.environ.get("VINDEX_WALK_LAYERS", ""),
        help='comma list (e.g. "12,13,14") or band name ("syntax","knowledge","output")',
    )
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--chat", action="store_true", help="Interactive REPL")
    args, rest = ap.parse_known_args(user_argv())
    if rest:
        _log(f"[vnxt] warning: ignored argv: {rest}")

    vdir = args.vindex.resolve()

    if args.build:
        em = (args.model or "").strip()
        if not em:
            raise SystemExit("--build requires --model or VINDEX_MODEL (HF id for extract).")
        build_browse_vindex(em)

    if args.infer_step:
        single_step_infer(vdir, args.prompt, top_k=5)
        return

    session = open_conversational_session(vdir)
    layers = _parse_layers(args.layers) if args.layers else None

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
        _log("[vnxt] entry: notebook default (empty argv)")
        run_default_cell()
    else:
        _log(f"[vnxt] entry: CLI argv={user_argv()!r}")
        main_cli()
