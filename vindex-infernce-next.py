#!/usr/bin/env python3
"""
Multi-token **chat / completion** aligned with `vindex-infer.py`.

The **HF model id** is taken only from the vindex (same rules as INFER):
`index.json` field `"model"`, optionally overridden by `VINDEX_HF_MODEL` / `--hf-model`.
We **load that model once** (tokenizer + weights) and reuse it for the whole conversation —
no per-message reload, and no separate `--model` for generation unless you only use
`--build` (extract still needs `VINDEX_MODEL` or `--model`).

Paths
-----
* **Transformers** (browse vindex or no `larql`): one `from_pretrained`, then
  `model.generate` until EOS / `max_new_tokens` — full multi-token chat.
* **Native `larql`** (inference vindex + bindings): no HF weights; greedy
  multi-step by repeating `v.infer(..., top_k_predictions=1)` (not identical to
  `generate()`, but uses the same mmap INFER path as `vindex-infer.py`).

Optional **browse vindex build** via `vindex.extract_browse`. Optional one-shot INFER
via `vindex-infer.run_infer`.

Dependencies
------------
  pip install transformers torch accelerate
  (+ `huggingface_hub` if you build from HF id via `vindex.py`)

Environment (cell, empty argv)
-------------------------------
  VINDEX_DIR         vindex path (default: ./.larql_colab/vindex_out)
  VINDEX_HF_MODEL    optional HF id override (same as vindex-infer)
  VINDEX_MODEL       only for VNEXT_BUILD=1 (extract source id)
  VNEXT_BUILD=1      run `extract_browse` first
  VNEXT_CHAT=1       REPL
  VNEXT_PROMPT       one-shot user text
  VNEXT_MAX_TOKENS   default 256
  VNEXT_TEMPERATURE  default 0.7
  VNEXT_STREAM=1     default on

CLI examples
------------
  python vindex-infernce-next.py --vindex ./.larql_colab/vindex_out -p "Hello" --max-tokens 128
  python vindex-infernce-next.py --vindex ./.larql_colab/vindex_out --chat
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

# Optional torch / transformers (lazy in session); type hints only where needed.

# ── Colab argv helpers (same pattern as vindex.py) ───────────────────────


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
    """Log generated throughput (elapsed excludes prompt processing unless noted)."""
    if elapsed_s <= 0:
        return
    if n_tokens <= 0:
        _log(f"[vnxt] {phase}: 0 new tokens in {elapsed_s:.2f}s")
        return
    tps = n_tokens / elapsed_s
    _log(f"[vnxt] {phase}: {n_tokens} new tokens in {elapsed_s:.2f}s → {tps:.2f} tok/s")


def _scripts_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


def _load_module_from_file(module_name: str, file_path: Path) -> Any:
    if not file_path.is_file():
        raise FileNotFoundError(f"Required script missing: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {file_path}")
    mod = importlib.util.module_from_spec(spec)
    # Required so dataclasses / typing can resolve cls.__module__ during exec (notebook-safe).
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_vindex_scripts() -> tuple[Any, Any]:
    """Return (vindex_py_module, vindex_infer_py_module)."""
    root = _scripts_dir()
    vix = _load_module_from_file("vindex_colab", root / "vindex.py")
    inf = _load_module_from_file("vindex_infer_colab", root / "vindex-infer.py")
    return vix, inf


# ── Public API: build + infer + multi-token chat ──────────────────────────


def build_browse_vindex(
    model_id: str,
    *,
    out_dir: Path | None = None,
    down_top_k: int = 10,
    **persist: Any,
) -> Path:
    """
    Run the pure-Python browse extract from `vindex.py`.
    `persist` forwards optional keys: save_zip, copy_to_dir, sync_colab_drive.
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
    hf_model: str | None = None,
) -> list[tuple[str, float]]:
    """Delegates to `vindex-infer.run_infer` (native larql INFER or HF top-k fallback)."""
    _, inf = load_vindex_scripts()
    return inf.run_infer(vindex_dir.resolve(), prompt, top_k, hf_model)


def resolve_hf_model_id_from_vindex(vindex_dir: Path, hf_model_override: str | None) -> str:
    """
    Same rule as `vindex-infer.run_infer` for the Transformers model id:
    `hf_model_override` if set, else `index.json` → `"model"`.
    """
    _, inf = load_vindex_scripts()
    meta = inf.load_index(vindex_dir.resolve())
    mid = (hf_model_override or "").strip() or str(meta.get("model", "")).strip()
    if not mid:
        raise SystemExit(
            "Cannot run chat: no HF model id. Set VINDEX_HF_MODEL / --hf-model, "
            'or ensure index.json has a non-empty "model" field (same as vindex-infer).'
        )
    return mid


def _try_larql_vindex_handle(vindex_dir: Path) -> Any | None:
    """Same gates as `vindex-infer.try_larql_infer`, but return the loaded handle or None."""
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


class VindexConversationalSession:
    """
    One resolved HF `model` id (from vindex + optional override), one load for the session.

    * If native `larql` + inference vindex works: tokenizer from HF id only; text is
      extended by repeated `infer(..., top_k_predictions=1)` (multi-token, greedy).
    * Else: full `AutoModelForCausalLM` load (same as `vindex-infer` fallback), then
      `generate()` for true multi-token sampling until EOS / cap.
    """

    def __init__(self, vindex_dir: Path, hf_model_override: str | None = None) -> None:
        self.vindex_dir = vindex_dir.resolve()
        self.model_id = resolve_hf_model_id_from_vindex(self.vindex_dir, hf_model_override)
        self._larql = _try_larql_vindex_handle(self.vindex_dir)
        self._tok: Any = None
        self._model: Any = None

        if self._larql is not None:
            _log(
                f"[vnxt] path: **native larql** — greedy multi-step INFER "
                f"(tokenizer from {self.model_id!r} for chat template only)"
            )
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            return

        _log(
            f"[vnxt] path: **Transformers** — same stack as vindex-infer fallback, "
            f"single load of {self.model_id!r}"
        )
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise SystemExit("pip install transformers torch accelerate") from e

        t0 = time.perf_counter()
        self._tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        _log(f"[vnxt] HF weights ready in {time.perf_counter() - t0:.1f}s")
        if self._tok.pad_token_id is None and self._tok.eos_token_id is not None:
            self._tok.pad_token = self._tok.eos_token

    def messages_to_prompt(self, messages: list[dict[str, str]]) -> str:
        if hasattr(self._tok, "apply_chat_template") and self._tok.chat_template is not None:
            return self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        parts = [f"{m['role'].upper()}: {m['content']}" for m in messages]
        parts.append("ASSISTANT:")
        return "\n".join(parts)

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
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
        assert self._model is not None
        return self._generate_hf(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
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
        """Repeat INFER top-1 (same primitive as vindex-infer) to build a longer reply."""
        text = prompt
        t0 = time.perf_counter()
        out_chunks: list[str] = []
        for _ in range(max_new_tokens):
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

    def _generate_hf(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        stream: bool,
        on_chunk: Callable[[str], None] | None,
    ) -> str:
        import torch
        from threading import Thread

        from transformers import TextIteratorStreamer

        inputs = self._tok(prompt, return_tensors="pt")
        dev = next(self._model.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        in_len = inputs["input_ids"].shape[1]

        gen_kw: dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": max(temperature, 1e-5) if do_sample else None,
            "top_p": top_p if do_sample else None,
            "pad_token_id": self._tok.pad_token_id,
            "eos_token_id": self._tok.eos_token_id,
        }
        gen_kw = {k: v for k, v in gen_kw.items() if v is not None}

        if stream:
            streamer = TextIteratorStreamer(self._tok, skip_prompt=True, skip_special_tokens=True)
            gen_kw["streamer"] = streamer
            t1 = time.perf_counter()
            th = Thread(target=self._model.generate, kwargs=gen_kw)
            th.start()
            parts: list[str] = []
            for chunk in streamer:
                parts.append(chunk)
                if on_chunk:
                    on_chunk(chunk)
                else:
                    print(chunk, end="", flush=True)
            th.join()
            if on_chunk is None:
                print(flush=True)
            elapsed = time.perf_counter() - t1
            full = "".join(parts)
            n_tok = len(self._tok.encode(full, add_special_tokens=False))
            _log(f"[vnxt] HF generation finished in {elapsed:.1f}s ({len(full)} chars)")
            _log_tokens_per_sec("HF generate (stream)", n_tok, elapsed)
            return full

        t1 = time.perf_counter()
        with torch.no_grad():
            out = self._model.generate(**gen_kw)
        new_ids = out[0, in_len:]
        elapsed = time.perf_counter() - t1
        text = self._tok.decode(new_ids, skip_special_tokens=True)
        n_tok = int(new_ids.shape[0])
        _log(f"[vnxt] HF generation finished in {elapsed:.1f}s ({len(text)} chars)")
        _log_tokens_per_sec("HF generate", n_tok, elapsed)
        return text


def open_conversational_session(
    vindex_dir: Path,
    hf_model_override: str | None = None,
) -> VindexConversationalSession:
    """Preferred entry: model id from vindex (+ optional override), single load."""
    return VindexConversationalSession(vindex_dir, hf_model_override)


def chat_loop(
    session: VindexConversationalSession,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
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
            temperature=temperature,
            stream=stream,
            on_chunk=(lambda c: print(c, end="", flush=True)) if stream else None,
        )
        if stream:
            print(flush=True)
        else:
            print(reply, flush=True)
        history.append({"role": "assistant", "content": reply})


def run_default_cell() -> None:
    vix, inf = load_vindex_scripts()
    base = vix.work_base()
    extract_model = os.environ.get("VINDEX_MODEL", "Qwen/Qwen3-0.6B").strip()
    if _env_truthy("VNEXT_BUILD"):
        if not extract_model:
            raise SystemExit("VNEXT_BUILD=1 requires VINDEX_MODEL (HF id for extract).")
        _log("[vnxt] VNEXT_BUILD=1 → building browse vindex …")
        build_browse_vindex(extract_model)

    vdir = Path(os.environ.get("VINDEX_DIR", str(base / "vindex_out"))).expanduser().resolve()
    hf_ov = os.environ.get("VINDEX_HF_MODEL", "").strip() or None
    session = open_conversational_session(vdir, hf_ov)

    if _env_truthy("VNEXT_CHAT"):
        chat_loop(
            session,
            max_new_tokens=int(os.environ.get("VNEXT_MAX_TOKENS", "256")),
            temperature=float(os.environ.get("VNEXT_TEMPERATURE", "0.7")),
            stream=_env_truthy("VNEXT_STREAM", default=True),
        )
        return

    prompt = os.environ.get("VNEXT_PROMPT", "Hello! Briefly introduce yourself.").strip()
    mx = int(os.environ.get("VNEXT_MAX_TOKENS", "256"))
    temp = float(os.environ.get("VNEXT_TEMPERATURE", "0.7"))
    st = _env_truthy("VNEXT_STREAM", default=True)
    _log(f"[vnxt] one-shot model_id={session.model_id!r}  max_new_tokens={mx}")
    _log("--- generation ---")
    out = session.generate(
        prompt,
        max_new_tokens=mx,
        temperature=temp,
        stream=st,
        on_chunk=None,
    )
    if not st:
        print(out, flush=True)
    _log("--- done ---")


def _env_truthy(name: str, *, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def main_cli() -> None:
    _, inf = load_vindex_scripts()
    ap = argparse.ArgumentParser(
        description="Conversational chat: model id from vindex (like vindex-infer), single HF load."
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
        help="Only for --build: HF id to extract into browse vindex (else VINDEX_MODEL)",
    )
    ap.add_argument("--build", action="store_true", help="Run vindex.extract_browse first (--model required)")
    ap.add_argument(
        "--hf-model",
        default=None,
        help="Override HF model id for chat (else VINDEX_HF_MODEL env or index.json model)",
    )
    ap.add_argument(
        "--infer-step",
        action="store_true",
        help="Run single-step INFER via vindex-infer.py then exit",
    )
    ap.add_argument("--prompt", "-p", default="Hello! Reply in one short sentence.")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
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

    hf = (args.hf_model or os.environ.get("VINDEX_HF_MODEL", "").strip() or None)

    if args.infer_step:
        single_step_infer(vdir, args.prompt, top_k=5, hf_model=hf)
        return

    session = open_conversational_session(vdir, hf)

    if args.chat:
        chat_loop(
            session,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            stream=not args.no_stream,
        )
        return

    out = session.generate(
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
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
