#!/usr/bin/env python3
"""
Multi-token **chat / completion** for le flux Colab **A‑Vindex**, aligné sur
``vindex-infernce-next.py`` mais avec la pile **avindex** (``avindex-infer`` +
``avindex_attention``).

* L’**id HF du modèle** vient du vindex (``index.json`` → ``"model"``), avec
  override optionnel ``VINDEX_HF_MODEL`` / ``--hf-model`` — comme
  ``vindex-infer`` / ``avindex-infer``.
* **Un seul chargement** tokenizer + poids HF pour toute la session (pas de
  reload par message).
* Avant ``AutoModelForCausalLM.from_pretrained``, on appelle le patch
  ``_transformers_skip_torchvision_for_text_models()`` de ``avindex_attention``
  (même idée que ``hf_next_token_topk``) pour éviter l’import torchvision
  incompatible sur certains Colabs.

Chemins
--------
* **Transformers** : ``generate()`` multi-tokens (échantillonnage / stream).
* **Native ``larql``** (vindex niveau inference + bindings) : greedy
  multi-étapes ``infer(..., top_k_predictions=1)`` comme ``vindex-infernce-next``.

Build browse optionnel via ``vindex.extract_browse``. ``--infer-step`` délègue
à la sonde **centroid + HF top-k** de ``avindex-infer`` (une passe diagnostic).

Dépendances
-----------
  pip install transformers torch accelerate
  (+ dépendances ``avindex_attention`` / ``vindex.py`` selon usage)

Notebook (argv vide)
--------------------
  VINDEX_DIR          répertoire vindex (défaut: ./.larql_colab/vindex_out)
  VINDEX_HF_MODEL     override id HF (optionnel)
  VINDEX_MODEL        uniquement si ANEXT_BUILD=1 (id HF pour extract)
  ANEXT_BUILD=1       lancer ``extract_browse`` d’abord
  ANEXT_CHAT=1        REPL conversationnel
  ANEXT_PROMPT        texte one-shot
  ANEXT_MAX_TOKENS    défaut 256
  ANEXT_TEMPERATURE   défaut 0.7
  ANEXT_STREAM=1      défaut activé

  AVINDEX_HF_CPU=1              ``device_map="cpu"`` pour le bloc HF session
  AVINDEX_ALLOW_TORCHVISION=1   désactive le patch torchvision (texte seul)

CLI
---
  python avindex-infer-next.py --vindex ./.larql_colab/vindex_out -p "Hello" --max-tokens 128
  python avindex-infer-next.py --vindex ./.larql_colab/vindex_out --chat
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


def _log_tokens_per_sec(phase: str, n_tokens: int, elapsed_s: float) -> None:
    if elapsed_s <= 0:
        return
    if n_tokens <= 0:
        _log(f"{phase}: 0 nouveaux tokens en {elapsed_s:.2f}s")
        return
    tps = n_tokens / elapsed_s
    _log(f"{phase}: {n_tokens} nouveaux tokens en {elapsed_s:.2f}s → {tps:.2f} tok/s")


def _scripts_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd().resolve()


def _load_module_from_file(module_name: str, file_path: Path) -> Any:
    if not file_path.is_file():
        raise FileNotFoundError(f"Script requis introuvable: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossible de charger le spec pour {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_vindex_scripts() -> Any:
    """Module ``vindex.py`` (extract_browse, work_base, resolve_model_dir)."""
    root = _scripts_dir()
    return _load_module_from_file("vindex_colab_anxt", root / "vindex.py")


def load_avindex_infer_scripts() -> Any:
    """``avindex-infer.py`` (load_avindex_attention, run_probe, default_vindex_dir, …)."""
    root = _scripts_dir()
    return _load_module_from_file("avindex_infer_colab_anxt", root / "avindex-infer.py")


def build_browse_vindex(
    model_id: str,
    *,
    out_dir: Path | None = None,
    down_top_k: int = 10,
    **persist: Any,
) -> Path:
    vix = load_vindex_scripts()
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


def resolve_hf_model_id_from_vindex(vindex_dir: Path, hf_model_override: str | None) -> str:
    main = json.loads((vindex_dir.resolve() / "index.json").read_text(encoding="utf-8"))
    mid = (hf_model_override or "").strip() or str(main.get("model", "")).strip()
    if not mid:
        raise SystemExit(
            "Chat impossible: pas d'id HF. Définir VINDEX_HF_MODEL / --hf-model, "
            'ou un champ "model" non vide dans index.json.'
        )
    return mid


def _try_larql_vindex_handle(vindex_dir: Path) -> Any | None:
    inf = load_avindex_infer_scripts()
    try:
        import larql  # type: ignore
    except ImportError:
        return None
    idx = json.loads((vindex_dir.resolve() / "index.json").read_text(encoding="utf-8"))
    if str(idx.get("extract_level", "browse")) == "browse":
        return None
    try:
        return larql.load(str(vindex_dir.resolve()))
    except Exception:  # noqa: BLE001
        return None


class AvindexConversationalSession:
    """
    Une session: id HF résolu depuis le vindex, un chargement tokenizer (+ modèle HF
    ou greedy larql). Patch torchvision appliqué sur le chemin Transformers.
    """

    def __init__(self, vindex_dir: Path, hf_model_override: str | None = None) -> None:
        self.vindex_dir = vindex_dir.resolve()
        self.model_id = resolve_hf_model_id_from_vindex(self.vindex_dir, hf_model_override)
        self._larql = _try_larql_vindex_handle(self.vindex_dir)
        self._tok: Any = None
        self._model: Any = None

        if self._larql is not None:
            _log(
                "chemin: **larql natif** — greedy multi-étapes INFER "
                f"(tokenizer depuis {self.model_id!r} pour le template chat)"
            )
            from transformers import AutoTokenizer

            self._tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            return

        _log(
            "chemin: **Transformers** — chargement unique "
            f"(patch torchvision comme avindex_attention) {self.model_id!r}"
        )
        inf = load_avindex_infer_scripts()
        av = inf.load_avindex_attention()
        av._transformers_skip_torchvision_for_text_models()
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise SystemExit("pip install transformers torch accelerate") from e

        use_cpu = os.environ.get("AVINDEX_HF_CPU", "").strip().lower() in ("1", "true", "yes", "on")
        device_map: str | dict[str, str] = "cpu" if use_cpu else "auto"
        dtype = torch.float32 if use_cpu else torch.float16

        t0 = time.perf_counter()
        self._tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        _log(f"poids HF prêts en {time.perf_counter() - t0:.1f}s (device_map={device_map!r})")
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
        _log(f"greedy larql terminé en {elapsed:.1f}s ({n_tok} étapes infer)")
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
            _log(f"génération HF terminée en {elapsed:.1f}s ({len(full)} car.)")
            _log_tokens_per_sec("HF generate (stream)", n_tok, elapsed)
            return full

        t1 = time.perf_counter()
        with torch.no_grad():
            out = self._model.generate(**gen_kw)
        new_ids = out[0, in_len:]
        elapsed = time.perf_counter() - t1
        text = self._tok.decode(new_ids, skip_special_tokens=True)
        n_tok = int(new_ids.shape[0])
        _log(f"génération HF terminée en {elapsed:.1f}s ({len(text)} car.)")
        _log_tokens_per_sec("HF generate", n_tok, elapsed)
        return text


def open_conversational_session(
    vindex_dir: Path,
    hf_model_override: str | None = None,
) -> AvindexConversationalSession:
    return AvindexConversationalSession(vindex_dir, hf_model_override)


def chat_loop(
    session: AvindexConversationalSession,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    stream: bool = True,
) -> None:
    _log("REPL chat — `/quit` ou ligne vide pour quitter")
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


def single_step_avindex_infer(
    vindex_dir: Path,
    prompt: str,
    *,
    hf_topk: int = 10,
    hf_model: str | None = None,
) -> None:
    """Centroid probe + HF top-k comme ``avindex-infer`` (une passe)."""
    inf = load_avindex_infer_scripts()
    vdir = vindex_dir.resolve()
    av = inf.load_avindex_attention()
    vx = av._import_vindex()
    main = json.loads((vdir / "index.json").read_text(encoding="utf-8"))
    mid = (hf_model or "").strip() or os.environ.get("VINDEX_MODEL", "").strip() or str(main.get("model", "")).strip()
    if not mid:
        raise SystemExit('Définir VINDEX_MODEL, --hf-model ou index.json "model".')
    mdir = vx.resolve_model_dir(mid, vx.work_base() / "hf_models").resolve()
    cache = av.KProjWeightCache(mdir)
    # run_probe exige un ``mid`` non vide dans index/env même si model_dir est passé
    prev_vm = os.environ.get("VINDEX_MODEL")
    try:
        os.environ["VINDEX_MODEL"] = mid
        out = inf.run_probe(
            vdir,
            prompt=prompt,
            layer=None,
            kv_head=0,
            model_dir=mdir,
            weight_cache=cache,
        )
    finally:
        if prev_vm is None:
            os.environ.pop("VINDEX_MODEL", None)
        else:
            os.environ["VINDEX_MODEL"] = prev_vm
    print(json.dumps({"centroid_probe": out}, indent=2), flush=True)
    if hf_topk > 0:
        try:
            tops = av.hf_next_token_topk(str(mdir), prompt, top_k=hf_topk)
            print(json.dumps({"hf_next_token_topk": tops}, indent=2), flush=True)
        except Exception as e:  # noqa: BLE001
            _log(f"HF top-k échoué: {e}")
            print(json.dumps({"hf_next_token_topk_error": str(e)}, indent=2), flush=True)


def run_default_cell() -> None:
    vix = load_vindex_scripts()
    base = vix.work_base()
    extract_model = os.environ.get("VINDEX_MODEL", "Qwen/Qwen3-0.6B").strip()
    if _env_truthy("ANEXT_BUILD"):
        if not extract_model:
            raise SystemExit("ANEXT_BUILD=1 requiert VINDEX_MODEL (id HF pour extract).")
        _log("ANEXT_BUILD=1 → construction browse vindex …")
        build_browse_vindex(extract_model)

    inf = load_avindex_infer_scripts()
    vdir = Path(os.environ.get("VINDEX_DIR", str(inf.default_vindex_dir()))).expanduser().resolve()
    hf_ov = os.environ.get("VINDEX_HF_MODEL", "").strip() or None
    session = open_conversational_session(vdir, hf_ov)

    if _env_truthy("ANEXT_CHAT"):
        chat_loop(
            session,
            max_new_tokens=int(os.environ.get("ANEXT_MAX_TOKENS", "256")),
            temperature=float(os.environ.get("ANEXT_TEMPERATURE", "0.7")),
            stream=_env_truthy("ANEXT_STREAM", default=True),
        )
        return

    prompt = os.environ.get("ANEXT_PROMPT", "Hello! Briefly introduce yourself.").strip()
    mx = int(os.environ.get("ANEXT_MAX_TOKENS", "256"))
    temp = float(os.environ.get("ANEXT_TEMPERATURE", "0.7"))
    st = _env_truthy("ANEXT_STREAM", default=True)
    _log(f"one-shot model_id={session.model_id!r}  max_new_tokens={mx}")
    _log("--- génération ---")
    out = session.generate(
        prompt,
        max_new_tokens=mx,
        temperature=temp,
        stream=st,
        on_chunk=None,
    )
    if not st:
        print(out, flush=True)
    _log("--- fin ---")


def _env_truthy(name: str, *, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def main_cli() -> None:
    inf = load_avindex_infer_scripts()
    ap = argparse.ArgumentParser(
        description="Chat multi-tokens: id HF depuis le vindex (comme avindex-infer), un chargement."
    )
    ap.add_argument(
        "--vindex",
        type=Path,
        default=inf.default_vindex_dir(),
        help="Répertoire vindex (défaut: VINDEX_DIR ou ./.larql_colab/vindex_out)",
    )
    ap.add_argument(
        "--model",
        default=os.environ.get("VINDEX_MODEL", ""),
        help="Uniquement avec --build: id HF pour extract_browse (sinon VINDEX_MODEL)",
    )
    ap.add_argument("--build", action="store_true", help="Lancer vindex.extract_browse d'abord (--model requis)")
    ap.add_argument(
        "--hf-model",
        default=None,
        help="Override id HF pour le chat (sinon VINDEX_HF_MODEL ou index.json model)",
    )
    ap.add_argument(
        "--infer-step",
        action="store_true",
        help="Sonde centroid + HF top-k (avindex-infer) puis quitter",
    )
    ap.add_argument("--hf-topk", type=int, default=10, help="Avec --infer-step: top-k HF (0 = skip)")
    ap.add_argument("--prompt", "-p", default="Hello! Reply in one short sentence.")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--chat", action="store_true", help="REPL interactif")
    args, rest = ap.parse_known_args(user_argv())
    if rest:
        _log(f"avertissement: argv ignoré: {rest}")

    vdir = args.vindex.resolve()

    if args.build:
        em = (args.model or "").strip()
        if not em:
            raise SystemExit("--build requiert --model ou VINDEX_MODEL (id HF pour extract).")
        build_browse_vindex(em)

    hf = (args.hf_model or os.environ.get("VINDEX_HF_MODEL", "").strip() or None)

    if args.infer_step:
        single_step_avindex_infer(vdir, args.prompt, hf_topk=args.hf_topk, hf_model=hf)
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
        _log("entrée: notebook défaut (argv vide)")
        run_default_cell()
    else:
        _log(f"entrée: CLI argv={user_argv()!r}")
        main_cli()
