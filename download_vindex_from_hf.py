#!/usr/bin/env python3
"""
Download a Vindex directory from the Hugging Face Hub (snapshot of all repo files).

Default: Gemma 3 4B IT Vindex — https://huggingface.co/cronos3k/gemma-3-4b-it-vindex
Requires: pip install huggingface_hub
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download a Vindex from Hugging Face Hub into a local folder."
    )
    ap.add_argument(
        "--repo-id",
        default="cronos3k/gemma-3-4b-it-vindex",
        help="Hugging Face dataset/model repo id",
    )
    ap.add_argument(
        "--local-dir",
        type=Path,
        default=Path("gemma3-4b.vindex"),
        help="Destination directory (created if missing)",
    )
    ap.add_argument(
        "--token",
        default=None,
        help="HF token (optional; else uses HF_TOKEN env or cached login)",
    )
    args = ap.parse_args()

    local_dir = args.local_dir.resolve()
    print(f"Downloading {args.repo_id} into {local_dir} …")

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(local_dir),
        resume_download=True,
        token=args.token,
    )

    print(f"Done. Vindex files are under: {local_dir}")


if __name__ == "__main__":
    main()
