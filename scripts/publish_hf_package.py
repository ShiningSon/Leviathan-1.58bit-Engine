"""Upload a local Leviathan model package folder to Hugging Face.

This helper intentionally does not manage tokens. Run `hf auth login` or
`huggingface-cli login` first and let huggingface_hub use the normal cache.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Hugging Face model repo and upload a local Leviathan package folder."
    )
    parser.add_argument("--repo-id", required=True, help="Target repo id, for example ShiningSon/name.")
    parser.add_argument("--folder", required=True, help="Local model package folder to upload.")
    parser.add_argument(
        "--private",
        nargs="?",
        const=True,
        default=False,
        type=parse_bool,
        help="Create the repo as private. Accepts true/false; default is false.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        raise SystemExit(f"folder does not exist or is not a directory: {folder}")

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(folder),
    )
    print(f"Uploaded {folder} to {args.repo_id}")


if __name__ == "__main__":
    main()
