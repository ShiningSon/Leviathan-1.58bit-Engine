#!/usr/bin/env python3
"""Download and verify a public Leviathan Hugging Face release package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_REPO_ID = "ShiningSon/Leviathan-MLGRU-200M-TinyStories-Instruct-v09a"
DEFAULT_DOWNLOAD_DIR = "hf_verify_v09a"
REQUIRED_PATHS = {
    "README.md",
    "leviathan_mlgru_200m_instruct_v09a.bin",
    "leviathan_mlgru_200m_instruct_v09a_meta.json",
    "leviathan_mlgru_tokenizer/tokenizer.json",
    "leviathan_mlgru_tokenizer/tokenizer_config.json",
    "report.json",
    "sample_outputs.txt",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_relative_path(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def normalize_card_text(text: str) -> str:
    """Normalize only line endings and trailing whitespace for card comparison."""
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")).rstrip() + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_package_files(download_dir: Path) -> set[str]:
    files: set[str] = set()
    for path in download_dir.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(download_dir)
        if ".cache" in relative.parts or "__pycache__" in relative.parts:
            continue
        files.add(normalize_relative_path(relative))
    return files


def load_json_object(path: Path, label: str, errors: list[str]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"{label} is not valid JSON: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{label} must contain a JSON object")
        return None
    return value


def validate_download(
    download_dir: Path,
    remote_files: Iterable[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    errors: list[str] = []
    remote_set = {normalize_relative_path(path) for path in remote_files}
    missing_remote = sorted(REQUIRED_PATHS - remote_set)
    if missing_remote:
        errors.append(f"public repository is missing required files: {', '.join(missing_remote)}")

    local_set = local_package_files(download_dir) if download_dir.is_dir() else set()
    missing_local = sorted(remote_set - local_set)
    extra_local = sorted(local_set - remote_set)
    if missing_local:
        errors.append(f"download is missing public files: {', '.join(missing_local)}")
    if extra_local:
        errors.append(f"download contains files not present in the public revision: {', '.join(extra_local)}")

    for relative in sorted(REQUIRED_PATHS & local_set):
        path = download_dir / Path(relative)
        if path.stat().st_size <= 0:
            errors.append(f"required file is empty: {relative}")

    meta_path = download_dir / "leviathan_mlgru_200m_instruct_v09a_meta.json"
    report_path = download_dir / "report.json"
    tokenizer_path = download_dir / "leviathan_mlgru_tokenizer" / "tokenizer.json"
    tokenizer_config_path = download_dir / "leviathan_mlgru_tokenizer" / "tokenizer_config.json"
    if meta_path.is_file():
        load_json_object(meta_path, "metadata JSON", errors)
    if report_path.is_file():
        load_json_object(report_path, "report.json", errors)
    if tokenizer_path.is_file():
        tokenizer = load_json_object(tokenizer_path, "tokenizer.json", errors)
        if tokenizer is not None and not tokenizer.get("model"):
            errors.append("tokenizer.json does not contain a tokenizer model")
    if tokenizer_config_path.is_file():
        load_json_object(tokenizer_config_path, "tokenizer_config.json", errors)

    records: list[dict[str, Any]] = []
    for relative in sorted(remote_set & local_set):
        path = download_dir / Path(relative)
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return errors, records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=repo_root() / DEFAULT_DOWNLOAD_DIR,
        help="Clean local directory to synchronize and verify.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        print("[FAIL] huggingface_hub is required; install requirements.txt")
        return 1

    api = HfApi(token=False)
    try:
        info = api.model_info(args.repo_id, revision=args.revision)
        resolved_revision = info.sha
        remote_files = sorted(api.list_repo_files(args.repo_id, revision=resolved_revision))
    except Exception as exc:
        print(f"[FAIL] cannot inspect public repository {args.repo_id}: {exc}")
        return 1

    print(f"Repository: {args.repo_id}")
    print(f"Requested revision: {args.revision}")
    print(f"Resolved revision: {resolved_revision}")
    print("Public files:")
    for relative in remote_files:
        print(f"- {relative}")

    download_dir = args.download_dir.resolve()
    download_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=args.repo_id,
            revision=resolved_revision,
            local_dir=str(download_dir),
            token=False,
        )
    except Exception as exc:
        print(f"[FAIL] public snapshot download failed: {exc}")
        return 1

    errors, records = validate_download(download_dir, remote_files)
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1

    total_bytes = sum(record["bytes"] for record in records)
    print("Verified files:")
    for record in records:
        print(f"[OK] {record['path']} bytes={record['bytes']} sha256={record['sha256']}")
    print(f"[OK] {len(records)} files, total bytes={total_bytes}")
    print(f"[OK] clean public package verified at {download_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
