#!/usr/bin/env python3
"""Small standard-library helpers shared by release checks and tests."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from urllib.parse import unquote


FORBIDDEN_ARTIFACT_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".log",
    ".npy",
    ".npz",
    ".pt",
    ".pth",
    ".safetensors",
    ".zip",
}
FORBIDDEN_ARTIFACT_DIRECTORIES = {
    ".cache",
    ".modal",
    ".venv",
    "benchmark_runs",
    "checkpoints",
    "exports",
    "runs",
    "venv",
}


def estimate_mlgru_params(
    vocab_size: int,
    hidden_size: int,
    n_layers: int,
    intermediate_size: int,
) -> int:
    """Match the tied-embedding MLGRU estimate used by the training scripts."""
    embedding = vocab_size * hidden_size
    linear_per_layer = (4 * hidden_size * hidden_size) + (
        3 * hidden_size * intermediate_size
    )
    norm_per_layer = (3 * hidden_size) + intermediate_size
    final_norm = hidden_size
    return embedding + n_layers * (linear_per_layer + norm_per_layer) + final_norm


def is_forbidden_tracked_path(path: str) -> bool:
    """Return whether a Git path looks like a generated or private artifact."""
    normalized = path.replace("\\", "/")
    pure_path = PurePosixPath(normalized)
    lowered_parts = [part.lower() for part in pure_path.parts]
    if pure_path.suffix.lower() in FORBIDDEN_ARTIFACT_SUFFIXES:
        return True
    if any(part in FORBIDDEN_ARTIFACT_DIRECTORIES for part in lowered_parts):
        return True
    return any(part.startswith("leviathan_mlgru_") for part in lowered_parts[:-1])


def find_developer_absolute_paths(text: str) -> list[str]:
    """Find workstation-specific absolute paths while allowing documented Modal paths."""
    patterns = (
        r"[A-Za-z]:\\Users\\[^\\\s]+",
        r"/Users/[^/\s]+",
        r"/home/[^/\s]+",
    )
    matches: set[str] = set()
    for pattern in patterns:
        matches.update(re.findall(pattern, text, flags=re.IGNORECASE))
    return sorted(matches)


def markdown_internal_targets(markdown_path: Path, text: str) -> list[Path]:
    """Resolve relative Markdown links that should point to repository files."""
    targets: list[Path] = []
    for raw_target in re.findall(r"!?(?:\[[^\]]*\])\(([^)]+)\)", text):
        target = raw_target.strip().strip("<>")
        if not target or target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        target = unquote(target.split("#", 1)[0].split("?", 1)[0]).strip()
        if not target:
            continue
        targets.append((markdown_path.parent / target).resolve())
    return targets
