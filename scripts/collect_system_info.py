#!/usr/bin/env python3
"""Collect privacy-limited hardware metadata for a Leviathan benchmark."""

from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MODEL_REPO = "ShiningSon/Leviathan-MLGRU-200M-TinyStories-Instruct-v09a"
DEFAULT_MODEL_REVISION = "116a857bdaf2a1118d479d52aedba7e65cbff960"
THREAD_ENV_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
PRIVATE_PATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/][^\s\"']+|(?<![:/.\w])/(?!/)[^\s\"']+)",
    re.IGNORECASE,
)
SECRET_VALUE_RE = re.compile(
    r"(?i)(?:(--?(?:token|password|secret|api[-_]?key)\s+)|"
    r"\b(?:HF_TOKEN|HUGGINGFACE_TOKEN|API_KEY|PASSWORD|SECRET)=)([^\s\"']+)"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_text(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = (completed.stdout or completed.stderr).strip()
    return output or None


def git_commit(root: Path | None = None) -> str | None:
    command = ["git", "rev-parse", "HEAD"]
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip().lower()
    return value if re.fullmatch(r"[0-9a-f]{40}", value) else None


def parse_linux_cpuinfo(text: str) -> dict[str, Any]:
    records: list[dict[str, str]] = []
    for block in text.strip().split("\n\n"):
        record: dict[str, str] = {}
        for line in block.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            record[key.strip()] = value.strip()
        if record:
            records.append(record)
    if not records:
        return {}
    first = records[0]
    physical_pairs = {
        (record.get("physical id"), record.get("core id"))
        for record in records
        if record.get("physical id") is not None and record.get("core id") is not None
    }
    return {
        "cpu_model": first.get("model name") or first.get("Hardware") or first.get("Processor"),
        "cpu_vendor": first.get("vendor_id") or first.get("CPU implementer"),
        "physical_cores": len(physical_pairs) or None,
    }


def linux_cpu_info() -> dict[str, Any]:
    path = Path("/proc/cpuinfo")
    if not path.is_file():
        return {}
    return parse_linux_cpuinfo(path.read_text(encoding="utf-8", errors="replace"))


def windows_cpu_info() -> dict[str, Any]:
    script = (
        "Get-CimInstance Win32_Processor | "
        "Select-Object Name,Manufacturer,NumberOfCores,NumberOfLogicalProcessors | "
        "ConvertTo-Json -Compress"
    )
    output = None
    for executable in ("powershell", "pwsh"):
        output = run_text([executable, "-NoProfile", "-Command", script])
        if output:
            break
    if not output:
        return {}
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return {}
    records = data if isinstance(data, list) else [data]
    records = [item for item in records if isinstance(item, dict)]
    if not records:
        return {}
    return {
        "cpu_model": str(records[0].get("Name") or "").strip() or None,
        "cpu_vendor": str(records[0].get("Manufacturer") or "").strip() or None,
        "physical_cores": sum(int(item.get("NumberOfCores") or 0) for item in records) or None,
        "logical_cores": sum(int(item.get("NumberOfLogicalProcessors") or 0) for item in records) or None,
    }


def generic_cpu_info() -> dict[str, Any]:
    system = platform.system()
    if system == "Linux":
        info = linux_cpu_info()
    elif system == "Windows":
        info = windows_cpu_info()
    else:
        info = {}
    info.setdefault("cpu_model", platform.processor().strip() or None)
    info.setdefault("cpu_vendor", None)
    info.setdefault("physical_cores", None)
    info.setdefault("logical_cores", os.cpu_count())
    return info


def total_memory_bytes() -> int | None:
    if platform.system() == "Windows":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        try:
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
        except (AttributeError, OSError):
            return None
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    total = int(pages) * int(page_size)
    return total if total > 0 else None


def configured_threads(explicit: int | None) -> int | None:
    if explicit is not None:
        return explicit
    for name in THREAD_ENV_NAMES:
        value = os.environ.get(name)
        if value and value.isdigit() and int(value) > 0:
            return int(value)
    return os.cpu_count()


def compiler_info(explicit_name: str | None, explicit_version: str | None) -> tuple[str | None, str | None]:
    if explicit_name:
        name = Path(explicit_name).name
        version = privacy_safe_command(explicit_version) if explicit_version else None
        return name, version
    candidates = (
        ("cl", ["cl"]),
        ("g++", ["g++", "--version"]),
        ("clang++", ["clang++", "--version"]),
    )
    for name, command in candidates:
        output = run_text(command)
        if output:
            return name, privacy_safe_command(output.splitlines()[0])
    return None, None


def pytorch_version() -> str | None:
    try:
        return importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return None


def privacy_safe_command(command: str) -> str:
    safe = PRIVATE_PATH_RE.sub("<redacted-absolute-path>", command.strip())
    return SECRET_VALUE_RE.sub(lambda match: f"{match.group(1) or '<redacted-secret>='}<redacted>", safe)


def build_manifest(args: argparse.Namespace, *, root: Path | None = None) -> dict[str, Any]:
    cpu = generic_cpu_info()
    compiler, compiler_version = compiler_info(args.compiler, args.compiler_version)
    commit = args.engine_commit or git_commit(root)
    return {
        "schema_version": 1,
        "timestamp_utc": utc_now(),
        "operating_system": platform.system() or "Unknown",
        "operating_system_version": platform.version() or None,
        "cpu_model": cpu.get("cpu_model"),
        "cpu_vendor": cpu.get("cpu_vendor"),
        "physical_cores": cpu.get("physical_cores"),
        "logical_cores": cpu.get("logical_cores"),
        "configured_threads": configured_threads(args.configured_threads),
        "ram_bytes": total_memory_bytes(),
        "compiler": compiler,
        "compiler_version": compiler_version,
        "python_version": platform.python_version(),
        "pytorch_version": pytorch_version(),
        "architecture": args.architecture,
        "model_repo": args.model_repo,
        "model_revision": args.model_revision.lower(),
        "engine_commit": commit,
        "benchmark_command": privacy_safe_command(args.benchmark_command),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, help="Write the manifest JSON to this path.")
    parser.add_argument("--architecture", default="mlgru")
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--engine-commit", help="Full Git commit; defaults to the current checkout.")
    parser.add_argument("--benchmark-command", default="not supplied")
    parser.add_argument("--configured-threads", type=int)
    parser.add_argument("--compiler")
    parser.add_argument("--compiler-version")
    return parser.parse_args()


def validate_cli_args(args: argparse.Namespace) -> None:
    if args.configured_threads is not None and args.configured_threads < 1:
        raise ValueError("--configured-threads must be at least 1")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.model_repo):
        raise ValueError("--model-repo must use owner/repository format")
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", args.model_revision):
        raise ValueError("--model-revision must be a pinned 40-64 character hex revision")
    if args.engine_commit and not re.fullmatch(r"[0-9a-fA-F]{40}", args.engine_commit):
        raise ValueError("--engine-commit must be a full 40-character Git SHA")


def print_summary(manifest: dict[str, Any]) -> None:
    print("Leviathan hardware manifest")
    for key in (
        "operating_system",
        "operating_system_version",
        "cpu_model",
        "cpu_vendor",
        "physical_cores",
        "logical_cores",
        "configured_threads",
        "ram_bytes",
        "compiler",
        "compiler_version",
        "python_version",
        "pytorch_version",
        "architecture",
        "model_repo",
        "model_revision",
        "engine_commit",
    ):
        print(f"- {key}: {manifest[key] if manifest[key] is not None else 'not available'}")
    print("Privacy: username, hostname, IP addresses, and local filesystem paths are not collected.")


def main() -> int:
    args = parse_args()
    try:
        validate_cli_args(args)
        manifest = build_manifest(args, root=Path(__file__).resolve().parents[1])
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if manifest["engine_commit"] is None:
        print("error: engine commit was not supplied and could not be read from Git", file=sys.stderr)
        return 2
    print_summary(manifest)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
