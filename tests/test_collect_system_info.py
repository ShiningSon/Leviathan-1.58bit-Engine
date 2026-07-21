from __future__ import annotations

import argparse
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from scripts import collect_system_info
from scripts.validate_benchmark_submission import validate_hardware_manifest


class CollectSystemInfoTests(unittest.TestCase):
    def test_privacy_safe_command_redacts_private_absolute_paths(self) -> None:
        command = (
            r"python C:\Users\example\repo\engine.py --model-dir /home/alice/model "
            r"--token hf_private HF_TOKEN=also_private"
        )
        safe = collect_system_info.privacy_safe_command(command)
        self.assertNotIn("example", safe)
        self.assertNotIn("alice", safe)
        self.assertNotIn("hf_private", safe)
        self.assertNotIn("also_private", safe)
        self.assertEqual(safe.count("<redacted-absolute-path>"), 2)

    @patch.object(collect_system_info, "generic_cpu_info")
    @patch.object(collect_system_info, "total_memory_bytes", return_value=16_000_000_000)
    @patch.object(collect_system_info, "compiler_info", return_value=("cl", "19.40"))
    @patch.object(collect_system_info, "pytorch_version", return_value="2.7.1")
    @patch.object(collect_system_info, "git_commit", return_value="a" * 40)
    def test_manifest_has_schema_fields_without_private_identity(
        self,
        _git_commit,
        _pytorch_version,
        _compiler_info,
        _total_memory,
        cpu_info,
    ) -> None:
        cpu_info.return_value = {
            "cpu_model": "Example CPU",
            "cpu_vendor": "Example Vendor",
            "physical_cores": 8,
            "logical_cores": 16,
        }
        args = argparse.Namespace(
            architecture="mlgru",
            model_repo="owner/model",
            model_revision="b" * 40,
            engine_commit=None,
            benchmark_command="python scripts/benchmark_engine.py",
            configured_threads=4,
            compiler=None,
            compiler_version=None,
        )
        manifest = collect_system_info.build_manifest(args, root=Path("."))
        self.assertEqual(manifest["configured_threads"], 4)
        self.assertEqual(manifest["engine_commit"], "a" * 40)
        self.assertEqual(validate_hardware_manifest(manifest), [])
        for private_key in ("username", "hostname", "ip", "home", "cwd"):
            self.assertNotIn(private_key, manifest)
        output = io.StringIO()
        with redirect_stdout(output):
            collect_system_info.print_summary(manifest)
        self.assertIn("Privacy:", output.getvalue())

    def test_invalid_identifiers_are_rejected(self) -> None:
        args = argparse.Namespace(
            configured_threads=0,
            model_repo="missing-slash",
            model_revision="main",
            engine_commit="short",
        )
        with self.assertRaises(ValueError):
            collect_system_info.validate_cli_args(args)

    def test_linux_cpu_parser_counts_physical_cores(self) -> None:
        text = """
processor : 0
vendor_id : GenuineIntel
model name : Example Linux CPU
physical id : 0
core id : 0

processor : 1
vendor_id : GenuineIntel
model name : Example Linux CPU
physical id : 0
core id : 1
"""
        info = collect_system_info.parse_linux_cpuinfo(text)
        self.assertEqual(info["cpu_model"], "Example Linux CPU")
        self.assertEqual(info["cpu_vendor"], "GenuineIntel")
        self.assertEqual(info["physical_cores"], 2)

    @patch.object(
        collect_system_info,
        "run_text",
        return_value=(
            '{"Name":"Example Windows CPU","Manufacturer":"Vendor",'
            '"NumberOfCores":8,"NumberOfLogicalProcessors":16}'
        ),
    )
    def test_windows_cpu_parser_uses_only_processor_fields(self, _run_text) -> None:
        info = collect_system_info.windows_cpu_info()
        self.assertEqual(info["cpu_model"], "Example Windows CPU")
        self.assertEqual(info["cpu_vendor"], "Vendor")
        self.assertEqual(info["physical_cores"], 8)
        self.assertEqual(info["logical_cores"], 16)


if __name__ == "__main__":
    unittest.main()
