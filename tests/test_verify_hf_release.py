from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_hf_release import (
    REQUIRED_PATHS,
    normalize_card_text,
    validate_download,
)
from scripts.check_release_readiness import validate_release_manifest


class HfReleaseVerificationTests(unittest.TestCase):
    def make_package(self, root: Path) -> set[str]:
        contents = {
            ".gitattributes": "*.bin filter=lfs diff=lfs merge=lfs -text\n",
            "README.md": "---\nlicense: mit\n---\n\n# Test\n",
            "leviathan_mlgru_200m_instruct_v09a.bin": b"binary",
            "leviathan_mlgru_200m_instruct_v09a_meta.json": json.dumps({"architecture": "mlgru"}),
            "leviathan_mlgru_tokenizer/tokenizer.json": json.dumps({"model": {"type": "BPE"}}),
            "leviathan_mlgru_tokenizer/tokenizer_config.json": json.dumps({"model_max_length": 96}),
            "report.json": json.dumps({"run_name": "v09a"}),
            "sample_outputs.txt": "USER> test\nENGINE> test\n",
        }
        for relative, content in contents.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, bytes):
                path.write_bytes(content)
            else:
                path.write_text(content, encoding="utf-8")
        return set(contents)

    def test_valid_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            remote_files = self.make_package(root)
            errors, records = validate_download(root, remote_files)
            self.assertEqual(errors, [])
            self.assertEqual({record["path"] for record in records}, remote_files)
            self.assertTrue(all(record["bytes"] > 0 for record in records))
            self.assertTrue(all(len(record["sha256"]) == 64 for record in records))

    def test_missing_required_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            remote_files = self.make_package(root)
            missing = "report.json"
            (root / missing).unlink()
            remote_files.remove(missing)
            errors, _ = validate_download(root, remote_files)
            self.assertTrue(any("public repository is missing" in error for error in errors))
            self.assertIn(missing, REQUIRED_PATHS)

    def test_extra_local_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            remote_files = self.make_package(root)
            (root / "unexpected.txt").write_text("unexpected", encoding="utf-8")
            errors, _ = validate_download(root, remote_files)
            self.assertTrue(any("not present in the public revision" in error for error in errors))

    def test_card_normalization_is_limited(self) -> None:
        left = "line one  \r\nline two\r\n"
        right = "line one\nline two\n"
        self.assertEqual(normalize_card_text(left), normalize_card_text(right))
        self.assertNotEqual(normalize_card_text("a\n"), normalize_card_text("b\n"))

    def test_committed_release_manifest_is_valid(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads(
            (root / "releases" / "v0.9.0_hf_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(validate_release_manifest(manifest), [])

    def test_release_manifest_rejects_changed_recommendation(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads(
            (root / "releases" / "v0.9.0_hf_manifest.json").read_text(encoding="utf-8")
        )
        manifest["recommended_top_k"] = 0.08
        errors = validate_release_manifest(manifest)
        self.assertTrue(any("recommended_top_k" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
