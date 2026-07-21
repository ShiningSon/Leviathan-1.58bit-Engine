from __future__ import annotations

import unittest

from scripts.release_utils import find_developer_absolute_paths, is_forbidden_tracked_path


class ReleaseUtilsTests(unittest.TestCase):
    def test_generated_artifacts_are_rejected(self) -> None:
        rejected = (
            "model.bin",
            "exports/model/report.json",
            "benchmark_runs/v09/results.json",
            "leviathan_mlgru_200m_instruct_v09a/README.md",
            "checkpoints/final.pt",
        )
        for path in rejected:
            with self.subTest(path=path):
                self.assertTrue(is_forbidden_tracked_path(path))

    def test_release_sources_are_allowed(self) -> None:
        allowed = (
            "benchmarks/results/scaling_summary.json",
            "training/configs/v09a_200m_mlgru.json",
            "hf_cards/Leviathan-MLGRU-200M-TinyStories-Instruct-v09a_README.md",
        )
        for path in allowed:
            with self.subTest(path=path):
                self.assertFalse(is_forbidden_tracked_path(path))

    def test_developer_paths_do_not_flag_modal_paths(self) -> None:
        text = "C:\\Users\\developer\\model.bin /home/alice/run /data/exports/model /root/seed.jsonl"
        matches = find_developer_absolute_paths(text)
        self.assertIn("C:\\Users\\developer", matches)
        self.assertIn("/home/alice", matches)
        self.assertNotIn("/data/exports/model", matches)
        self.assertNotIn("/root/seed.jsonl", matches)


if __name__ == "__main__":
    unittest.main()
