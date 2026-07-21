from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.release_utils import estimate_mlgru_params


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "training" / "configs"
PROMPTS_PATH = ROOT / "benchmarks" / "prompts_v02b_qa.json"
REQUIRED_CONFIG_KEYS = {
    "run_name",
    "dataset",
    "vocab_size",
    "hidden_size",
    "n_layers",
    "intermediate_size",
    "seq_len",
    "batch_size",
    "steps",
    "lr",
    "qa_ratio",
    "training_stage",
    "model_display_name",
}


class ConfigTests(unittest.TestCase):
    def test_active_configs_have_required_structure(self) -> None:
        names = (
            "v07b_70m_qa_repair.json",
            "v08a_100m_mlgru.json",
            "v09a_200m_mlgru.json",
            "v09a_200m_mlgru_h100_fast.json",
            "v09a_200m_mlgru_h100_safe.json",
            "v09a_200m_mlgru_h200_fast.json",
        )
        for name in names:
            with self.subTest(name=name):
                data = json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))
                self.assertFalse(REQUIRED_CONFIG_KEYS - data.keys())
                for key in ("vocab_size", "hidden_size", "n_layers", "intermediate_size", "steps"):
                    self.assertGreater(data[key], 0)
                self.assertGreater(data["qa_ratio"], 0)
                self.assertLessEqual(data["qa_ratio"], 1)

    def test_v09_parameter_estimate(self) -> None:
        data = json.loads((CONFIG_DIR / "v09a_200m_mlgru.json").read_text(encoding="utf-8"))
        estimate = estimate_mlgru_params(
            data["vocab_size"],
            data["hidden_size"],
            data["n_layers"],
            data["intermediate_size"],
        )
        self.assertEqual(estimate, 233_388_800)

    def test_strict_qa_prompt_schema(self) -> None:
        prompts = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(len(prompts), 20)
        for item in prompts:
            self.assertIsInstance(item.get("prompt"), str)
            self.assertTrue(item["prompt"])
            self.assertIsInstance(item.get("expected"), list)
            self.assertTrue(item["expected"])
            self.assertIsInstance(item.get("forbidden"), list)
            self.assertIsInstance(item.get("max_words"), int)
            self.assertGreater(item["max_words"], 0)


if __name__ == "__main__":
    unittest.main()
