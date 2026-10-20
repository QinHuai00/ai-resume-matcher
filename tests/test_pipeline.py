"""流水线单测：返回契约、缓存命中、3 次稳定性、报告结构与数据校验。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config, store
from app.pipeline import analyze
from app.resume_parser import extract_text
from app.schemas import Capability, MatchResult, ResumeProfile

JD_TEXT = """# 数据运营专员

## 任职要求
1. 本科及以上学历，熟悉 SQL 与 Excel；
2. 熟悉 Python，能写简单的数据清洗脚本；
3. 具备良好的沟通能力。
"""

RESUME_TEXT = (
    "王一\n\n教育背景\n某大学 本科\n\n技能清单\n熟悉 SQL 多表查询与 Excel 数据透视，"
    "熟悉 Python 数据处理。\n\n项目经历\n校园活动数据统计\n- 用 SQL 和 Excel 统计 2000 条报名数据，"
    "报表产出时间从 3 小时缩短到 20 分钟。\n\n个人优势\n沟通能力好，做过 2 次社团宣讲。\n"
)


class AnalyzeContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = analyze(JD_TEXT, RESUME_TEXT, use_cache=False)

    def test_required_keys(self) -> None:
        for key in ("capability", "profile", "score", "advices", "markdown", "meta"):
            self.assertIn(key, self.result)

    def test_score_range_and_weights(self) -> None:
        score = self.result["score"]
        self.assertIsInstance(score["total"], int)
        self.assertGreaterEqual(score["total"], 0)
        self.assertLessEqual(score["total"], 100)
        self.assertEqual(score["weights"], config.WEIGHTS)

    def test_meta_fields(self) -> None:
        meta = self.result["meta"]
        self.assertIn(meta["mode"], {"offline", "llm"})
        self.assertEqual(meta["prompt_version"], config.PROMPT_VERSION)
        self.assertFalse(meta["cache_hit"])
        self.assertGreaterEqual(meta["elapsed_ms"], 0)

    def test_result_validates(self) -> None:
        self.assertEqual(Capability.validate_dict(self.result["capability"]), [])
        self.assertEqual(ResumeProfile.validate_dict(self.result["profile"]), [])
        self.assertEqual(MatchResult.validate_dict(self.result), [])

    def test_markdown_contains_key_sections(self) -> None:
        markdown = self.result["markdown"]
        self.assertIn("总分", markdown)
        self.assertIn("三维评分", markdown)
        self.assertIn("改写建议", markdown)
        self.assertIn("局限", markdown)

    def test_advices_count_and_shape(self) -> None:
        advices = self.result["advices"]
        self.assertGreaterEqual(len(advices), 3)
        self.assertLessEqual(len(advices), 5)
        for advice in advices:
            self.assertEqual(set(advice), {"where", "problem", "rewrite", "priority"})
            self.assertIn(advice["priority"], {"urgent", "important", "suggest"})
            self.assertTrue(all(str(value).strip() for value in advice.values()))


class CacheAndHistoryTest(unittest.TestCase):
    def test_cache_hit_on_second_call(self) -> None:
        # 第一次（可能从未缓存过）写入缓存，第二次应当命中
        first = analyze(JD_TEXT, RESUME_TEXT, use_cache=True)
        second = analyze(JD_TEXT, RESUME_TEXT, use_cache=True)
        self.assertTrue(second["meta"]["cache_hit"])
        self.assertEqual(first["score"]["total"], second["score"]["total"])

    def test_store_stats(self) -> None:
        analyze(JD_TEXT, RESUME_TEXT, use_cache=True)
        stats = store.stats()
        self.assertGreaterEqual(stats["cache_entries"], 1)
        self.assertIn("cache_hit_rate", stats)
        self.assertTrue(Path(stats["db_path"]).exists())


class StabilityOnSamplesTest(unittest.TestCase):
    """同一份输入跑 3 次，总分波动必须 < 5（离线规则引擎应为 0）。"""

    def test_three_runs_on_samples(self) -> None:
        jd_text = (config.SAMPLES_DIR / "jd_ai_app.md").read_text(encoding="utf-8")
        resume_text = extract_text(config.SAMPLES_DIR / "resume_zhang.md")
        totals = [analyze(jd_text, resume_text, use_cache=False)["score"]["total"] for _ in range(3)]
        self.assertLess(max(totals) - min(totals), 5, f"3 次总分波动过大：{totals}")
        self.assertEqual(len(set(totals)), 1, f"离线模式应当完全确定，实际：{totals}")


class InputValidationTest(unittest.TestCase):
    def test_short_jd_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analyze("太短", RESUME_TEXT, use_cache=False)

    def test_short_resume_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analyze(JD_TEXT, "太短的简历", use_cache=False)


if __name__ == "__main__":
    unittest.main()
