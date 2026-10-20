"""评分纯函数单测：极端样本（完全匹配 / 完全无关）+ 稳定性 + 权重契约。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config
from app.schemas import Capability, ResumeProfile
from app.scoring import exp_relevance, must_hit_rate, score, skill_coverage, tokens


def make_capability() -> Capability:
    return Capability(
        job_title="AI 应用开发实习生",
        must_have=["熟悉 Python", "熟悉 FastAPI", "熟悉 SQL 多表查询"],
        nice_to_have=["了解 Docker 优先"],
        skills=["python", "fastapi", "sql", "docker"],
        years=None,
        education="本科及以上",
    )


def make_perfect_profile() -> ResumeProfile:
    return ResumeProfile(
        name="测试候选人",
        education="本科",
        skills=["python", "fastapi", "sql", "docker"],
        projects=["用 Python 和 FastAPI 写接口，用 SQL 做多表查询，用 Docker 部署服务"],
        highlights=["接口 P95 从 800ms 降到 200ms"],
    )


def make_unrelated_profile() -> ResumeProfile:
    return ResumeProfile(
        name="无关候选人",
        education="大专",
        skills=["excel"],
        projects=["负责门店排班和库存盘点"],
        highlights=["每月盘点 300 件商品"],
    )


class TokensTest(unittest.TestCase):
    def test_extracts_english_and_chinese(self) -> None:
        token_set = tokens("熟悉 Python 与 SQL，了解容器化部署")
        self.assertIn("python", token_set)
        self.assertIn("sql", token_set)
        self.assertTrue(any("容器" in token for token in token_set))

    def test_stopwords_removed(self) -> None:
        token_set = tokens("熟悉 精通 掌握 了解 优先 加分")
        self.assertEqual(token_set, set())

    def test_empty_input(self) -> None:
        self.assertEqual(tokens(""), set())


class ExtremeCaseTest(unittest.TestCase):
    def test_perfect_match_hits_all_must(self) -> None:
        capability = make_capability()
        profile = make_perfect_profile()
        resume_text = "熟悉 Python，熟悉 FastAPI，熟悉 SQL 多表查询，了解 Docker，本科"
        rate, hit, miss = must_hit_rate(capability, profile, resume_text)
        self.assertEqual(rate, 1.0)
        self.assertEqual(len(hit), 3)
        self.assertEqual(miss, [])

    def test_unrelated_hits_none(self) -> None:
        capability = make_capability()
        profile = make_unrelated_profile()
        rate, hit, miss = must_hit_rate(capability, profile, "负责门店排班和库存盘点")
        self.assertEqual(rate, 0.0)
        self.assertEqual(hit, [])
        self.assertEqual(len(miss), 3)

    def test_perfect_score_approaches_100(self) -> None:
        capability = make_capability()
        profile = make_perfect_profile()
        # 让简历项目描述覆盖 JD 的 must_have + skills，保证经历相关度 Jaccard = 1
        profile.projects = [" ".join(capability.must_have + capability.skills)]
        resume_text = " ".join(capability.must_have + capability.skills)
        result = score(capability, profile, resume_text)
        self.assertEqual(result["total"], 100)
        self.assertEqual(result["must_hit_rate"], 1.0)
        self.assertEqual(result["skill_coverage"], 1.0)
        self.assertEqual(result["exp_relevance"], 1.0)

    def test_unrelated_score_near_zero(self) -> None:
        capability = make_capability()
        profile = make_unrelated_profile()
        result = score(capability, profile, "负责门店排班和库存盘点，每月盘点 300 件商品")
        self.assertLess(result["total"], 10)
        self.assertEqual(result["must_hit_count"], 0)
        self.assertEqual(result["missing_skills"], ["docker", "fastapi", "python", "sql"])


class DimensionTest(unittest.TestCase):
    def test_skill_coverage_partial(self) -> None:
        capability = Capability(job_title="x", skills=["python", "sql", "docker", "git"])
        profile = ResumeProfile(skills=["python", "sql"])
        coverage, missing = skill_coverage(capability, profile)
        self.assertAlmostEqual(coverage, 0.5)
        self.assertEqual(missing, ["docker", "git"])

    def test_empty_jd_skills(self) -> None:
        coverage, missing = skill_coverage(Capability(job_title="x"), ResumeProfile())
        self.assertEqual((coverage, missing), (0.0, []))

    def test_skill_alias_normalized(self) -> None:
        """别名归一：JD 写「kubernetes」、简历写「k8s」应判为命中而非缺失。"""
        capability = Capability(job_title="x", skills=["kubernetes", "llm"])
        profile = ResumeProfile(skills=["k8s", "大模型"])
        coverage, missing = skill_coverage(capability, profile)
        self.assertAlmostEqual(coverage, 1.0)
        self.assertEqual(missing, [])

    def test_exp_relevance_range(self) -> None:
        capability = make_capability()
        self.assertGreaterEqual(exp_relevance(capability, make_perfect_profile()), 0.0)
        self.assertLessEqual(exp_relevance(capability, make_perfect_profile()), 1.0)
        self.assertEqual(exp_relevance(capability, ResumeProfile()), 0.0)

    def test_weights_sum_to_one(self) -> None:
        self.assertAlmostEqual(sum(config.WEIGHTS.values()), 1.0)
        self.assertEqual(config.WEIGHTS, {"must": 0.65, "skill": 0.25, "exp": 0.10})


class StabilityTest(unittest.TestCase):
    def test_same_input_three_runs_identical(self) -> None:
        capability = make_capability()
        profile = make_perfect_profile()
        totals = [
            score(capability, profile, "熟悉 Python、FastAPI、SQL 多表查询").get("total") for _ in range(3)
        ]
        self.assertEqual(len(set(totals)), 1)

    def test_score_is_pure(self) -> None:
        capability = make_capability()
        profile = make_perfect_profile()
        first = score(capability, profile, "正文")
        second = score(capability, profile, "正文")
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
