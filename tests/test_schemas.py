"""数据契约单测：validate_dict 报错可读、from_dict 容错、to_dict 归一。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas import Capability, MatchResult, ResumeProfile


class CapabilitySchemaTest(unittest.TestCase):
    def test_valid_dict_passes(self) -> None:
        data = {
            "job_title": "AI 应用开发实习生",
            "must_have": ["熟悉 Python"],
            "nice_to_have": ["了解 Docker 优先"],
            "skills": ["Python", "FASTAPI"],
            "years": None,
            "education": "本科及以上",
        }
        self.assertEqual(Capability.validate_dict(data), [])
        capability = Capability.from_dict(data)
        self.assertEqual(capability.skills, ["python", "fastapi"])  # 统一小写
        self.assertEqual(capability.to_dict()["skills"], ["fastapi", "python"])  # 排序稳定

    def test_invalid_types_reported(self) -> None:
        errors = Capability.validate_dict(
            {"job_title": "", "must_have": "不是数组", "skills": [1, "  "], "years": 3}
        )
        self.assertGreaterEqual(len(errors), 3)
        self.assertTrue(any("job_title" in item for item in errors))
        self.assertTrue(any("must_have" in item for item in errors))
        self.assertTrue(any("years" in item for item in errors))

    def test_non_dict_reported(self) -> None:
        errors = Capability.validate_dict("JSON 字符串")
        self.assertEqual(len(errors), 1)
        self.assertIn("JSON 对象", errors[0])

    def test_from_dict_tolerates_dirty_data(self) -> None:
        capability = Capability.from_dict(
            {
                "job_title": "  ",
                "must_have": ["  ", "熟悉 SQL", "熟悉 SQL", None, 3],
                "skills": ["Python", "python"],
                "years": "",
            }
        )
        self.assertEqual(capability.job_title, "")
        self.assertEqual(capability.must_have, ["熟悉 SQL"])
        self.assertEqual(capability.skills, ["python"])
        self.assertIsNone(capability.years)


class ResumeSchemaTest(unittest.TestCase):
    def test_valid_resume(self) -> None:
        data = {
            "name": "张沐阳",
            "education": "本科",
            "skills": ["python"],
            "projects": ["知识库问答项目"],
            "highlights": ["可用率从 62% 提升到 84%"],
        }
        self.assertEqual(ResumeProfile.validate_dict(data), [])
        self.assertEqual(ResumeProfile.from_dict(data).name, "张沐阳")

    def test_invalid_resume_reported(self) -> None:
        errors = ResumeProfile.validate_dict({"skills": "python", "projects": [1], "highlights": {}})
        self.assertEqual(len(errors), 3)


class MatchResultSchemaTest(unittest.TestCase):
    def test_roundtrip(self) -> None:
        result = MatchResult(
            capability=Capability(job_title="测试岗位", skills=["python"]),
            profile=ResumeProfile(name="测试人", skills=["python"]),
            score={"total": 66, "weights": {"must": 0.6, "skill": 0.25, "exp": 0.15}},
            advices=[{"where": "技能清单", "problem": "缺 Docker", "rewrite": "补一行 Docker"}],
            markdown="# 报告",
            meta={"mode": "offline", "cache_hit": False},
        )
        restored = MatchResult.from_dict(result.to_dict())
        self.assertEqual(restored.capability.job_title, "测试岗位")
        self.assertEqual(restored.score["total"], 66)
        self.assertEqual(restored.meta["mode"], "offline")
        self.assertEqual(MatchResult.validate_dict(result.to_dict()), [])

    def test_bad_total_reported(self) -> None:
        errors = MatchResult.validate_dict(
            {
                "capability": {"job_title": "x"},
                "profile": {},
                "score": {"total": 120},
                "advices": [],
            }
        )
        self.assertTrue(any("total" in item for item in errors))


if __name__ == "__main__":
    unittest.main()
