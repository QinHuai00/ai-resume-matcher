"""解析单测：JD / 简历规则解析、文本读取与错误提示。"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config
from app.jd_parser import parse_jd
from app.resume_parser import extract_text, parse_resume

SAMPLES = config.SAMPLES_DIR


def read_sample(*parts: str) -> str:
    return (SAMPLES.joinpath(*parts)).read_text(encoding="utf-8")


class JdParserTest(unittest.TestCase):
    def test_ai_jd_fields(self) -> None:
        capability = parse_jd(read_sample("jd_ai_app.md"))
        self.assertIn("AI 应用开发", capability.job_title)
        self.assertIn("本科", capability.education or "")
        self.assertIsNone(capability.years)  # 原文没有年限要求
        self.assertIn("python", capability.skills)
        self.assertIn("fastapi", capability.skills)
        self.assertGreaterEqual(len(capability.must_have), 5)
        self.assertGreaterEqual(len(capability.nice_to_have), 3)

    def test_ai_jd_no_heading_pollution(self) -> None:
        capability = parse_jd(read_sample("jd_ai_app.md"))
        for item in capability.must_have + capability.nice_to_have:
            self.assertNotIn("##", item)
            self.assertFalse(item.startswith("工作地点"))

    def test_data_jd_years_and_skills(self) -> None:
        capability = parse_jd(read_sample("jd_data_analyst.md"))
        self.assertEqual(capability.years, "3年")
        self.assertIn("sql", capability.skills)
        self.assertIn("tableau", capability.skills)
        self.assertTrue(any("SQL" in item for item in capability.must_have))

    def test_empty_jd_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_jd("   ")


class ResumeParserTest(unittest.TestCase):
    def test_zhang_resume_fields(self) -> None:
        text = extract_text(SAMPLES / "resume_zhang.md")
        profile = parse_resume(text)
        self.assertEqual(profile.name, "张沐阳")
        self.assertIn("python", profile.skills)
        self.assertIn("fastapi", profile.skills)
        self.assertGreaterEqual(len(profile.projects), 3)
        self.assertGreaterEqual(len(profile.highlights), 3)

    def test_projects_exclude_personal_info(self) -> None:
        profile = parse_resume(extract_text(SAMPLES / "resume_zhang.md"))
        for project in profile.projects:
            self.assertNotIn("求职意向", project)
            self.assertFalse(project.startswith("#"))

    def test_highlights_all_contain_digits(self) -> None:
        profile = parse_resume(extract_text(SAMPLES / "resume_li.md"))
        self.assertTrue(profile.highlights)
        for highlight in profile.highlights:
            self.assertRegex(highlight, r"\d")

    def test_text_too_short_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            short_file = Path(temp_dir) / "short.txt"
            short_file.write_text("太短了，没有有效内容。", encoding="utf-8")
            with self.assertRaises(ValueError) as context:
                extract_text(short_file)
            self.assertIn("过短", str(context.exception))

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            extract_text(SAMPLES / "不存在的简历.txt")

    def test_pdf_without_pypdf_gives_clear_hint(self) -> None:
        if importlib.util.find_spec("pypdf") is not None:
            self.skipTest("当前环境已安装 pypdf，跳过缺库分支")
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_pdf = Path(temp_dir) / "fake.pdf"
            fake_pdf.write_bytes(b"%PDF-1.4 fake")
            with self.assertRaises(RuntimeError) as context:
                extract_text(fake_pdf)
            self.assertIn("pypdf", str(context.exception))

    def test_old_doc_rejected_with_hint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            old_doc = Path(temp_dir) / "old.doc"
            old_doc.write_bytes(b"legacy binary")
            with self.assertRaises(ValueError) as context:
                extract_text(old_doc)
            self.assertIn(".docx", str(context.exception))


class SampleDataTest(unittest.TestCase):
    def test_samples_exist(self) -> None:
        expected = [
            "jd_ai_app.md",
            "jd_data_analyst.md",
            "resume_zhang.md",
            "resume_li.md",
            "variants/resume_zhang_lite.md",
            "variants/resume_li_backend.md",
            "variants/resume_li_senior.md",
        ]
        for name in expected:
            self.assertTrue((SAMPLES / name).exists(), f"缺少示例数据：{name}")

    def test_samples_long_enough(self) -> None:
        for path in SAMPLES.rglob("*.md"):
            self.assertGreater(len(path.read_text(encoding="utf-8")), config.MIN_TEXT_LEN, path.name)


if __name__ == "__main__":
    unittest.main()
