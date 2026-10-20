"""缓存键单测：评分配置指纹必须参与 key。

背景：缓存键原来只有 ``PROMPT_VERSION``，改了权重/缩放/阈值但 Prompt 没改时，
同一 JD+简历会命中旧缓存、回放旧分数。这里把「改配置 → key 变化 → 旧缓存失效」
整条链路都测到，防止以后有人把指纹从 key 里删掉。

所有会改 ``config`` 的用例都用 ``mock.patch.object``，退出 with 自动恢复，
不污染其它测试；端到端用例用临时目录里的 DB，不碰 ``data/matcher.db``。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config, store
from app.pipeline import analyze

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


class FingerprintTest(unittest.TestCase):
    def test_fingerprint_is_stable_and_short(self) -> None:
        first = config.scoring_fingerprint()
        second = config.scoring_fingerprint()
        self.assertEqual(first, second, "同一评分配置的指纹必须稳定")
        self.assertRegex(first, r"^[0-9a-f]{12}$")

    def test_same_input_same_key(self) -> None:
        self.assertEqual(
            store.make_key(JD_TEXT, RESUME_TEXT),
            store.make_key(JD_TEXT, RESUME_TEXT),
        )


class ConfigSensitivityTest(unittest.TestCase):
    """改任一评分参数 / 版本号，缓存键都必须变化。"""

    def _changed_key(self, attr: str, value: object) -> str:
        base = store.make_key(JD_TEXT, RESUME_TEXT)
        with mock.patch.object(config, attr, value):
            changed = store.make_key(JD_TEXT, RESUME_TEXT)
        self.assertNotEqual(base, changed, f"改 {attr} 后缓存键应当变化")
        self.assertEqual(base, store.make_key(JD_TEXT, RESUME_TEXT), "退出 patch 后应恢复")
        return changed

    def test_weights_change_key(self) -> None:
        tuned = {"must": 0.60, "skill": 0.30, "exp": 0.10}
        self._changed_key("WEIGHTS", tuned)

    def test_scale_and_overlap_change_key(self) -> None:
        self._changed_key("EXP_RELEVANCE_SCALE", config.EXP_RELEVANCE_SCALE + 1.0)
        self._changed_key("MUST_HIT_OVERLAP", config.MUST_HIT_OVERLAP + 0.1)

    def test_prompt_version_change_key(self) -> None:
        self._changed_key("PROMPT_VERSION", "v-test-next")

    def test_scoring_version_change_key(self) -> None:
        # 人工版本号是公式改动的兜底：参数没变也要能强制失效
        self._changed_key("SCORING_VERSION", "v-test-forced")


class CacheInvalidationEndToEndTest(unittest.TestCase):
    """端到端：缓存后改权重，同一输入必须 cache_hit=False（用临时 DB）。"""

    def test_weights_change_invalidates_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with mock.patch.object(config, "DB_PATH", Path(tmp_dir) / "cache.db"):
                first = analyze(JD_TEXT, RESUME_TEXT, use_cache=True)
                self.assertFalse(first["meta"]["cache_hit"], "首次分析不应命中缓存")

                again = analyze(JD_TEXT, RESUME_TEXT, use_cache=True)
                self.assertTrue(again["meta"]["cache_hit"], "同配置再跑应当命中缓存")

                tuned = {"must": 0.60, "skill": 0.30, "exp": 0.10}
                with mock.patch.object(config, "WEIGHTS", tuned):
                    changed = analyze(JD_TEXT, RESUME_TEXT, use_cache=True)
                self.assertFalse(
                    changed["meta"]["cache_hit"],
                    "改权重后必须重算，不能回放旧缓存",
                )
                self.assertNotEqual(changed["score"]["weights"], again["score"]["weights"])

                # 权重恢复了，旧 key 仍然有效（没有覆盖旧缓存，只是新键另存一份）
                restored = analyze(JD_TEXT, RESUME_TEXT, use_cache=True)
                self.assertTrue(restored["meta"]["cache_hit"])


if __name__ == "__main__":
    unittest.main()
