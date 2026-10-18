"""JD → Capability：真实模式走 LLM JSON + 校验回灌，离线模式走规则引擎。

设计取舍：规则引擎只保证"在示例 JD 上稳定可用"，不追求通用。它的价值有三个：
（1）无网络、无 Key 时项目仍能演示、能评测；（2）作为 LLM 失败/输出不合法时的兜底降级；
（3）给权重校准提供确定性的基线——同样的 JD 永远解析出同样的 must_have，实验才可复现。

规则引擎的做法很朴素：按句子切分，命中"必须/要求/熟悉/精通/掌握/至少/具备"的进
must_have，命中"优先/加分/了解/最好"的进 nice_to_have；技能、年限、学历单独用词表/正则抽。
"""

from __future__ import annotations

import json
import logging
import re

from app import config
from app.llm import LLMError, OfflineLLMError, chat
from app.schemas import Capability, capability_schema

logger = logging.getLogger(__name__)

_MUST_TRIGGERS = ("必须", "要求", "熟悉", "精通", "掌握", "至少", "具备", "熟练", "能独立")
_NICE_TRIGGERS = ("优先", "加分", "了解", "最好", "更佳")
_EDU_TRIGGERS = ("学历", "专业", "届", "毕业")
_HEADINGS = {
    "岗位职责", "工作职责", "职位描述", "任职要求", "岗位要求", "加分项", "我们希望你",
    "岗位名称", "职位名称", "工作地点", "公司", "部门", "福利待遇", "关于我们",
}
_BULLET_RE = re.compile(r"^(?:[-*•·]|\d+\s*[.、)）]|[（(]\d+[)）])\s*")
# 岗位信息行（公司/地点/薪资……）不是"要求"，避免"工作地点……至少 4 天"被当成硬性要求
_META_LINE_RE = re.compile(r"^(?:公司|部门|工作地点|地点|坐标|薪资|福利|待遇|岗位名称|职位名称|汇报对象)\s*[:：]")
_YEARS_RE = re.compile(r"(\d+\s*(?:[-~至到]\s*\d+)?)\s*年(?:以上)?[^\n。；;]{0,12}?(?:经验|工作)")
_EDU_ORDER = ("博士", "硕士", "研究生", "本科", "大专", "专科")

_JD_PROMPT = """你是资深 HR，擅长把招聘说明拆成结构化的岗位能力画像。

只输出 JSON，不要解释，不要 Markdown 代码块。

字段定义（JSON 对象，键名必须完全一致）：
{schema}

判定规则：
1. must_have 只放原文明确写"必须/要求/熟悉/精通/掌握/至少/具备"的条目，保留原文措辞；
2. nice_to_have 放"优先/加分/了解/最好"的条目；
3. skills 只输出技术或能力名词，统一小写，去重，不要整句话；
4. 原文没有的信息填 null 或空数组，绝对不要编造。

招聘说明原文：
---
{jd}
---"""


def _clean(raw: str) -> str:
    """清掉模型爱加的 Markdown 代码块包裹。"""
    text = (raw or "").strip()
    for prefix in ("```json", "```JSON", "```"):
        text = text.removeprefix(prefix)
    return text.removesuffix("```").strip()


def _pattern_hit(low_text: str, pattern: str) -> bool:
    """判断一个词表写法是否出现在文本里。

    纯 ASCII 的模式按"单词边界"匹配，避免 java 命中 javascript、git 命中 digital；
    中文模式直接子串匹配（中文没有天然词边界）。
    """
    if all(ord(ch) < 128 for ch in pattern):
        return re.search(rf"(?<![a-z0-9]){re.escape(pattern)}(?![a-z0-9])", low_text) is not None
    return pattern in low_text


def extract_skills(text: str) -> list[str]:
    """按词表从文本中抽技能，返回小写、去重、顺序稳定的列表。"""
    low = (text or "").lower()
    found: list[str] = []
    for canonical, patterns in config.SKILL_PATTERNS.items():
        if any(_pattern_hit(low, pattern) for pattern in patterns):
            found.append(canonical)
    return found


def normalize_skills(skills: list[str]) -> list[str]:
    """把模型/词表给的技能列表统一：小写、去重、别名归一。"""
    seen: set[str] = set()
    out: list[str] = []
    for raw in skills or []:
        if not isinstance(raw, str):
            continue
        skill = config.SKILL_ALIASES.get(raw.strip().lower(), raw.strip().lower())
        if skill and skill not in seen:
            seen.add(skill)
            out.append(skill)
    return out


def _split_segment(segment: str) -> list[str]:
    """把一句话再拆细：只有同时含"硬性触发词"和"加分触发词"时才按逗号拆。

    例："本科及以上学历，2027 届毕业生优先" → 拆成两条，学历进 must、届别进 nice；
    而"熟悉 Python，具备良好的编码习惯"不拆，保持一条完整的硬性要求。
    """
    has_must = any(t in segment for t in _MUST_TRIGGERS) or any(e in segment for e in _EDU_TRIGGERS)
    has_nice = any(t in segment for t in _NICE_TRIGGERS)
    parts = re.split(r"[，,]", segment) if (has_must and has_nice) else [segment]
    return [part.strip(" \t：:") for part in parts if part.strip(" \t：:")]


def _classify(part: str) -> str | None:
    """判断一句话属于 must / nice / 丢弃，并顺手过滤标题行与岗位信息行。"""
    text = _BULLET_RE.sub("", part).strip(" 。；;，,：:")
    if len(text) < 4 or text in _HEADINGS:
        return None
    if text.startswith("#") or _META_LINE_RE.match(text):
        return None
    if any(trigger in text for trigger in _NICE_TRIGGERS):
        return "nice"
    if any(trigger in text for trigger in _MUST_TRIGGERS) or any(e in text for e in _EDU_TRIGGERS):
        return "must"
    return None


def _extract_years(text: str) -> str | None:
    """抽年限要求；要求后面紧跟"经验/工作"，避免把"2026 年毕业"误判为年限。"""
    match = _YEARS_RE.search(text)
    return f"{match.group(1).strip()}年" if match else None


def _extract_education(text: str) -> str | None:
    """抽学历要求，返回如 "本科及以上"；找不到返回 None。"""
    for keyword in _EDU_ORDER:
        if keyword not in text:
            continue
        for line in text.splitlines():
            if keyword in line and "学历" in line:
                return f"{keyword}及以上" if "及以上" in line else keyword
        return keyword
    return None


def _extract_job_title(text: str) -> str:
    """岗位名称：优先取"岗位名称：xxx"，否则取第一个有内容的行（去掉 Markdown 标记）。"""
    for line in text.splitlines():
        line = re.sub(r"^#+\s*", "", line.strip())
        if not line:
            continue
        match = re.search(r"(?:岗位名称|职位名称|岗位|职位)\s*[:：]\s*(.+)", line)
        if match:
            return match.group(1).strip(" 【】[]")[:40]
        if len(line) >= 4:
            return line.strip("【】[]# ")[:40]
    return "未知岗位"


def _parse_jd_rules(jd_text: str) -> Capability:
    """离线规则解析：不打网络，确定性输出。"""
    must_have: list[str] = []
    nice_to_have: list[str] = []
    for line in jd_text.splitlines():
        line = _BULLET_RE.sub("", line.strip())
        if not line:
            continue
        for segment in re.split(r"[。；;\n]", line):
            for part in _split_segment(segment):
                kind = _classify(part)
                if kind is None:
                    continue
                text = _BULLET_RE.sub("", part).strip(" 。；;，,：:")[:120]
                bucket = must_have if kind == "must" else nice_to_have
                if text not in bucket:
                    bucket.append(text)

    return Capability(
        job_title=_extract_job_title(jd_text),
        must_have=must_have,
        nice_to_have=nice_to_have,
        skills=normalize_skills(extract_skills(jd_text)),
        years=_extract_years(jd_text),
        education=_extract_education(jd_text),
    )


def parse_jd(jd_text: str, max_retry: int = 2) -> Capability:
    """JD 解析入口：离线直接走规则；真实模式先 LLM，失败后降级规则并记日志。"""
    text = (jd_text or "").strip()
    if not text:
        raise ValueError("JD 文本为空，无法解析")

    if not config.OFFLINE:
        prompt = _JD_PROMPT.format(
            schema=json.dumps(capability_schema(), ensure_ascii=False),
            jd=text[:6000],
        )
        for attempt in range(max_retry + 1):
            try:
                raw = chat(prompt, temperature=0, json_mode=True, max_retry=1)
            except (OfflineLLMError, LLMError) as exc:
                logger.warning("JD 解析调用模型失败，降级规则引擎：%s", exc)
                break
            try:
                data = json.loads(_clean(raw))
            except json.JSONDecodeError as exc:
                prompt += f"\n\n你上次的输出不是合法 JSON（错误：{exc}），请只输出合法 JSON。"
                logger.warning("JD 输出非法 JSON（attempt=%s）：%s", attempt, exc)
                continue
            errors = Capability.validate_dict(data)
            if not errors:
                capability = Capability.from_dict(data)
                if capability.job_title and (capability.must_have or capability.skills):
                    return capability
                errors = ["job_title / must_have / skills 抽出来的内容为空"]
            logger.warning("JD 输出校验失败（attempt=%s）：%s", attempt, errors)
            prompt += "\n\n你上次的输出不符合格式，错误如下：\n- " + "\n- ".join(errors) + (
                "\n请重新只输出合法 JSON，字段结构必须与说明完全一致。"
            )

    return _parse_jd_rules(text)
