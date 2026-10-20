"""改写建议：真实模式走 LLM（temperature=0.5），离线模式走模板规则。

两条设计原则：
1. **每条建议必须指明"改哪里 / 改成什么样"**。泛泛而谈"建议提升能力"没有价值，
   所以无论 LLM 还是模板，输出结构固定为 ``{where, problem, rewrite}``。
2. **离线模板也得能用**。没 Key 时报告仍然给出 3-5 条可操作建议，项目才能离线演示；
   建议里的数字一律用 ``__`` 占位，绝不编造用户没提供的数据。
"""

from __future__ import annotations

import json
import logging

from app import config
from app.jd_parser import _clean
from app.llm import LLMError, OfflineLLMError, chat

logger = logging.getLogger(__name__)

_ADVICE_PROMPT = """你是资深 HR，正在帮应届生改简历。基于下面的差距分析，给出 3-5 条可执行的改写建议。

要求：
1. 每条建议必须指明改哪里（技能清单 / 项目描述 / 个人优势 / 教育背景 / 投递策略）；
2. 每条给出"改成什么样"的具体句式，涉及数字的地方用 __ 占位，不要编造数字；
3. 不要"建议提升能力"这类空话；
4. 只输出 JSON：{{"advices": [{{"where": "...", "problem": "...", "rewrite": "..."}}]}}

差距分析：
{diff}
"""


def _clean_advices(raw: object, limit: int = 5) -> list[dict]:
    """校验并收敛模型给的建议列表；结构不对的条目直接丢掉。

    LLM 输出不保证带 priority，这里按 ``where`` 映射一个默认分级（与规则模板口径一致），
    模型显式给了 priority 且合法时优先用它。
    """
    if not isinstance(raw, list):
        return []
    cleaned: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        where = str(item.get("where") or "").strip()
        problem = str(item.get("problem") or "").strip()
        rewrite = str(item.get("rewrite") or "").strip()
        if not (where and problem and rewrite):
            continue
        priority = _map_priority(str(item.get("priority") or "").strip(), where)
        cleaned.append(
            {
                "where": where[:30],
                "problem": problem[:160],
                "rewrite": rewrite[:240],
                "priority": priority,
            }
        )
        if len(cleaned) >= limit:
            break
    return cleaned


def _map_priority(explicit: str, where: str) -> str:
    """把模型给的（可能是空的）priority 收敛到 urgent/important/suggest。

    模型没给或给了非法值时，按 where 兜底：涉及技能清单/硬性要求的默认 urgent，
    项目经历/描述默认 important，其余 suggest。
    """
    if explicit in {"urgent", "important", "suggest"}:
        return explicit
    if "技能" in where or "硬性" in where:
        return "urgent"
    if "项目" in where or "经历" in where:
        return "important"
    return "suggest"


def _rule_advices(diff: dict, limit: int = 5) -> list[dict]:
    """离线模板建议：按"缺失技能 → 未命中硬性要求 → 量化成果 → 投递策略"优先级生成。

    V2 起每条建议带 ``priority``：urgent（直接决定能不能过初筛，先改）/
    important（明显拉分，但改起来要时间）/ suggest（锦上添花）。分级依据：
    - 缺失技能、未命中硬性要求 → urgent：这两项对应评分里权重最大的维度（must 65%），
      不补齐分根本上不去；
    - 量化成果不足、经历相关度低 → important：影响 exp 维度，但权重已降到 10%；
    - 简历结构、投递策略 → suggest：不影响匹配分，只影响命中率。
    """
    advices: list[dict] = []
    job_title = str(diff.get("job_title") or "目标岗位")
    missing_skills = list(diff.get("missing_skills") or [])
    must_miss = list(diff.get("must_miss") or [])
    total = int(diff.get("total") or 0)
    highlight_count = int(diff.get("highlight_count") or 0)

    for skill in missing_skills[:3]:
        advices.append(
            {
                "where": "技能清单",
                "problem": f"JD 技能表里有「{skill}」，简历中完全没有出现，技能覆盖被判为缺失。",
                "rewrite": f"在技能清单补一行「{skill}」，并注明熟练度与使用场景："
                f"「{skill}（熟练，用于 __ 项目）」；如果只是了解过，写「了解 {skill}」也可以，别硬冒充熟练。",
                "priority": "urgent",
            }
        )

    for requirement in must_miss[:2]:
        advices.append(
            {
                "where": "项目描述 / 个人优势",
                "problem": f"硬性要求「{requirement}」没有命中：简历里缺少对应关键词或经历。",
                "rewrite": "把最接近的一段经历改写成「负责 __，使用 __，结果 __」，并把这项要求的关键词"
                f"（{requirement[:40]}）自然写进去；确实没做过就不要编，改投要求更接近的岗位。",
                "priority": "urgent",
            }
        )

    if highlight_count < 3:
        advices.append(
            {
                "where": "项目经历",
                "problem": f"可量化的成果只有 {highlight_count} 条，经历相关度很难拉开差距。",
                "rewrite": "每个项目结尾补一行数字：「接口 P95 从 __ms 降到 __ms」「回答可用率从 __% 提升到 __%」"
                "「处理 __ 万行数据」；数字按真实情况填，没有指标就写规模与耗时。",
                "priority": "important",
            }
        )

    if len(advices) < limit:
        relevance = float(diff.get("exp_relevance") or 0.0)
        if relevance < 0.12:
            advices.append(
                {
                    "where": "项目描述",
                    "problem": f"项目描述与 JD 用词几乎不重叠（经历相关度 {relevance:.2f}）。",
                    "rewrite": "按 JD 的措辞重写项目里的技术名词：JD 写「容器化」就用 Docker、写「检索增强」就提 RAG，"
                    "让做过的事和岗位语言对齐（前提是真做过）。",
                    "priority": "important",
                }
            )

    if len(advices) < 3:
        advices.append(
            {
                "where": "简历结构",
                "problem": f"当前匹配分 {total}，求职方向与这份 JD 的一致性需要判断。",
                "rewrite": f"如果决定投「{job_title}」：把一页简历按「教育背景-技能清单-项目经历-实习经历」排序，"
                "把与 JD 最相关的两条成果放第一屏；如果匹配分长期低于 60，优先补硬性要求再投。",
                "priority": "suggest",
            }
        )
        advices.append(
            {
                "where": "投递策略",
                "problem": "单份简历投所有岗位，匹配度会被平均拉低。",
                "rewrite": "按岗位方向准备 2 个版本（AI 应用 / 数据分析），每版技能清单顺序跟着 JD 调整，"
                "投递前用本工具跑一遍，优先投分数 ≥ 70 的岗位。",
                "priority": "suggest",
            }
        )

    return advices[:limit]


def build_advices(diff: dict, max_retry: int = 1) -> list[dict]:
    """生成 3-5 条改写建议；LLM 失败或输出不合法时降级模板。"""
    if not config.OFFLINE:
        prompt = _ADVICE_PROMPT.format(diff=json.dumps(diff, ensure_ascii=False))
        for attempt in range(max_retry + 1):
            try:
                raw = chat(prompt, temperature=0.5, json_mode=True, max_retry=1)
            except (OfflineLLMError, LLMError) as exc:
                logger.warning("建议生成调用模型失败，降级模板：%s", exc)
                break
            try:
                data = json.loads(_clean(raw))
            except json.JSONDecodeError as exc:
                logger.warning("建议输出非法 JSON（attempt=%s）：%s", attempt, exc)
                prompt += "\n\n上次输出不是合法 JSON，请严格按 {\"advices\": [...]} 重新输出。"
                continue
            advices = _clean_advices(data.get("advices") if isinstance(data, dict) else data)
            if advices:
                return advices
            prompt += "\n\n上次输出的 advices 为空或字段不全，请重新输出合法 JSON。"
    return _rule_advices(diff)
