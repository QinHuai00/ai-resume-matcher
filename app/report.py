"""Markdown 报告：总分 + 三维表 + 未命中硬性要求 + 缺失技能 + 改写建议 + 局限说明。

报告是给"人"看的，所以三件事必须做到：
1. 每个分数都带分子/分母（"8/11" 比 "73%" 更可核对）；
2. 末尾主动写清局限（关键词近似、同义漏判、两栏简历）——承认边界比吹"很准"可信；
3. 带生成信息（模式/耗时/缓存/Prompt 版本），方便排查"同样的输入为什么分数变了"。
"""

from __future__ import annotations

from app import config
from app.schemas import Capability, ResumeProfile


def _mode_text(meta: dict) -> str:
    """把 meta 里的模式翻译成人话，并明确标注离线结果不是真实模型指标。"""
    if meta.get("mode") == "offline":
        return "离线规则引擎（未调用大模型，结果为确定性规则输出）"
    return f"大模型 {meta.get('model') or ''}".strip()


_PRIORITY_LABEL = {
    "urgent": "【高优·先改】",
    "important": "【重要】",
    "suggest": "【可选】",
}


def to_markdown(
    capability: Capability,
    profile: ResumeProfile,
    score_result: dict,
    advices: list[dict],
    meta: dict | None = None,
) -> str:
    """渲染完整报告 Markdown（纯函数，不依赖缓存/网络）。"""
    meta = meta or {}
    must_rate = float(score_result.get("must_hit_rate") or 0.0)
    coverage = float(score_result.get("skill_coverage") or 0.0)
    relevance = float(score_result.get("exp_relevance") or 0.0)
    weights = score_result.get("weights") or config.WEIGHTS

    lines: list[str] = [
        f"# 简历匹配报告：{capability.job_title or '未知岗位'}",
        "",
        f"**总分：{score_result.get('total', 0)} / 100**",
        "",
        f"- 候选人：{profile.name or '未识别'}"
        + (f" ｜ 学历：{profile.education}" if profile.education else ""),
        f"- 岗位要求："
        + (f"学历 {capability.education}；" if capability.education else "")
        + (f"年限 {capability.years}；" if capability.years else "")
        + f"硬性要求 {len(capability.must_have)} 条、加分项 {len(capability.nice_to_have)} 条",
        "",
        "## 三维评分",
        "",
        "| 维度 | 数值 | 权重 | 说明 |",
        "|------|------|------|------|",
        f"| 硬性要求命中 | {must_rate:.0%}（{score_result.get('must_hit_count', 0)}/{score_result.get('must_total', 0)}）"
        f" | {weights.get('must', config.WEIGHTS['must']):.0%} | 逐条判断关键词重叠是否 ≥ 1/3 |",
        f"| 技能覆盖 | {coverage:.0%}（{score_result.get('skill_hit_count', 0)}/{score_result.get('skill_total', 0)}）"
        f" | {weights.get('skill', config.WEIGHTS['skill']):.0%} | JD 技能表与简历技能的交集 |",
        f"| 经历相关度 | {relevance:.2f}（放大后 {score_result.get('exp_relevance_scaled', 0):.2f}）"
        f" | {weights.get('exp', config.WEIGHTS['exp']):.0%} | 项目描述与 JD 用词的 Jaccard ×{config.EXP_RELEVANCE_SCALE:.0f} 截断 |",
        "",
    ]

    must_miss = score_result.get("must_miss") or []
    if must_miss:
        lines.append("## 未命中的硬性要求")
        lines.append("")
        lines.extend(f"- {item}" for item in must_miss)
        lines.append("")

    missing_skills = score_result.get("missing_skills") or []
    if missing_skills:
        lines.append("## 缺失技能（JD 技能表中简历未出现）")
        lines.append("")
        lines.append("、".join(missing_skills))
        lines.append("")

    lines.append("## 改写建议")
    lines.append("")
    if advices:
        for index, advice in enumerate(advices, start=1):
            label = _PRIORITY_LABEL.get(advice.get("priority", ""), "")
            lines.append(
                f"{index}. {label}**{advice.get('where', '')}**：{advice.get('problem', '')}"
            )
            lines.append(f"   - 建议改为：{advice.get('rewrite', '')}")
    else:
        lines.append("- 暂无建议（可能是简历已经足够贴合该 JD）。")
    lines.append("")

    lines.extend(
        [
            "## 说明与局限",
            "",
            f"- 匹配分 = 硬性要求命中 × {weights.get('must', config.WEIGHTS['must']):.0%} + 技能覆盖 × {weights.get('skill', config.WEIGHTS['skill']):.0%}"
            f" + 经历相关度 × {weights.get('exp', config.WEIGHTS['exp']):.0%}，权重由留一交叉验证校准（见 eval/weight_calibration.md）。",
            "- 技能匹配基于内置词表的关键词近似，**同义表述会漏判**（「容器化」与 Docker、「大模型」与 LLM 已做别名归一，但词表外仍会漏）。",
            "- 简历解析假设一栏式排版；两栏模板抽出来的文本可能串行，导致经历相关度偏低。",
            "- 本报告是**辅助判断**，不是招聘结论；人工标注样本量小，分数不代表客观真值。",
            "- 离线规则引擎不调用大模型，报告中的分数与建议均为确定性规则输出，不能当作真实模型效果。",
            "",
            "---",
            "",
            f"生成信息：模式 {_mode_text(meta)} ｜ 耗时 {meta.get('elapsed_ms', 0)} ms ｜ "
            f"缓存 {'命中' if meta.get('cache_hit') else '未命中'} ｜ Prompt 版本 {meta.get('prompt_version', '')}",
            "",
        ]
    )
    return "\n".join(lines)
