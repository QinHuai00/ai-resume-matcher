"""三维可解释评分：硬性要求命中 60% + 技能覆盖 25% + 经历相关度 15%。

为什么不"让模型直接打个分"：那样每次分数都不一样，也说不清为什么；面试问"78 分
怎么来的"会直接卡住。拆成三个可计算的维度之后，每个数字都有公式，权重是公开参数，
可以争议、可以校准——这是设计的价值。

本模块是纯函数集合：不读文件、不连网络、不碰 SQLite，输入 Capability / ResumeProfile，
输出 dict。因此可以直接单测，同输入永远同输出，离线评测的"稳定性"就来自这里。
"""

from __future__ import annotations

import re

from app import config
from app.schemas import Capability, ResumeProfile

# 停用词：虚词 + 招聘套话。它们几乎每个 JD 都有，留着只会稀释重叠比例。
_STOP = set(
    "的 了 和 与 及 或 在 对 有 是 能 会 要 就 都 而 你 我 他 这 那 个 吧 吗 呢 "
    "熟悉 精通 掌握 了解 具备 优先 加分 经验 能力 要求 负责 参与 使用 相关 良好 以上 "
    "岗位 职责 公司 团队 工作 任职 希望 我们 可以 能够 以及 进行 通过 完成".split()
)
_EN_RE = re.compile(r"[a-z0-9][a-z0-9\.\+#/]*")
_CN_RE = re.compile(r"[\u4e00-\u9fa5]{2,}")

# 匹配前要剔除的"模板词"：它们对语义贡献很小，却会在滑窗分词里生成大量 n-gram，
# 把分母撑大导致真实命中被误判为未命中（例："本科及以上学历" vs 简历里的"本科"）。
# 只做模板词归一，不做同义词映射——同义漏判仍是初版的已知局限，不在这里假装解决。
_FILLER_WORDS = (
    "及以上", "以上", "相关", "良好的", "良好", "具备", "能够", "能独立",
    "至少", "左右", "其他", "能力",
)


def _match_text(text: str) -> str:
    """去掉模板词并补空格分隔，供 must_hit_rate 用。"""
    cleaned = (text or "")
    for filler in _FILLER_WORDS:
        cleaned = cleaned.replace(filler, " ")
    return cleaned


def tokens(text: str) -> set[str]:
    """极简分词：英文/数字按词切，中文按 2-4 字滑窗粗切，去停用词。

    为什么用滑窗：中文没有空格，滑窗能让"多表关联查询"和"关联查询"共享 token，
    实现成本极低。局限是会产生大量无意义 n-gram，所以命中阈值取 1/3 而非精确相等。
    """
    low = (text or "").lower()
    words = set(_EN_RE.findall(low))
    for segment in _CN_RE.findall(low):
        for n in (2, 3, 4):
            for index in range(len(segment) - n + 1):
                words.add(segment[index:index + n])
    return {word for word in words if word and word not in _STOP}


# 兼容参考文档里的命名（_tokens）；新代码请用 tokens
_tokens = tokens


def must_hit_rate(
    capability: Capability,
    profile: ResumeProfile,
    resume_text: str,
) -> tuple[float, list[str], list[str]]:
    """硬性要求命中率，返回 (命中率, 命中列表, 未命中列表)。

    判定：某条硬性要求与"简历全文 + 技能列表"的 token 重叠 ≥ 1/3 就算命中。
    用全文而不是只用技能列表：项目描述里写了"用 FastAPI 搭后端"也应该算命中。
    """
    resume_blob = f"{resume_text} {' '.join(profile.skills)}"
    resume_tokens = tokens(resume_blob)
    hit: list[str] = []
    miss: list[str] = []
    for requirement in capability.must_have:
        requirement_tokens = tokens(_match_text(requirement))
        if not requirement_tokens:
            continue
        overlap = len(requirement_tokens & resume_tokens) / len(requirement_tokens)
        (hit if overlap >= config.MUST_HIT_OVERLAP else miss).append(requirement)
    total = len(hit) + len(miss)
    return (len(hit) / total if total else 0.0), hit, miss


def skill_coverage(capability: Capability, profile: ResumeProfile) -> tuple[float, list[str]]:
    """技能覆盖率（集合交并）+ 缺失技能列表（排序保证输出稳定）。"""
    jd_skills = {s.lower().strip() for s in capability.skills if s and s.strip()}
    resume_skills = {s.lower().strip() for s in profile.skills if s and s.strip()}
    if not jd_skills:
        return 0.0, []
    missing = sorted(jd_skills - resume_skills)
    return len(jd_skills & resume_skills) / len(jd_skills), missing


def exp_relevance(capability: Capability, profile: ResumeProfile) -> float:
    """经历相关度：JD 要求/技能 与 简历项目描述 的词汇 Jaccard。"""
    jd_tokens = tokens(" ".join(capability.must_have + capability.skills))
    exp_tokens = tokens(" ".join(profile.projects))
    if not jd_tokens or not exp_tokens:
        return 0.0
    return len(jd_tokens & exp_tokens) / len(jd_tokens | exp_tokens)


def score(capability: Capability, profile: ResumeProfile, resume_text: str) -> dict:
    """算总分与各维度明细；返回 dict 直接进报告和缓存（JSON 可序列化）。"""
    must_rate, must_hit, must_miss = must_hit_rate(capability, profile, resume_text)
    coverage, missing_skills = skill_coverage(capability, profile)
    relevance = exp_relevance(capability, profile)
    relevance_scaled = min(relevance * config.EXP_RELEVANCE_SCALE, 1.0)

    weights = dict(config.WEIGHTS)
    total = round(
        100
        * (
            weights["must"] * must_rate
            + weights["skill"] * coverage
            + weights["exp"] * relevance_scaled
        )
    )
    jd_skill_total = len({s.lower().strip() for s in capability.skills if s and s.strip()})
    return {
        "total": total,
        "must_hit_rate": round(must_rate, 3),
        "must_hit": must_hit,
        "must_miss": must_miss,
        "must_hit_count": len(must_hit),
        "must_total": len(must_hit) + len(must_miss),
        "skill_coverage": round(coverage, 3),
        "skill_hit_count": jd_skill_total - len(missing_skills),
        "skill_total": jd_skill_total,
        "missing_skills": missing_skills,
        "exp_relevance": round(relevance, 3),
        "exp_relevance_scaled": round(relevance_scaled, 3),
        "weights": weights,
    }
