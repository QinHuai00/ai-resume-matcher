"""把"解析 → 评分 → 建议 → 报告 → 缓存/历史"串成一次分析。

``analyze()`` 是整个项目的唯一主流程入口：CLI、Streamlit、批量、评测都调它，
保证"网页里看到的分数"和"评测脚本里的分数"是同一套逻辑算出来的。

返回结构固定为 dict（含 capability / profile / score / advices / markdown / meta），
meta 里记录 cache_hit、elapsed_ms、mode——这两个字段是排查"分数怎么变了"的入口。
"""

from __future__ import annotations

import logging
import time

from app import config, store
from app.advisor import build_advices
from app.jd_parser import parse_jd
from app.report import to_markdown
from app.resume_parser import parse_resume
from app.schemas import MatchResult
from app.scoring import score

logger = logging.getLogger(__name__)

_MIN_JD_LEN = 20


def _build_meta(key: str, cache_hit: bool, elapsed_ms: int) -> dict:
    """统一构造 meta，避免各处手写漏字段。"""
    return {
        "cache_hit": cache_hit,
        "elapsed_ms": elapsed_ms,
        "mode": config.MODE,
        "key": key,
        "prompt_version": config.PROMPT_VERSION,
    }


def analyze(jd_text: str, resume_text: str, use_cache: bool = True) -> dict:
    """分析一份 JD + 一份简历，返回完整结果 dict。

    参数：
        jd_text: JD 原文（粘贴或读文件后的文本）。
        resume_text: 简历解析后的文本（``resume_parser.extract_text`` 的输出）。
        use_cache: False 时强制重算并**不读缓存**（评测/稳定性测试要关掉它）。
    """
    jd_text = (jd_text or "").strip()
    resume_text = (resume_text or "").strip()
    if len(jd_text) < _MIN_JD_LEN:
        raise ValueError(f"JD 文本过短（{len(jd_text)} 字，要求 ≥ {_MIN_JD_LEN}），无法可靠解析。")
    if len(resume_text) < config.MIN_TEXT_LEN:
        raise ValueError(
            f"简历文本过短（{len(resume_text)} 字，要求 ≥ {config.MIN_TEXT_LEN}），"
            "请检查文件是否为扫描件或内容是否完整。"
        )

    started = time.perf_counter()
    key = store.make_key(jd_text, resume_text)

    if use_cache:
        cached = store.get_cache(key)
        if cached:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            result = MatchResult.from_dict(cached)
            result.meta = _build_meta(key, cache_hit=True, elapsed_ms=elapsed_ms)
            result.markdown = to_markdown(
                result.capability, result.profile, result.score, result.advices, result.meta
            )
            store.add_history(result.capability.job_title, int(result.score.get("total", 0)), config.MODE, True)
            logger.info("缓存命中 key=%s", key[:8])
            return result.to_dict()

    capability = parse_jd(jd_text)
    profile = parse_resume(resume_text)
    score_result = score(capability, profile, resume_text)

    diff = {
        **score_result,
        "job_title": capability.job_title,
        "resume_skills": profile.skills,
        "project_count": len(profile.projects),
        "highlight_count": len(profile.highlights),
    }
    advices = build_advices(diff)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    meta = _build_meta(key, cache_hit=False, elapsed_ms=elapsed_ms)
    markdown = to_markdown(capability, profile, score_result, advices, meta)

    result = MatchResult(
        capability=capability,
        profile=profile,
        score=score_result,
        advices=advices,
        markdown=markdown,
        meta=meta,
    )
    payload = result.to_dict()
    if use_cache:
        store.cache(key, {k: v for k, v in payload.items() if k != "meta"})
    store.add_history(capability.job_title, int(score_result["total"]), config.MODE, False)
    logger.info(
        "分析完成 岗位=%s 总分=%s 模式=%s 耗时=%sms",
        capability.job_title,
        score_result["total"],
        config.MODE,
        elapsed_ms,
    )
    return payload
