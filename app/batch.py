"""并发批量分析：asyncio.Semaphore 限流 + 单条失败不影响整批。

为什么不是 for 循环直接调：循环会把 N 份分析串成 N 倍耗时；无限制并发又会把接口
打爆（429），而且批量任务里"一条失败让整批崩"是最常见的低级事故。所以：
Semaphore 限制同时在飞的数量，asyncio.to_thread 把同步流水线丢进线程池，
每条任务单独 try/except，失败位置记日志、最终汇总里标出来。

离线模式下同样能跑（没有网络等待，并发主要用来演示结构和陪练测试）。
"""

from __future__ import annotations

import asyncio
import csv
import logging
from pathlib import Path

from app import config
from app.pipeline import analyze
from app.resume_parser import extract_text

logger = logging.getLogger(__name__)


def load_pairs(path: str | Path) -> list[dict]:
    """读取 pairs CSV（列：jd, resume, 可选 label），路径相对项目根解析。"""
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = config.PROJECT_ROOT / file_path
    if not file_path.exists():
        raise FileNotFoundError(f"批量清单不存在：{file_path}")

    pairs: list[dict] = []
    with file_path.open(encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            jd_name = (row.get("jd") or "").strip()
            resume_name = (row.get("resume") or "").strip()
            if not jd_name or not resume_name:
                raise ValueError(f"批量清单第 {line_number} 行缺少 jd 或 resume 列")
            pairs.append(
                {"jd": jd_name, "resume": resume_name, "label": (row.get("label") or "").strip()}
            )
    if not pairs:
        raise ValueError(f"批量清单为空：{file_path}")
    return pairs


def _resolve(path_str: str) -> Path:
    """相对路径以项目根为准，绝对路径原样使用。"""
    path = Path(path_str)
    return path if path.is_absolute() else config.PROJECT_ROOT / path


def _read_jd(path_str: str) -> str:
    """JD 一律按 UTF-8 文本读（示例 JD 是 .md）。"""
    path = _resolve(path_str)
    if not path.exists():
        raise FileNotFoundError(f"JD 文件不存在：{path}")
    return path.read_text(encoding="utf-8", errors="replace")


async def analyze_pair(jd_path: str, resume_path: str, use_cache: bool = True) -> dict:
    """分析一对文件：JD 读文本、简历走 extract_text（校验长度/格式），再丢线程池。"""
    jd_text = _read_jd(jd_path)
    resume_text = await asyncio.to_thread(extract_text, _resolve(resume_path))
    return await asyncio.to_thread(analyze, jd_text, resume_text, use_cache)


async def analyze_many(
    pairs: list[dict],
    concurrency: int = 4,
    use_cache: bool = True,
) -> list[dict | None]:
    """并发分析多对文件；返回顺序与输入一致，失败位置为 None。"""
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _one(pair: dict) -> dict:
        async with semaphore:
            result = await analyze_pair(pair["jd"], pair["resume"], use_cache=use_cache)
            return {**pair, **result}

    tasks = [asyncio.create_task(_one(pair)) for pair in pairs]
    results: list[dict | None] = [None] * len(tasks)
    for index, task in enumerate(tasks):
        try:
            results[index] = await task
        except Exception as exc:  # noqa: BLE001 —— 单条失败不能让整批崩
            logger.error("批量分析第 %s 条失败（%s）：%s", index + 1, pairs[index], exc)
    return results


def run_batch(
    pairs_path: str | Path = "eval/pairs.csv",
    concurrency: int = 4,
    use_cache: bool = True,
) -> dict:
    """同步入口：读清单 → 并发分析 → 返回汇总（含每行明细与平均分）。"""
    pairs = load_pairs(pairs_path)
    results = asyncio.run(analyze_many(pairs, concurrency=concurrency, use_cache=use_cache))
    rows: list[dict] = []
    for pair, result in zip(pairs, results):
        if result is None:
            rows.append({"jd": pair["jd"], "resume": pair["resume"], "total": None, "error": "分析失败"})
            continue
        rows.append(
            {
                "jd": pair["jd"],
                "resume": pair["resume"],
                "label": pair.get("label", ""),
                "job_title": result["capability"]["job_title"],
                "total": result["score"]["total"],
                "cache_hit": result["meta"]["cache_hit"],
            }
        )
    ok_rows = [row for row in rows if row["total"] is not None]
    average = sum(row["total"] for row in ok_rows) / len(ok_rows) if ok_rows else 0.0
    return {
        "pairs_path": str(pairs_path),
        "total": len(pairs),
        "ok": len(ok_rows),
        "failed": len(pairs) - len(ok_rows),
        "average_total": round(average, 1),
        "rows": rows,
    }
