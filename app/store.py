"""SQLite 缓存 + 历史：用内容哈希当主键，键里带 PROMPT_VERSION 与评分配置指纹。

为什么用 SQLite 而不是内存字典：缓存要跨进程复用（CLI 跑一次、Streamlit 再跑一次
应该命中同一份结果），也要能在演示时用 ``cache-stats`` 展示命中行为。
为什么用参数化 SQL：拼接 SQL 是注入漏洞，哪怕本地项目也不该养成习惯。

缓存键 = ``md5(PROMPT_VERSION | scoring_fingerprint | jd_text | resume_text)``。
**版本号必须进键**：只带 Prompt 版本不够——改了 Prompt / 评分权重 / 缩放 / 阈值但键不变，
旧结果会被当成新结果返回（"我明明改了权重，怎么分数没变"）。
``scoring_fingerprint()`` 覆盖权重、缩放、阈值等评分参数，``PROMPT_VERSION`` 覆盖 Prompt，
两者任一变化都让旧缓存自然 miss；这也是面试里能讲清楚的缓存失效设计。
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from datetime import datetime

from app import config

logger = logging.getLogger(__name__)


def _now() -> str:
    """统一时间格式：秒级 ISO 字符串，便于排序与人工阅读。"""
    return datetime.now().isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    """建库/建表并返回连接；表已存在时是幂等的。"""
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cache("
        "key TEXT PRIMARY KEY, payload TEXT NOT NULL, created_at TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS history("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, job_title TEXT, total INTEGER, "
        "mode TEXT, cache_hit INTEGER, created_at TEXT NOT NULL)"
    )
    return conn


def make_key(jd_text: str, resume_text: str) -> str:
    """生成缓存键：内容哈希 + Prompt 版本 + 评分配置指纹。

    指纹每次调用时现算，所以改完 ``config.WEIGHTS`` / 缩放 / 阈值再跑，
    键会变、旧缓存不会命中——不需要手工清库。
    """
    raw = (
        f"{config.PROMPT_VERSION}|{config.scoring_fingerprint()}|{jd_text}|{resume_text}"
    ).encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def get_cache(key: str) -> dict | None:
    """按 key 读缓存；不存在或内容损坏都返回 None（缓存坏了不能影响主流程）。"""
    conn = _connect()
    try:
        row = conn.execute("SELECT payload FROM cache WHERE key = ?", (key,)).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    try:
        payload = json.loads(row[0])
    except json.JSONDecodeError:
        logger.warning("缓存内容损坏，忽略：%s", key[:8])
        return None
    return payload if isinstance(payload, dict) else None


def cache(key: str, payload: dict, created_at: str | None = None) -> None:
    """写入缓存（存在则替换），payload 必须可 JSON 序列化。"""
    conn = _connect()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO cache(key, payload, created_at) VALUES(?,?,?)",
            (key, json.dumps(payload, ensure_ascii=False), created_at or _now()),
        )
        conn.commit()
    finally:
        conn.close()


# 兼容参考代码里的命名，二者等价
set_cache = cache


def add_history(job_title: str, total: int, mode: str, cache_hit: bool) -> None:
    """记一条分析历史，供 cache-stats 展示与后续统计用。"""
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO history(job_title, total, mode, cache_hit, created_at) VALUES(?,?,?,?,?)",
            (job_title, int(total), mode, 1 if cache_hit else 0, _now()),
        )
        conn.commit()
    finally:
        conn.close()


def history(limit: int = 10) -> list[dict]:
    """最近的分析历史（新到旧）。"""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT job_title, total, mode, cache_hit, created_at FROM history "
            "ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "job_title": row[0],
            "total": row[1],
            "mode": row[2],
            "cache_hit": bool(row[3]),
            "created_at": row[4],
        }
        for row in rows
    ]


def stats() -> dict:
    """缓存与历史概况，给 ``cli cache-stats`` 用。"""
    conn = _connect()
    try:
        cache_count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        history_count = conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
        hit_count = conn.execute("SELECT COUNT(*) FROM history WHERE cache_hit = 1").fetchone()[0]
    finally:
        conn.close()
    hit_rate = (hit_count / history_count) if history_count else 0.0
    return {
        "db_path": str(config.DB_PATH),
        "cache_entries": int(cache_count),
        "history_entries": int(history_count),
        "cache_hit_rate": round(hit_rate, 3),
        "prompt_version": config.PROMPT_VERSION,
        "scoring_fingerprint": config.scoring_fingerprint(),
    }


def clear_cache() -> int:
    """清空缓存表并返回删除条数（调试/演示用）。"""
    conn = _connect()
    try:
        deleted = conn.execute("DELETE FROM cache").rowcount
        conn.commit()
    finally:
        conn.close()
    logger.info("已清空缓存 %s 条", deleted)
    return int(deleted)
