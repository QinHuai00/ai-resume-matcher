"""命令行入口：analyze / report / batch / cache-stats。

面向用户的中文输出允许直接 print（业务日志仍走 logging）。三处工程细节：
1. Windows 控制台默认 cp936，中文/符号容易炸：启动时把 stdout/stderr 重设为 UTF-8；
2. 缺库（pypdf/streamlit 等）时只打印一条中文提示 + 退出码，不打印堆栈；
3. 所有路径以项目根为基准解析，允许在任意目录执行 ``python -m app.cli``。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from app import config, store
from app.batch import run_batch
from app.llm import provider_info
from app.pipeline import analyze
from app.resume_parser import extract_text


def _init_console() -> None:
    """把标准输出/错误切成 UTF-8，避免 Windows cp936 下中文乱码或抛异常。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def _resolve(path_str: str) -> Path:
    """相对路径以项目根为准（这样在任意目录下执行命令都能找到 data/samples）。"""
    path = Path(path_str)
    return path if path.is_absolute() else config.PROJECT_ROOT / path


def _read_jd(path_str: str) -> str:
    path = _resolve(path_str)
    if not path.exists():
        raise FileNotFoundError(f"JD 文件不存在：{path}")
    return path.read_text(encoding="utf-8", errors="replace")


def _load_inputs(jd_path: str, resume_path: str) -> tuple[str, str]:
    return _read_jd(jd_path), extract_text(_resolve(resume_path))


def _mode_tag() -> str:
    return "[OFFLINE]" if config.OFFLINE else "[LLM]"


def _print_summary(result: dict) -> None:
    """打印面向用户的结果摘要（分数三维 + 缺失技能 + 前三条建议 + 生成信息）。"""
    capability = result["capability"]
    score = result["score"]
    meta = result["meta"]

    # 权重一律读评分结果里的值，不在展示层硬编码——否则校准过的权重会在摘要里显示成旧值。
    weights = score.get("weights") or config.WEIGHTS
    print(f"{_mode_tag()} 岗位：{capability['job_title']}")
    print(f"候选人：{result['profile'].get('name') or '未识别'}")
    print(f"总分：{score['total']} / 100")
    print(
        f"├─ 硬性要求命中  {score['must_hit_count']}/{score['must_total']}"
        f" ({score['must_hit_rate']:.0%})   权重 {weights['must']:.0%}"
    )
    print(
        f"├─ 技能覆盖      {score['skill_hit_count']}/{score['skill_total']}"
        f" ({score['skill_coverage']:.0%})   权重 {weights['skill']:.0%}"
    )
    print(f"└─ 经历相关度    {score['exp_relevance']:.2f}           权重 {weights['exp']:.0%}")
    if score["missing_skills"]:
        print(f"缺失技能：{'、'.join(score['missing_skills'])}")
    if score["must_miss"]:
        print(f"未命中硬性要求：{len(score['must_miss'])} 条")
    print(
        f"模式：{provider_info()['mode']}（{provider_info()['reason']}）"
        f" ｜ 缓存：{'命中' if meta['cache_hit'] else '未命中'}"
        f" ｜ 耗时：{meta['elapsed_ms']} ms ｜ Prompt：{meta['prompt_version']}"
    )
    print()
    print(result["markdown"])


def _run_analysis(args: argparse.Namespace, force_write: bool = False) -> int:
    jd_text, resume_text = _load_inputs(args.jd, args.resume)
    result = analyze(jd_text, resume_text, use_cache=not args.no_cache)
    _print_summary(result)

    out = getattr(args, "out", None)
    if out or force_write:
        out_path = Path(out or "match_report.md")
        if not out_path.is_absolute():
            out_path = Path.cwd() / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(result["markdown"], encoding="utf-8")
        print(f"报告已写入：{out_path}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """analyze：分析一对 JD/简历并打印完整报告。"""
    return _run_analysis(args)


def cmd_report(args: argparse.Namespace) -> int:
    """report：分析并强制把 Markdown 报告写到文件（默认 match_report.md）。"""
    return _run_analysis(args, force_write=True)


def cmd_batch(args: argparse.Namespace) -> int:
    """batch：并发跑一批（JD, 简历）组合，打印每行总分与汇总。"""
    summary = run_batch(args.pairs, concurrency=args.concurrency, use_cache=not args.no_cache)
    print(f"{_mode_tag()} 批量分析：{summary['pairs_path']}")
    print(f"{'JD':<32}{'简历':<32}{'总分':>6}  缓存")
    for row in summary["rows"]:
        total = "-" if row["total"] is None else str(row["total"])
        cache_text = "命中" if row.get("cache_hit") else "未命中"
        print(f"{row['jd']:<32}{row['resume']:<32}{total:>6}  {cache_text}")
    print(
        f"\n共 {summary['total']} 组，成功 {summary['ok']}，失败 {summary['failed']}，"
        f"平均分 {summary['average_total']}"
    )
    return 0 if summary["failed"] == 0 else 1


def cmd_cache_stats(args: argparse.Namespace) -> int:  # noqa: ARG001 —— argparse 需要统一签名
    """cache-stats：缓存条数、命中率、最近历史与当前 provider。"""
    info = provider_info()
    stats = store.stats()
    print(f"{_mode_tag()} 缓存统计")
    print(f"数据库：{stats['db_path']}")
    print(f"缓存条目：{stats['cache_entries']} ｜ 历史记录：{stats['history_entries']} ｜ 缓存命中率：{stats['cache_hit_rate']:.0%}")
    print(f"Prompt 版本：{stats['prompt_version']} ｜ 评分指纹：{stats['scoring_fingerprint']}")
    print(f"当前 provider：{info['model']}（{info['reason']}）")
    records = store.history(limit=5)
    if records:
        print("\n最近 5 次分析：")
        for record in records:
            cache_text = "命中" if record["cache_hit"] else "未命中"
            print(f"- {record['created_at']}  {record['job_title']}  {record['total']} 分  [{record['mode']}/{cache_text}]")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器（子命令与 SPEC 一致）。"""
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="AI 简历—岗位匹配助手（默认离线可用，配置 .env 后自动切真实模型）",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze", help="分析一对 JD / 简历")
    analyze_parser.add_argument("--jd", required=True, help="JD 文件路径（.md/.txt）")
    analyze_parser.add_argument("--resume", required=True, help="简历文件路径（.md/.txt/.docx/.pdf）")
    analyze_parser.add_argument("--no-cache", action="store_true", help="强制重新计算，不读缓存")
    analyze_parser.add_argument("--out", default=None, help="另外把 Markdown 报告写到该路径")
    analyze_parser.set_defaults(func=cmd_analyze)

    report_parser = subparsers.add_parser("report", help="分析并把报告写入文件")
    report_parser.add_argument("--jd", required=True, help="JD 文件路径")
    report_parser.add_argument("--resume", required=True, help="简历文件路径")
    report_parser.add_argument("--out", default="match_report.md", help="报告输出路径（默认 match_report.md）")
    report_parser.add_argument("--no-cache", action="store_true", help="强制重新计算")
    report_parser.set_defaults(func=cmd_report)

    batch_parser = subparsers.add_parser("batch", help="并发分析一批组合")
    batch_parser.add_argument("--pairs", default="eval/pairs.csv", help="批量清单 CSV（列：jd, resume）")
    batch_parser.add_argument("--concurrency", type=int, default=4, help="并发上限（默认 4）")
    batch_parser.add_argument("--no-cache", action="store_true", help="不读缓存")
    batch_parser.set_defaults(func=cmd_batch)

    stats_parser = subparsers.add_parser("cache-stats", help="查看缓存与历史统计")
    stats_parser.set_defaults(func=cmd_cache_stats)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI 主函数：统一兜住可预期异常，打印中文提示而不是堆栈。"""
    _init_console()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n已取消。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
