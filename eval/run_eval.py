"""项目二评测脚本：与人工标注对照算 MAE，同时验证稳定性与结构化成功率。

运行（在 resume-matcher 目录下，或任意目录用绝对路径）::

    D:\\python\\python.exe eval/run_eval.py

三个数字对应 SPEC 的验收标准：
1. **MAE**：系统分与 eval/labels.csv 里人工分的平均绝对误差；
2. **稳定性**：同一份输入连续跑 3 次，总分极差（要求 < 5）；
3. **结构化成功率**：Capability / ResumeProfile 通过 validate_dict 的比例
   （离线规则引擎恒为 100%，真实模式用它观察模型输出是否合规）。

注意：离线模式的结果是确定性规则输出，**不能当作真实模型指标**；报告里会显式标注模式。
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import config  # noqa: E402
from app.llm import provider_info  # noqa: E402
from app.pipeline import analyze  # noqa: E402
from app.resume_parser import extract_text  # noqa: E402
from app.schemas import Capability, ResumeProfile  # noqa: E402


def _init_console() -> None:
    """Windows 控制台按 UTF-8 输出中文。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_text(path_str: str) -> str:
    return _resolve(path_str).read_text(encoding="utf-8", errors="replace")


def load_labels(path: Path) -> list[dict]:
    """读人工标注；列：jd, resume, human_score, note(可选)。"""
    with path.open(encoding="utf-8", newline="") as handle:
        labels: list[dict] = []
        for line_number, row in enumerate(csv.DictReader(handle), start=2):
            try:
                human_score = int((row.get("human_score") or "").strip())
            except ValueError as exc:
                raise ValueError(f"labels.csv 第 {line_number} 行 human_score 不是整数") from exc
            labels.append(
                {
                    "jd": (row.get("jd") or "").strip(),
                    "resume": (row.get("resume") or "").strip(),
                    "human_score": human_score,
                    "note": (row.get("note") or "").strip(),
                }
            )
    if not labels:
        raise ValueError(f"标注文件为空：{path}")
    return labels


def _stability(jd_text: str, resume_text: str, runs: int = 3) -> list[int]:
    """同一输入重复跑（禁用缓存），返回每次的总分。"""
    return [analyze(jd_text, resume_text, use_cache=False)["score"]["total"] for _ in range(runs)]


def _spearman(pairs: list[tuple[int, int]]) -> float:
    """Spearman 秩相关系数：衡量系统分与人工分的"排序一致性"。

    为什么除了 MAE 还要看它：MAE 惩罚"数值差多少"，但招聘场景更关心"谁该排前面"。
    两份简历系统分一个 90 一个 60、人工分一个 95 一个 70，MAE 都有误差，但排序完全一致，
    这对"筛简历"才是更有意义的指标。用秩而非原始值，也避免受"人工分整体偏高"这种
    系统偏移的干扰。样本 < 3 或存在并列时退化为 Pearson（退化场景这里直接用秩差公式兜底）。
    """
    n = len(pairs)
    if n < 2:
        return 0.0

    def _rank(values: list[int]) -> list[float]:
        # 平均秩：并列值取平均名次
        sorted_pairs = sorted((value, index) for index, value in enumerate(values))
        ranks = [0.0] * n
        index = 0
        while index < n:
            j = index
            while j + 1 < n and sorted_pairs[j + 1][0] == sorted_pairs[index][0]:
                j += 1
            avg = (index + j) / 2 + 1
            for k in range(index, j + 1):
                ranks[sorted_pairs[k][1]] = avg
            index = j + 1
        return ranks

    sys_rank = _rank([p[0] for p in pairs])
    human_rank = _rank([p[1] for p in pairs])
    mean_sys = (n + 1) / 2
    mean_human = (n + 1) / 2
    num = sum((a - mean_sys) * (b - mean_human) for a, b in zip(sys_rank, human_rank))
    den_sys = (sum((a - mean_sys) ** 2 for a in sys_rank)) ** 0.5
    den_human = (sum((b - mean_human) ** 2 for b in human_rank)) ** 0.5
    if den_sys == 0 or den_human == 0:
        return 0.0
    return num / (den_sys * den_human)


def _band_accuracy(pairs: list[tuple[int, int]]) -> tuple[float, list[int]]:
    """分档准确率：把 0-100 分切三档（高 ≥70 / 中 40-69 / 低 <40），
    系统分与人工分落在同一档才算对。招聘里"要不要进下一轮"是分档决策，比精确分更贴近真实用法。

    返回 (准确率, 各样本是否分档正确)。"""
    def _band(score: int) -> str:
        if score >= 70:
            return "high"
        if score >= 40:
            return "mid"
        return "low"

    correct = [1 if _band(sys) == _band(human) else 0 for sys, human in pairs]
    return sum(correct) / len(correct), correct


def run() -> dict:
    """执行评测并返回汇总 dict（同时写 eval/last_eval.md）。"""
    labels = load_labels(PROJECT_ROOT / "eval" / "labels.csv")
    rows: list[dict] = []
    structured_ok = 0

    for item in labels:
        jd_text = _read_text(item["jd"])
        resume_text = extract_text(_resolve(item["resume"]))
        result = analyze(jd_text, resume_text, use_cache=False)
        total = result["score"]["total"]
        rows.append(
            {
                "jd": Path(item["jd"]).name,
                "resume": Path(item["resume"]).name,
                "sys": total,
                "human": item["human_score"],
                "diff": abs(total - item["human_score"]),
                "elapsed_ms": result["meta"]["elapsed_ms"],
                "note": item["note"],
            }
        )
        capability_errors = Capability.validate_dict(result["capability"])
        profile_errors = ResumeProfile.validate_dict(result["profile"])
        if not capability_errors and not profile_errors:
            structured_ok += 1

    mae = sum(row["diff"] for row in rows) / len(rows)
    avg_elapsed = sum(row["elapsed_ms"] for row in rows) / len(rows)
    pairs = [(row["sys"], row["human"]) for row in rows]
    spearman = _spearman(pairs)
    band_acc, band_correct = _band_accuracy(pairs)

    first = labels[0]
    stability_totals = _stability(_read_text(first["jd"]), extract_text(_resolve(first["resume"])))
    stability_spread = max(stability_totals) - min(stability_totals)

    provider = provider_info()
    summary = {
        "mode": provider["mode"],
        "model": provider["model"],
        "sample_count": len(rows),
        "mae": round(mae, 2),
        "spearman": round(spearman, 3),
        "band_accuracy": round(band_acc, 3),
        "stability_spread": stability_spread,
        "stability_totals": stability_totals,
        "structured_success_rate": structured_ok / len(rows),
        "avg_elapsed_ms": round(avg_elapsed, 1),
        "rows": rows,
    }

    lines = [
        f"# 项目二评测结果（{datetime.now().isoformat(timespec='seconds')}）",
        "",
        f"- 模式：{provider['mode']}（{provider['model']}）；Prompt 版本：{config.PROMPT_VERSION}",
        f"- 样本数：{len(rows)} 组（2 份 JD × 5 份简历，人工分来自 eval/labels.csv）",
        f"- **MAE（平均绝对误差）**：{summary['mae']} 分",
        f"- **Spearman 秩相关**：{summary['spearman']}（越接近 1 说明系统分与人工分排序越一致）",
        f"- **分档准确率**：{summary['band_accuracy']:.0%}（高 ≥70 / 中 40-69 / 低 <40 三档，同档算对）",
        f"- **稳定性**：同一输入跑 3 次总分 {stability_totals}，极差 {stability_spread}（要求 < 5）",
        f"- **结构化成功率**：{structured_ok}/{len(rows)} = {summary['structured_success_rate']:.0%}",
        f"- 单组平均耗时：{summary['avg_elapsed_ms']} ms",
        "",
        "| # | JD | 简历 | 系统分 | 人工分 | 绝对误差 | 分档 |",
        "|---|----|------|--------|--------|----------|------|",
    ]
    band_label = {"high": "高", "mid": "中", "low": "低"}
    for index, row in enumerate(rows, start=1):
        sys_band = band_label["high" if row["sys"] >= 70 else "mid" if row["sys"] >= 40 else "low"]
        human_band = band_label["high" if row["human"] >= 70 else "mid" if row["human"] >= 40 else "low"]
        ok = "✓" if sys_band == human_band else "✗"
        lines.append(
            f"| {index} | {row['jd']} | {row['resume']} | {row['sys']} | {row['human']} "
            f"| {row['diff']} | {ok} |"
        )
    lines.extend(
        [
            "",
            "> 说明：人工分由我们自己逐条对照 JD 要求给出，不是行业金标；样本量 10 组，只用于发现明显偏差。",
        ]
    )
    if provider["offline"]:
        lines.append("> 离线规则引擎不调用大模型，以上数字是确定性规则输出，不代表真实模型效果。")

    report = "\n".join(lines)
    (PROJECT_ROOT / "eval" / "last_eval.md").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\n报告已写入：{PROJECT_ROOT / 'eval' / 'last_eval.md'}")
    return summary


if __name__ == "__main__":
    _init_console()
    run()
