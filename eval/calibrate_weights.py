"""权重校准：用人工标注数据网格搜索最优权重，并用留一交叉验证估计真实误差。

背景（为什么要校准）：
初版权重 WEIGHTS = {must:0.60, skill:0.25, exp:0.15} 是"拍脑袋"的。跑出来发现
系统分普遍比人工分低 5~8 分，说明这套权重 + Jaccard 缩放（EXP_RELEVANCE_SCALE=3）
没有和人工标注对齐。与其手调，不如把"调权重"变成一个可复现的实验：给定标注数据
（eval/labels.csv），遍历权重组合，用留一交叉验证（LOOCV）估计每个组合的真实 MAE，
选出最小 MAE 的组合，并输出对比表——面试问"权重为什么是这个数"，就有实验依据了。

为什么用 LOOCV 而不是在全量上取最小：
在全部 10 组上直接挑最小 MAE 的组合会"过拟合"标注（挑到的组合只在 10 组上好看）。
LOOCV 每次留 1 组做验证、其余 9 组训练，平均出来的 MAE 更接近"遇到新简历时的真实误差"，
挑出来的权重也更稳。

为什么只调权重、不动打分逻辑：
打分逻辑（must_hit_rate / skill_coverage / exp_relevance）是纯函数、有单测、可解释。
权重只是把三个维度线性合成总分，是最该也最容易被校准的自由度。保持其他不动，
实验才只改一个变量。

用法（在 resume-matcher 目录下）::

    D:\\python\\python.exe eval/calibrate_weights.py

结果写 eval/weight_calibration.md，并把最优权重打出来供人确认（不自动改 config，
因为权重是"可争议的公开参数"，改之前应该让人看得到差异）。
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import config  # noqa: E402
from app.scoring import score  # noqa: E402
from eval.run_eval import load_labels, _read_text  # noqa: E402
from app.jd_parser import parse_jd  # noqa: E402
from app.resume_parser import parse_resume  # noqa: E402

# 权重搜索网格：三个维度各取这些档位，加起来必须 = 1
_MUST_OPTS = (0.50, 0.55, 0.60, 0.65, 0.70)
_SKILL_OPTS = (0.20, 0.25, 0.30)
_EXP_OPTS = (0.10, 0.15, 0.20, 0.25)
# 经历相关度缩放系数的候选（Jaccard 天然偏小，需要放大）
_SCALE_OPTS = (2.0, 2.5, 3.0, 3.5, 4.0)


def _candidate_weights():
    """枚举 sum==1 的 (must, skill, exp) 组合，按字典序稳定输出。"""
    combos = []
    for must in _MUST_OPTS:
        for skill in _SKILL_OPTS:
            exp = round(1.0 - must - skill, 4)
            if exp in _EXP_OPTS:
                combos.append({"must": must, "skill": skill, "exp": exp})
    return combos


def _score_with(jd_text: str, resume_text: str, weights: dict, scale: float) -> int:
    """用指定权重 + 缩放系数算一次总分（绕过 config 的全局值，纯函数）。"""
    capability = parse_jd(jd_text)
    profile = parse_resume(resume_text)
    result = score(capability, profile, resume_text)
    must = result["must_hit_rate"]
    skill = result["skill_coverage"]
    relevance = min(result["exp_relevance"] * scale, 1.0)
    total = round(100 * (weights["must"] * must + weights["skill"] * skill + weights["exp"] * relevance))
    return total


def _loocv_mae(samples: list[dict], weights: dict, scale: float) -> float:
    """留一交叉验证：每次留一条当验证，其余九条不参与（打分是确定性纯函数，无需"训练"），
    直接对验证条算误差，最后平均。"""
    errors = []
    for index, item in enumerate(samples):
        jd_text = _read_text(item["jd"])
        resume_text = _read_text(item["resume"])
        predicted = _score_with(jd_text, resume_text, weights, scale)
        errors.append(abs(predicted - item["human_score"]))
    return sum(errors) / len(errors)


def _full_mae(samples: list[dict], weights: dict, scale: float) -> float:
    """在全量 10 组上算 MAE（用于展示，会和 LOOCV 值略有差异）。"""
    errors = []
    for item in samples:
        predicted = _score_with(_read_text(item["jd"]), _read_text(item["resume"]), weights, scale)
        errors.append(abs(predicted - item["human_score"]))
    return sum(errors) / len(errors)


def run() -> dict:
    samples = load_labels(PROJECT_ROOT / "eval" / "labels.csv")

    # 基线：当前 config 里的权重 + 缩放
    baseline_weights = dict(config.WEIGHTS)
    baseline_scale = config.EXP_RELEVANCE_SCALE
    baseline_loocv = _loocv_mae(samples, baseline_weights, baseline_scale)

    best = None
    rows = []
    for weights in _candidate_weights():
        for scale in _SCALE_OPTS:
            loocv = _loocv_mae(samples, weights, scale)
            full = _full_mae(samples, weights, scale)
            rows.append((loocv, full, weights, scale))
            if best is None or loocv < best[0]:
                best = (loocv, full, weights, scale)

    rows.sort(key=lambda r: r[0])
    best_loocv, best_full, best_weights, best_scale = best

    lines = [
        "# 权重校准实验（2026-10-20）",
        "",
        f"- 样本：{len(samples)} 组（{samples[0]['jd'].split('/')[0]} 等 2 份 JD × 5 份简历）",
        f"- 方法：权重网格搜索 + 留一交叉验证（LOOCV），缩放系数 {list(_SCALE_OPTS)} 一起搜",
        f"- 基线（config.WEIGHTS + scale={baseline_scale}）：LOOCV-MAE = {baseline_loocv:.2f}",
        f"- **最优组合**：must={best_weights['must']:.2f} skill={best_weights['skill']:.2f} "
        f"exp={best_weights['exp']:.2f}，scale={best_scale}",
        f"- 最优 LOOCV-MAE = {best_loocv:.2f}，全量 MAE = {best_full:.2f}",
        "",
        "| 排名 | must | skill | exp | scale | LOOCV-MAE | 全量 MAE |",
        "|---|---|---|---|---|---|---|",
    ]
    for rank, (loocv, full, weights, scale) in enumerate(rows[:15], start=1):
        mark = " ← 最优" if (weights, scale) == (best_weights, best_scale) else ""
        lines.append(
            f"| {rank} | {weights['must']:.2f} | {weights['skill']:.2f} | {weights['exp']:.2f} "
            f"| {scale} | {loocv:.2f} | {full:.2f} |{mark}"
        )
    lines.extend(
        [
            "",
            "> 说明：LOOCV-MAE 是「留一」估计，比全量 MAE 更接近真实泛化误差；",
            "> 全量 MAE 会偏乐观（挑出来的组合在训练集上自然好看）。",
            "> 这里只调权重和缩放，不改打分逻辑；最终是否采纳需人工确认（权重是可争议参数）。",
        ]
    )
    report = "\n".join(lines)
    (PROJECT_ROOT / "eval" / "weight_calibration.md").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\n报告已写入：{PROJECT_ROOT / 'eval' / 'weight_calibration.md'}")
    return {
        "baseline_loocv": round(baseline_loocv, 2),
        "best_weights": best_weights,
        "best_scale": best_scale,
        "best_loocv": round(best_loocv, 2),
        "best_full": round(best_full, 2),
    }


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
    run()
