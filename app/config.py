"""集中管理配置：模型、Prompt 版本、权重、路径，以及"离线 / 真实"模式判定。

为什么集中：权重、Prompt 版本、阈值这些值会反复调整，散落在各模块里迟早出现
"改了 Prompt 忘了改版本号"这类隐蔽 bug（缓存键里带着旧版本还能命中旧结果）。
测试也需要能整体替换配置而不碰业务逻辑。

离线判定放在这里而不是各模块各自探测：其它模块只读 ``OFFLINE`` /
``OFFLINE_REASON``，不需要 import openai，也不需要每个文件重复写 try/except。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------- 路径：一律以项目根为基准，允许从任意工作目录运行 ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
EVAL_DIR = PROJECT_ROOT / "eval"
DB_PATH = DATA_DIR / "matcher.db"

# ---------- 评分参数（可公开、可争议、可调；见 README「评分设计」） ----------
# V2 起：权重不再拍脑袋，由 eval/calibrate_weights.py 在 10 组人工标注上
# 用留一交叉验证网格搜索得出（LOOCV-MAE 6.50 → 4.40）。结论：硬性要求权重应更高、
# 经历相关度权重应更低、Jaccard 缩放应更大。样本量小，作为方向性结论，详见 eval/weight_calibration.md。
WEIGHTS = {"must": 0.65, "skill": 0.25, "exp": 0.10}
MUST_HIT_OVERLAP = 1 / 3      # 硬性要求：token 重叠 ≥ 1/3 视为命中（经验值，20 份样本上调出来的）
EXP_RELEVANCE_SCALE = 4.0     # Jaccard 天然偏小，放大 4 倍再截断到 1（校准实验得出）

# 评分配置指纹要进缓存键：评分参数（权重/缩放/阈值）一变，旧缓存必须自然失效。
# SCORING_VERSION 是人工兜底：将来改动评分**公式**（参数没变但行为变了）时手动 +1 强制失效。
SCORING_VERSION = "v1"


def scoring_fingerprint() -> str:
    """把影响分数的常量稳定序列化后取 md5 前 12 位。

    为什么需要它：缓存键里只放 PROMPT_VERSION 时，改了 ``WEIGHTS`` /
    ``EXP_RELEVANCE_SCALE`` / ``MUST_HIT_OVERLAP`` 但 Prompt 没改，同一份 JD+简历
    仍会命中旧缓存，把**旧分数**当新结果返回（"我明明调了权重，怎么分数没变"）。
    这里每次调用都重新读模块级常量，所以运行时改动也会反映到指纹上（测试可验证）。

    稳定性：``sort_keys`` + 固定分隔符 + 统一 float 转换，保证同配置在任何进程、
    任何机器上得到同一个指纹；人工版本号也放进 payload，便于强制失效。
    """
    payload = {
        "scoring_version": SCORING_VERSION,
        "weights": {key: float(value) for key, value in sorted(WEIGHTS.items())},
        "exp_relevance_scale": float(EXP_RELEVANCE_SCALE),
        "must_hit_overlap": float(MUST_HIT_OVERLAP),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]

# ---------- 解析与缓存参数 ----------
MIN_TEXT_LEN = 100            # 解析出的文本短于它直接报错（多半是扫描件/加密 PDF）
PROMPT_VERSION = "v1"         # 改 Prompt 必须 +1，它会进缓存键，否则旧结果会被错误命中


def _load_dotenv() -> None:
    """读取项目根目录的 ``.env``，不依赖 python-dotenv。

    为什么要自己写：本机离线装不上任何第三方库，而 .env 又是接真实模型的必要入口。
    规则：``KEY=VALUE``、``#`` 注释、空行忽略；**已存在的环境变量优先**，这样在命令行
    或 CI 里显式传入的配置不会被 .env 悄悄覆盖（排障时少一个变量）。
    """
    path = PROJECT_ROOT / ".env"
    if not path.exists():
        return
    try:
        # utf-8-sig：兼容 Windows 记事本/PowerShell 写入的 BOM，否则第一个 key 会带上 \ufeff
        content = path.read_text(encoding="utf-8-sig", errors="replace")
        for raw in content.splitlines():
            line = raw.strip().lstrip("\ufeff")
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError as exc:  # 文件存在但读不了，也不该拖垮整个程序
        logger.warning("读取 .env 失败，按无配置处理：%s", exc)


def _detect_offline(api_key: str) -> tuple[bool, str]:
    """判定是否走离线规则引擎，并返回一句人话原因（CLI/报告要能解释清楚）。"""
    forced = os.getenv("RESUME_MATCHER_OFFLINE", "").strip().lower()
    if forced in {"1", "true", "yes", "on"}:
        return True, "环境变量 RESUME_MATCHER_OFFLINE 强制离线"
    if not api_key:
        return True, "未配置 SF_API_KEY（无 .env 或值为空）"
    try:
        import openai  # noqa: F401  # 只在判定时探测一次，不参与模块顶部导入
    except ImportError:
        return True, "已配置 SF_API_KEY，但未安装 openai SDK"
    return False, "已配置 Key 且 openai SDK 可用"


_load_dotenv()

API_KEY = os.getenv("SF_API_KEY", "").strip()
BASE_URL = os.getenv("SF_BASE_URL", "https://api.siliconflow.cn/v1").strip()
CHAT_MODEL = os.getenv("CHAT_MODEL", "Qwen/Qwen2.5-7B-Instruct").strip()

OFFLINE, OFFLINE_REASON = _detect_offline(API_KEY)
MODE = "offline" if OFFLINE else "llm"

# ---------- 技能词表：JD 与简历共用同一份，保证两头口径一致 ----------
# 结构：标准名（小写）→ 命中它的若干写法。英文按单词边界匹配，中文按子串匹配。
# 为什么用词表而不是自由抽取：初版要的是"可复现、可解释"；词表命中能明确解释
# "为什么算缺这个技能"。局限：词表外的技能会漏（见 README「已知局限」）。
SKILL_PATTERNS: dict[str, tuple[str, ...]] = {
    "python": ("python",),
    "java": ("java",),
    "javascript": ("javascript", "js"),
    "typescript": ("typescript",),
    "sql": ("sql",),
    "mysql": ("mysql",),
    "sqlite": ("sqlite",),
    "postgresql": ("postgresql", "postgres"),
    "redis": ("redis",),
    "mongodb": ("mongodb", "mongo"),
    "docker": ("docker", "容器化"),
    "kubernetes": ("kubernetes", "k8s"),
    "git": ("git",),
    "linux": ("linux",),
    "fastapi": ("fastapi",),
    "flask": ("flask",),
    "django": ("django",),
    "restful api": ("restful",),
    "pandas": ("pandas",),
    "numpy": ("numpy",),
    "pytorch": ("pytorch",),
    "tensorflow": ("tensorflow",),
    "机器学习": ("机器学习",),
    "深度学习": ("深度学习",),
    "spark": ("spark",),
    "hadoop": ("hadoop",),
    "hive": ("hive",),
    "airflow": ("airflow",),
    "kafka": ("kafka",),
    "消息队列": ("消息队列", "rabbitmq"),
    "tableau": ("tableau",),
    "power bi": ("power bi", "powerbi"),
    "excel": ("excel",),
    "数据分析": ("数据分析",),
    "数据仓库": ("数据仓库", "数仓"),
    "数据建模": ("数据建模", "维度建模"),
    "etl": ("etl", "数据清洗"),
    "ab test": ("ab 测试", "ab测试", "a/b 测试", "a/b测试", "a/b 实验", "ab 实验", "ab test"),
    "数据可视化": ("数据可视化", "可视化"),
    "rag": ("rag", "检索增强"),
    "llm": ("llm", "大模型", "大语言模型"),
    "prompt engineering": ("prompt 工程", "prompt工程", "提示词工程", "prompt engineering"),
    "向量检索": ("向量检索", "向量数据库", "语义检索"),
    "embedding": ("embedding", "向量化"),
    "langchain": ("langchain",),
    "微服务": ("微服务",),
    "高并发": ("高并发",),
    "敏捷开发": ("敏捷开发", "scrum"),
}

# 常见同义/缩写归一，避免 "k8s" 与 "kubernetes" 被当成两个技能
SKILL_ALIASES = {
    "k8s": "kubernetes",
    "postgres": "postgresql",
    "powerbi": "power bi",
    "a/b 测试": "ab test",
    "a/b测试": "ab test",
    "ab测试": "ab test",
    "数据清洗": "etl",
    "检索增强": "rag",
    "大模型": "llm",
}
