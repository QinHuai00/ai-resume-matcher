"""数据契约：Capability / ResumeProfile / MatchResult（dataclass + 手工校验）。

为什么不用 Pydantic：本机离线装不上 pydantic，而"大模型输出必须被校验"是这个项目的
立身之本，不能省。这里用 dataclass 承担"结构定义"，用 ``validate_dict`` 承担"校验"，
返回的中文错误列表可以直接回灌给模型让它重写——和 Pydantic 的效果一致。
真实项目里换成 pydantic 只需替换本文件（``to_dict / from_dict / validate_dict``
这组接口保持不变，调用方不用改）。

另一个设计点：``from_dict`` 对脏数据是宽容的（类型不对就当空、去重、转小写），
因为校验和清洗的职责分开：``validate_dict`` 负责"报错让人/模型改"，
``from_dict`` 负责"取能用的部分继续跑"，避免一次小瑕疵让整条流水线崩掉。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _clean_str_list(value: Any) -> list[str]:
    """把任意输入整理成"非空字符串、去重、保序"的列表。"""
    if not isinstance(value, list):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _clean_skill_list(value: Any) -> list[str]:
    """技能列表归一：**先去空白转小写再去重**（"Python" 与 "python" 是同一个技能）。"""
    if not isinstance(value, list):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        skill = item.strip().lower()
        if skill and skill not in seen:
            seen.add(skill)
            out.append(skill)
    return out


def _validate_str_list(name: str, value: Any, errors: list[str], required: bool = False) -> None:
    """校验"字符串数组"字段，错误信息写清楚是哪个字段第几项。"""
    if not isinstance(value, list):
        errors.append(f"{name} 必须是数组，实际是 {type(value).__name__}")
        return
    if required and not value:
        errors.append(f"{name} 不能为空数组（原文没写时可以放空数组，但这里一条都没抽到）")
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            errors.append(f"{name}[{index}] 必须是非空字符串")


def _optional_str(data: dict, key: str) -> str | None:
    """可空字符串字段：None / 空串 / 非字符串统一收敛成 None。"""
    value = data.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass
class Capability:
    """JD 抽出来的岗位能力画像。"""

    job_title: str = ""
    must_have: list[str] = field(default_factory=list)
    nice_to_have: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    years: str | None = None
    education: str | None = None

    def to_dict(self) -> dict:
        """转成可 JSON 序列化的 dict（skills 统一小写）。"""
        data = asdict(self)
        data["skills"] = sorted({s.lower().strip() for s in self.skills if s.strip()})
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Capability":
        """容错构造：字段缺失/类型不对时退化为空值，不抛异常。"""
        data = data if isinstance(data, dict) else {}
        return cls(
            job_title=str(data.get("job_title") or "").strip(),
            must_have=_clean_str_list(data.get("must_have")),
            nice_to_have=_clean_str_list(data.get("nice_to_have")),
            skills=_clean_skill_list(data.get("skills")),
            years=_optional_str(data, "years"),
            education=_optional_str(data, "education"),
        )

    @classmethod
    def validate_dict(cls, data: Any) -> list[str]:
        """校验一个"疑似 Capability"的 dict，返回中文错误列表（空表示通过）。"""
        if not isinstance(data, dict):
            return [f"顶层必须是 JSON 对象，实际是 {type(data).__name__}"]
        errors: list[str] = []
        if not isinstance(data.get("job_title"), str) or not data.get("job_title", "").strip():
            errors.append("job_title 必须是非空字符串（岗位名称）")
        _validate_str_list("must_have", data.get("must_have", []), errors)
        _validate_str_list("nice_to_have", data.get("nice_to_have", []), errors)
        _validate_str_list("skills", data.get("skills", []), errors)
        for key in ("years", "education"):
            value = data.get(key)
            if value is not None and not isinstance(value, str):
                errors.append(f"{key} 必须是字符串或 null")
        return errors


@dataclass
class ResumeProfile:
    """简历抽出来的候选人画像。"""

    name: str | None = None
    education: str | None = None
    skills: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)
    highlights: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["skills"] = sorted({s.lower().strip() for s in self.skills if s.strip()})
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "ResumeProfile":
        data = data if isinstance(data, dict) else {}
        return cls(
            name=_optional_str(data, "name"),
            education=_optional_str(data, "education"),
            skills=_clean_skill_list(data.get("skills")),
            projects=_clean_str_list(data.get("projects")),
            highlights=_clean_str_list(data.get("highlights")),
        )

    @classmethod
    def validate_dict(cls, data: Any) -> list[str]:
        if not isinstance(data, dict):
            return [f"顶层必须是 JSON 对象，实际是 {type(data).__name__}"]
        errors: list[str] = []
        for key in ("name", "education"):
            value = data.get(key)
            if value is not None and not isinstance(value, str):
                errors.append(f"{key} 必须是字符串或 null")
        _validate_str_list("skills", data.get("skills", []), errors)
        _validate_str_list("projects", data.get("projects", []), errors)
        _validate_str_list("highlights", data.get("highlights", []), errors)
        return errors


@dataclass
class MatchResult:
    """一次完整分析的结果：画像 + 评分 + 建议 + 报告 + 元信息。"""

    capability: Capability = field(default_factory=Capability)
    profile: ResumeProfile = field(default_factory=ResumeProfile)
    score: dict = field(default_factory=dict)
    advices: list[dict] = field(default_factory=list)
    markdown: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "capability": self.capability.to_dict(),
            "profile": self.profile.to_dict(),
            "score": dict(self.score),
            "advices": [dict(a) for a in self.advices],
            "markdown": self.markdown,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MatchResult":
        data = data if isinstance(data, dict) else {}
        return cls(
            capability=Capability.from_dict(data.get("capability", {})),
            profile=ResumeProfile.from_dict(data.get("profile", {})),
            score=dict(data.get("score") or {}),
            advices=[a for a in (data.get("advices") or []) if isinstance(a, dict)],
            markdown=str(data.get("markdown") or ""),
            meta=dict(data.get("meta") or {}),
        )

    @classmethod
    def validate_dict(cls, data: Any) -> list[str]:
        if not isinstance(data, dict):
            return [f"顶层必须是 JSON 对象，实际是 {type(data).__name__}"]
        errors = Capability.validate_dict(data.get("capability", {}))
        errors += ResumeProfile.validate_dict(data.get("profile", {}))
        score = data.get("score")
        if not isinstance(score, dict):
            errors.append("score 必须是对象")
        else:
            total = score.get("total")
            if not isinstance(total, int) or not 0 <= total <= 100:
                errors.append("score.total 必须是 0-100 的整数")
        if not isinstance(data.get("advices", []), list):
            errors.append("advices 必须是数组")
        return errors


def capability_schema() -> dict:
    """给 Prompt 用的字段说明（真实项目里由 pydantic 的 model_json_schema 生成）。"""
    return {
        "job_title": "岗位名称，字符串",
        "must_have": ["硬性要求，逐条字符串"],
        "nice_to_have": ["加分项，逐条字符串"],
        "skills": ["技能关键词，小写，去重，不要整句话"],
        "years": "年限要求，如 '3年'；原文没写填 null",
        "education": "学历要求，如 '本科及以上'；原文没写填 null",
    }


def resume_schema() -> dict:
    """给 Prompt 用的简历字段说明。"""
    return {
        "name": "候选人姓名；没写填 null",
        "education": "教育背景一行摘要；没写填 null",
        "skills": ["技能关键词，小写，去重"],
        "projects": ["每段经历一句话摘要：项目名 + 做了什么 + 用了什么技术"],
        "highlights": ["只放原文中出现数字/比例的成果，原文没有就留空数组"],
    }
