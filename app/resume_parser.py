"""简历解析：文件 → 文本 → ResumeProfile。

三条硬性约束：
1. **缺库不崩**：PDF 需要 ``pypdf``、DOCX 需要 ``python-docx``，本机只有后者。
   缺库时抛的是带安装指引的中文 RuntimeError，CLI 打一条提示就退出，不打印堆栈。
2. **文本过短直接拒绝**：扫描件/图片版 PDF 抽出来是空，放它进流水线后面全是垃圾输入。
   所以 ``MIN_TEXT_LEN`` 是硬门槛，报错里说明可能原因和替代方案。
3. **一栏式假设**：按行/空行切段落，两栏排版抽出来会串行（这是初版的已知局限，
   写进 README，不在初版做坐标重排）。
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from app import config
from app.jd_parser import _clean, extract_skills
from app.llm import LLMError, OfflineLLMError, chat
from app.schemas import ResumeProfile, resume_schema

logger = logging.getLogger(__name__)

_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".text"}
_PROJECT_HINTS = ("项目", "实习", "工作", "实践", "竞赛")
_PROJECT_HINT_SCAN_HEAD = 120  # 只在段落开头这么多字里找项目关键词，避免误收"个人优势"整段
# 段落级黑名单：章节标题、求职意向/个人信息这类段落不是"项目经历"
_HEADING_LINE_RE = re.compile(r"^#{1,6}\s")
_PERSONAL_LINE_RE = re.compile(r"^(?:求职意向|个人信息|联系方式|电话|手机|邮箱|微信|姓名|年龄|现居|住址)")
_CONTACT_RE = re.compile(r"@|\d{5,}")

_RESUME_PROMPT = """你是资深 HR。请从下面的简历文本中抽取结构化信息。

只输出 JSON，不要解释，不要 Markdown 代码块。字段定义（JSON 对象，键名必须完全一致）：
{schema}

规则：
1. skills 只放技术名词或能力名词，统一小写，去重；
2. projects 每项一句话：项目名 + 做了什么 + 用了什么技术；
3. highlights 只放原文中出现数字/比例的成果，原文没有就留空数组，禁止编造；
4. 简历没写的字段填 null。

简历文本：
---
{text}
---"""


def _clean_text(text: str) -> str:
    """清洗文本：全角空格、连续空格、过多空行；保留空行（段落切分依赖它）。"""
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _read_pdf(path: Path) -> str:
    """逐页抽 PDF 文本；缺 pypdf 时给安装指引而不是堆栈。"""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "解析 PDF 需要 pypdf，当前环境未安装。请执行 pip install pypdf；"
            "或把简历另存为 .docx / .txt 后再试（离线演示推荐 .md/.txt 样本）。"
        ) from exc
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _read_docx(path: Path) -> str:
    """抽 DOCX 段落 + 表格文本（表格里的时间线常被漏掉，必须单独遍历）。"""
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError(
            "解析 DOCX 需要 python-docx，当前环境未安装。请执行 pip install python-docx；"
            "或把简历另存为 .txt 后再试。"
        ) from exc
    document = Document(str(path))
    lines = [p.text.strip() for p in document.paragraphs if p.text and p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text and cell.text.strip()]
            if cells:
                lines.append(" | ".join(cells))
    return "\n".join(lines)


def extract_text(path: str | Path) -> str:
    """读简历文件并返回清洗后的文本；文本过短抛 ValueError。"""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"简历文件不存在：{file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        raw = _read_pdf(file_path)
    elif suffix == ".docx":
        raw = _read_docx(file_path)
    elif suffix == ".doc":
        raise ValueError("不支持旧版 .doc 二进制格式，请在 Word/WPS 里另存为 .docx 后再试。")
    else:
        raw = file_path.read_text(encoding="utf-8", errors="replace")

    raw = _clean_text(raw)
    if len(raw) < config.MIN_TEXT_LEN:
        raise ValueError(
            f"解析出的文本过短（{len(raw)} 字，要求 ≥ {config.MIN_TEXT_LEN}）："
            "常见原因是扫描件/图片版 PDF、加密文件或上传错了文件。"
            "请换一份带文本层的文件，或另存为 .docx / .txt 再试。"
        )
    logger.info("简历解析成功：%s（%s 字）", file_path.name, len(raw))
    return raw


def _find_name(text: str) -> str | None:
    """姓名：取开头几行里最短的"像人名"的一行，排除联系方式/求职意向/章节标题。"""
    for line in text.splitlines()[:8]:
        candidate = line.strip(" #*【】")
        if not candidate or len(candidate) > 20:
            continue
        if re.search(r"@|\d{5,}|电话|邮箱|微信|求职|简历|意向|github", candidate, re.IGNORECASE):
            continue
        if re.search(r"[:：]", candidate):
            continue
        if re.search(r"(项目|技能|教育|经历|优势|背景|信息)", candidate):
            continue
        return candidate
    return None


def _find_education(text: str) -> str | None:
    """教育背景：取第一行含学校/学历/专业的行，截断到 80 字。"""
    for line in text.splitlines():
        cleaned = line.strip(" #*")
        if len(cleaned) >= 4 and re.search(r"(大学|学院|本科|硕士|博士|研究生|大专|专科|学历|GPA|专业)", cleaned):
            return cleaned[:80]
    return None


def _split_blocks(text: str) -> list[str]:
    """按空行切段落；段落是项目经历的最小单元（样本简历用空行分隔项目）。"""
    return [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]


def _parse_projects(text: str, limit: int = 6) -> list[str]:
    """项目/经历摘要：段落开头出现"项目/实习/工作/实践/竞赛"即收，取前三行合并。

    过滤三类误收：章节标题段落（"## 项目经历"）、求职意向/个人信息段落、
    纯联系方式段落——它们都只是"碰巧含关键词"，不是经历本身。
    """
    projects: list[str] = []
    for block in _split_blocks(text):
        first_line = block.splitlines()[0].strip()
        if _HEADING_LINE_RE.match(first_line) or _PERSONAL_LINE_RE.match(first_line):
            continue
        if _CONTACT_RE.search(block) and len(block) < 120:
            continue
        head = block[:_PROJECT_HINT_SCAN_HEAD]
        if not any(hint in head for hint in _PROJECT_HINTS):
            continue
        lines = [line.strip(" -•*·") for line in block.splitlines() if line.strip()]
        if len(lines) < 2:  # 只有一个标题行，没有内容，不算经历
            continue
        summary = "；".join(lines[:3])
        projects.append(summary[:180])
        if len(projects) >= limit:
            break
    if not projects:  # 兜底：没有空行分隔、但行内含"项目"的文本
        projects = [line.strip(" -•*·")[:180] for line in text.splitlines() if "项目" in line][:limit]
    return projects


def _parse_highlights(text: str, limit: int = 8) -> list[str]:
    """亮点：只保留含数字的行（量化成果），过滤联系方式与以日期开头的教育/在职行。"""
    highlights: list[str] = []
    for line in text.splitlines():
        raw = line.strip(" -•*·#")
        if len(raw) < 6 or not re.search(r"\d", raw):
            continue
        if _CONTACT_RE.search(raw):
            continue
        if _PERSONAL_LINE_RE.match(raw):
            continue
        if re.match(r"^(?:20\d\d)\s*[./年-]", raw):
            continue
        if raw in highlights:
            continue
        highlights.append(raw[:120])
        if len(highlights) >= limit:
            break
    return highlights


def _parse_resume_rules(text: str) -> ResumeProfile:
    """离线规则解析：词表匹配技能、空行切项目、含数字行当亮点。"""
    return ResumeProfile(
        name=_find_name(text),
        education=_find_education(text),
        skills=extract_skills(text),
        projects=_parse_projects(text),
        highlights=_parse_highlights(text),
    )


def parse_resume(text: str, max_retry: int = 1) -> ResumeProfile:
    """简历解析入口：离线走规则；真实模式先 LLM，校验失败回灌重试，最终降级规则。"""
    content = (text or "").strip()
    if len(content) < config.MIN_TEXT_LEN:
        raise ValueError(
            f"简历文本过短（{len(content)} 字，要求 ≥ {config.MIN_TEXT_LEN}），"
            "无法可靠解析；请检查文件是否正常、是否上传错了文件。"
        )

    if not config.OFFLINE:
        prompt = _RESUME_PROMPT.format(
            schema=json.dumps(resume_schema(), ensure_ascii=False),
            text=content[:8000],
        )
        for attempt in range(max_retry + 1):
            try:
                raw = chat(prompt, temperature=0, json_mode=True, max_retry=1)
            except (OfflineLLMError, LLMError) as exc:
                logger.warning("简历解析调用模型失败，降级规则引擎：%s", exc)
                break
            try:
                data = json.loads(_clean(raw))
            except json.JSONDecodeError as exc:
                prompt += f"\n\n你上次的输出不是合法 JSON（错误：{exc}），请只输出合法 JSON。"
                logger.warning("简历输出非法 JSON（attempt=%s）：%s", attempt, exc)
                continue
            errors = ResumeProfile.validate_dict(data)
            if not errors:
                profile = ResumeProfile.from_dict(data)
                if profile.skills or profile.projects:
                    return profile
                errors = ["skills 与 projects 同时为空，至少要有其中一项"]
            logger.warning("简历输出校验失败（attempt=%s）：%s", attempt, errors)
            prompt += "\n\n你上次的输出不符合格式，错误如下：\n- " + "\n- ".join(errors) + (
                "\n请重新只输出合法 JSON。"
            )

    return _parse_resume_rules(content)
