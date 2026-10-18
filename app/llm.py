"""大模型统一入口：真实模式（OpenAI 兼容协议）+ 离线模式的异常约定。

为什么要统一封装：换模型、加日志、加重试、统计 token 都只改这一处。
为什么离线模式抛异常而不是返回假 JSON：如果离线时也"假装模型返回了 JSON"，
线上的日志、报告、评测就分不清"结果来自模型还是规则"，调试和诚信都会出问题。
调用方看到 ``OfflineLLMError`` 会明确降级到本地规则引擎，并在 meta 里标记模式。
"""

from __future__ import annotations

import logging
import time

from app import config

logger = logging.getLogger(__name__)

_client = None


class OfflineLLMError(RuntimeError):
    """离线模式下调模型时抛出；调用方据此降级到规则引擎。"""


class LLMError(RuntimeError):
    """真实模式下重试后仍失败；调用方据此降级到规则引擎或向上抛。"""


def _is_retryable(exc: Exception) -> bool:
    """判断异常值不值得重试：429/5xx/超时值得，401/403/400 重试也不会变好。"""
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if status is None:
        text = str(exc).lower()
        return any(k in text for k in ("timeout", "timed out", "429", "502", "503", "504", "connection reset"))
    try:
        code = int(status)
    except (TypeError, ValueError):
        return True
    return code in (408, 409, 429) or 500 <= code < 600


def _get_client():
    """延迟导入 openai 并复用连接。

    缺库时只有真实模式受影响：模块顶部不 import，离线运行、单测、CLI 都不受影响。
    """
    global _client
    if _client is None:
        from openai import OpenAI  # 缺库时这里才抛 ImportError，由调用方兜住

        _client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL, timeout=60)
    return _client


def provider_info() -> dict:
    """当前 provider 摘要；CLI、报告、评测统一从这里取，避免各模块自己判断。"""
    return {
        "mode": config.MODE,
        "offline": config.OFFLINE,
        "model": config.CHAT_MODEL if not config.OFFLINE else "离线规则引擎",
        "base_url": config.BASE_URL if not config.OFFLINE else "",
        "prompt_version": config.PROMPT_VERSION,
        "reason": config.OFFLINE_REASON,
    }


def chat(
    prompt: str,
    temperature: float = 0.2,
    json_mode: bool = False,
    max_retry: int = 2,
) -> str:
    """调一次对话模型，返回纯文本；失败按 1s / 2s / 4s 退避重试。

    - 抽取类任务（JD / 简历结构化）用 ``temperature=0``，要的是可复现；
    - 生成建议类用 ``temperature=0.5``，要的是措辞有变化但不发散；
    - ``json_mode=True`` 时请求平台强制 JSON，Prompt 里也必须出现 "JSON" 字样。

    失败抛 ``LLMError``；离线模式立即抛 ``OfflineLLMError``，不做无意义等待。
    """
    if config.OFFLINE:
        raise OfflineLLMError(f"离线模式不调用大模型：{config.OFFLINE_REASON}")

    try:
        client = _get_client()
    except ImportError as exc:
        raise OfflineLLMError("未安装 openai SDK，无法调用真实模型") from exc

    kwargs: dict = {
        "model": config.CHAT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": 2000,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    last_err: Exception | None = None
    for attempt in range(max_retry + 1):
        started = time.perf_counter()
        try:
            resp = client.chat.completions.create(**kwargs)
            text = (resp.choices[0].message.content or "").strip()
            logger.info(
                "LLM 调用成功 attempt=%s 耗时=%.2fs usage=%s",
                attempt,
                time.perf_counter() - started,
                getattr(resp, "usage", None),
            )
            if not text:
                raise LLMError("模型返回空文本")
            return text
        except Exception as exc:  # noqa: BLE001 —— SDK 异常层级不稳定，先收口再判断是否重试
            last_err = exc
            if not _is_retryable(exc) or attempt >= max_retry:
                logger.error("LLM 调用失败（不再重试）：%s", exc)
                break
            wait = 2**attempt
            logger.warning("LLM 调用失败（%s），%ss 后重试（第 %s/%s 次）", exc, wait, attempt + 1, max_retry)
            time.sleep(wait)

    raise LLMError(f"大模型调用失败：{last_err}")
