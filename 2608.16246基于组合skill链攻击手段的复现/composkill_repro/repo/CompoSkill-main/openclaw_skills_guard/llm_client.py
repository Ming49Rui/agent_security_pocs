"""
OpenAI-compatible LLM API client.
Supports streaming / non-streaming, with auto-retry on context overflow.
"""

import json
import logging
import re
from typing import AsyncIterator, Dict, List, Optional, Tuple

import httpx

from config import settings

logger = logging.getLogger("openclaw_skills_guard.llm")


def _build_payload(
    messages: List[Dict[str, str]], stream: bool, max_tokens: int = 0
) -> dict:
    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "repetition_penalty": settings.repetition_penalty,
        "stream": stream,
    }
    effective_max = max_tokens if max_tokens > 0 else settings.max_output_tokens
    if effective_max > 0:
        payload["max_tokens"] = effective_max
    return payload


def _build_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"
    return headers


def parse_context_limit(error_text: str) -> Optional[int]:
    """
    Parse the actual context limit from a 400 error message.
    Example: "maximum context length is 8192 tokens. However, you requested 9271 tokens (5175 in the messages, 4096 in the completion)"
    """
    match = re.search(r"maximum context length is (\d+)", error_text)
    if match:
        return int(match.group(1))
    return None


def parse_requested_tokens(error_text: str) -> Optional[Tuple[int, int]]:
    """Parse (input_tokens, completion_tokens) from error message."""
    match = re.search(r"(\d+) in the messages.*?(\d+) in the completion", error_text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


async def stream_chat(
    messages: List[Dict[str, str]], max_tokens: int = 0,
) -> AsyncIterator[str]:
    """
    Call the LLM API. Yields content chunks.
    If streaming fails, falls back to blocking mode.
    """
    if settings.llm_stream:
        async for chunk in _stream_mode(messages, max_tokens):
            yield chunk
    else:
        result = await _blocking_mode(messages, max_tokens)
        yield result


async def _stream_mode(
    messages: List[Dict[str, str]], max_tokens: int = 0
) -> AsyncIterator[str]:
    payload = _build_payload(messages, stream=True, max_tokens=max_tokens)
    headers = _build_headers()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            async with client.stream(
                "POST", settings.llm_api_url, json=payload, headers=headers
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    body_str = body.decode("utf-8", errors="ignore")
                    logger.error("LLM stream error %d: %s", resp.status_code, body_str[:500])
                    # fall back to blocking, which handles retry
                    logger.info("Falling back to non-streaming mode")
                    result = await _blocking_mode(messages, max_tokens)
                    yield result
                    return

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, IndexError, KeyError):
                        continue
    except httpx.ConnectError:
        yield "\n\n> ⚠️ 无法连接到大模型API，请检查 LLM_API_URL 配置。\n"
    except httpx.ReadTimeout:
        yield "\n\n> ⚠️ 大模型API响应超时（300s），请检查模型是否正常运行。\n"
    except Exception as exc:
        logger.exception("LLM streaming error")
        yield f"\n\n> ⚠️ 大模型调用异常: {exc}\n"


async def _blocking_mode(messages: List[Dict[str, str]], max_tokens: int = 0) -> str:
    """
    Non-streaming call. If 400 with context limit error,
    returns a special marker so the caller can retry with reduced content.
    """
    payload = _build_payload(messages, stream=False, max_tokens=max_tokens)
    headers = _build_headers()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            resp = await client.post(
                settings.llm_api_url, json=payload, headers=headers
            )
            if resp.status_code == 400:
                error_text = resp.text
                ctx_limit = parse_context_limit(error_text)
                if ctx_limit:
                    req_info = parse_requested_tokens(error_text)
                    raise ContextOverflowError(ctx_limit, error_text, req_info)
                logger.error("LLM 400 error: %s", error_text[:500])
                return f"\n\n> ⚠️ 大模型API返回错误 (HTTP 400): {error_text[:200]}\n"

            if resp.status_code != 200:
                logger.error("LLM API error %d: %s", resp.status_code, resp.text[:500])
                return f"\n\n> ⚠️ 大模型API返回错误 (HTTP {resp.status_code}): {resp.text[:200]}\n"

            data = resp.json()
            return data["choices"][0]["message"]["content"]
    except ContextOverflowError:
        raise
    except httpx.ConnectError:
        return "\n\n> ⚠️ 无法连接到大模型API，请检查 LLM_API_URL 配置。\n"
    except httpx.ReadTimeout:
        return "\n\n> ⚠️ 大模型API响应超时（300s），请检查模型是否正常运行。\n"
    except Exception as exc:
        logger.exception("LLM blocking call error")
        return f"\n\n> ⚠️ 大模型调用异常: {exc}\n"


class ContextOverflowError(Exception):
    """Raised when model reports context length exceeded."""
    def __init__(
        self, max_tokens: int, raw_message: str,
        requested: Optional[Tuple[int, int]] = None,
    ):
        self.max_tokens = max_tokens
        self.raw_message = raw_message
        self.input_tokens = requested[0] if requested else None
        self.completion_tokens = requested[1] if requested else None
        super().__init__(f"Context overflow: model max is {max_tokens} tokens")


async def test_connection(timeout: float = 30.0) -> dict:
    """
    Send a short test prompt to verify model connectivity.
    Returns {"success": bool, "message": str, "response_preview": str}.
    """
    test_messages = [{"role": "user", "content": "请回复OK"}]
    payload = _build_payload(test_messages, stream=False, max_tokens=32)
    headers = _build_headers()

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            resp = await client.post(
                settings.llm_api_url, json=payload, headers=headers
            )

            if resp.status_code == 400:
                error_text = resp.text
                ctx_limit = parse_context_limit(error_text)
                if ctx_limit:
                    return {
                        "success": False,
                        "message": f"模型上下文限制为 {ctx_limit} tokens，但连接本身是通的",
                        "response_preview": "",
                    }
                return {
                    "success": False,
                    "message": f"模型返回错误 (HTTP 400): {error_text[:300]}",
                    "response_preview": "",
                }

            if resp.status_code != 200:
                return {
                    "success": False,
                    "message": f"模型返回错误 (HTTP {resp.status_code}): {resp.text[:300]}",
                    "response_preview": "",
                }

            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return {
                "success": True,
                "message": "模型连接成功",
                "response_preview": content[:200],
            }

    except httpx.ConnectError:
        return {
            "success": False,
            "message": f"无法连接到 {settings.llm_api_url}，请检查 API URL 是否正确",
            "response_preview": "",
        }
    except httpx.ReadTimeout:
        return {
            "success": False,
            "message": f"连接超时（{timeout}秒），请检查模型服务是否正常运行",
            "response_preview": "",
        }
    except httpx.ConnectTimeout:
        return {
            "success": False,
            "message": f"连接超时（{timeout}秒），请检查 API URL 和网络连通性",
            "response_preview": "",
        }
    except Exception as exc:
        logger.exception("Connection test error")
        return {
            "success": False,
            "message": f"连接测试异常: {exc}",
            "response_preview": "",
        }
