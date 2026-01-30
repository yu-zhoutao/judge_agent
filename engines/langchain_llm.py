import json
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage

import logging

from judge_agent.config import Config
from judge_agent.engines.langchain_model import build_chat_model
from judge_agent.utils.json_utils import JSONUtils

logger = logging.getLogger("judge_agent.langchain_llm")


def _to_messages(messages: Sequence[Dict[str, Any]]) -> List[BaseMessage]:
    converted: List[BaseMessage] = []
    for msg in messages:
        if isinstance(msg, BaseMessage):
            converted.append(msg)
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            converted.append(SystemMessage(content=content))
        elif role == "assistant":
            converted.append(AIMessage(content=content))
        else:
            converted.append(HumanMessage(content=content))
    return converted


def build_visual_messages(text: str, base64_images: Iterable[str]) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    for b64 in base64_images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    return [{"role": "user", "content": content}]


@lru_cache(maxsize=8)
def _get_model_cached(streaming: bool, temperature: float, response_format: Optional[str]) -> Any:
    model = build_chat_model(streaming=streaming, temperature=temperature)
    if response_format == "json":
        try:
            model = model.bind(response_format={"type": "json_object"})
        except Exception:
            pass
    return model


async def async_chat_response(
    messages: Union[str, Sequence[Dict[str, Any]]],
    *,
    temperature: float = 0.3,
) -> str:
    if isinstance(messages, str):
        msg_list: Sequence[Dict[str, Any]] = [{"role": "user", "content": messages}]
    else:
        msg_list = messages

    model = _get_model_cached(False, temperature, None)
    response = await model.ainvoke(_to_messages(msg_list))
    content = response.content or ""
    logger.info("llm_text_response", extra={"content": content})
    return content


async def async_get_json_response(
    messages: Union[str, Sequence[Dict[str, Any]]],
    *,
    temperature: float = 0.1,
) -> Optional[Dict[str, Any]]:
    if isinstance(messages, str):
        msg_list: Sequence[Dict[str, Any]] = [{"role": "user", "content": messages}]
    else:
        msg_list = messages

    response_format = "json" if "json" in Config.MODEL_NAME.lower() else None
    model = _get_model_cached(False, temperature, response_format)
    response = await model.ainvoke(_to_messages(msg_list))
    content = response.content or ""
    logger.info("llm_json_raw_response", extra={"content": content})

    if not content:
        return None

    if isinstance(content, dict):
        return content

    if isinstance(content, list):
        # If content blocks, try to extract text
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        content = "".join(parts)

    result = JSONUtils.safe_json_loads(content)
    if isinstance(result, dict):
        return result

    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return None
