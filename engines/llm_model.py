import inspect
import json
import logging
import os
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage

from judge_agent.config import Config
from judge_agent.utils.json_utils import JSONUtils

logger = logging.getLogger("judge_agent.llm_model")


def build_chat_model(streaming: bool = True, temperature: float = 0.1) -> Any:
    try:
        from langchain_openai import ChatOpenAI
    except Exception as exc:
        raise RuntimeError("langchain-openai is required to build the LangChain chat model.") from exc

    # Ensure OpenAI client picks up the correct base URL/key even if kwargs are ignored.
    if Config.API_URL:
        os.environ["OPENAI_BASE_URL"] = Config.API_URL
        os.environ["OPENAI_API_BASE"] = Config.API_URL
    if Config.API_KEY:
        os.environ["OPENAI_API_KEY"] = Config.API_KEY

    sig = inspect.signature(ChatOpenAI)
    params = sig.parameters

    kwargs = {"model": Config.MODEL_NAME}

    if "openai_api_key" in params:
        kwargs["openai_api_key"] = Config.API_KEY
    elif "api_key" in params:
        kwargs["api_key"] = Config.API_KEY

    if "openai_api_base" in params:
        kwargs["openai_api_base"] = Config.API_URL
    elif "base_url" in params:
        kwargs["base_url"] = Config.API_URL
    elif "api_base" in params:
        kwargs["api_base"] = Config.API_URL

    if "streaming" in params:
        kwargs["streaming"] = streaming

    if "temperature" in params:
        kwargs["temperature"] = temperature

    return ChatOpenAI(**kwargs)


def _mask_image_url(url: str) -> str:
    if url.startswith("data:image"):
        return f"data:image/<base64>({len(url)} chars)"
    return url


def _safe_messages_for_log(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    safe: List[Dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, list):
            safe_content: List[Any] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    image_url = dict(item.get("image_url") or {})
                    url = image_url.get("url")
                    if isinstance(url, str):
                        image_url["url"] = _mask_image_url(url)
                    safe_content.append({"type": "image_url", "image_url": image_url})
                else:
                    safe_content.append(item)
            content = safe_content
        safe.append({"role": role, "content": content})
    return safe


def _pretty_json(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return str(data)


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

    safe_msgs = _safe_messages_for_log(msg_list)
    logger.info("llm_request_messages:\n%s", _pretty_json(safe_msgs))

    model = _get_model_cached(False, temperature, None)
    response = await model.ainvoke(_to_messages(msg_list))
    content = response.content or ""
    logger.info("llm_text_response:\n%s", _pretty_json(content))
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

    safe_msgs = _safe_messages_for_log(msg_list)
    logger.info("llm_request_messages:\n%s", _pretty_json(safe_msgs))

    response_format = "json" if "json" in Config.MODEL_NAME.lower() else None
    model = _get_model_cached(False, temperature, response_format)
    response = await model.ainvoke(_to_messages(msg_list))
    content = response.content or ""
    logger.info("llm_json_raw_response:\n%s", _pretty_json(content))

    if not content:
        return None

    if isinstance(content, dict):
        return content

    if isinstance(content, list):
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
