import inspect
import os
from typing import Any

from judge_agent.config import Config


def build_chat_model(streaming: bool = True, temperature: float = 0.1) -> Any:
    try:
        from langchain_openai import ChatOpenAI
    except Exception as exc:
        raise RuntimeError("langchain-openai is required to build the LangChain chat model.") from exc

    # Ensure OpenAI client picks up the correct base URL/key even if kwargs are ignored.
    if Config.API_URL:
        os.environ.setdefault("OPENAI_BASE_URL", Config.API_URL)
        os.environ.setdefault("OPENAI_API_BASE", Config.API_URL)
    if Config.API_KEY:
        os.environ.setdefault("OPENAI_API_KEY", Config.API_KEY)

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
