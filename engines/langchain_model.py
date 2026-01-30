import inspect
from typing import Any

from judge_agent.config import Config


def build_chat_model(streaming: bool = True, temperature: float = 0.1) -> Any:
    try:
        from langchain_openai import ChatOpenAI
    except Exception as exc:
        raise RuntimeError("langchain-openai is required to build the LangChain chat model.") from exc

    sig = inspect.signature(ChatOpenAI)
    params = sig.parameters

    kwargs = {"model": Config.MODEL_NAME}

    if "openai_api_key" in params:
        kwargs["openai_api_key"] = Config.VLLM_API_KEY
    elif "api_key" in params:
        kwargs["api_key"] = Config.VLLM_API_KEY

    if "openai_api_base" in params:
        kwargs["openai_api_base"] = Config.VLLM_API_URL
    elif "base_url" in params:
        kwargs["base_url"] = Config.VLLM_API_URL
    elif "api_base" in params:
        kwargs["api_base"] = Config.VLLM_API_URL

    if "streaming" in params:
        kwargs["streaming"] = streaming

    if "temperature" in params:
        kwargs["temperature"] = temperature

    return ChatOpenAI(**kwargs)
