import inspect
from typing import Any, Dict, Iterable, Optional

from judge_agent.agent.langchain_state import GraphAgentState
from judge_agent.agent.prompts import SYSTEM_PROMPT_LC


def _load_create_agent():
    try:
        from langchain.agents import create_agent
    except Exception as exc:
        raise RuntimeError("LangChain v1 create_agent is required. Install langchain>=1.0.0.") from exc
    return create_agent


def build_prompt(system_prompt: str = SYSTEM_PROMPT_LC):
    try:
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    except Exception as exc:
        raise RuntimeError("langchain-core is required to build prompts.") from exc

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("messages"),
        ]
    )


def build_agent(
    *,
    model: Any,
    tools: Iterable[Any],
    system_prompt: str = SYSTEM_PROMPT_LC,
    checkpointer: Optional[Any] = None,
    store: Optional[Any] = None,
    middlewares: Optional[Iterable[Any]] = None,
    response_format: Optional[Any] = None,
):
    create_agent = _load_create_agent()
    prompt = build_prompt(system_prompt)

    kwargs: Dict[str, Any] = {
        "model": model,
        "tools": list(tools),
        "prompt": prompt,
        "state_schema": GraphAgentState,
        "checkpointer": checkpointer,
        "store": store,
        "response_format": response_format,
    }

    sig = inspect.signature(create_agent)
    supported = set(sig.parameters.keys())

    if middlewares:
        if "middleware" in supported:
            kwargs["middleware"] = list(middlewares)
        elif "middlewares" in supported:
            kwargs["middlewares"] = list(middlewares)

    filtered = {k: v for k, v in kwargs.items() if k in supported and v is not None}

    try:
        return create_agent(**filtered)
    except TypeError as exc:
        raise TypeError("create_agent signature mismatch. Check LangChain version and params.") from exc


def build_initial_state(
    *,
    file_path: str,
    file_type: str,
    s3_url: Optional[str] = None,
    messages: Optional[list] = None,
    remaining_steps: int = 10,
) -> GraphAgentState:
    return {
        "messages": messages or [],
        "remaining_steps": remaining_steps,
        "file_path": file_path,
        "file_type": file_type,
        "s3_url": s3_url,
        "evidences": [],
        "client_events": [],
        "visual_frames": [],
        "visual_frames_file_path": None,
        "visual_face_findings": [],
        "visual_behavior_findings": [],
        "visual_ocr_findings": [],
        "visual_marked_images": [],
        "visual_marked_images_file_path": None,
    }
