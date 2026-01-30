import json
from typing import Any, Dict

from langchain_core.messages import ToolMessage
from langgraph.types import Command


def serialize_tool_output(output: Dict[str, Any]) -> str:
    return json.dumps(output, ensure_ascii=False)


def command_with_update(
    tool_call_id: str,
    output: Dict[str, Any],
    update: Dict[str, Any],
) -> Command:
    update_payload = dict(update)
    update_payload["messages"] = [
        ToolMessage(content=serialize_tool_output(output), tool_call_id=tool_call_id)
    ]
    return Command(update=update_payload)
