from typing import Any, Dict, List, Optional, Literal, TypedDict, Annotated

from langgraph.graph.message import add_messages

from judge_agent.schemas import Evidence


def _append_list(existing: List[Any], new: Optional[List[Any]]) -> List[Any]:
    if not new:
        return existing
    if isinstance(new, list):
        return existing + new
    return existing + [new]


def _replace_value(existing: Any, new: Any) -> Any:
    if new is None:
        return existing
    return new


class GraphAgentState(TypedDict):
    messages: Annotated[List[Dict[str, Any]], add_messages]
    remaining_steps: int
    file_path: str
    file_type: Literal["video", "image", "audio"]
    s3_url: Optional[str]
    evidences: Annotated[List[Evidence], _append_list]
    client_events: Annotated[List[Dict[str, Any]], _append_list]
    visual_frames: Annotated[List[Dict[str, Any]], _replace_value]
    visual_frames_file_path: Annotated[Optional[str], _replace_value]
    visual_face_findings: Annotated[List[Dict[str, Any]], _replace_value]
    visual_behavior_findings: Annotated[List[Dict[str, Any]], _replace_value]
    visual_ocr_findings: Annotated[List[Dict[str, Any]], _replace_value]
    visual_marked_images: Annotated[List[str], _replace_value]
    visual_marked_images_file_path: Annotated[Optional[str], _replace_value]
