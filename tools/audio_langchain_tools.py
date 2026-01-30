import os
import json
import logging
from typing import Any, Dict, List, Optional, Annotated

try:
    from langchain.tools import tool, InjectedState, InjectedToolCallId
except Exception:
    from langchain_core.tools import tool  # type: ignore
    try:
        from langchain_core.tools import InjectedToolCallId  # type: ignore
    except Exception:  # pragma: no cover
        from langchain.tools import InjectedToolCallId  # type: ignore
    try:
        from langgraph.prebuilt import InjectedState  # type: ignore
    except Exception:  # pragma: no cover
        from langchain.tools import InjectedState  # type: ignore

from judge_agent.utils.file_utils import FileUtils
from judge_agent.tools.audio_tools import (
    asr_transcribe,
    correct_text as audio_correct_text,
    violation_check as audio_violation_check,
    slice_evidence as audio_slice_evidence,
)
from judge_agent.tools.tool_utils import command_with_update

logger = logging.getLogger("judge_agent.tools")


@tool("audio_asr_transcribe")
async def audio_asr_transcribe(
    file_path: str,
    state: Annotated[Dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Dict[str, Any]:
    """音频转写（ASR）。"""
    file_type = state.get("file_type") if isinstance(state, dict) else None
    if not file_type:
        file_type = FileUtils.detect_file_type(file_path)

    if file_type not in {"audio", "video"}:
        payload = {
            "status": "skipped",
            "reason": "not_audio_or_video",
            "file_type": file_type,
            "file_path": file_path,
        }
        logger.info("audio_asr_skip:\n%s", json.dumps(payload, ensure_ascii=False, indent=2))
        return payload

    output = await asr_transcribe(file_path)
    update = {
        "audio_raw_text": output.get("raw_text", ""),
        "audio_segments": output.get("segments", []),
    }
    return command_with_update(tool_call_id, output, update)


@tool("audio_correct_text")
async def audio_correct_text_tool(
    state: Annotated[Dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    raw_text: str = "",
) -> Dict[str, Any]:
    """音频文本纠错。"""
    if not raw_text and isinstance(state, dict):
        raw_text = state.get("audio_raw_text") or ""

    output = await audio_correct_text(raw_text)
    update = {"audio_corrected_text": output.get("corrected_text", "")}
    return command_with_update(tool_call_id, output, update)


@tool("audio_violation_check")
async def audio_violation_check_tool(
    state: Annotated[Dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    segments: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """音频违规检测（不切片）。"""
    if not segments and isinstance(state, dict):
        segments = state.get("audio_segments") or []

    output = await audio_violation_check(segments or [])
    update = {"audio_violation_report": output.get("violation_check")}
    return command_with_update(tool_call_id, output, update)


@tool("audio_slice_evidence")
async def audio_slice_evidence_tool(
    file_path: str,
    state: Annotated[Dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    time_anchors: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """根据违规时间轴切片并上传。"""
    if not os.path.exists(file_path):
        return {"error": f"文件不存在: {file_path}"}
    if not time_anchors and isinstance(state, dict):
        report = state.get("audio_violation_report") or {}
        time_anchors = report.get("segments") or []

    output = await audio_slice_evidence(file_path, time_anchors or [])
    update = {"audio_violation_report": output.get("violation_check")}
    return command_with_update(tool_call_id, output, update)
