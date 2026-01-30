# judge_agent/tools/audio_tools.py

import os
import uuid
import asyncio
import json
import logging
import imageio_ffmpeg as ffmpeg
from typing import Dict, List, Any, Optional

from judge_agent.config import Config
from judge_agent.engines.whisper_engine import WhisperEngine
from judge_agent.engines.langchain_llm import async_chat_response, async_get_json_response
from judge_agent.engines.minio_engine import MinioEngine
from judge_agent.prompts.templates import PromptTemplates
from judge_agent.utils.json_utils import JSONUtils

logger = logging.getLogger("judge_agent.audio_tools")


async def asr_transcribe(file_path: str) -> Dict[str, Any]:
    """ASR 转写，返回原始文本与分段时间轴。"""
    if not os.path.exists(file_path):
        return {"status": "error", "error": "文件不存在"}

    loop = asyncio.get_running_loop()
    try:
        raw_text, segments = await loop.run_in_executor(
            None, WhisperEngine.transcribe, file_path
        )
    except Exception as e:
        logger.error("asr_error:\n%s", str(e))
        return {"status": "error", "error": f"转写失败: {e}"}

    payload = {"status": "success", "raw_text": raw_text, "segments": segments}
    logger.info("asr_result:\n%s", json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


async def correct_text(raw_text: str) -> Dict[str, Any]:
    """文本纠错，返回更通顺的文本。"""
    if not raw_text:
        return {"status": "skipped", "corrected_text": ""}

    corrected_text = raw_text
    try:
        candidate = await async_chat_response(
            PromptTemplates.audio_correction_prompt(raw_text), temperature=0.3
        )
        if candidate:
            corrected_text = candidate
    except Exception as e:
        logger.warning("audio_correction_failed:\n%s", str(e))

    payload = {"status": "success", "corrected_text": corrected_text}
    logger.info("asr_corrected_text:\n%s", json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


async def violation_check(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """违规检测，仅返回违规时间轴。"""
    if not segments:
        return {"status": "success", "violation_check": {"is_violation": False, "segments": []}}

    formatted_text = WhisperEngine.format_segments_for_llm(segments)
    judge_prompt = PromptTemplates.text_review_and_correct_json_template(formatted_text)

    report = {"is_violation": False, "segments": []}
    try:
        violation_data = await async_get_json_response([
            {"role": "user", "content": judge_prompt}
        ])
        if violation_data and violation_data.get("is_violation"):
            report["is_violation"] = True
            merged_anchors = JSONUtils.merge_intervals(violation_data.get("time_anchors", []))
            report["segments"] = merged_anchors
    except Exception as e:
        logger.error("audio_violation_check_failed:\n%s", str(e))

    payload = {"status": "success", "violation_check": report}
    logger.info("audio_violation_report:\n%s", json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


async def slice_evidence(file_path: str, time_anchors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """对违规时间轴切片，并上传 MinIO，返回带 clip_url 的结果。"""
    if not time_anchors:
        return {"status": "skipped", "violation_check": {"is_violation": False, "segments": []}}

    clip_tasks = []
    for anchor in time_anchors:
        clip_tasks.append(_slice_media(file_path, anchor["start"], anchor["end"]))

    clip_filenames = await asyncio.gather(*clip_tasks)
    merged_anchors = []

    for anchor, fname in zip(time_anchors, clip_filenames):
        item = dict(anchor)
        if fname:
            clip_path = os.path.join(Config.FIXED_TEMP_DIR, fname)
            try:
                minio_url = MinioEngine.upload_file(clip_path)
                item["clip_url"] = minio_url
                logger.info(
                    "audio_clip_uploaded:\n%s",
                    json.dumps({"clip_url": minio_url}, ensure_ascii=False, indent=2),
                )
            except Exception as e:
                logger.warning("audio_clip_upload_failed:\n%s", str(e))
                item["clip_url"] = f"/static_temp/{fname}"
        merged_anchors.append(item)

    payload = {"status": "success", "violation_check": {"is_violation": True, "segments": merged_anchors}}
    logger.info("audio_clip_report:\n%s", json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


async def _slice_media(input_path: str, start: float, end: float) -> str:
    """调用 ffmpeg 切割媒体文件（自动识别视频或音频）"""
    try:
        if not os.path.exists(Config.FIXED_TEMP_DIR):
            os.makedirs(Config.FIXED_TEMP_DIR)

        input_ext = os.path.splitext(input_path)[1].lower()
        audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma'}
        is_audio_mode = input_ext in audio_extensions

        if is_audio_mode:
            output_ext = ".mp3"
            encoding_args = ['-vn', '-c:a', 'libmp3lame', '-q:a', '2']
        else:
            output_ext = ".mp4"
            encoding_args = ['-c:v', 'libx264', '-preset', 'ultrafast', '-c:a', 'aac', '-strict', 'experimental']

        output_filename = f"evidence_{uuid.uuid4().hex[:8]}{output_ext}"
        output_path = os.path.join(Config.FIXED_TEMP_DIR, output_filename)

        duration = max(end - start, 1.0)
        cmd = [
            ffmpeg.get_ffmpeg_exe(), '-y',
            '-ss', str(start),
            '-t', str(duration),
            '-i', input_path,
        ] + encoding_args + [output_path]

        process = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            return output_filename

        logger.error(
            "slice_failed:\n%s",
            json.dumps(
                {"stderr": stderr.decode(), "start": start, "end": end},
                ensure_ascii=False,
                indent=2,
            ),
        )
        return ""
    except Exception as e:
        logger.error("slice_exception:\n%s", str(e))
        return ""
