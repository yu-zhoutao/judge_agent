import os
import json
import uuid
import asyncio
import logging
from typing import Any, Dict, List, Optional, Annotated, Tuple

import cv2

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
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from judge_agent.config import Config
from judge_agent.engines.face_engine import FaceEngine
from judge_agent.engines.minio_engine import MinioEngine
from judge_agent.engines.ocr_engine import OcrEngine
from judge_agent.engines.yolo_engine import YoloEngine
from judge_agent.engines.langchain_llm import async_get_json_response, build_visual_messages
from judge_agent.prompts.templates import PromptTemplates
from judge_agent.tools.audio_tools import AudioTranscribeTool
from judge_agent.tools.search_tools import WebSearchTool
from judge_agent.utils.image_utils import ImageUtils
from judge_agent.utils.file_utils import FileUtils

logger = logging.getLogger("judge_agent.tools")

_FRAME_CACHE: Dict[str, List[Dict[str, Any]]] = {}
_FRAME_LOCKS: Dict[str, asyncio.Lock] = {}


def _get_frame_lock(file_path: str) -> asyncio.Lock:
    lock = _FRAME_LOCKS.get(file_path)
    if lock is None:
        lock = asyncio.Lock()
        _FRAME_LOCKS[file_path] = lock
    return lock


async def _upload_frames_concurrently(frames_data: List[Dict[str, Any]]) -> None:
    tasks = []

    async def _save_and_upload(frame_item: Dict[str, Any]):
        if frame_item.get("minio_url"):
            return
        try:
            local_path = frame_item.get("local_path")
            if not local_path or not os.path.exists(local_path):
                img = frame_item.get("img")
                if img is None:
                    return
                local_path = _save_frame_to_temp(img)
                if not local_path:
                    return
                frame_item["local_path"] = local_path

            url = await asyncio.to_thread(MinioEngine.upload_file, local_path)
            frame_item["minio_url"] = url
        except Exception as exc:
            print(f"frame upload failed: {exc}")
            frame_item["minio_url"] = None

    for item in frames_data:
        tasks.append(_save_and_upload(item))

    if tasks:
        await asyncio.gather(*tasks)

def _ensure_cache_dir() -> None:
    if not os.path.exists(Config.FIXED_TEMP_DIR):
        os.makedirs(Config.FIXED_TEMP_DIR)


def _save_frame_to_temp(img: Any) -> str:
    _ensure_cache_dir()
    temp_filename = f"frame_cache_{uuid.uuid4().hex}.jpg"
    temp_filepath = os.path.join(Config.FIXED_TEMP_DIR, temp_filename)
    ok = cv2.imwrite(temp_filepath, img)
    if not ok:
        return ""
    return temp_filepath


def _normalize_frames(frames_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in frames_data:
        local_path = item.get("local_path")
        if not local_path:
            img = item.get("img")
            if img is None:
                continue
            local_path = _save_frame_to_temp(img)
            if not local_path:
                continue
        normalized.append(
            {
                "index": item.get("index"),
                "local_path": local_path,
                "minio_url": item.get("minio_url"),
            }
        )
    return normalized


def _serialize_tool_output(output: Dict[str, Any]) -> str:
    return json.dumps(output, ensure_ascii=False)


def _command_with_update(
    tool_call_id: str,
    output: Dict[str, Any],
    update: Dict[str, Any],
) -> Command:
    update_payload = dict(update)
    update_payload["messages"] = [
        ToolMessage(content=_serialize_tool_output(output), tool_call_id=tool_call_id)
    ]
    return Command(update=update_payload)


def _build_bbox_map(findings: List[Dict[str, Any]], key: str) -> Dict[int, List[Any]]:
    bbox_map: Dict[int, List[Any]] = {}
    for item in findings or []:
        frame_index = item.get("frame_index")
        if frame_index is None:
            continue
        bboxes = item.get(key) or []
        if not bboxes:
            continue
        bbox_map.setdefault(frame_index, []).extend(bboxes)
    return bbox_map


async def _get_or_prepare_frames(
    file_path: str,
    state: Optional[Dict[str, Any]],
) -> Tuple[Optional[List[Dict[str, Any]]], bool]:
    cached_frames = None
    cached_path = None
    if isinstance(state, dict):
        cached_frames = state.get("visual_frames")
        cached_path = state.get("visual_frames_file_path")

    if cached_frames and cached_path == file_path:
        valid = True
        for frame_item in cached_frames:
            local_path = frame_item.get("local_path")
            if not local_path or not os.path.exists(local_path):
                valid = False
                break
        if valid:
            missing_upload = [f for f in cached_frames if not f.get("minio_url")]
            if missing_upload:
                await _upload_frames_concurrently(missing_upload)
                return cached_frames, True
            return cached_frames, False

    cached_global = _FRAME_CACHE.get(file_path)
    if cached_global:
        valid = True
        for frame_item in cached_global:
            local_path = frame_item.get("local_path")
            if not local_path or not os.path.exists(local_path):
                valid = False
                break
        if valid:
            missing_upload = [f for f in cached_global if not f.get("minio_url")]
            if missing_upload:
                await _upload_frames_concurrently(missing_upload)
            return cached_global, True

    lock = _get_frame_lock(file_path)
    async with lock:
        cached_global = _FRAME_CACHE.get(file_path)
        if cached_global:
            valid = True
            for frame_item in cached_global:
                local_path = frame_item.get("local_path")
                if not local_path or not os.path.exists(local_path):
                    valid = False
                    break
            if valid:
                missing_upload = [f for f in cached_global if not f.get("minio_url")]
                if missing_upload:
                    await _upload_frames_concurrently(missing_upload)
                return cached_global, True

        frames_data = ImageUtils.extract_frames(file_path)
        if not frames_data:
            return None, False
        normalized = _normalize_frames(frames_data)
        if not normalized:
            return None, False
        await _upload_frames_concurrently(normalized)
        _FRAME_CACHE[file_path] = normalized
        return normalized, True


@tool("visual_prepare_frames")
async def visual_prepare_frames(
    file_path: str,
    state: Annotated[Dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Dict[str, Any]:
    """抽帧并缓存本地路径与 MinIO URL，供视觉工具复用。"""
    if not os.path.exists(file_path):
        return {"error": f"文件不存在: {file_path}"}

    frames_data, updated = await _get_or_prepare_frames(file_path, state)
    if not frames_data:
        return {"error": "无法提取图像帧"}

    minio_ready = sum(1 for f in frames_data if f.get("minio_url"))
    output = {
        "status": "success",
        "frame_count": len(frames_data),
        "minio_ready": minio_ready,
    }

    if updated:
        return _command_with_update(
            tool_call_id,
            output,
            {
                "visual_frames": frames_data,
                "visual_frames_file_path": file_path,
            },
        )

    return output


@tool("visual_render_marks")
async def visual_render_marks(
    file_path: str,
    state: Annotated[Dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Any:
    """生成标记后的帧图并上传，返回可预览图片 URL 列表。"""
    if not os.path.exists(file_path):
        return {"error": f"文件不存在: {file_path}"}

    cached_images = None
    cached_path = None
    if isinstance(state, dict):
        cached_images = state.get("visual_marked_images")
        cached_path = state.get("visual_marked_images_file_path")

    if cached_images and cached_path == file_path:
        output = {"status": "success", "preview_images": cached_images}
        return _command_with_update(
            tool_call_id,
            output,
            {
                "visual_marked_images": cached_images,
                "visual_marked_images_file_path": file_path,
            },
        )

    frames_data, updated = await _get_or_prepare_frames(file_path, state)
    if not frames_data:
        return {"error": "无法提取图像帧"}

    face_findings = []
    behavior_findings = []
    ocr_findings = []
    if isinstance(state, dict):
        face_findings = state.get("visual_face_findings") or []
        behavior_findings = state.get("visual_behavior_findings") or []
        ocr_findings = state.get("visual_ocr_findings") or []

    face_map: Dict[int, List[Any]] = {}
    for item in face_findings:
        frame_index = item.get("frame_index")
        if frame_index is None:
            continue
        for person in item.get("persons") or []:
            bbox = person.get("bbox")
            if bbox:
                face_map.setdefault(frame_index, []).append({"bbox": bbox})

    behavior_map = _build_bbox_map(behavior_findings, "bboxes")

    ocr_map: Dict[int, List[Any]] = {}
    for item in ocr_findings:
        frame_index = item.get("frame_index")
        if frame_index is None:
            continue
        boxes = item.get("boxes") or []
        if boxes:
            ocr_map.setdefault(frame_index, []).extend(
                [{"box": b} for b in boxes if b]
            )

    preview_images: List[str] = []

    for frame_item in frames_data:
        local_path = frame_item.get("local_path")
        if not local_path or not os.path.exists(local_path):
            continue

        target_img = cv2.imread(local_path)
        if target_img is None:
            continue

        frame_index = frame_item.get("index")

        detections = []
        detections.extend(face_map.get(frame_index, []))
        detections.extend([{"bbox": b} for b in behavior_map.get(frame_index, [])])

        if detections:
            target_img = ImageUtils.draw_detections(target_img, detections, color=(0, 0, 255), thickness=3)

        ocr_boxes = ocr_map.get(frame_index, [])
        if ocr_boxes:
            target_img = ImageUtils.draw_ocr_boxes(target_img, ocr_boxes, color=(0, 255, 0))

        temp_filename = f"frame_marked_{frame_index}_{uuid.uuid4().hex}.jpg"
        temp_filepath = os.path.join(Config.FIXED_TEMP_DIR, temp_filename)
        try:
            cv2.imwrite(temp_filepath, target_img)
            minio_url = MinioEngine.upload_file(temp_filepath)
            logger.info(
                "marked_image_uploaded:\n%s",
                json.dumps({"minio_url": minio_url}, ensure_ascii=False, indent=2),
            )
            preview_images.append('/' + minio_url.split('/', 3)[-1])
        except Exception as exc:
            print(f"marked frame upload failed: {exc}")
            final_b64 = ImageUtils.encode_to_base64(target_img)
            if final_b64:
                preview_images.append(final_b64)
        finally:
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except Exception:
                    pass

    output = {"status": "success", "preview_images": preview_images}
    update = {
        "visual_marked_images": preview_images,
        "visual_marked_images_file_path": file_path,
    }
    if updated:
        update["visual_frames"] = frames_data
        update["visual_frames_file_path"] = file_path

    return _command_with_update(tool_call_id, output, update)


@tool("visual_face_check")
async def visual_face_check(
    file_path: str,
    state: Annotated[Dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Any:
    """视觉人脸识别。识别黑名单人物并返回人物信息。"""
    if not os.path.exists(file_path):
        return {"error": f"文件不存在: {file_path}"}

    frames_data, updated = await _get_or_prepare_frames(file_path, state)
    if not frames_data:
        return {"error": "无法提取图像帧"}

    detected_persons = set()
    findings: List[Dict[str, Any]] = []

    async def _detect_in_frame(frame_item: Dict[str, Any]):
        minio_url = frame_item.get("minio_url")
        if not minio_url:
            return None
        try:
            return await asyncio.to_thread(FaceEngine.identify_face, minio_url)
        except Exception as exc:
            print(f"face identify failed: {exc}")
            return None

    tasks = [
        _detect_in_frame(frame_item)
        for frame_item in frames_data
    ]

    results = await asyncio.gather(*tasks)

    for frame_item, persons in zip(frames_data, results):
        if not persons:
            continue
        frame_findings = []
        for person in persons:
            name = person.get("name", "未知")
            tag = person.get("tag", "")
            label = f"{name} ({tag})" if tag else name
            detected_persons.add(label)
            frame_findings.append({
                "name": name,
                "tag": tag,
                "similarity": person.get("similarity", 0),
                "bbox": person.get("bbox", []),
            })
        findings.append({
            "frame_index": frame_item.get("index"),
            "minio_url": frame_item.get("minio_url"),
            "persons": frame_findings,
        })

    output = {
        "status": "success",
        "detected_persons": list(detected_persons),
        "face_findings": findings,
    }
    logger.info("face_findings:\n%s", json.dumps(output, ensure_ascii=False, indent=2))

    update = {
        "visual_face_findings": findings,
    }
    if updated:
        update["visual_frames"] = frames_data
        update["visual_frames_file_path"] = file_path

    return _command_with_update(tool_call_id, output, update)


@tool("visual_behavior_check")
async def visual_behavior_check(
    file_path: str,
    state: Annotated[Dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Any:
    """视觉非人脸行为与标识检测（旗帜、暴力、低俗等）。"""
    if not os.path.exists(file_path):
        return {"error": f"文件不存在: {file_path}"}

    frames_data, updated = await _get_or_prepare_frames(file_path, state)
    if not frames_data:
        return {"error": "无法提取图像帧"}

    bad_type = "违规行为、敏感标识、阴暗内容、同性低俗、擦边、卖腐、性暗示、国民党党旗、台独、台湾旗帜、丑化嘲讽领导人，歧视中国人"

    visual_risks: List[str] = []
    findings: List[Dict[str, Any]] = []

    for frame_item in frames_data:
        local_path = frame_item.get("local_path")
        if not local_path or not os.path.exists(local_path):
            continue
        target_img = cv2.imread(local_path)
        if target_img is None:
            continue
        raw_detections = YoloEngine.detect(target_img, conf=0.3)
        merged_candidates = ImageUtils.merge_overlapping_boxes(raw_detections, target_img.shape)
        if not merged_candidates:
            continue

        slices_b64 = [
            ImageUtils.encode_to_base64(
                ImageUtils.get_single_object_crop(target_img, d["bbox"])
            )
            for d in merged_candidates
        ]

        behavior_prompt = PromptTemplates.get_image_prompt(bad_type)
        msgs = build_visual_messages(behavior_prompt, slices_b64)
        behavior_res = await async_get_json_response(msgs)

        if behavior_res and behavior_res.get("image"):
            valid_ids = [i for i in behavior_res["image"] if 0 < i <= len(merged_candidates)]
            if valid_ids:
                visual_risks.append(
                    f"frame {frame_item.get('index')} objects {valid_ids} flagged"
                )
                bboxes = [merged_candidates[i - 1]["bbox"] for i in valid_ids if 0 < i <= len(merged_candidates)]
                findings.append({
                    "frame_index": frame_item.get("index"),
                    "object_ids": valid_ids,
                    "bboxes": bboxes,
                })

    output = {
        "status": "success",
        "visual_risks": list(set(visual_risks)),
        "visual_findings": findings,
    }
    logger.info("behavior_findings:\n%s", json.dumps(output, ensure_ascii=False, indent=2))

    update = {
        "visual_behavior_findings": findings,
    }
    if updated:
        update["visual_frames"] = frames_data
        update["visual_frames_file_path"] = file_path

    return _command_with_update(tool_call_id, output, update)


@tool("visual_ocr_check")
async def visual_ocr_check(
    file_path: str,
    state: Annotated[Dict[str, Any], InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Any:
    """视觉 OCR 文本检测与违规判定。"""
    if not os.path.exists(file_path):
        return {"error": f"文件不存在: {file_path}"}

    frames_data, updated = await _get_or_prepare_frames(file_path, state)
    if not frames_data:
        return {"error": "无法提取图像帧"}

    ocr_risks: List[str] = []
    findings: List[Dict[str, Any]] = []

    for frame_item in frames_data:
        local_path = frame_item.get("local_path")
        if not local_path or not os.path.exists(local_path):
            continue
        target_img = cv2.imread(local_path)
        if target_img is None:
            continue
        ocr_results = OcrEngine.detect_text(target_img)
        if not ocr_results:
            continue

        text_A = " ".join(o["text"] for o in ocr_results)
        text_B = {o["id"]: o["text"] for o in ocr_results}
        text_match = await async_get_json_response([
            {"role": "user", "content": PromptTemplates.ocr_judge_prompt(text_A, text_B)}
        ])

        if text_match and text_match.get("id"):
            bad_ocr = [o for o in ocr_results if o["id"] in text_match["id"]]
            bad_texts = [o["text"] for o in bad_ocr]
            if bad_texts:
                ocr_risks.extend(bad_texts)
                findings.append({
                    "frame_index": frame_item.get("index"),
                    "texts": bad_texts,
                    "boxes": [o.get("box") for o in bad_ocr if o.get("box")],
                })

    output = {
        "status": "success",
        "ocr_risks": list(set(ocr_risks)),
        "ocr_findings": findings,
    }
    logger.info("ocr_findings:\n%s", json.dumps(output, ensure_ascii=False, indent=2))

    update = {
        "visual_ocr_findings": findings,
    }
    if updated:
        update["visual_frames"] = frames_data
        update["visual_frames_file_path"] = file_path

    return _command_with_update(tool_call_id, output, update)


_audio_tool = AudioTranscribeTool()
_search_tool = WebSearchTool()


@tool("audio_transcribe")
async def audio_transcribe(
    file_path: str,
    state: Annotated[Dict[str, Any], InjectedState],
) -> Dict[str, Any]:
    """音频转写与违规检测。"""
    file_type = None
    if isinstance(state, dict):
        file_type = state.get("file_type")
    if not file_type:
        file_type = FileUtils.detect_file_type(file_path)

    if file_type not in {"audio", "video"}:
        payload = {
            "status": "skipped",
            "reason": "not_audio_or_video",
            "file_type": file_type,
            "file_path": file_path,
        }
        logger.info("audio_transcribe_skip:\n%s", json.dumps(payload, ensure_ascii=False, indent=2))
        return payload

    return await _audio_tool.run(file_path=file_path)


@tool("web_search")
async def web_search(query: str = "", image_path: str = "", image_url: str = "") -> Dict[str, Any]:
    """网络以图搜图。"""
    return await _search_tool.run(query=query, image_path=image_path, image_url=image_url)
