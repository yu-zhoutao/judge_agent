# judge_agent/tools/visual_tools.py

import os
import cv2
import uuid
import json
import asyncio
import time
import numpy as np
from typing import Dict, List, Any
from judge_agent.config import Config
from judge_agent.tools.base import BaseTool
from judge_agent.engines.yolo_engine import YoloEngine
from judge_agent.engines.face_engine import FaceEngine
from judge_agent.engines.ocr_engine import OcrEngine
from judge_agent.engines.minio_engine import MinioEngine
from judge_agent.engines.llm_client import LLMClient
from judge_agent.utils.image_utils import ImageUtils
from judge_agent.prompts.templates import PromptTemplates

class VisualScanTool(BaseTool):
    name = "visual_scan"
    description = "视觉扫描工具。用于分析视频或图片，识别黑名单人物、OCR违规文字及敏感行为（如旗帜、暴力）。会返回带有红色违规标记的图片证据。"

    async def _upload_frames_concurrently(self, frames_data: List[Dict]) -> None:
        """
        并发保存并上传所有帧到 MinIO (备用逻辑)
        """
        tasks = []
        temp_files = []

        async def _save_and_upload(frame_item):
            if frame_item.get("minio_url"): return # 已经有URL则跳过

            try:
                temp_filename = f"{uuid.uuid4().hex}.jpg"
                temp_filepath = os.path.join(Config.FIXED_TEMP_DIR, temp_filename)
                
                await asyncio.to_thread(cv2.imwrite, temp_filepath, frame_item["img"])
                temp_files.append(temp_filepath)
                
                url = await asyncio.to_thread(MinioEngine.upload_file, temp_filepath)
                frame_item["minio_url"] = url
            except Exception as e:
                print(f"⚠️ 帧 {frame_item['index']} 上传失败: {e}")
                frame_item["minio_url"] = None

        print(f"🚀 开始补传 {len(frames_data)} 张图片到 MinIO...")
        for item in frames_data:
            tasks.append(_save_and_upload(item))
            
        await asyncio.gather(*tasks)
        
        for f in temp_files:
            if os.path.exists(f):
                try: os.remove(f)
                except: pass

    async def run(self, file_path: str, scan_mode: str = "fast", frames_url_map: Dict[int, str] = None) -> Dict[str, Any]:
        """
        :param frames_url_map: 可选，外部传入的 {index: url} 映射表，如果存在则直接使用，不再上传
        """
        if not os.path.exists(file_path):
            return {"error": f"文件不存在: {file_path}"}

        # 1. 抽帧
        frames_data = ImageUtils.extract_frames(file_path)
        if not frames_data:
            return {"error": "无法提取图像帧"}
            
        # 2. 关联预上传的 URL
        if frames_url_map:
            for item in frames_data:
                idx = item["index"]
                if idx in frames_url_map:
                    item["minio_url"] = frames_url_map[idx]
        
        # 3. 检查是否有缺失 URL 的帧，如果有则补传
        missing_upload = [f for f in frames_data if not f.get("minio_url")]
        if missing_upload:
            await self._upload_frames_concurrently(missing_upload)

        results_summary = {
            "person_names": set(),
            "visual_risks": [],
            "ocr_risks": [],
            "preview_images": []
        }

        # 4. 逐帧深度分析
        for frame_item in frames_data:
            target_img = frame_item["img"] 
            minio_url = frame_item.get("minio_url")
            
            frame_violated = False 
            violation_bboxes = []  
            
            # --- [A] YOLO 全量提取 ---
            yolo_start_time = time.perf_counter()
            raw_detections = YoloEngine.detect(target_img, conf=0.3)
            merged_candidates = ImageUtils.merge_overlapping_boxes(raw_detections, target_img.shape)
            yolo_elapsed_time = time.perf_counter() - yolo_start_time
            print(f"⏱️ YOLO 检测耗时: {yolo_elapsed_time:.2f} 秒")
            
            slices_b64 = []
            if merged_candidates:
                slices_b64 = [ImageUtils.encode_to_base64(ImageUtils.get_single_object_crop(target_img, d["bbox"])) for d in merged_candidates]
            
            blacklist_idxs = set() 

            # --- [B] API 身份识别 ---
            if minio_url:
                face_start_time = time.perf_counter()
                try:
                    # 现在返回的是 List[Dict]
                    person_results = await asyncio.to_thread(FaceEngine.identify_face, minio_url)
                    face_elapsed_time = time.perf_counter() - face_start_time
                    print(f"⏱️ 人脸识别 API 耗时: {face_elapsed_time:.2f} 秒")

                    if person_results:
                        for p in person_results:
                            p_name = p.get("name", "未知")
                            p_tag = p.get("tag", "")
                            p_info = f"{p_name} ({p_tag})"
                            
                            results_summary["person_names"].add(p_info)
                            results_summary["visual_risks"].append(f"发现黑名单人物: {p_info} (置信度: {p.get('similarity', 0)})")
                            frame_violated = True

                            p_bbox = p.get("bbox", [])
                            if p_bbox:
                                violation_bboxes.append({
                                    "bbox": p_bbox,
                                    "label": p_name,
                                    "score": p.get("similarity", 0)
                                })
                except Exception as e:
                    print(f"⚠️ 人脸识别请求异常: {e}")
            
            # --- [C] 行为与敏感标识研判 (LLM) ---
            if slices_b64:
                behavior_prompt = PromptTemplates.get_image_prompt("违规行为、敏感标识、阴暗内容、同性低俗、擦边、卖腐、性暗示、国民党党旗、台独、台湾旗帜、丑化嘲讽领导人，歧视中国人")
                msgs = LLMClient.build_visual_message(behavior_prompt, slices_b64)
                behavior_res = await LLMClient.get_json_response(msgs)
                
                if behavior_res and behavior_res.get("image"):
                    valid_ids = [i for i in behavior_res["image"] if 0 < i <= len(merged_candidates)]
                    for vid in valid_ids:
                        c_idx = vid - 1
                        if c_idx not in blacklist_idxs:
                            violation_bboxes.append(merged_candidates[c_idx])
                            frame_violated = True
                            results_summary["visual_risks"].append(f"发现敏感行为/标识 (对象ID: {vid})")

            # --- [D] OCR 敏感文字检测 ---
            ocr_start_time = time.perf_counter()
            ocr_results = OcrEngine.detect_text(target_img)
            ocr_elapsed_time = time.perf_counter() - ocr_start_time
            print(f"⏱️ OCR 检测耗时: {ocr_elapsed_time:.2f} 秒")
            
            if ocr_results:
                text_A = " ".join(o["text"] for o in ocr_results)
                text_B = {o["id"]: o["text"] for o in ocr_results}
                text_match = await LLMClient.get_json_response([
                    {"role": "user", "content": PromptTemplates.ocr_judge_prompt(text_A, text_B)}
                ])
                
                if text_match and text_match.get("id"):
                    bad_ocr = [o for o in ocr_results if o["id"] in text_match["id"]]
                    if bad_ocr:
                        target_img = ImageUtils.draw_ocr_boxes(target_img, bad_ocr)
                        frame_violated = True
                        bad_texts = [o['text'] for o in bad_ocr]
                        results_summary["ocr_risks"].extend(bad_texts)

            # --- [E] 最终绘图与保存 ---
            if violation_bboxes:
                final_dets = ImageUtils.merge_overlapping_boxes(violation_bboxes, target_img.shape)
                target_img = ImageUtils.draw_detections(target_img, final_dets, color=(0, 0, 255), thickness=3)

            # 保存临时文件并上传到 MinIO
            temp_filename = f"frame_{frame_item['index']}_{uuid.uuid4().hex}.jpg"
            temp_filepath = os.path.join(Config.FIXED_TEMP_DIR, temp_filename)
            
            try:
                # 保存图片到临时文件
                cv2.imwrite(temp_filepath, target_img)
                
                # 上传到 MinIO 并获取 URL
                minio_url = MinioEngine.upload_file(temp_filepath)
                results_summary["preview_images"].append('/' + minio_url.split('/', 3)[-1])
                # results_summary["preview_images"].append(minio_url)   # 本地测试显示图片

                print(f"✅ 帧 {frame_item['index']} 已上传到 MinIO: {minio_url}")
            except Exception as e:
                print(f"⚠️ 帧 {frame_item['index']} 上传到 MinIO 失败: {e}")
                # 上传失败时回退到 base64
                final_b64 = ImageUtils.encode_to_base64(target_img)
                results_summary["preview_images"].append(final_b64)
            finally:
                # 清理临时文件
                if os.path.exists(temp_filepath):
                    try:
                        os.remove(temp_filepath)
                    except Exception as e:
                        print(f"⚠️ 清理临时文件失败: {e}")
            

        return {
            "status": "success",
            "detected_persons": list(results_summary["person_names"]),
            "ocr_risks": list(set(results_summary["ocr_risks"])),
            "visual_risks": list(set(results_summary["visual_risks"])),
            "preview_images": results_summary["preview_images"] 
        }

    def _get_args_schema(self) -> Dict:
        return {
            "file_path": {"type": "string", "description": "媒体文件的本地绝对路径"},
            "scan_mode": {"type": "string", "enum": ["fast", "deep"], "description": "扫描模式"}
        }

    def _get_required_args(self) -> List[str]:
        return ["file_path"]