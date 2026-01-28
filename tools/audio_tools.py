# judge_agent/tools/audio_tools.py

import os
import uuid
import asyncio
import time
import imageio_ffmpeg as ffmpeg
from typing import Dict, List, Any
from judge_agent.config import Config
from judge_agent.tools.base import BaseTool
from judge_agent.engines.whisper_engine import WhisperEngine
from judge_agent.engines.llm_client import LLMClient
from judge_agent.engines.minio_engine import MinioEngine
from judge_agent.prompts.templates import PromptTemplates
from judge_agent.utils.json_utils import JSONUtils

class AudioTranscribeTool(BaseTool):
    name = "audio_transcribe"
    description = "音频转写与违规检测工具。提取语音转文字，并自动检测政治敏感/低俗/暴恐言论。若发现违规，会自动对视频进行切片存证。"

    async def run(self, file_path: str) -> Dict[str, Any]:
        """执行音频转写与违规检测。此方法会先通过 yield 流式输出音频文本，最后返回一个包含检测结果的字典。"""
        if not os.path.exists(file_path):
            return {"error": "文件不存在"}

        # 1. 语音转录 (Whisper)
        loop = asyncio.get_running_loop()
        whisper_start_time = time.perf_counter()
        try:
            raw_text, segments = await loop.run_in_executor(
                None, WhisperEngine.transcribe, file_path
            )
            whisper_elapsed_time = time.perf_counter() - whisper_start_time
            print(f"⏱️ Whisper 转录耗时: {whisper_elapsed_time:.2f} 秒")
        except Exception as e:
            whisper_elapsed_time = time.perf_counter() - whisper_start_time
            print(f"❌ Whisper 转录失败，耗时: {whisper_elapsed_time:.2f} 秒: {e}")
            return {"error": f"转写失败: {e}"}

        if not raw_text.strip():
            return {"status": "success", "text_content": "无有效语音", "violation_segments": []}

        # 2. 文本纠错 (生成适合阅读的字幕)
        # 这一步是为了生成前端需要的"更通顺的文本"
        correction_prompt = PromptTemplates.audio_correction_prompt(raw_text)
        corrected_text = raw_text # 默认回退
        try:
            # 调用非流式接口获取纠错结果
            client = LLMClient.get_async_client()
            llm_start_time = time.perf_counter()
            resp = await client.chat.completions.create(
                model=Config.MODEL_NAME,
                messages=[{"role": "user", "content": correction_prompt}],
                temperature=0.3
            )
            llm_elapsed_time = time.perf_counter() - llm_start_time
            print(f"⏱️ LLM 文本纠错耗时: {llm_elapsed_time:.2f} 秒")
            if resp.choices[0].message.content:
                corrected_text = resp.choices[0].message.content
        except Exception as e:
            print(f"文本纠错失败，使用原文: {e}")

        # 3. 违规判定 (LLM)
        formatted_text = WhisperEngine.format_segments_for_llm(segments)
        judge_prompt = PromptTemplates.text_review_and_correct_json_template(formatted_text)
        
        violation_report = {"is_violation": False, "segments": []}

        try:
            llm_judge_start_time = time.perf_counter()
            violation_data = await LLMClient.get_json_response([
                {"role": "user", "content": judge_prompt}
            ])
            llm_judge_elapsed_time = time.perf_counter() - llm_judge_start_time
            print(f"⏱️ LLM 违规判定耗时: {llm_judge_elapsed_time:.2f} 秒")
            
            if violation_data and violation_data.get("is_violation"):
                violation_report["is_violation"] = True
                merged_anchors = JSONUtils.merge_intervals(violation_data.get("time_anchors", []))
                
                # 4. 执行视频切片
                clip_tasks = []
                for anchor in merged_anchors:
                    clip_tasks.append(self._slice_video(
                        file_path, anchor['start'], anchor['end']
                    ))
                
                # 并发执行切片
                clip_filenames = await asyncio.gather(*clip_tasks)
                
                for i, fname in enumerate(clip_filenames):
                    if fname:
                        # 构造完整的文件路径
                        clip_path = os.path.join(Config.FIXED_TEMP_DIR, fname)
                        
                        try:
                            # 上传到 MinIO 并获取 URL
                            minio_url = MinioEngine.upload_file(clip_path)
                            merged_anchors[i]["clip_url"] = '/' + minio_url.split('/', 3)[-1]
                            print(f"✅ 视频切片已上传到 MinIO: {minio_url}")
                        except Exception as e:
                            print(f"⚠️ 视频切片上传到 MinIO 失败: {e}")
                            # 上传失败时回退到本地路径
                            merged_anchors[i]["clip_url"] = f"/static_temp/{fname}"
                
                violation_report["segments"] = merged_anchors
        
        except Exception as e:
            print(f"音频合规性检测失败: {e}")


        # 返回包含所有前端所需信息的字典 (不再包含 corrected_text)
        return {
            "status": "success",
            "text_content": raw_text,         # 原始文本
            "corrected_text": corrected_text,
            "violation_check": violation_report,
            "evidence": {
                "audio_risk": violation_report["is_violation"],
                "clips": violation_report["segments"]
            }
        }

    async def _slice_video(self, input_path: str, start: float, end: float) -> str:
        """调用 ffmpeg 切割视频"""
        try:
            if not os.path.exists(Config.FIXED_TEMP_DIR):
                os.makedirs(Config.FIXED_TEMP_DIR)
            
            ext = os.path.splitext(input_path)[1]
            if not ext: ext = ".mp4"
            
            output_filename = f"evidence_{uuid.uuid4().hex[:8]}{ext}"
            output_path = os.path.join(Config.FIXED_TEMP_DIR, output_filename)
            
            # 保证最少切1秒，避免ffmpeg报错
            duration = max(end - start, 1.0)
            
            # 使用 ultrafast 预设加速
            cmd = [
                ffmpeg.get_ffmpeg_exe(), '-y',
                '-ss', str(start), 
                '-t', str(duration),
                '-i', input_path,
                '-c:v', 'libx264', '-preset', 'ultrafast',
                '-c:a', 'aac', 
                '-strict', 'experimental',
                output_path
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            slice_start_time = time.perf_counter()
            await process.communicate()
            slice_elapsed_time = time.perf_counter() - slice_start_time
            
            if process.returncode == 0:
                print(f"✅ 视频切片成功，耗时: {slice_elapsed_time:.2f} 秒")
                return output_filename
            else:
                print(f"❌ 视频切片失败，耗时: {slice_elapsed_time:.2f} 秒")
                return ""
        except Exception as e:
            print(f"❌ 视频切片异常: {e}")
            return ""

    def _get_args_schema(self) -> Dict:
        return {"file_path": {"type": "string", "description": "媒体文件路径"}}

    def _get_required_args(self) -> List[str]:
        return ["file_path"]