# judge_agent/tools/audio_tools.py

import os
import uuid
import asyncio
import time
import json
import logging
import imageio_ffmpeg as ffmpeg
from typing import Dict, List, Any
from judge_agent.config import Config
from judge_agent.engines.whisper_engine import WhisperEngine
from judge_agent.engines.langchain_llm import async_chat_response, async_get_json_response
from judge_agent.engines.minio_engine import MinioEngine
from judge_agent.prompts.templates import PromptTemplates
from judge_agent.utils.json_utils import JSONUtils

logger = logging.getLogger("judge_agent.audio_tools")

class AudioTranscribeTool:
    name = "audio_transcribe"
    description = "音频转写与违规检测工具。提取语音转文字，并自动检测政治敏感/低俗/暴恐言论。若发现违规，会自动对视频进行切片存证。"

    async def run(self, file_path: str) -> Dict[str, Any]:
        """执行音频转写与违规检测。此方法会先通过 yield 流式输出音频文本，最后返回一个包含检测结果的字典。"""
        if not os.path.exists(file_path):
            return {"error": "文件不存在"}

        # 1. 语音转录 (Whisper)
        loop = asyncio.get_running_loop()
        try:
            raw_text, segments = await loop.run_in_executor(
                None, WhisperEngine.transcribe, file_path
            )
        except Exception as e:
            logger.error("asr_error:\n%s", str(e))
            return {"error": f"转写失败: {e}"}

        if not raw_text.strip():
            return {"status": "success", "text_content": "无有效语音", "violation_segments": []}

        logger.info(
            "asr_result:\n%s",
            json.dumps({"text": raw_text, "segments": segments}, ensure_ascii=False, indent=2),
        )

        # 2. 文本纠错 (生成适合阅读的字幕)
        # 这一步是为了生成前端需要的"更通顺的文本"
        correction_prompt = PromptTemplates.audio_correction_prompt(raw_text)
        corrected_text = raw_text # 默认回退
        try:
            candidate = await async_chat_response(correction_prompt, temperature=0.3)
            if candidate:
                corrected_text = candidate
        except Exception as e:
            logger.warning("audio_correction_failed:\n%s", str(e))

        logger.info(
            "asr_corrected_text:\n%s",
            json.dumps({"corrected_text": corrected_text}, ensure_ascii=False, indent=2),
        )

        # 3. 违规判定 (LLM)
        formatted_text = WhisperEngine.format_segments_for_llm(segments)
        judge_prompt = PromptTemplates.text_review_and_correct_json_template(formatted_text)
        
        violation_report = {"is_violation": False, "segments": []}

        try:
            violation_data = await async_get_json_response([
                {"role": "user", "content": judge_prompt}
            ])
            
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
                            # merged_anchors[i]["clip_url"] = minio_url      # 本地测试显示视频和音频
                            logger.info(
                                "audio_clip_uploaded:\n%s",
                                json.dumps({"clip_url": minio_url}, ensure_ascii=False, indent=2),
                            )
                        except Exception as e:
                            logger.warning("audio_clip_upload_failed:\n%s", str(e))
                            # 上传失败时回退到本地路径
                            merged_anchors[i]["clip_url"] = f"/static_temp/{fname}"
                
                violation_report["segments"] = merged_anchors
        
        except Exception as e:
            logger.error("audio_violation_check_failed:\n%s", str(e))

        logger.info(
            "audio_violation_report:\n%s",
            json.dumps(violation_report, ensure_ascii=False, indent=2),
        )


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
        """调用 ffmpeg 切割媒体文件（自动识别视频或音频）"""
        try:
            if not os.path.exists(Config.FIXED_TEMP_DIR):
                os.makedirs(Config.FIXED_TEMP_DIR)

            # 1. 获取输入文件后缀，判断是否为纯音频文件
            input_ext = os.path.splitext(input_path)[1].lower()
            audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.wma'}
            is_audio_mode = input_ext in audio_extensions

            # 2. 根据类型确定输出后缀和编码参数
            if is_audio_mode:
                output_ext = ".mp3"
                # -vn: 禁用视频流
                # -c:a libmp3lame: 指定 MP3 编码器
                # -q:a 2: 动态码率 V2 (高质量，约 170-210kbps)，比固定码率更好
                encoding_args = [
                    '-vn',
                    '-c:a', 'libmp3lame',
                    '-q:a', '2'
                ]
            else:
                output_ext = ".mp4"
                # 保持原有的视频压缩参数
                encoding_args = [
                    '-c:v', 'libx264',
                    '-preset', 'ultrafast',
                    '-c:a', 'aac',
                    '-strict', 'experimental'
                ]

            output_filename = f"evidence_{uuid.uuid4().hex[:8]}{output_ext}"
            output_path = os.path.join(Config.FIXED_TEMP_DIR, output_filename)

            # 保证最少切1秒
            duration = max(end - start, 1.0)

            # 3. 组装 FFmpeg 命令
            # 注意：-ss 放在 -i 之前是为了启用输入流跳转（Input Seeking），速度极快
            cmd = [
                      ffmpeg.get_ffmpeg_exe(), '-y',
                      '-ss', str(start),
                      '-t', str(duration),
                      '-i', input_path,
                  ] + encoding_args + [output_path]

            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            slice_start_time = time.perf_counter()
            stdout, stderr = await process.communicate()
            slice_elapsed_time = time.perf_counter() - slice_start_time

            if process.returncode == 0:
                logger.info(
                    "slice_success:\n%s",
                    json.dumps(
                        {
                            "type": "audio" if is_audio_mode else "video",
                            "duration_sec": slice_elapsed_time,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                )
                return output_filename
            else:
                logger.error(
                    "slice_failed:\n%s",
                    json.dumps(
                        {
                            "type": "audio" if is_audio_mode else "video",
                            "duration_sec": slice_elapsed_time,
                            "stderr": stderr.decode(),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                )
                return ""
        except Exception as e:
            logger.error("slice_exception:\n%s", str(e))
            return ""
