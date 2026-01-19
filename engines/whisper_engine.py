import os
import io
import time
import json
import uuid
import wave
import math
import base64
import requests
import subprocess
import imageio_ffmpeg
import urllib3
from typing import List, Dict, Any, Tuple
from multiprocessing.pool import ThreadPool
from judge_agent.config import Config

# 屏蔽 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class WhisperEngine:
    """
    在线 ASR 语音转写引擎 (并发版)
    集成音频转码、切片、并发请求与结果合并
    """

    @classmethod
    def _convert_to_16k_wav(cls, source_path: str) -> str:
        """
        使用 FFmpeg 将任意音频/视频转换为 16k采样率、单声道 WAV
        """
        if not os.path.exists(Config.FIXED_TEMP_DIR):
            os.makedirs(Config.FIXED_TEMP_DIR)

        filename = f"temp_asr_{uuid.uuid4().hex[:8]}.wav"
        output_path = os.path.join(Config.FIXED_TEMP_DIR, filename)
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        # -ar 16000: 采样率 16k
        # -ac 1: 单声道
        # -c:a pcm_s16le: 16位 PCM 编码
        cmd = [
            ffmpeg_exe, '-y',
            '-i', source_path,
            '-ar', '16000',
            '-ac', '1',
            '-c:a', 'pcm_s16le',
            output_path
        ]

        try:
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True
            )
            return output_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"音频转码失败: {e.stderr.decode() if e.stderr else 'unknown error'}")

    @classmethod
    def _split_wav(cls, byte_data: bytes, segment_length=60):
        """
        将 WAV 二进制数据按时长切片
        :param segment_length: 切片时长(秒)，默认 60s
        """
        wf = wave.open(io.BytesIO(byte_data), "rb")
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()

        duration = nframes / framerate
        data_length = int(segment_length * framerate)  # 必须转为 int

        segments = []
        # 计算切片数量
        num_chunks = math.ceil(1.0 * duration / segment_length)

        for i in range(num_chunks):
            wf.setpos(i * data_length)
            data = wf.readframes(data_length)

            tmpf = io.BytesIO()
            with wave.open(tmpf, "wb") as new_wf:
                new_wf.setnchannels(nchannels)
                new_wf.setsampwidth(sampwidth)
                new_wf.setframerate(framerate)
                new_wf.writeframes(data)

            segments.append(tmpf.getvalue())

        wf.close()
        return segments, duration

    @classmethod
    def _asr_infer(cls, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        单次 API 请求任务
        """
        try:
            bdata = task["bdata"]
            url = Config.ASR_API_URL

            # 构造 Payload
            payload = {
                "request_id": f"req_{uuid.uuid4().hex[:8]}",
                "audio_type": "wav",
                "audio_data": base64.b64encode(bdata).decode(),
                "stream": False,
                "audio_fs": 16000
            }

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {Config.ASR_API_KEY}",
                "Connection": "keep-alive",
            }

            # 发起请求 (包含 SSL 跳过和内网代理设置)
            # debug_st = time.time()
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                verify=False,
                proxies={"http": None, "https": None},
                timeout=60
            )
            # print(f"Chunk processed in {time.time() - debug_st:.2f}s")

            if resp.status_code != 200:
                print(f"❌ ASR Chunk Failed: {resp.status_code} - {resp.text[:100]}")
                return {}

            jsres = resp.json()
            # 提取结果，不同 API 结构可能略有不同，这里沿用参考代码的路径
            # 假设结构: {"result": {"result": [{"text":..., "timestamp":...}]}}
            if "result" in jsres and "result" in jsres["result"]:
                result_list = jsres["result"]["result"]
                if result_list:
                    result = result_list[0]
                    # 修正相对时间戳为绝对时间戳 (API返回的是分片内的偏移，需要加上切片起始时间)
                    # 注意：参考代码中 task["bg"] 是秒，这里转为毫秒
                    result["start"] = task["bg"] * 1000
                    return result

            return {}

        except Exception as e:
            print(f"❌ ASR Infer Exception: {e}")
            return {}

    @classmethod
    def _merge_asr_results(cls, results: List[Dict], punctuation="。！？；，、", ts_unit="ms"):
        """
        合并结果并根据标点符号进行断句
        """
        full_text = ""
        timestamp_list = []  # [[start, end], ...]
        segments = []  # [{"start":, "end":, "text":}, ...]

        # 1. 扁平化合并所有分片的文本和时间戳
        # 按照 start 时间排序，防止线程乱序
        sorted_results = sorted([r for r in results if r], key=lambda x: x.get("start", 0))

        def to_sec(t):
            # 将毫秒转为秒
            return float(t) / 1000.0 if ts_unit == "ms" else float(t)

        for r in sorted_results:
            full_text += r.get("text", "")

            # 处理每一个字的时间戳
            chunk_start_ms = r.get("start", 0)
            for t in r.get("timestamp", []):
                # t[0], t[1] 是相对于该分片起始的偏移量
                t_start = to_sec(t[0] + chunk_start_ms)
                t_end = to_sec(t[1] + chunk_start_ms)
                timestamp_list.append([t_start, t_end])

        # 2. 根据标点符号重新切分句子 (Logic from reference)
        current_sentence = ""
        sent_start = None
        ts_idx = 0

        # 遍历全文字符
        for char in full_text:
            if ts_idx < len(timestamp_list):
                if sent_start is None:
                    sent_start = timestamp_list[ts_idx][0]
                sent_end = timestamp_list[ts_idx][1]
                ts_idx += 1
            else:
                # 容错：文字比时间戳多
                if sent_start is None: sent_start = 0.0
                sent_end = sent_start

            current_sentence += char

            # 遇到标点，结束当前句
            if char in punctuation:
                clean_sentence = current_sentence.strip()
                if clean_sentence:
                    segments.append({
                        "start": round(sent_start, 2),
                        "end": round(sent_end, 2),
                        "text": clean_sentence
                    })
                current_sentence = ""
                sent_start = None

        # 处理末尾剩余文本
        if current_sentence.strip():
            segments.append({
                "start": round(sent_start if sent_start else 0.0, 2),
                "end": round(sent_end if sent_end else 0.0, 2),
                "text": current_sentence.strip()
            })

        return full_text, segments

    @classmethod
    def transcribe(cls, audio_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        主入口：执行音频转写
        :return: (完整文本, 详细分段列表)
        """
        wav_path = None
        try:
            # 1. 格式转换 (转为 16k WAV)
            # print(f"🔄 正在转换音频: {os.path.basename(audio_path)}")
            wav_path = cls._convert_to_16k_wav(audio_path)

            # 2. 读取二进制数据
            with open(wav_path, "rb") as f:
                wav_bytes = f.read()

            # 3. 切片 (每 60 秒一片)
            segment_length = 60
            wav_chunks, duration = cls._split_wav(wav_bytes, segment_length)

            # 4. 构造任务列表
            tasks = []
            for i, chunk_data in enumerate(wav_chunks):
                tasks.append({
                    "bg": i * segment_length,  # 这里的 bg 单位是秒
                    "ed": min((i + 1) * segment_length, duration),
                    "bdata": chunk_data
                })

            # 5. 并发请求
            pool_size = min(len(tasks), Config.ASR_THREAD_POOL_SIZE)
            # print(f"🚀 开始并发识别: {len(tasks)} 个分片, 线程数: {pool_size}")

            with ThreadPool(pool_size) as p:
                raw_results = p.map(cls._asr_infer, tasks)

            # 6. 合并结果
            full_text, segments = cls._merge_asr_results(raw_results)

            return full_text, segments

        except Exception as e:
            print(f"❌ ASR 转写异常: {e}")
            import traceback
            traceback.print_exc()
            return "", []

        finally:
            # 7. 清理临时文件
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except:
                    pass

    @classmethod
    def format_segments_for_llm(cls, segments: List[Dict[str, Any]]) -> str:
        """格式化为 LLM 易读的字符串"""
        formatted = []
        for s in segments:
            formatted.append(f"[{s['start']} - {s['end']}] {s['text']}")
        return "\n".join(formatted)