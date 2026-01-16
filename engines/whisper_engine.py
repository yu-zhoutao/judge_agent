import os
from faster_whisper import WhisperModel
from typing import List, Dict, Any, Tuple
from judge_agent.config import Config

class WhisperEngine:
    """Faster-Whisper 语音转写引擎 (单例)"""
    
    _model = None

    @classmethod
    def get_model(cls) -> WhisperModel:
        """初始化或获取 Faster-Whisper 模型"""
        if cls._model is None:
            print(f"🚀 正在加载 Whisper 模型: {Config.WHISPER_MODEL_PATH} ...")
            # device: cuda / cpu
            # compute_type: float16 (GPU 推荐) / int8 (CPU 推荐)
            cls._model = WhisperModel(
                Config.WHISPER_MODEL_PATH,
                device=Config.DEVICE,
                compute_type=Config.COMPUTE_TYPE
            )
        return cls._model

    @classmethod
    def transcribe(cls, audio_path: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        将音频文件转录为文本
        :param audio_path: 音频或视频文件路径
        :return: (完整文本, 带有时间戳的分段列表)
        """
        model = cls.get_model()
        
        # beam_size: 5 是平衡速度与准确度的常用值
        segments, info = model.transcribe(
            audio_path, 
            beam_size=5,
            vad_filter=True,  # 开启静音过滤，提高转录效率
            word_timestamps=False
        )

        full_text = []
        detailed_segments = []

        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            
            full_text.append(text)
            detailed_segments.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": text
            })

        return " ".join(full_text), detailed_segments

    @classmethod
    def format_segments_for_llm(cls, segments: List[Dict[str, Any]]) -> str:
        """
        将时间轴分段格式化为易于 LLM 理解的字符串
        例如: [0.0 - 2.5] 大家好，欢迎收看...
        """
        formatted = []
        for s in segments:
            formatted.append(f"[{s['start']} - {s['end']}] {s['text']}")
        return "\n".join(formatted)