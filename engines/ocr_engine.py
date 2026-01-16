import numpy as np
from typing import List, Dict, Any, Union
from rapidocr import RapidOCR
from judge_agent.config import Config

class OcrEngine:
    """RapidOCR 文字识别引擎 (单例)"""
    
    _engine = None

    @classmethod
    def get_engine(cls) -> RapidOCR:
        """初始化或获取 RapidOCR 实例"""
        if cls._engine is None:
            print("🚀 正在初始化 RapidOCR 引擎...")
            # 可以通过参数配置是否使用 GPU
            cls._engine = RapidOCR()
        return cls._engine

    @classmethod
    def detect_text(cls, image_source: Union[str, np.ndarray]) -> List[Dict[str, Any]]:
        """
        识别图像中的文字
        :param image_source: 图像路径或 OpenCV 图像数组
        :return: 结构化结果列表
        """
        engine = cls.get_engine()
        
        # 执行推理
        # 新版本 RapidOCR 返回的是 RapidOCROutput 对象，不能直接 result, _ 解包
        output = engine(image_source)
        ocr_results = []
        if output.boxes is not None:
            for idx, (box, text) in enumerate(zip(output.boxes, output.txts)):
                ocr_results.append({
                    "id": idx + 1,
                    "text": text,
                    "box": box.tolist()
                })
        return ocr_results

    @classmethod
    def get_full_text(cls, ocr_results: List[Dict[str, Any]]) -> str:
        """
        将 OCR 结果合并为纯文本字符串
        """
        return " ".join([item['text'] for item in ocr_results])