import base64
import json
import ast
import requests
import numpy as np
import cv2
from typing import List, Dict, Any, Union

import urllib3

from judge_agent.config import Config

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
class OcrEngine:
    """
    在线 OCR 文字识别引擎
    """

    @classmethod
    def _encode_image(cls, image_source: Union[str, np.ndarray]) -> str:
        """
        将图像（路径或 numpy 数组）转换为 Base64 字符串
        """
        img_data = None

        # 1. 如果是文件路径
        if isinstance(image_source, str):
            with open(image_source, "rb") as f:
                img_data = f.read()

        # 2. 如果是 OpenCV/Numpy 图像数组
        elif isinstance(image_source, np.ndarray):
            # 将 numpy 数组编码为 jpg 格式的字节流
            success, encoded_img = cv2.imencode('.jpg', image_source)
            if not success:
                raise ValueError("无法将 Numpy 数组编码为图像")
            img_data = encoded_img.tobytes()
        else:
            raise TypeError(f"不支持的图像类型: {type(image_source)}")

        # 进行 Base64 编码并解码为 utf-8 字符串
        return base64.b64encode(img_data).decode("utf-8")

    @classmethod
    def detect_text(cls, image_source: Union[str, np.ndarray]) -> List[Dict[str, Any]]:
        """
        识别图像中的文字 (调用线上 API)
        :param image_source: 图像路径或 OpenCV 图像数组
        :return: 结构化结果列表
        """
        ocr_results = []

        try:
            # 1. 准备 Base64 数据
            encoded_image = cls._encode_image(image_source)

            # 2. 构造请求参数
            url = Config.OCR_API_URL
            payload = {
                "IMAGE": encoded_image,
                "base64_list": ["IMAGE"]
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {Config.OCR_API_KEY}",
            }

            # 3. 发起请求
            # print(f"🚀 正在调用 OCR API: {url}...")
            response = requests.post(url, headers=headers, json=payload, timeout=30, verify=False, proxies={"http": None, "https": None})

            # 4. 解析响应
            if response.ok:
                outer_response = response.json()

                # API 返回的 bridge_output0 是一个字符串形式的 Python 字典，需要解析
                # 使用 ast.literal_eval 比 eval 更安全
                if "bridge_output0" in outer_response:
                    bridge_output = outer_response["bridge_output0"]
                    # 检查是否为空或 None
                    if bridge_output:
                        output = ast.literal_eval(bridge_output)

                        # 提取坐标和文本
                        extra_bbox = output.get("extra_bbox", [])
                        extra_info = output.get("extra_info", [])

                        for idx, (box, text) in enumerate(zip(extra_bbox, extra_info)):
                            ocr_results.append({
                                "id": idx + 1,
                                "text": text,
                                "box": box
                            })
                else:
                    print(f"⚠️ OCR API 响应格式异常: {outer_response}")
            else:
                print(f"❌ OCR API 请求失败: {response.status_code} - {response.text}")

        except Exception as e:
            print(f"❌ OCR 识别过程中发生错误: {e}")
            # 根据需要决定是否 raise 异常，或者返回空列表
            # raise e

        return ocr_results

    @classmethod
    def get_full_text(cls, ocr_results: List[Dict[str, Any]]) -> str:
        """
        将 OCR 结果合并为纯文本字符串 (保持不变)
        """
        return " ".join([item['text'] for item in ocr_results])