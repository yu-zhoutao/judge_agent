import requests
import json
from typing import List, Dict, Any
from judge_agent.config import Config

class FaceEngine:
    """API based Face Recognition Engine (Optimized for specific Face API)"""

    @staticmethod
    def identify_face(image_url: str) -> List[Dict[str, Any]]:
        """
        通过 API 识别图片中的黑名单人物
        :param image_url: 图片的 URL (MinIO)
        :return: 命中人员的详细信息列表 [{'name': 'xxx', 'tag': 'xxx', 'bbox': [x1, y1, x2, y2], 'similarity': 0.9}, ...]
        """
        url = Config.FACE_API_URL
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        
        payload = {
            "ability": ["face"],
            "tasks": [
                {
                    "dataId": "audit_task",
                    "url": image_url
                }
            ],
            "rule": []
        }
        
        found_results = []
        
        try:
            print(f"🚀 调用人脸识别 API: {image_url}")
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            print(f"🚀 人脸识别结果: \n{response.json()}")
            if response.status_code == 200:
                res_json = response.json()
                
                # 根据 face.txt 中的结构解析
                if res_json.get("code") == 200:
                    results_list = res_json.get("result", [])
                    for task_res in results_list:
                        face_data = task_res.get("face", {})
                        detail = face_data.get("detail", {})
                        extra_info = detail.get("extra_info", [])
                        
                        for info in extra_info:
                            name = info.get("name")
                            if name and name != "unknown":
                                first_class = info.get("first_class", "")
                                second_class = info.get("second_class", "")
                                similarity = info.get("similarity", 0)
                                bbox = info.get("bbox", []) # [y1, x1, y2, x2]? 或 [x1, y1, x2, y2]?
                                
                                # 构造返回对象
                                found_results.append({
                                    "name": name,
                                    "tag": f"{first_class} | {second_class}",
                                    "similarity": similarity,
                                    "bbox": bbox
                                })
                else:
                    print(f"⚠️ Face API 返回错误状态码: {res_json.get('code')} - {res_json.get('msg')}")
            else:
                print(f"❌ Face API 请求失败: {response.status_code}")

        except Exception as e:
            print(f"❌ Face API 请求异常: {e}")
            
        return found_results