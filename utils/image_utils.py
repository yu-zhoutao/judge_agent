import cv2
import base64
import numpy as np
from typing import List, Dict, Any

class ImageUtils:
    """图像处理与视觉标注工具类"""

    @staticmethod
    def extract_frames(file_path: str, sample_count: int = 8) -> List[Dict]:
        """
        统一抽帧逻辑
        :param file_path: 文件路径
        :param sample_count: 视频采样帧数
        :return: [{'img': np.ndarray, 'index': int}, ...]
        """
        import os
        frames = []
        if not os.path.exists(file_path):
            return []
            
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext in ['.jpg', '.png', '.jpeg', '.webp', '.bmp']:
            img = cv2.imread(file_path)
            if img is not None:
                frames.append({"img": img, "index": 0})
        else:
            # 视频
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return []
                
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(total // sample_count, 1)
            
            for i in range(0, total, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frames.append({"img": frame, "index": i})
                if len(frames) >= sample_count:
                    break
            cap.release()
        
        return frames

    @staticmethod
    def encode_to_base64(image: np.ndarray, quality: int = 90) -> str:
        """
        将 OpenCV 的 BGR 图像转换为 Base64 编码的 JPEG 字符串
        :param image: numpy 数组格式的图像
        :param quality: JPG 压缩质量 (1-100)
        """
        if image is None:
            return ""
        # 压缩图像以减少网络传输负担
        success, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not success:
            return ""
        return base64.b64encode(buffer).decode("utf-8")

    @staticmethod
    def decode_from_base64(base64_str: str) -> np.ndarray:
        """
        将 Base64 字符串解码回 OpenCV 格式的 BGR 图像
        """
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    @staticmethod
    def draw_detections(image: np.ndarray, detections: List[Dict[str, Any]], color=(0, 0, 255), thickness=2) -> np.ndarray:
        """
        在图像上绘制 YOLO 检测到的目标框，仅保留边框，去除文本标签
        :param detections: [{'bbox': [x1, y1, x2, y2]}]
        """
        if image is None:
            return None
            
        temp_img = image.copy()
        for det in detections:
            # 提取坐标
            x1, y1, x2, y2 = det['bbox']
            
            # 仅绘制主矩形框
            cv2.rectangle(temp_img, (x1, y1), (x2, y2), color, thickness)
            
        return temp_img

    @staticmethod
    def draw_ocr_boxes(image: np.ndarray, ocr_results: List[Dict[str, Any]], color=(0, 255, 0)) -> np.ndarray:
        """
        在图像上绘制 OCR 文字区域多边形（默认绿色）
        :param ocr_results: [{'box': [[x1,y1], [x2,y2], ...], 'text': '...'}]
        """
        temp_img = image.copy()
        for ocr in ocr_results:
            # RapidOCR 返回的通常是 4 个点的列表
            pts = np.array(ocr['box'], np.int32).reshape((-1, 1, 2))
            cv2.polylines(temp_img, [pts], isClosed=True, color=color, thickness=2)
        return temp_img

    @staticmethod
    def get_single_object_crop(image: np.ndarray, bbox: List[int], padding: int = 10) -> np.ndarray:
        """
        根据 bbox 裁剪出单个目标区域（用于 Qwen3-VL 1:N 识别）
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        # 适当扩充边界，方便 AI 识别特征
        x1_p = max(0, x1 - padding)
        y1_p = max(0, y1 - padding)
        x2_p = min(w, x2 + padding)
        y2_p = min(h, y2 + padding)
        return image[y1_p:y2_p, x1_p:x2_p]
    
    @staticmethod
    def boxes_overlap(b1, b2):
        """判断两个矩形框是否重叠"""
        return max(b1[0], b2[0]) < min(b1[2], b2[2]) and \
               max(b1[1], b2[1]) < min(b1[3], b2[3])
               
    @staticmethod
    def crop_image(image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        get_single_object_crop 的简化别名，方便在业务流中调用
        """
        return ImageUtils.get_single_object_crop(image, bbox, padding=0)

    @staticmethod
    def merge_overlapping_boxes(detections, img_shape):
        """
        输入: 原始检测列表 [{'bbox': [x1,y1,x2,y2], ...}, ...]
        输出: 合并后的检测列表
        """
        if not detections: return []
        h, w = img_shape[:2]
        n = len(detections)
        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry: parent[ry] = rx

        # 只要重叠就视为同一组
        for i in range(n):
            for j in range(i + 1, n):
                if ImageUtils.boxes_overlap(detections[i]["bbox"], detections[j]["bbox"]):
                    union(i, j)

        clusters = {}
        for i in range(n):
            root = find(i)
            clusters.setdefault(root, []).append(detections[i])

        merged = []
        for idx, cluster in enumerate(clusters.values()):
            boxes = [c["bbox"] for c in cluster]
            x1 = max(0, min(b[0] for b in boxes))
            y1 = max(0, min(b[1] for b in boxes))
            x2 = min(w, max(b[2] for b in boxes))
            y2 = min(h, max(b[3] for b in boxes))
            
            # 保留该簇中置信度最高或最主要的标签
            main_label = cluster[0]["label"] 
            merged.append({
                "id": idx + 1,
                "bbox": [x1, y1, x2, y2],
                "label": main_label
            })
        return merged