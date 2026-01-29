import torch
from ultralytics import YOLO
from typing import List, Dict, Any
from judge_agent.config import Config

class YoloEngine:
    """YOLOv8 目标检测引擎 (单例)"""
    
    _model = None

    @classmethod
    def get_model(cls) -> YOLO:
        """获取或初始化 YOLO 模型实例"""
        if cls._model is None:
            print(f"🚀 正在加载 YOLO 模型: {Config.YOLO_MODEL_PATH} ...")
            # 加载模型并移动到指定设备 (CUDA/CPU)
            cls._model = YOLO(Config.YOLO_MODEL_PATH).to(Config.DEVICE)
        return cls._model

    @classmethod
    def detect(cls, image_path_or_array: Any, conf: float = 0.25) -> List[Dict[str, Any]]:
        """
        执行目标检测
        :param image_path_or_array: 图片路径或 OpenCV 图像数组
        :param conf: 置信度阈值
        :return: 检测结果列表 [{'label': 'person', 'conf': 0.9, 'bbox': [x1, y1, x2, y2]}]
        """
        model = cls.get_model()
        
        # 执行推理，设置 verbose=False 减少控制台日志抖动
        results = model(image_path_or_array, conf=conf, verbose=False)
        
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            
            # 提取坐标、置信度和类别索引
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            
            for box, score, cls_idx in zip(boxes, scores, classes):
                x1, y1, x2, y2 = map(int, box)
                label = model.names[cls_idx]
                
                detections.append({
                    "label": label,
                    "conf": float(score),
                    "bbox": [x1, y1, x2, y2]
                })
        
        return detections

    @classmethod
    def detect_and_filter(cls, image_path: str, target_labels: List[str]) -> List[Dict[str, Any]]:
        """
        检测并过滤出特定类别的目标（例如只看人或旗帜）
        """
        all_dets = cls.detect(image_path)
        return [d for d in all_dets if d['label'] in target_labels]