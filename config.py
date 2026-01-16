import os
import torch
from dotenv import load_dotenv
from pathlib import Path

# 加载 .env 环境变量
load_dotenv()

class Config:
    # --- API 配置 ---
    SERPAPI_KEY = os.getenv("SERPAPI_KEY")
    VLLM_API_URL = os.getenv("API_URL", "http://127.0.0.1:8008/v1") 
    VLLM_API_KEY = os.getenv("API_KEY", "EMPTY") 
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen3-VL-30B-A3B-Instruct") 
    
    # --- MinIO 配置 ---
    MINIO_ENDPOINT = "minio.di.qihoo.net:9000"
    MINIO_ACCESS_KEY = "zhangshuhao"
    MINIO_SECRET_KEY = "MinIO@2025.qihoo"
    MINIO_BUCKET = "facerun-content-detect"
    MINIO_SECURE = False

    # --- Face API 配置 ---
    FACE_API_URL = "http://hpcinf01.aitc.bjwdt.qihoo.net:6980/api/v1/image/sync"

    # --- 模型路径配置 ---
    WHISPER_MODEL_PATH = "./faster-whisper-medium"
    YOLO_MODEL_PATH = "./yolov8n.pt"

    # --- 业务目录配置 ---
    # 使用 Path 对象自动处理不同操作系统的路径分隔符
    BASE_DIR = Path(__file__).resolve().parent.parent
    FIXED_TEMP_DIR = os.path.join(BASE_DIR, "upload_cache")

    # --- 硬件配置 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
    
    # DEVICE = "cpu"
    # # 计算类型: CUDA 用 float16 提速，CPU 用 int8 节省资源
    # COMPUTE_TYPE = "int8"

    # --- 初始化检查 ---
    @classmethod
    def init_directories(cls):
        """确保必要的目录存在"""
        if not os.path.exists(cls.FIXED_TEMP_DIR):
            os.makedirs(cls.FIXED_TEMP_DIR)
            print(f"📁 已创建目录: {cls.FIXED_TEMP_DIR}")

# 执行初始化
Config.init_directories()