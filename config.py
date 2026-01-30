import os
import torch
from dotenv import load_dotenv
from pathlib import Path

# 加载 .env 环境变量
load_dotenv()

class Config:
    # --- API 配置 ---
    SERPAPI_KEY = os.getenv("SERPAPI_KEY")
    _RAW_API_URL = (
        os.getenv("API_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or "http://127.0.0.1:8008/v1"
    )
    API_URL = _RAW_API_URL.rstrip("/") + "/v1" if not _RAW_API_URL.rstrip("/").endswith("/v1") else _RAW_API_URL
    API_KEY = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen3-VL-30B-A3B-Instruct")
    
    # --- MinIO 配置 ---
    MINIO_ENDPOINT = "minio.di.qihoo.net:9000"
    MINIO_ACCESS_KEY = "zhangshuhao"
    MINIO_SECRET_KEY = "MinIO@2025.qihoo"
    MINIO_BUCKET = "facerun-content-detect"
    MINIO_SECURE = False

    # --- MongoDB 配置 ---
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://admin:MongoDB%40qihoo.360@merger522.add.zzzc.qihoo.net:27017/")
    MONGO_DATABASE = os.getenv("MONGO_DATABASE", "judge_agent")
    MONGO_MAX_POOL_SIZE = int(os.getenv("MONGO_MAX_POOL_SIZE", "100"))
    MONGO_MIN_POOL_SIZE = int(os.getenv("MONGO_MIN_POOL_SIZE", "10"))
    MONGO_MAX_IDLE_TIME_MS = int(os.getenv("MONGO_MAX_IDLE_TIME_MS", "10000"))
    MONGO_SERVER_SELECTION_TIMEOUT_MS = int(os.getenv("MONGO_SERVER_SELECTION_TIMEOUT_MS", "5000"))
    MONGO_CONNECT_TIMEOUT_MS = int(os.getenv("MONGO_CONNECT_TIMEOUT_MS", "5000"))
    MONGO_INDEX_TTL_SECONDS = int(os.getenv("MONGO_INDEX_TTL_SECONDS", "86400"))

    # --- Face API 配置 ---
    FACE_API_URL = "http://hpcinf01.aitc.bjwdt.qihoo.net:6980/api/v1/image/sync"

    # --- 模型路径配置 ---
    YOLO_MODEL_PATH = "./yolov8n.pt"

    # --- ASR (语音转写) API 配置 ---
    ASR_API_URL = os.getenv("ASR_API_URL")
    ASR_API_KEY = os.getenv("ASR_API_KEY")
    # ASR 并发线程数
    ASR_THREAD_POOL_SIZE = 6

    # --- OCR API 配置 ---
    OCR_API_URL = os.getenv("OCR_API_URL")
    OCR_API_KEY = os.getenv("OCR_API_KEY")

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
