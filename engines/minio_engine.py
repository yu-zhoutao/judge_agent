import os
import uuid
import hashlib
from minio import Minio
from minio.error import S3Error
from judge_agent.config import Config

class MinioEngine:
    """MinIO 文件存储引擎 (支持内容去重)"""
    
    _client = None
    _url_cache = {} # 本地路径 -> URL 缓存

    @classmethod
    def get_client(cls) -> Minio:
        if cls._client is None:
            cls._client = Minio(
                Config.MINIO_ENDPOINT,
                access_key=Config.MINIO_ACCESS_KEY,
                secret_key=Config.MINIO_SECRET_KEY,
                secure=Config.MINIO_SECURE
            )
        return cls._client

    @classmethod
    def _calculate_md5(cls, file_path: str) -> str:
        """计算文件的 MD5 值"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    @classmethod
    def upload_file(cls, file_path: str) -> str:
        """
        上传文件到 MinIO (带去重逻辑)
        :param file_path: 本地文件路径
        :return: 文件 URL
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 1. 检查路径缓存
        if file_path in cls._url_cache:
            return cls._url_cache[file_path]

        client = cls.get_client()
        bucket_name = Config.MINIO_BUCKET
        
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)

        # 2. 计算哈希值作为对象名，确保相同内容不重复存储
        file_hash = cls._calculate_md5(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        object_name = f"image/{file_hash}{ext}"
        
        # 3. 检查 MinIO 中是否已存在该对象
        protocol = "https" if Config.MINIO_SECURE else "http"
        url = f"{protocol}://{Config.MINIO_ENDPOINT}/{bucket_name}/{object_name}"
        
        try:
            client.stat_object(bucket_name, object_name)
            # print(f"✨ 文件已存在，跳过上传: {object_name}")
            cls._url_cache[file_path] = url
            return url
        except:
            # 不存在则继续上传
            pass

        content_type = "application/octet-stream"
        if ext in ['.jpg', '.jpeg']:
            content_type = "image/jpeg"
        elif ext == '.png':
            content_type = "image/png"
        
        client.fput_object(
            bucket_name=bucket_name,
            object_name=object_name,
            file_path=file_path,
            content_type=content_type
        )

        cls._url_cache[file_path] = url
        return url
