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
    def _get_content_type(cls, ext: str) -> str:
        """根据文件扩展名获取 Content-Type"""
        content_type = "application/octet-stream"
        
        # 图片类型
        if ext in ['.jpg', '.jpeg']:
            content_type = "image/jpeg"
        elif ext == '.png':
            content_type = "image/png"
        elif ext in ['.gif', '.webp', '.bmp']:
            content_type = f"image/{ext[1:]}"
        
        # 音频类型
        elif ext in ['.mp3']:
            content_type = "audio/mpeg"
        elif ext == '.wav':
            content_type = "audio/wav"
        elif ext == '.aac':
            content_type = "audio/aac"
        elif ext == '.flac':
            content_type = "audio/flac"
        elif ext == '.m4a':
            content_type = "audio/mp4"
        
        # 视频类型
        elif ext == '.mp4':
            content_type = "video/mp4"
        elif ext == '.avi':
            content_type = "video/x-msvideo"
        elif ext == '.mov':
            content_type = "video/quicktime"
        elif ext == '.mkv':
            content_type = "video/x-matroska"
        elif ext in ['.flv', '.webm']:
            content_type = f"video/{ext[1:]}"
        
        return content_type

    @classmethod
    def _get_storage_path(cls, ext: str) -> str:
        """根据文件扩展名获取存储路径前缀"""
        # 图片类型
        if ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']:
            return "image"
        # 音频类型
        elif ext in ['.mp3', '.wav', '.aac', '.flac', '.m4a']:
            return "audio"
        # 视频类型
        elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm']:
            return "video"
        # 其他类型
        else:
            return "other"

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
        
        # 3. 根据文件类型选择存储路径
        storage_path = cls._get_storage_path(ext)
        object_name = f"{storage_path}/{file_hash}{ext}"
        
        # 4. 检查 MinIO 中是否已存在该对象
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

        # 5. 获取正确的 Content-Type
        content_type = cls._get_content_type(ext)
        
        client.fput_object(
            bucket_name=bucket_name,
            object_name=object_name,
            file_path=file_path,
            content_type=content_type
        )

        cls._url_cache[file_path] = url
        return url
