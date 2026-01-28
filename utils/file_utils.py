import os
import shutil
import time
import uuid
import hashlib
from pathlib import Path
from typing import Optional, Tuple
from judge_agent.config import Config
from judge_agent.engines.minio_engine import MinioEngine
import aiohttp

class FileUtils:
    """文件系统操作工具类"""

    @staticmethod
    def detect_file_type(filename: str) -> str:
        """
        根据扩展名探测媒体类型
        """
        ext = Path(filename).suffix.lower()
        if ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']:
            return "image"
        if ext in ['.mp3', '.wav', '.aac', '.flac', '.m4a']:
            return "audio"
        if ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm']:
            return "video"
        return "unknown"

    @staticmethod
    def _calculate_md5_from_upload(upload_file) -> str:
        """
        计算 FastAPI UploadFile 对象的 MD5 值
        :param upload_file: FastAPI 的 UploadFile 对象
        :return: MD5 哈希值（十六进制字符串）
        """
        hash_md5 = hashlib.md5()
        # 保存当前位置
        original_position = upload_file.file.tell()
        
        try:
            # 重置到文件开头
            upload_file.file.seek(0)
            
            # 分块读取计算 MD5
            for chunk in iter(lambda: upload_file.file.read(4096), b""):
                hash_md5.update(chunk)
            
            # 返回文件开头，以便后续读取
            upload_file.file.seek(0)
            
            return hash_md5.hexdigest()
        except Exception as e:
            # 发生异常时恢复文件位置
            upload_file.file.seek(original_position)
            raise e

    @staticmethod
    def save_upload_file(upload_file, custom_name: Optional[str] = None, upload_to_minio: bool = True) -> Tuple[str, Optional[str]]:
        """
        将 FastAPI 的 UploadFile 对象保存到临时目录，并可选地上传到 MinIO
        :param upload_file: FastAPI 的 UploadFile 对象
        :param custom_name: 自定义文件名（可选），如果未提供则使用 MD5 作为文件名
        :param upload_to_minio: 是否上传到 MinIO（默认 True）
        :return: (本地文件路径, MinIO URL) 元组，如果不上传则 MinIO URL 为 None
        """
        if not os.path.exists(Config.FIXED_TEMP_DIR):
            os.makedirs(Config.FIXED_TEMP_DIR)
            
        # 获取文件扩展名
        ext = Path(upload_file.filename).suffix
        
        # 如果没有提供自定义文件名，则使用 MD5 作为文件名
        if custom_name is None:
            try:
                file_hash = FileUtils._calculate_md5_from_upload(upload_file)
                filename = f"{file_hash}{ext}"
                print(f"📝 文件 MD5: {file_hash}")
            except Exception as e:
                print(f"⚠️ 计算 MD5 失败，使用 UUID 作为文件名: {str(e)}")
                filename = f"{uuid.uuid4().hex}{ext}"
        else:
            filename = custom_name
        
        file_path = os.path.join(Config.FIXED_TEMP_DIR, filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        
        # 上传到 MinIO
        minio_url = None
        if upload_to_minio:
            try:
                minio_url = MinioEngine.upload_file(file_path)
                print(f"✅ 文件已上传到 MinIO: {minio_url}")
            except Exception as e:
                print(f"⚠️ 文件上传到 MinIO 失败: {str(e)}")
                # 即使上传失败，也继续使用本地文件
            
        return file_path, minio_url

    @staticmethod
    def clear_temp_dir(age_seconds: int = 3600):
        """
        清理临时目录中超过一定时间的文件 (默认1小时)
        防止服务器硬盘被上传的视频撑爆
        """
        now = time.time()
        if not os.path.exists(Config.FIXED_TEMP_DIR):
            return

        for item in os.listdir(Config.FIXED_TEMP_DIR):
            item_path = os.path.join(Config.FIXED_TEMP_DIR, item)
            # 检查文件修改时间
            if os.path.getmtime(item_path) < now - age_seconds:
                try:
                    if os.path.isfile(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                    print(f"🧹 已自动清理过期文件: {item}")
                except Exception as e:
                    print(f"❌ 清理文件失败 {item}: {e}")

    @staticmethod
    def get_static_url(file_path: str) -> str:
        """
        将本地路径转换为前端可访问的静态 URL 路径
        例如: /static_temp/violation_123.mp4
        """
        filename = os.path.basename(file_path)
        return f"/static_temp/{filename}"

    @staticmethod
    async def async_serper_search(image_url: str, extra_query: str = "") -> str:
        if not image_url or not Config.SERPAPI_KEY: return "未启用搜索。"
        params = {
            "engine": "google_reverse_image", "image_url": image_url,
            "api_key": Config.SERPAPI_KEY, "hl": "zh-CN", "gl": "cn"
        }
        if extra_query: params["q"] = extra_query
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("https://serpapi.com/search.json", params=params) as response:
                    data = await response.json()
            
            results_text = []
            if "knowledge_graph" in data:
                results_text.append(f"【知识卡片】: {data['knowledge_graph'].get('title', '')}")
            
            results = data.get("image_results", []) + data.get("inline_images", [])
            for item in results[:6]:
                title = item.get("title", "")
                source = item.get("source", "")
                if title: results_text.append(f"- [{source}] {title}")
                
            return "\n".join(results_text) if results_text else "未搜索到相关结果。"
        except Exception as e:
            return f"搜索服务错误: {str(e)}"