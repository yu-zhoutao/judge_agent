import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional
from judge_agent.config import Config
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
    def save_upload_file(upload_file, custom_name: Optional[str] = None) -> str:
        """
        将 FastAPI 的 UploadFile 对象保存到临时目录
        :return: 保存后的绝对路径
        """
        if not os.path.exists(Config.FIXED_TEMP_DIR):
            os.makedirs(Config.FIXED_TEMP_DIR)
            
        # 防止文件名冲突，建议使用 UUID
        ext = Path(upload_file.filename).suffix
        filename = custom_name or f"{uuid.uuid4().hex}{ext}"
        file_path = os.path.join(Config.FIXED_TEMP_DIR, filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
            
        return file_path

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