import os
import time
import asyncio
from typing import Dict, List, Any, Optional
from judge_agent.tools.base import BaseTool
from judge_agent.utils.file_utils import FileUtils
from judge_agent.engines.minio_engine import MinioEngine

class WebSearchTool(BaseTool):
    name = "web_search"
    description = "网络搜索工具。支持以图搜图或关键词搜索。用于核实未知人物身份、旗帜含义或确认新闻事实。"

    async def run(self, query: str = "", image_path: str = "", image_url: str = "") -> Dict[str, Any]:
        """
        :param query: 搜索关键词 (可选)
        :param image_path: 本地图片路径 (可选，用于以图搜图)
        :param image_url: 图片 URL (可选，如果有则直接使用，不再上传)
        """
        if not query and not image_path and not image_url:
            return {"error": "必须提供 query, image_path 或 image_url"}

        search_result = ""

        # 模式 A: 以图搜图
        if image_path or image_url:
            target_url = image_url
            
            # 如果没有直接提供 URL，但有本地路径，则尝试上传
            if not target_url and image_path:
                if not os.path.exists(image_path):
                    return {"error": f"图片文件不存在: {image_path}"}
                
                # 1. 上传图片到 MinIO
                upload_start_time = time.perf_counter()
                try:
                    target_url = await asyncio.to_thread(MinioEngine.upload_file, image_path)
                    print(f"MinIO 上传成功,Url: {target_url}")
                except Exception as e:
                    print(f"❌ MinIO 上传失败: {e}")
                    target_url = None
                    
                upload_elapsed_time = time.perf_counter() - upload_start_time
                print(f"⏱️ MinIO 上传耗时: {upload_elapsed_time:.2f} 秒")
            
            if not target_url:
                return {"error": "无法获取有效的图片 URL 进行搜索"}
            
            # 2. 调用 SerpApi
            search_start_time = time.perf_counter()
            search_result = await FileUtils.async_serper_search(target_url, extra_query=query)
            search_elapsed_time = time.perf_counter() - search_start_time
            print(f"⏱️ 以图搜图搜索耗时: {search_elapsed_time:.2f} 秒")

        # 模式 B: 纯文本搜索 (如果没有图片)
        else:
            # 这里需要 FileUtils 实现一个纯文本搜索的方法，或者直接复用 serper_search 传空 url
            # 假设 async_serper_search 支持仅传 query
            # search_result = await FileUtils.async_google_search(query) 
            # 暂时复用现有逻辑，如果没有图片搜图功能，这里可以返回提示
            search_start_time = time.perf_counter()
            search_result = f"收到纯文本搜索请求: {query}。当前底层引擎暂仅支持'以图搜图'，请提供相关截图。"
            search_elapsed_time = time.perf_counter() - search_start_time
            print(f"⏱️ 纯文本搜索耗时: {search_elapsed_time:.2f} 秒")

        return {
            "status": "success",
            "search_findings": search_result
        }

    def _get_args_schema(self) -> Dict:
        return {
            "query": {"type": "string", "description": "搜索关键词，例如人物姓名、事件描述"},
            "image_path": {"type": "string", "description": "需要搜索的图片本地路径（用于以图搜图）"},
            "image_url": {"type": "string", "description": "已上传的图片 URL（可选，优先使用）"}
        }

    def _get_required_args(self) -> List[str]:
        # 两个参数选其一，但在 Schema 定义中通常不方便表达 "OR" 逻辑，
        # 所以这里把它们都设为可选，但在 run 方法里做校验
        return ["query", "image_path", "image_url"]