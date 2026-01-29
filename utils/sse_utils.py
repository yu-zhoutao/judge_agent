import asyncio
import json
import time
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from judge_agent.agent.core import MongoAgentMemory

class SSEUtils:
    """标准服务器发送事件 (Server-Sent Events) 格式工具"""

    @staticmethod
    def format_event(event_type: str, content: Any) -> str:
        """
        将数据包装成标准的 SSE 格式字符串
        :param event_type: 事件类型 (如 'log', 'images', 'token', 'violation_data' 等)
        :param content: 发送的内容 (字符串、列表或字典)
        """
        data = {
            "type": event_type,
            "content": content
        }
        # ensure_ascii=False 确保中文不被转义，方便前端展示
        json_str = json.dumps(data, ensure_ascii=False)
        return f"data: {json_str}\n\n"

    @staticmethod
    def log(message: str, start_time: Optional[float] = None) -> str:
        """
        生成带有时间戳的日志事件
        :param message: 日志消息内容
        :param start_time: 任务开始的时间戳，如果不传则不计算耗时
        """
        if start_time is not None:
            elapsed = time.time() - start_time
            message = f"[{elapsed:.1f}s] {message}"
        
        return SSEUtils.format_event("log", message)

    @staticmethod
    def error(message: str) -> str:
        """生成错误事件"""
        return SSEUtils.format_event("error", message)

    @staticmethod
    def token(content: str) -> str:
        """生成大模型流式输出的 Token 事件"""
        return SSEUtils.format_event("token", content)

    @staticmethod
    def images(image_list: list) -> str:
        """生成图片列表事件 (Base64 列表)"""
        return SSEUtils.format_event("images", image_list)

    @staticmethod
    def violation(data: dict) -> str:
        """生成违规研判数据事件"""
        return SSEUtils.format_event("violation_data", data)


class CacheSSEUtils:
    """
    带缓存功能的 SSE 工具类
    在生成 SSE 事件的同时，自动将事件缓存到 MongoDB 的 client_history 中
    注意：所有方法都是同步的，内部使用 asyncio.create_task 异步缓存
    """
    
    def __init__(self, memory: 'MongoAgentMemory'):
        """
        初始化缓存 SSE 工具
        :param memory: MongoAgentMemory 实例，用于缓存对话历史
        """
        self.memory = memory
    
    def _cache_async(self, event_type: str, content: Any):
        """
        异步缓存到 MongoDB（不阻塞主流程）
        """
        async def _do_cache():
            try:
                await self.memory.add_client_history(event_type, content)
            except Exception as e:
                # 缓存失败不影响 SSE 事件生成
                print(f"⚠️ 缓存 SSE 事件失败: {str(e)}")
        
        # 创建后台任务，不等待完成
        asyncio.create_task(_do_cache())
    
    def format_event(self, event_type: str, content: Any) -> str:
        """
        将数据包装成标准的 SSE 格式字符串，并异步缓存到 MongoDB
        :param event_type: 事件类型
        :param content: 发送的内容
        """
        # 生成 SSE 格式字符串
        sse_str = SSEUtils.format_event(event_type, content)
        
        # 异步缓存到 MongoDB（不阻塞）
        self._cache_async(event_type, content)
        
        return sse_str
    
    def log(self, message: str, start_time: Optional[float] = None) -> str:
        """
        生成带有时间戳的日志事件，并异步缓存到 MongoDB
        :param message: 日志消息内容
        :param start_time: 任务开始的时间戳
        """
        if start_time is not None:
            elapsed = time.time() - start_time
            message = f"[{elapsed:.1f}s] {message}"
        
        return self.format_event("log", message)
    
    def error(self, message: str) -> str:
        """生成错误事件，并异步缓存到 MongoDB"""
        return self.format_event("error", message)
    
    def token(self, content: str) -> str:
        """生成大模型流式输出的 Token 事件，并异步缓存到 MongoDB"""
        return self.format_event("token", content)
    
    def images(self, image_list: list) -> str:
        """生成图片列表事件，并异步缓存到 MongoDB"""
        return self.format_event("images", image_list)
    
    def violation(self, data: dict) -> str:
        """生成违规研判数据事件，并异步缓存到 MongoDB"""
        return self.format_event("violation_data", data)