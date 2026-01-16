import json
import time
from typing import Any, Optional

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