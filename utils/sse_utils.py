import asyncio
import json
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from judge_agent.utils.sse_cache import MongoSSECache

class SSEUtils:
    """标准服务器发送事件 (Server-Sent Events) 格式工具"""

    @staticmethod
    def _extract_text_from_chunk(chunk: Any) -> str:
        if chunk is None:
            return ""
        if isinstance(chunk, str):
            return chunk
        content = getattr(chunk, "content", None)
        if content is None and isinstance(chunk, dict):
            content = chunk.get("content") or chunk.get("text") or chunk.get("delta")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "".join(parts)
        return ""

    @staticmethod
    def _extract_tool_outputs(output: Any) -> List[Dict[str, Any]]:
        if isinstance(output, dict):
            return [output]

        update = None
        if hasattr(output, "update"):
            update = getattr(output, "update")
        elif isinstance(output, dict):
            update = output.get("update")

        if not isinstance(update, dict):
            return []

        messages = update.get("messages") or []
        outputs: List[Dict[str, Any]] = []
        for msg in messages:
            content = getattr(msg, "content", None)
            if content is None and isinstance(msg, dict):
                content = msg.get("content")
            if isinstance(content, dict):
                outputs.append(content)
                continue
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                except Exception:
                    continue
                if isinstance(parsed, dict):
                    outputs.append(parsed)
        return outputs

    @staticmethod
    def _tool_output_to_payloads(output: Any) -> List[Dict[str, Any]]:
        payloads: List[Dict[str, Any]] = []
        outputs = SSEUtils._extract_tool_outputs(output)
        if not outputs:
            return payloads

        for out in outputs:
            payloads.extend(SSEUtils._tool_output_dict_to_payloads(out))
        return payloads

    @staticmethod
    def _tool_output_dict_to_payloads(output: Dict[str, Any]) -> List[Dict[str, Any]]:
        payloads: List[Dict[str, Any]] = []

        if "preview_images" in output:
            payloads.append({"type": "images", "content": output.get("preview_images")})

        if "frame_count" in output:
            frame_count = output.get("frame_count")
            minio_ready = output.get("minio_ready")
            if minio_ready is not None:
                payloads.append({
                    "type": "log",
                    "content": f"帧准备完成: {frame_count} 帧，已上传 {minio_ready} 帧",
                })
            else:
                payloads.append({
                    "type": "log",
                    "content": f"帧准备完成: {frame_count} 帧",
                })

        if "corrected_text" in output:
            payloads.append({"type": "audio_text_start", "content": ""})
            for char in output.get("corrected_text", ""):
                payloads.append({"type": "audio_text_chunk", "content": char})

        if "violation_check" in output:
            v_data = output.get("violation_check", {})
            if isinstance(v_data, dict) and v_data.get("is_violation"):
                payloads.append(
                    {
                        "type": "violation_data",
                        "content": {
                            "is_violation": True,
                            "time_anchors": v_data.get("segments", []),
                        },
                    }
                )

        return payloads

    @staticmethod
    def langgraph_event_to_payloads(event: Dict[str, Any], include_tool_payloads: bool = True) -> List[Dict[str, Any]]:
        payloads: List[Dict[str, Any]] = []
        if not event:
            return payloads

        kind = event.get("event")
        name = event.get("name") or "unknown"
        data = event.get("data") or {}

        if kind in ("on_chat_model_stream", "on_llm_stream"):
            chunk = data.get("chunk") if isinstance(data, dict) else None
            token = SSEUtils._extract_text_from_chunk(chunk)
            if token:
                payloads.append({"type": "token", "content": token})
            return payloads

        if kind == "on_chat_model_start":
            payloads.append({"type": "log", "content": f"chat_model_start:{name}"})
            return payloads

        if kind == "on_chat_model_end":
            payloads.append({"type": "log", "content": f"chat_model_end:{name}"})
            return payloads

        if kind == "on_chain_start":
            payloads.append({"type": "log", "content": f"chain_start:{name}"})
            return payloads

        if kind == "on_chain_end":
            payloads.append({"type": "log", "content": f"chain_end:{name}"})
            return payloads

        if kind == "on_tool_start":
            payloads.append({"type": "log", "content": f"tool_start:{name}"})
            return payloads

        if kind == "on_tool_end":
            payloads.append({"type": "log", "content": f"tool_end:{name}"})
            if include_tool_payloads and isinstance(data, dict):
                payloads.extend(SSEUtils._tool_output_to_payloads(data.get("output")))
            return payloads

        return payloads

    @staticmethod
    def format_langgraph_event(event: Dict[str, Any], include_tool_payloads: bool = True) -> List[str]:
        payloads = SSEUtils.langgraph_event_to_payloads(event, include_tool_payloads=include_tool_payloads)
        return [SSEUtils.format_event(p["type"], p["content"]) for p in payloads]

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
    
    def __init__(self, memory: 'MongoSSECache'):
        """
        初始化缓存 SSE 工具
        :param memory: MongoSSECache 实例，用于缓存对话历史
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

    def format_langgraph_event(self, event: Dict[str, Any], include_tool_payloads: bool = True) -> List[str]:
        payloads = SSEUtils.langgraph_event_to_payloads(event, include_tool_payloads=include_tool_payloads)
        return [self.format_event(p["type"], p["content"]) for p in payloads]
