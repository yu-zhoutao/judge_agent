# app/tools/base.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List

class BaseTool(ABC):
    """所有 Agent 工具的基类"""
    name: str = ""
    description: str = ""

    @abstractmethod
    async def run(self, **kwargs) -> Dict[str, Any]:
        """
        执行工具的具体逻辑
        :return: 必须返回一个字典，便于序列化反馈给 LLM
        """
        pass

    def to_schema(self) -> Dict:
        """
        生成 OpenAI Function Calling 格式的 JSON Schema
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self._get_args_schema(),
                    "required": self._get_required_args()
                }
            }
        }

    @abstractmethod
    def _get_args_schema(self) -> Dict:
        """定义工具参数的结构 (JSON Schema properties)"""
        pass

    @abstractmethod
    def _get_required_args(self) -> List[str]:
        """定义哪些参数是必填的"""
        pass