from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal

# --- 基础证据模型 ---
class Evidence(BaseModel):
    source: Literal["visual", "audio", "ocr", "search"]
    content: str  # 描述发现了什么
    timestamp: Optional[str] = None # 时间戳范围
    confidence: float = 0.0
    images: List[str] = [] # 相关的 base64 图片或路径

# --- Agent 的状态 ---
class AgentState(BaseModel):
    file_path: str
    file_type: str
    thoughts: List[str] = [] # 思考过程记录
    evidences: List[Evidence] = [] # 收集到的证据
    finished: bool = False
    final_conclusion: Optional[str] = None