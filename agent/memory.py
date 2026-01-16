# app/agent/memory.py

from typing import List, Dict, Any, Optional
from judge_agent.schemas import AgentState, Evidence

class AgentMemory:
    """
    Agent 的记忆管理器：
    1. 维护与 LLM 的对话历史 (Messages)
    2. 维护当前的审核状态 (AgentState)
    """
    
    def __init__(self, file_path: str, file_type: str):
        self.state = AgentState(
            file_path=file_path, 
            file_type=file_type
        )
        self.messages: List[Dict[str, Any]] = []

    def add_message(self, role: str, content: str, tool_calls: Optional[List] = None, tool_call_id: Optional[str] = None):
        """添加一条对话记录"""
        msg = {"role": role, "content": content}
        
        # 处理 Function Calling 的特殊字段
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if tool_call_id:
            msg["tool_call_id"] = tool_call_id
            
        self.messages.append(msg)

    def add_evidence(self, evidence: Evidence):
        """添加一条确凿的证据"""
        self.state.evidences.append(evidence)

    def add_thought(self, thought: str):
        """记录 Agent 的思考过程"""
        self.state.thoughts.append(thought)

    def get_messages(self) -> List[Dict[str, Any]]:
        """获取当前完整的对话上下文"""
        return self.messages

    def mark_finished(self, final_conclusion: str):
        """标记任务结束"""
        self.state.finished = True
        self.state.final_conclusion = final_conclusion
    
    def is_finished(self) -> bool:
        return self.state.finished