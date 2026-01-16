# judge_agent/agent/core.py

import json
import uuid
import asyncio
import re
from typing import List, AsyncGenerator, Dict, Any

from judge_agent.config import Config
from judge_agent.engines.llm_client import LLMClient
from judge_agent.utils.sse_utils import SSEUtils
from judge_agent.tools.base import BaseTool
from judge_agent.agent.prompts import SYSTEM_PROMPT
from judge_agent.schemas import Evidence


class AgentMemory:
    """智能体记忆管理类"""
    def __init__(self, file_path: str, file_type: str):
        self.messages = []
        self.file_path = file_path
        self.file_type = file_type
        self._finished = False
        self._final_content = ""
    
    def add_message(self, role: str, content: str = None, tool_calls=None, tool_call_id: str = None):
        """添加消息到记忆"""
        msg = {"role": role}
        if content is not None:
            msg["content"] = content
        if tool_calls is not None:
            msg["tool_calls"] = tool_calls
        if tool_call_id is not None:
            msg["tool_call_id"] = tool_call_id
            
        self.messages.append(msg)
    
    def get_messages(self) -> List[Dict[str, Any]]:
        """获取记忆中的所有消息"""
        return self.messages
    
    def mark_finished(self, content: str):
        """标记任务完成"""
        self._finished = True
        self._final_content = content
    
    def is_finished(self) -> bool:
        """检查是否已完成"""
        return self._finished
    
    def get_final_content(self) -> str:
        """获取最终内容"""
        return self._final_content

class AuditAgent:
    def __init__(self, tools: List[BaseTool]):
        # 注册工具箱
        self.tools_map = {t.name: t for t in tools}
        self.tools_schemas = [t.to_schema() for t in tools]
        
        # 获取 LLM 客户端
        self.client = LLMClient.get_async_client()
        self.model_name = Config.MODEL_NAME

    async def execute(self, file_path: str, file_type: str) -> AsyncGenerator[str, None]:
        """
        Agent 主执行循环
        """
        # 1. 初始化记忆
        memory = AgentMemory(file_path, file_type)
        
        # 2. 设置 System Prompt
        memory.add_message("system", SYSTEM_PROMPT)
        memory.add_message("user", f"请开始审核该文件。文件路径: {file_path}, 类型: {file_type}")

        yield SSEUtils.log(f"🤖 智能体启动，正在加载工具箱 ({len(self.tools_map)}个工具)...")

        # 3. 思考-行动循环 (最大 10 步，防止死循环)
        max_steps = 10
        step_count = 0

        while not memory.is_finished() and step_count < max_steps:
            step_count += 1
            yield SSEUtils.log(f"🤔 智能体正在进行第 {step_count} 轮思考...", start_time=None)

            try:
                # --- [A] 调用 LLM 进行决策 ---
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=memory.get_messages(),
                    tools=self.tools_schemas,
                    tool_choice="auto", 
                    temperature=0.1,    # 降低随机性
                )
                
                ai_message = response.choices[0].message
                
                # 将 AI 的回复（包含思考或工具调用）加入记忆
                memory.add_message(
                    role="assistant", 
                    content=ai_message.content, 
                    tool_calls=ai_message.tool_calls
                )
                
                # --- [B] 分支 1：模型决定调用工具 (并行执行优化版) ---
                if ai_message.tool_calls:
                    # 记录思考过程（如果有）
                    if ai_message.content:
                        yield SSEUtils.token(f"\n> **思考**: {ai_message.content}\n\n")

                    # 1. 准备任务列表
                    tasks = []
                    tool_call_meta = [] # 存储对应的 tool_call 信息，用于后续匹配结果

                    yield SSEUtils.log(f"⚡️ 启动并行执行: 将同时运行 {len(ai_message.tool_calls)} 个工具任务...")

                    for tool_call in ai_message.tool_calls:
                        fn_name = tool_call.function.name
                        fn_args_str = tool_call.function.arguments
                        
                        # 解析参数
                        try:
                            fn_args = json.loads(fn_args_str)
                        except:
                            try:
                                import ast
                                fn_args = ast.literal_eval(fn_args_str)
                            except:
                                yield SSEUtils.error(f"❌ 参数解析失败: {fn_args_str}")
                                continue
                                
                        if fn_name == "web_search":
                            # 提取 query 或 image_path 简写
                            q = fn_args.get('query', '无词')
                            img = "有图" if fn_args.get('image_path') else "无图"
                            log_msg = f"🚀 [启动] 搜索: {q} ({img})"
                        else:
                            log_msg = f"🚀 [启动] 工具: {fn_name}"
                        yield SSEUtils.log(log_msg)
                        
                        if fn_name in self.tools_map:
                            tool_instance = self.tools_map[fn_name]
                            # 创建协程任务，但不立即 await
                            tasks.append(tool_instance.run(**fn_args))
                            tool_call_meta.append({
                                "valid": True,
                                "tool_call": tool_call,
                                "name": fn_name
                            })
                        else:
                            # 占位，防止索引错位
                            tasks.append(None)
                            tool_call_meta.append({
                                "valid": False,
                                "tool_call": tool_call,
                                "name": fn_name,
                                "error": f"Tool {fn_name} not found"
                            })

                    # 2. 并行执行所有任务 (Gather)
                    # return_exceptions=True 确保一个工具报错不会炸掉所有工具
                    if tasks:
                        # 过滤掉无效任务(None)进行gather，或者手动处理
                        valid_coroutines = [t for t in tasks if t is not None]
                        
                        # === 核心：并行等待 ===
                        results = await asyncio.gather(*valid_coroutines, return_exceptions=True)
                        
                        # 将结果映射回 tool_call_meta
                        result_iter = iter(results)
                        
                        # 3. 处理结果并推送 SSE
                        for meta in tool_call_meta:
                            tool_call = meta["tool_call"]
                            fn_name = meta["name"]
                            
                            if not meta["valid"]:
                                tool_result_str = json.dumps({"error": meta["error"]})
                                yield SSEUtils.error(f"❌ {meta['error']}")
                            else:
                                # 获取 gather 的结果
                                res = next(result_iter)
                                
                                if isinstance(res, Exception):
                                    # 工具内部报错
                                    tool_result_str = json.dumps({"error": str(res)})
                                    yield SSEUtils.error(f"❌ 工具 {fn_name} 执行异常: {str(res)}")
                                else:
                                    # 工具执行成功，res 是 result_dict
                                    result_dict = res
                                    
                                    # ----------------- 前端交互适配层 -----------------
                                    # 1. 图片预览
                                    if "preview_images" in result_dict:
                                        yield SSEUtils.images(result_dict["preview_images"])
                                        del result_dict["preview_images"]

                                    # 2. 音频文本 - 实现流式输出
                                    if "corrected_text" in result_dict:
                                        text = result_dict["corrected_text"]
                                        yield SSEUtils.format_event("audio_text_start", "")
                                        # 将文本按字符流式发送
                                        for char in text:
                                            yield SSEUtils.format_event("audio_text_chunk", char)
                                            await asyncio.sleep(0.005)  # 短暂暂停，模拟自然流式效果

                                    # 3. 违规证据
                                    if "violation_check" in result_dict:
                                        v_data = result_dict["violation_check"]
                                        if v_data.get("is_violation"):
                                            frontend_data = {
                                                "is_violation": True,
                                                "time_anchors": v_data.get("segments", [])
                                            }
                                            yield SSEUtils.violation(frontend_data)
                                    # ---------------------------------------------------

                                    tool_result_str = json.dumps(result_dict, ensure_ascii=False)
                                    yield SSEUtils.log(f"✅ [完成] 工具 {fn_name}")

                            # 4. 写入记忆 (Memory)
                            memory.add_message(
                                role="tool",
                                content=tool_result_str,
                                tool_call_id=tool_call.id
                            )

                # --- [C] 分支 2：模型没有调用工具，给出了最终回答 ---
                else:
                    final_content = ai_message.content or ""
                    memory.mark_finished(final_content)
                    
                    yield SSEUtils.log("📝 智能体已完成研判，正在生成最终报告...")

                    # 流式输出最终报告
                    if final_content:
                        yield SSEUtils.format_event("final_report_start", "")  # 添加开始事件
                        for char in final_content:
                            yield SSEUtils.token(char)
                            await asyncio.sleep(0.005)  # 短暂暂停，模拟流式效果
                        yield SSEUtils.format_event("final_report_end", "")  # 添加结束事件

            except Exception as e:
                import traceback
                traceback.print_exc()
                yield SSEUtils.error(f"智能体运行发生致命错误: {str(e)}")
                break
        
        if step_count >= max_steps:
            yield SSEUtils.error("⚠️ 审核任务过于复杂，已达到最大推理步数，强制结束。")