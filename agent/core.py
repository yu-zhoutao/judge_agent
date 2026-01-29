# judge_agent/agent/core.py

import json
import uuid
import asyncio
import re
from typing import List, AsyncGenerator, Dict, Any, Optional
from datetime import datetime
from bson import ObjectId

from judge_agent.config import Config
from judge_agent.engines.llm_client import LLMClient
from judge_agent.utils.sse_utils import SSEUtils, CacheSSEUtils
from judge_agent.tools.base import BaseTool
from judge_agent.agent.prompts import SYSTEM_PROMPT
from judge_agent.schemas import Evidence
from judge_agent.utils.mongo_utils import AsyncMongoUtils


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


class MongoAgentMemory:
    """智能体记忆管理类 - MongoDB版本"""

    # 集合名称
    COLLECTION_NAME = "agent_memories"

    def __init__(self, file_path: str, file_type: str, s3_url: str = "", memory_id: Optional[str] = None):
        self.file_path = file_path
        self.file_type = file_type
        self.s3_url = s3_url
        self._finished = False
        self._final_content = ""
        self._memory_id = memory_id
        self._messages: List[Dict[str, Any]] = []
        self._mongo = AsyncMongoUtils()

    async def _initialize_memory(self):
        """初始化MongoDB中的记忆记录"""
        if self._memory_id is None:
            memory_doc = {
                "file_path": self.file_path,
                "file_type": self.file_type,
                "s3_url": self.s3_url,
                "file_id": self.s3_url.split('/')[-1].split('.')[0],
                "messages": [],
                "client_history": [],
                "finished": False,
                "final_content": "",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            self._memory_id = await self._mongo.insert_one(self.COLLECTION_NAME, memory_doc)

    async def add_message(self, role: str, content: str = None, tool_calls=None, tool_call_id: str = None):
        """添加消息到记忆"""
        # 确保记忆已初始化
        await self._initialize_memory()

        msg = {"role": role}
        if content is not None:
            msg["content"] = content
        if tool_calls is not None:
            # 检查 tool_calls 是否是对象列表，如果是，则转为字典
            # OpenAI 的 tool_calls 通常是一个列表
            serialized_tool_calls = []
            for tc in tool_calls:
                if hasattr(tc, 'model_dump'):
                    # Pydantic v2 / OpenAI SDK v1+ 标准方法
                    serialized_tool_calls.append(tc.model_dump())
                elif hasattr(tc, 'dict'):
                    # Pydantic v1 旧方法
                    serialized_tool_calls.append(tc.dict())
                elif isinstance(tc, dict):
                    # 如果已经是字典了，直接用
                    serialized_tool_calls.append(tc)
                else:
                    # 兜底：尝试转 dict 或者 str，防止报错
                    try:
                        serialized_tool_calls.append(dict(tc))
                    except:
                        serialized_tool_calls.append(str(tc))

            msg["tool_calls"] = serialized_tool_calls

        if tool_call_id is not None:
            msg["tool_call_id"] = tool_call_id

        # 添加到内存缓存
        self._messages.append(msg)

        # 更新MongoDB
        print(f"更新数据 - {self._memory_id} - {msg}")
        await self._mongo.update_one(
            self.COLLECTION_NAME,
            {"_id": ObjectId(self._memory_id)},
            {"$push": {"messages": msg}}
        )

    async def add_client_history(self, type: str, content: str = "",):
        """添加客户端SSE会话响应历史"""
        # 确保记忆已初始化
        await self._initialize_memory()

        msg = {"type": type}
        if content is not None:
            msg["content"] = content

        # 更新MongoDB
        # print(f"更新数据 - {self._memory_id} - {msg}")
        await self._mongo.update_one(
            self.COLLECTION_NAME,
            {"_id": ObjectId(self._memory_id)},
            {"$push": {"client_history": msg}}
        )
    async def get_messages(self) -> List[Dict[str, Any]]:
        """获取记忆中的所有消息"""
        # 优先从内存缓存返回
        if self._messages:
            return self._messages

        # 如果内存缓存为空，从MongoDB加载
        if self._memory_id:
            doc = await self._mongo.find_one(
                self.COLLECTION_NAME,
                {"_id": self._memory_id}
            )
            if doc:
                self._messages = doc.get("messages", [])
                return self._messages

        return []

    async def mark_finished(self, content: str):
        """标记任务完成"""
        await self._initialize_memory()

        self._finished = True
        self._final_content = content

        # 更新MongoDB
        await self._mongo.update_one(
            self.COLLECTION_NAME,
            {"_id": ObjectId(self._memory_id)},
            {
                "$set": {
                    "finished": True,
                    "final_content": content,
                    "updated_at": datetime.utcnow()
                }
            }
        )

    def is_finished(self) -> bool:
        """检查是否已完成"""
        return self._finished

    def get_final_content(self) -> str:
        """获取最终内容"""
        return self._final_content

    async def get_memory_id(self) -> Optional[str]:
        """获取记忆ID"""
        return self._memory_id

    async def load_from_mongo(self, memory_id: str):
        """从MongoDB加载已有的记忆"""
        doc = await self._mongo.find_one(
            self.COLLECTION_NAME,
            {"_id": ObjectId(memory_id)}
        )

        if doc:
            self._memory_id = memory_id
            self.file_path = doc.get("file_path", "")
            self.file_type = doc.get("file_type", "")
            self._messages = doc.get("messages", [])
            self._finished = doc.get("finished", False)
            self._final_content = doc.get("final_content", "")
            return True
        return False

    async def delete_memory(self):
        """删除当前记忆"""
        if self._memory_id:
            await self._mongo.delete_one(
                self.COLLECTION_NAME,
                {"_id": ObjectId(self._memory_id)}
            )
            self._memory_id = None
            self._messages = []
            self._finished = False
            self._final_content = ""

    @staticmethod
    async def find_finished_memory_by_file_id(file_id: str) -> Optional[Dict[str, Any]]:
        """
        根据 file_id 查询已完成的最新记录
        :param file_id: 文件 ID（从 s3_url 提取的 MD5）
        :return: 包含 client_history 的文档，如果不存在则返回 None
        """
        mongo = AsyncMongoUtils()
        
        # 查询条件：file_id 匹配且 finished 为 true
        query = {
            "file_id": file_id,
            "finished": True
        }
        
        # 按 created_at 降序排序，获取最新的记录
        sort = [("created_at", -1)]
        
        try:
            # 使用 find_many 并限制返回 1 条记录
            docs = await mongo.find_many(
                MongoAgentMemory.COLLECTION_NAME,
                query,
                sort=sort,
                limit=1
            )
            
            if docs:
                doc = docs[0]
                print(f"✅ 找到已完成的记录: {doc.get('_id')}")
                return doc
            else:
                print(f"ℹ️ 未找到 file_id={file_id} 的已完成记录")
                return None
        except Exception as e:
            print(f"❌ 查询已完成记录失败: {str(e)}")
            return None

class AuditAgent:
    def __init__(self, tools: List[BaseTool]):
        # 注册工具箱
        self.tools_map = {t.name: t for t in tools}
        self.tools_schemas = [t.to_schema() for t in tools]
        
        # 获取 LLM 客户端
        self.client = LLMClient.get_async_client()
        self.model_name = Config.MODEL_NAME

    async def execute(self, file_path: str, file_type: str, s3_url: str = "", enable_cache: bool = False) -> AsyncGenerator[str, None]:
        """
        Agent 主执行循环
        :param file_path: 文件路径
        :param file_type: 文件类型
        :param s3_url: MinIO/S3 文件 URL（可选）
        :param enable_cache: 是否启用 SSE 事件缓存到 MongoDB（默认 False）
        """
        # 0. 检查是否有已完成的记录（从 s3_url 提取 file_id）
        if False:
            try:
                # 从 s3_url 提取 file_id（MD5）
                file_id = s3_url.split('/')[-1].split('.')[0]
                finished_doc = await MongoAgentMemory.find_finished_memory_by_file_id(file_id)
                
                if finished_doc:
                    # 找到已完成的记录，重放 client_history
                    client_history = finished_doc.get("client_history", [])
                    print(f"📜 重放 {len(client_history)} 条历史记录...")
                    
                    for history_item in client_history:
                        event_type = history_item.get("type")
                        content = history_item.get("content", "")
                        
                        # 使用 SSEUtils.format_event 重新生成 SSE 事件
                        yield SSEUtils.format_event(event_type, content)
                        await asyncio.sleep(0.035)
                    
                    # 重放完成后直接返回，不再执行新的审核
                    return
            except Exception as e:
                print(f"⚠️ 查询历史记录失败，继续执行新审核: {str(e)}")
        
        # 1. 初始化记忆
        # memory = AgentMemory(file_path, file_type)
        memory = MongoAgentMemory(file_path, file_type, s3_url)  # 数据落盘

        # 2. 设置 System Prompt
        await memory.add_message("system", SYSTEM_PROMPT)
        await memory.add_message("user", f"请开始审核该文件。文件路径: {file_path}, 类型: {file_type}")

        # 3. 选择 SSE 工具类
        if enable_cache:
            sse = CacheSSEUtils(memory)
        else:
            sse = SSEUtils

        yield sse.log(f"🤖 智能体启动，正在加载工具箱 ({len(self.tools_map)}个工具)...")

        # 3. 思考-行动循环 (最大 10 步，防止死循环)
        max_steps = 10
        step_count = 0

        while not memory.is_finished() and step_count < max_steps:
            step_count += 1
            yield sse.log(f"🤔 智能体正在进行第 {step_count} 轮思考...", start_time=None)

            try:
                current_messages = await memory.get_messages()
                # --- [A] 调用 LLM 进行决策 ---
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=current_messages,
                    tools=self.tools_schemas,
                    tool_choice="auto", 
                    temperature=0.1,    # 降低随机性
                )
                
                ai_message = response.choices[0].message
                
                # 将 AI 的回复（包含思考或工具调用）加入记忆
                await memory.add_message(
                    role="assistant", 
                    content=ai_message.content, 
                    tool_calls=ai_message.tool_calls
                )
                
                # --- [B] 分支 1：模型决定调用工具 (并行执行优化版) ---
                if ai_message.tool_calls:
                    # 记录思考过程（如果有）
                    if ai_message.content:
                        yield sse.token(f"\n> **思考**: {ai_message.content}\n\n")

                    # 1. 准备任务列表
                    tasks = []
                    tool_call_meta = [] # 存储对应的 tool_call 信息，用于后续匹配结果

                    yield sse.log(f"⚡️ 启动并行执行: 将同时运行 {len(ai_message.tool_calls)} 个工具任务...")

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
                                yield sse.error(f"❌ 参数解析失败: {fn_args_str}")
                                continue
                                
                        if fn_name == "web_search":
                            # 提取 query 或 image_path 简写
                            q = fn_args.get('query', '无词')
                            img = "有图" if fn_args.get('image_path') else "无图"
                            log_msg = f"🚀 [启动] 搜索: {q} ({img})"
                        else:
                            log_msg = f"🚀 [启动] 工具: {fn_name}"
                        yield sse.log(log_msg)
                        
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
                                yield sse.error(f"❌ {meta['error']}")
                            else:
                                # 获取 gather 的结果
                                res = next(result_iter)
                                
                                if isinstance(res, Exception):
                                    # 工具内部报错
                                    tool_result_str = json.dumps({"error": str(res)})
                                    yield sse.error(f"❌ 工具 {fn_name} 执行异常: {str(res)}")
                                else:
                                    # 工具执行成功，res 是 result_dict
                                    result_dict = res
                                    
                                    # ----------------- 前端交互适配层 -----------------
                                    # 1. 图片预览
                                    if "preview_images" in result_dict:
                                        yield sse.images(result_dict["preview_images"])
                                        del result_dict["preview_images"]

                                    # 2. 音频文本 - 实现流式输出
                                    if "corrected_text" in result_dict:
                                        text = result_dict["corrected_text"]
                                        yield sse.format_event("audio_text_start", "")
                                        # 将文本按字符流式发送
                                        for char in text:
                                            yield sse.format_event("audio_text_chunk", char)
                                            await asyncio.sleep(0.005)  # 短暂暂停，模拟自然流式效果

                                    # 3. 违规证据
                                    if "violation_check" in result_dict:
                                        v_data = result_dict["violation_check"]
                                        if v_data.get("is_violation"):
                                            frontend_data = {
                                                "is_violation": True,
                                                "time_anchors": v_data.get("segments", [])
                                            }
                                            yield sse.violation(frontend_data)
                                    # ---------------------------------------------------

                                    tool_result_str = json.dumps(result_dict, ensure_ascii=False)
                                    yield sse.log(f"✅ [完成] 工具 {fn_name}")

                            # 4. 写入记忆 (Memory)
                            await memory.add_message(
                                role="tool",
                                content=tool_result_str,
                                tool_call_id=tool_call.id
                            )

                # --- [C] 分支 2：模型没有调用工具，给出了最终回答 ---
                else:
                    final_content = ai_message.content or ""
                    await memory.mark_finished(final_content)
                    
                    yield sse.log("📝 智能体已完成研判，正在生成最终报告...")

                    # 流式输出最终报告
                    if final_content:
                        yield sse.format_event("final_report_start", "")  # 添加开始事件
                        for char in final_content:
                            yield sse.token(char)
                            await asyncio.sleep(0.005)  # 短暂暂停，模拟流式效果
                        yield sse.format_event("final_report_end", "")  # 添加结束事件

            except Exception as e:
                import traceback
                traceback.print_exc()
                yield sse.error(f"智能体运行发生致命错误: {str(e)}")
                break
        
        if step_count >= max_steps:
            yield sse.error("⚠️ 审核任务过于复杂，已达到最大推理步数，强制结束。")
