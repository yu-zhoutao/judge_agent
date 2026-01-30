import os
import time
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# --- 修改引用路径为 judge_agent ---
from judge_agent.config import Config
from judge_agent.utils.file_utils import FileUtils
import logging
from judge_agent.utils.sse_utils import SSEUtils, CacheSSEUtils
from judge_agent.utils.sse_cache import MongoSSECache
from judge_agent.engines.langchain_model import build_chat_model
from judge_agent.agent import build_agent, build_initial_state
from judge_agent.agent.prompts import SYSTEM_PROMPT_LC

from judge_agent.tools.langchain_tools import (
    visual_prepare_frames,
    visual_face_check,
    visual_behavior_check,
    visual_ocr_check,
    visual_render_marks,
    audio_transcribe,
    web_search,
)

# 初始化 FastAPI 应用
logger = logging.getLogger("judge_agent")

app = FastAPI(
    title="JianceAI Audit Agent",
    description="基于 ReAct 架构的多模态内容安全审核智能体",
    version="3.0.0" # Agent 版本
)


# --- 中间件配置 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 静态资源挂载 ---
# 确保临时目录存在，用于访问切片证据或临时图片
if not os.path.exists(Config.FIXED_TEMP_DIR):
    os.makedirs(Config.FIXED_TEMP_DIR)
app.mount("/static_temp", StaticFiles(directory=Config.FIXED_TEMP_DIR), name="static_temp")


# --- 路由定义 ---

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy", "mode": "Agent", "timestamp": time.time()}

@app.post("/analyze")
async def analyze_media(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    enable_search: bool = Form(True),
    enable_cache: bool = Form(True)
):
    """
    智能体审核主接口 (SSE 流式响应)
    :param enable_cache: 是否启用 SSE 事件缓存到 MongoDB（默认 True）
    """
    # 1. 文件预处理：保存到临时目录并识别类型，同时上传到 MinIO
    try:
        file_path, minio_url = FileUtils.save_upload_file(file)
        file_type = FileUtils.detect_file_type(file.filename)
        
        # 记录 MinIO URL（可用于后续访问或存储）
        if minio_url:
            print(f"📦 文件已存储到 MinIO: {minio_url}")
    except Exception as e:
        # 如果文件保存就失败了，直接返回错误流
        async def error_handler():
            yield SSEUtils.error(f"文件接收失败: {str(e)}")
        return StreamingResponse(error_handler(), media_type="text/event-stream")

    # 2. 组装 Agent 的工具箱 (Toolkit)
    tools = [
        visual_prepare_frames,
        visual_face_check,
        visual_behavior_check,
        visual_ocr_check,
        visual_render_marks,
        audio_transcribe,
    ]
    if enable_search:
        tools.append(web_search)

    # 3. 初始化 LangGraph 智能体
    model = build_chat_model()
    langgraph_agent = build_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT_LC,
    )

    # 4. 定义流式生成器
    async def stream_factory():
        if enable_cache:
            memory = MongoSSECache(file_path, file_type, minio_url)
            sse = CacheSSEUtils(memory)
        else:
            sse = SSEUtils

        try:
            initial_messages = [
                {"role": "user", "content": f"请开始审核该文件。文件路径: {file_path}, 类型: {file_type}"}
            ]
            state = build_initial_state(
                file_path=file_path,
                file_type=file_type,
                s3_url=minio_url,
                messages=initial_messages,
                remaining_steps=10,
            )

            yield sse.log("🤖 LangGraph 智能体启动，正在流式推理...")

            async for event in langgraph_agent.astream_events(state, version="v2"):
                for sse_event in sse.format_langgraph_event(event):
                    yield sse_event
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield SSEUtils.error(f"智能体运行异常: {str(e)}")
        finally:
            # 可以在这里做一些针对本次请求的立即清理工作（可选）
            pass

    # 5. 注册背景任务：定时清理过期临时文件
    # 不会阻塞当前的 SSE 响应
    background_tasks.add_task(FileUtils.clear_temp_dir, age_seconds=3600)

    return StreamingResponse(
        stream_factory(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no" # 禁用 Nginx 缓存，确保 Agent 的思考过程实时展示
        }
    )

# --- 启动配置 (调试用) ---
if __name__ == "__main__":
    import uvicorn
    # 启动命令示例: uvicorn judge_agent.main:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8001)
