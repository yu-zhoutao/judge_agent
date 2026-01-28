import os
import time
import asyncio
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# --- 修改引用路径为 judge_agent ---
from judge_agent.config import Config
from judge_agent.utils.file_utils import FileUtils
from judge_agent.utils.sse_utils import SSEUtils

# --- 引入新的 Agent 和 Tools ---
from judge_agent.agent.core import AuditAgent
from judge_agent.tools.visual_tools import VisualScanTool
from judge_agent.tools.audio_tools import AudioTranscribeTool
from judge_agent.tools.search_tools import WebSearchTool

# 初始化 FastAPI 应用
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
    # 在这里，我们将具体的“能力”实例化
    tools = [
        VisualScanTool(),       # 视觉能力 (YOLO/Face/OCR)
        AudioTranscribeTool(),  # 听觉能力 (Whisper)
    ]
    
    # 根据用户选项决定是否给予 Agent 联网搜索能力
    if enable_search:
        tools.append(WebSearchTool())


    # 3. 初始化智能体
    agent = AuditAgent(tools=tools)

    # 4. 定义流式生成器
    async def stream_factory():
        try:
            # 启动 Agent 的思考与执行循环
            async for event in agent.execute(
                file_path=file_path,
                file_type=file_type,
                s3_url=minio_url,  # 缓存结果
                enable_cache=enable_cache  # sse响应缓存
            ):
                yield event
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