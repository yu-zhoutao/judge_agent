judge_agent/
├── .env                        # 环境变量配置
├── __init__.py                 # 模块初始化
├── config.py                   # 全局配置
├── schemas.py                  # Pydantic 数据模型定义
├── main.py                     # FastAPI 应用入口
├── agent/                      # 智能体核心逻辑
│   ├── __init__.py
│   ├── core.py                 # 智能体主循环逻辑
│   ├── memory.py               # 记忆管理（存储证据链）
│   └── prompts.py              # 系统提示词
├── tools/                      # 工具封装层
│   ├── __init__.py
│   ├── base.py                 # 工具基类
│   ├── visual_tools.py         # 视觉分析工具
│   ├── audio_tools.py          # 音频分析工具
│   └── search_tools.py         # 网络搜索工具
├── engines/                    # 底层引擎
│   ├── __init__.py
│   ├── face_engine.py          # InsightFace 引擎
│   ├── ocr_engine.py           # RapidOCR 引擎
│   ├── yolo_engine.py          # YOLO 引擎
│   ├── whisper_engine.py       # Whisper 引擎
│   └── llm_client.py           # LLM 客户端
├── utils/                      # 工具函数
│   ├── __init__.py
│   ├── file_utils.py
│   ├── image_utils.py
│   ├── json_utils.py
│   └── sse_utils.py
├── prompts/                    # 提示词模板
│   ├── __init__.py
│   └── templates.py
├── index.html               # 前端页面
└── readme.md                  # 项目文档