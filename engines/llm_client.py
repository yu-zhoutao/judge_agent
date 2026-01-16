import json
from typing import List, Dict, Any, AsyncGenerator, Optional
from openai import AsyncOpenAI, OpenAI
from judge_agent.config import Config
from judge_agent.utils.json_utils import JSONUtils

# 定义一些颜色代码，让控制台输出更清晰
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

class LLMClient:
    """大语言模型（LLM/VLM）通信引擎"""
    
    _async_client = None
    _sync_client = None

    @classmethod
    def get_async_client(cls) -> AsyncOpenAI:
        if cls._async_client is None:
            cls._async_client = AsyncOpenAI(
                api_key=Config.VLLM_API_KEY,
                base_url=Config.VLLM_API_URL
            )
        return cls._async_client

    @classmethod
    def get_sync_client(cls) -> OpenAI:
        if cls._sync_client is None:
            cls._sync_client = OpenAI(
                api_key=Config.VLLM_API_KEY,
                base_url=Config.VLLM_API_URL
            )
        return cls._sync_client

    # --- 新增：调试打印辅助函数 ---
    @staticmethod
    def _debug_print(title: str, content: Any, color: str = Colors.CYAN):
        """在控制台打印带边框的调试信息"""
        print(f"\n{color}{'='*20} [LLM {title}] {'='*20}{Colors.ENDC}")
        if isinstance(content, list):
            # 打印 Messages 列表
            for msg in content:
                role = msg.get('role', 'unknown')
                text = msg.get('content', '')
                # 如果 content 是列表（多模态图片），只打印文本部分，简化显示
                if isinstance(text, list):
                    text_part = [t for t in text if t.get('type') == 'text']
                    text = text_part[0]['text'] if text_part else "[Image Content]"
                
                print(f"{Colors.HEADER}{role.upper()}:{Colors.ENDC} {text}")
        else:
            # 打印普通字符串或字典
            print(f"{content}")
        print(f"{color}{'='*50}{Colors.ENDC}\n")

    @classmethod
    async def chat_stream(cls, messages: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
        """
        流式对话接口：用于最终研判结果的实时展示
        """
        # [Debug] 打印输入
        cls._debug_print("INPUT (Stream)", messages)

        client = cls.get_async_client()
        full_response = "" # 用于收集完整响应以便打印

        try:
            stream = await client.chat.completions.create(
                model=Config.MODEL_NAME,
                messages=messages,
                temperature=0.6,
                max_tokens=4096,
                stream=True
            )
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    full_response += content
                    yield content
            
            # [Debug] 打印完整输出
            cls._debug_print("OUTPUT (Stream Result)", full_response, Colors.GREEN)

        except Exception as e:
            print(f"{Colors.FAIL}[LLM Error]: {str(e)}{Colors.ENDC}")
            yield f"\n[LLM Error]: {str(e)}"

    @classmethod
    async def get_json_response(cls, messages: List[Dict[str, Any]]) -> Optional[Dict]:
        """
        非流式接口：自动解析 AI 返回的 JSON 结果
        用于：违规时间点提取、1:N 视觉比对、OCR 判定
        """
        # [Debug] 打印输入
        cls._debug_print("INPUT (JSON Mode)", messages)

        client = cls.get_async_client()
        try:
            response = await client.chat.completions.create(
                model=Config.MODEL_NAME,
                messages=messages,
                temperature=0.1,  # 低随机性确保 JSON 格式稳定
                response_format={"type": "json_object"} if "json" in Config.MODEL_NAME.lower() else None
            )
            content = response.choices[0].message.content
            
            # [Debug] 打印原始输出
            cls._debug_print("OUTPUT (Raw JSON)", content, Colors.GREEN)

            result = JSONUtils.safe_json_loads(content)
            
            # [Debug] 打印解析后的结果
            cls._debug_print("OUTPUT (Parsed JSON)", result, Colors.GREEN)
            
            return result
        except Exception as e:
            print(f"❌ LLM JSON 获取失败: {e}")
            return None

    @staticmethod
    def build_visual_message(text: str, base64_images: List[str]) -> List[Dict]:
        """
        构造符合 OpenAI 格式的视觉输入消息
        """
        content = [{"type": "text", "text": text}]
        for b64 in base64_images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
        return [{"role": "user", "content": content}]