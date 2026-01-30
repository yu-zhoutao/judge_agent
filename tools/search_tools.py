import os
import time
import asyncio
import uuid
import cv2  # 需要导入 opencv 来保存图片
import numpy as np
from typing import Dict, List, Any, Optional

try:
    from langchain.tools import tool
except Exception:
    from langchain_core.tools import tool  # type: ignore

from judge_agent.config import Config  # 假设你有 Config，如果没有，后面会自动降级到 static_temp
from judge_agent.utils.file_utils import FileUtils
from judge_agent.engines.minio_engine import MinioEngine

try:
    from judge_agent.utils.image_utils import ImageUtils
except ImportError:
    ImageUtils = None


class WebSearchTool:
    name = "web_search"
    description = "网络搜索工具。仅支持以图搜图（单图或视频抽帧）。"

    async def _process_single_frame(self, index: int, img_path: str, current_url: Optional[str], query: str) -> Dict[
        str, Any]:
        """
        内部并发单元：负责单帧图片的 MinIO 上传 + SerpApi 搜索
        """

        return {"idx": index, "error": "搜索次数有限，暂停搜索功能"}


        # # --- 再次防御：确保进来的 img_path 必须是字符串路径 ---
        # if not isinstance(img_path, str):
        #     return {"idx": index, "error": f"处理逻辑错误：期望文件路径(str)，实际得到 {type(img_path)}"}

        # target_url = current_url

        # # 1. 检查并上传图片
        # if not target_url:
        #     if not img_path:
        #         return {"idx": index, "error": "图片路径为空"}

        #     if not os.path.exists(img_path):
        #         return {"idx": index, "error": f"文件不存在: {img_path}"}

        #     try:
        #         # 放入线程池执行上传
        #         target_url = await asyncio.to_thread(MinioEngine.upload_file, img_path)
        #     except Exception as e:
        #         print(f"❌ [Task-{index}] MinIO 上传失败: {e}")
        #         return {"idx": index, "error": f"上传失败: {e}"}

        # if not target_url:
        #     return {"idx": index, "error": "无法获取有效的图片 URL"}

        # # 2. 调用以图搜图
        # search_start = time.perf_counter()
        # try:
        #     print(f"🔍 [Task-{index}] 开始搜索：{target_url}")
        #     search_result = await FileUtils.async_serper_search(target_url, extra_query=query)
        #     cost = time.perf_counter() - search_start
        #     print(f"⏱️ [Task-{index}] 搜索耗时: {cost:.2f}s")
        #     print(f"🔍 搜索结果：\n{search_result}")
        #     return {
        #         "idx": index,
        #         "status": "success",
        #         "finding": search_result
        #     }
        # except Exception as e:
        #     print(f"❌ [Task-{index}] 搜索异常: {e}")
        #     return {"idx": index, "error": str(e)}

    def _save_numpy_to_temp_file(self, img_data: np.ndarray) -> str:
        """
        辅助函数：将内存中的 NumPy 图片保存为本地临时文件
        """
        try:
            # 确定临时目录 (优先使用配置的目录，否则用 static_temp)
            temp_dir = getattr(Config, "FIXED_TEMP_DIR", "static_temp")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            # 生成唯一文件名
            filename = f"frame_search_{uuid.uuid4().hex[:8]}.jpg"
            file_path = os.path.join(temp_dir, filename)

            # 使用 opencv 保存
            cv2.imwrite(file_path, img_data)
            return file_path
        except Exception as e:
            print(f"❌ 图片保存失败: {e}")
            return ""

    async def run(self, query: str = "", image_path: str = "", image_url: str = "") -> Dict[str, Any]:
        # --- 1. 基础校验 ---
        if not image_path and not image_url:
            return {"error": "本工具仅支持以图搜图，请务必提供 image_path 或 image_url"}

        if image_path and image_path.lower().endswith(('.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a')):
            return {"error": "输入为纯音频文件，未进行搜索。"}

        # --- 2. 准备数据 ---
        is_video = image_path and image_path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))

        if is_video:
            if not ImageUtils:
                return {"error": "未找到 ImageUtils 工具"}

            # 这里 extract_frames 返回的 img 是 numpy 数组
            raw_frames_data = ImageUtils.extract_frames(image_path)

            # 校验返回是否有效
            if raw_frames_data is None or (isinstance(raw_frames_data, list) and len(raw_frames_data) == 0):
                return {"error": "视频抽帧结果为空"}

            items_to_process = raw_frames_data
        else:
            # 单图模式
            items_to_process = [{
                "index": 0,
                "img": image_path,
                "minio_url": image_url
            }]

        # --- 3. 预处理：将 NumPy 数组转为文件路径 ---
        tasks = []
        for i, item in enumerate(items_to_process):
            idx = item.get("index", i)
            raw_img = item.get("img")
            p_url = item.get("minio_url")

            final_path = ""

            # 情况 A: img 已经是字符串路径（单图模式或 ImageUtils 修改了实现）
            if isinstance(raw_img, str):
                final_path = raw_img

            # 情况 B: img 是 NumPy 数组（视频抽帧模式）
            elif isinstance(raw_img, (np.ndarray, list)):
                # print(f"🔄 [Task-{idx}] 检测到内存图片数据，正在保存为临时文件...")
                final_path = self._save_numpy_to_temp_file(np.array(raw_img))
                if not final_path:
                    print(f"⚠️ [Task-{idx}] 图片保存失败，跳过此帧")
                    continue

            # 创建任务：此时 final_path 必然是字符串，不会再报 truth value ambiguous 错误
            tasks.append(self._process_single_frame(idx, final_path, p_url, query))

        if not tasks:
            return {"error": "没有有效的图像帧可供处理"}

        # --- 4. 并发执行 ---
        count = len(tasks)
        print(f"🚀 开始并发执行 {count} 个以图搜图任务...")
        total_start_time = time.perf_counter()

        results = await asyncio.gather(*tasks)

        total_cost = time.perf_counter() - total_start_time
        print(f"✅ 所有搜索任务完成，总耗时: {total_cost:.2f} 秒")

        # --- 5. 结果聚合 ---
        valid_findings = []
        for res in results:
            if res.get("status") == "success":
                finding = res.get("finding", "").strip()
                if finding:
                    prefix = f"[第{res['idx']}帧搜图结果] " if count > 1 else ""
                    valid_findings.append(f"{prefix}{finding}")

        if not valid_findings:
            search_findings_agg = "未找到有效的搜索结果。"
        else:
            search_findings_agg = "\n\n".join(valid_findings)

        return {
            "status": "success",
            "search_findings": search_findings_agg
        }


_search_tool = WebSearchTool()


@tool("web_search")
async def web_search(query: str = "", image_path: str = "", image_url: str = "") -> Dict[str, Any]:
    """网络以图搜图。"""
    return await _search_tool.run(query=query, image_path=image_path, image_url=image_url)
