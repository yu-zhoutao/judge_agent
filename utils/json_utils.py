import re
import json
from typing import Any, List, Dict

class JSONUtils:
    """处理不规范的 JSON 解析及违规时间轴合并"""

    @staticmethod
    def safe_json_loads(text: str) -> Any:
        """
        鲁棒性 JSON 解析器：
        1. 自动去除 Markdown 代码块包裹
        2. 清理不可见字符
        3. 处理 AI 可能生成的额外解释文字
        """
        if not text:
            return None
            
        # 尝试匹配 Markdown 代码块内容
        json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        match = re.search(json_pattern, text)
        if match:
            text = match.group(1)
        
        # 清除首尾空白及可能的控制字符
        text = text.strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # 最后的倔强：尝试通过正则强行提取最外层的 {} 或 []
            try:
                fallback_match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
                if fallback_match:
                    return json.loads(fallback_match.group(1))
            except:
                pass
            print(f"❌ JSON 解析失败，原始文本: {text[:100]}...")
            return None

    @staticmethod
    def merge_intervals(intervals: List[Dict[str, Any]], gap: float = 1.0) -> List[Dict[str, Any]]:
        """
        合并视频审核中的违规时间片段：
        :param intervals: 待合并的列表，如 [{'start': 1.0, 'end': 2.0, 'reason': '...'}]
        :param gap: 允许合并的最大间隔时间（秒）。如果两段违规时间靠得很近，合并为一段。
        """
        if not intervals:
            return []

        # 按开始时间排序
        sorted_intervals = sorted(intervals, key=lambda x: x['start'])
        
        merged = []
        if not sorted_intervals:
            return merged

        current = sorted_intervals[0].copy()

        for next_int in sorted_intervals[1:]:
            # 如果下一段的开始时间 小于 当前段结束时间 + 允许间隔
            if next_int['start'] <= current['end'] + gap:
                # 更新结束时间为较大者
                current['end'] = max(current['end'], next_int['end'])
                # 合并原因描述（去重）
                reasons = set(current.get('reason', '').split('; '))
                reasons.add(next_int.get('reason', ''))
                current['reason'] = "; ".join(filter(None, reasons))
            else:
                merged.append(current)
                current = next_int.copy()
        
        merged.append(current)
        return merged

    @staticmethod
    def format_time_range(seconds: float) -> str:
        """将秒转换为 mm:ss 格式，方便前端展示"""
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"