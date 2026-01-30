try:
    from langchain.tools import tool
except Exception:
    from langchain_core.tools import tool  # type: ignore

from judge_agent.tools.search_tools import WebSearchTool

_search_tool = WebSearchTool()


@tool("web_search")
async def web_search(query: str = "", image_path: str = "", image_url: str = ""):
    """网络以图搜图。"""
    return await _search_tool.run(query=query, image_path=image_path, image_url=image_url)
