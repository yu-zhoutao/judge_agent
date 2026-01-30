from typing import Any, Dict, Optional


class AgentMiddleware:
    name = "agent_middleware"

    def before_agent(self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return state

    def after_agent(self, result: Any, config: Optional[Dict[str, Any]] = None) -> Any:
        return result

    def before_model(self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return state

    def after_model(self, result: Any, config: Optional[Dict[str, Any]] = None) -> Any:
        return result

    def before_tool(self, *args, **kwargs) -> Any:
        if args:
            return args[0]
        return kwargs.get("tool_input") or kwargs.get("input") or kwargs.get("tool_call")

    def after_tool(self, *args, **kwargs) -> Any:
        if args:
            return args[0]
        return kwargs.get("output") or kwargs.get("result")

    def wrap_tool_call(self, tool_call: Any, tool: Any, config: Optional[Dict[str, Any]] = None) -> Any:
        return tool_call

    async def awrap_tool_call(self, tool_call: Any, tool: Any, config: Optional[Dict[str, Any]] = None) -> Any:
        return self.wrap_tool_call(tool_call, tool, config)

    def before_tool_call(self, tool_call: Any, tool: Any, config: Optional[Dict[str, Any]] = None) -> Any:
        return tool_call

    def after_tool_call(self, result: Any, tool: Any, config: Optional[Dict[str, Any]] = None) -> Any:
        return result


class SSEMiddleware(AgentMiddleware):
    name = "sse_middleware"
    def __init__(self, stream_writer):
        self.stream_writer = stream_writer

    def before_model(self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.stream_writer:
            self.stream_writer({"type": "log", "content": "model_call_start"})
        return state

    def after_model(self, result: Any, config: Optional[Dict[str, Any]] = None) -> Any:
        if self.stream_writer:
            self.stream_writer({"type": "log", "content": "model_call_end"})
        return result


class TraceMiddleware(AgentMiddleware):
    name = "trace_middleware"
    def __init__(self, logger):
        self.logger = logger

    def before_model(self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.logger:
            self.logger.info("agent_before_model", extra={"keys": list(state.keys())})
        return state

    def after_model(self, result: Any, config: Optional[Dict[str, Any]] = None) -> Any:
        if self.logger:
            content = getattr(result, "content", None)
            if content is None and isinstance(result, dict):
                content = result.get("content")
            self.logger.info("agent_after_model", extra={"content": content})
        return result


def build_middlewares(stream_writer=None, logger=None):
    middlewares = []
    if stream_writer:
        middlewares.append(SSEMiddleware(stream_writer))
    if logger:
        middlewares.append(TraceMiddleware(logger))
    return middlewares
