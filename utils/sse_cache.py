from datetime import datetime
from typing import Optional, Any

from bson import ObjectId

from judge_agent.utils.mongo_utils import AsyncMongoUtils


class MongoSSECache:
    """SSE 事件缓存（MongoDB 版本）。"""

    COLLECTION_NAME = "agent_memories"

    def __init__(self, file_path: str, file_type: str, s3_url: str = "") -> None:
        self.file_path = file_path
        self.file_type = file_type
        self.s3_url = s3_url
        self._memory_id: Optional[str] = None
        self._mongo = AsyncMongoUtils()

    async def _initialize(self) -> None:
        if self._memory_id is not None:
            return
        file_id = ""
        if self.s3_url:
            try:
                file_id = self.s3_url.split("/")[-1].split(".")[0]
            except Exception:
                file_id = ""

        memory_doc = {
            "file_path": self.file_path,
            "file_type": self.file_type,
            "s3_url": self.s3_url,
            "file_id": file_id,
            "messages": [],
            "client_history": [],
            "finished": False,
            "final_content": "",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        self._memory_id = await self._mongo.insert_one(self.COLLECTION_NAME, memory_doc)

    async def add_client_history(self, event_type: str, content: Any) -> None:
        await self._initialize()
        if not self._memory_id:
            return
        await self._mongo.update_one(
            self.COLLECTION_NAME,
            {"_id": ObjectId(self._memory_id)},
            {
                "$push": {"client_history": {"type": event_type, "content": content}},
                "$set": {"updated_at": datetime.utcnow()},
            },
        )
