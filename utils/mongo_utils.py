"""
MongoDB工具类 - 支持异步和同步操作
提供统一的MongoDB操作接口，包含连接池管理、CRUD操作等
"""

import asyncio
from typing import Optional, Dict, List, Any, Union
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
import logging

# 同步MongoDB驱动
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, OperationFailure, DuplicateKeyError

# 异步MongoDB驱动
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection

from judge_agent.config import Config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MongoConfig:
    """MongoDB配置类"""

    # 从环境变量或配置文件读取
    MONGO_URI = getattr(Config, 'MONGO_URI', 'mongodb://localhost:27017/')
    MONGO_DATABASE = getattr(Config, 'MONGO_DATABASE', 'judge_agent')

    # 连接池配置
    MAX_POOL_SIZE = getattr(Config, 'MONGO_MAX_POOL_SIZE', 100)
    MIN_POOL_SIZE = getattr(Config, 'MONGO_MIN_POOL_SIZE', 10)
    MAX_IDLE_TIME_MS = getattr(Config, 'MONGO_MAX_IDLE_TIME_MS', 10000)
    SERVER_SELECTION_TIMEOUT_MS = getattr(Config, 'MONGO_SERVER_SELECTION_TIMEOUT_MS', 5000)
    CONNECT_TIMEOUT_MS = getattr(Config, 'MONGO_CONNECT_TIMEOUT_MS', 5000)

    # 索引配置
    INDEX_TTL_SECONDS = getattr(Config, 'MONGO_INDEX_TTL_SECONDS', 86400)  # 默认24小时


class MongoUtils:
    """
    MongoDB同步工具类
    提供同步的MongoDB操作接口
    """

    _instance: Optional['MongoUtils'] = None
    _client: Optional[MongoClient] = None
    _db: Optional[Database] = None

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化MongoDB连接"""
        if self._client is None:
            self._connect()

    def _connect(self):
        """建立MongoDB连接"""
        try:
            self._client = MongoClient(
                MongoConfig.MONGO_URI,
                maxPoolSize=MongoConfig.MAX_POOL_SIZE,
                minPoolSize=MongoConfig.MIN_POOL_SIZE,
                maxIdleTimeMS=MongoConfig.MAX_IDLE_TIME_MS,
                serverSelectionTimeoutMS=MongoConfig.SERVER_SELECTION_TIMEOUT_MS,
                connectTimeoutMS=MongoConfig.CONNECT_TIMEOUT_MS
            )
            self._db = self._client[MongoConfig.MONGO_DATABASE]

            # 测试连接
            self._client.admin.command('ping')
            logger.info(f"✅ MongoDB同步连接成功: {MongoConfig.MONGO_DATABASE}")

        except ConnectionFailure as e:
            logger.error(f"❌ MongoDB同步连接失败: {e}")
            raise

    def get_collection(self, collection_name: str) -> Collection:
        """获取集合对象"""
        if self._db is None:
            self._connect()
        return self._db[collection_name]

    def get_database(self) -> Database:
        """获取数据库对象"""
        if self._db is None:
            self._connect()
        return self._db

    def close(self):
        """关闭连接"""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            logger.info("🔌 MongoDB同步连接已关闭")

    # ==================== CRUD 操作 ====================

    def insert_one(self, collection_name: str, document: Dict[str, Any]) -> str:
        """
        插入单个文档
        :param collection_name: 集合名称
        :param document: 文档数据
        :return: 插入的文档ID
        """
        try:
            # 添加创建时间
            if 'created_at' not in document:
                document['created_at'] = datetime.utcnow()

            collection = self.get_collection(collection_name)
            result = collection.insert_one(document)
            logger.debug(f"📝 插入文档到 {collection_name}: {result.inserted_id}")
            return str(result.inserted_id)

        except DuplicateKeyError as e:
            logger.warning(f"⚠️ 重复键错误: {e}")
            raise
        except OperationFailure as e:
            logger.error(f"❌ 插入操作失败: {e}")
            raise

    def insert_many(self, collection_name: str, documents: List[Dict[str, Any]]) -> List[str]:
        """
        批量插入文档
        :param collection_name: 集合名称
        :param documents: 文档列表
        :return: 插入的文档ID列表
        """
        try:
            # 添加创建时间
            for doc in documents:
                if 'created_at' not in doc:
                    doc['created_at'] = datetime.utcnow()

            collection = self.get_collection(collection_name)
            result = collection.insert_many(documents)
            logger.debug(f"📝 批量插入 {len(result.inserted_ids)} 个文档到 {collection_name}")
            return [str(oid) for oid in result.inserted_ids]

        except OperationFailure as e:
            logger.error(f"❌ 批量插入操作失败: {e}")
            raise

    def find_one(self, collection_name: str, query: Dict[str, Any],
                 projection: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        查询单个文档
        :param collection_name: 集合名称
        :param query: 查询条件
        :param projection: 投影字段
        :return: 文档数据或None
        """
        try:
            collection = self.get_collection(collection_name)
            document = collection.find_one(query, projection)

            if document:
                # 转换ObjectId为字符串
                document['_id'] = str(document['_id'])

            return document

        except OperationFailure as e:
            logger.error(f"❌ 查询操作失败: {e}")
            raise

    def find_many(self, collection_name: str, query: Dict[str, Any],
                  projection: Optional[Dict[str, Any]] = None,
                  sort: Optional[List[tuple]] = None,
                  limit: int = 0,
                  skip: int = 0) -> List[Dict[str, Any]]:
        """
        查询多个文档
        :param collection_name: 集合名称
        :param query: 查询条件
        :param projection: 投影字段
        :param sort: 排序字段，例如 [('field', 1), ('field2', -1)]
        :param limit: 限制返回数量
        :param skip: 跳过数量
        :return: 文档列表
        """
        try:
            collection = self.get_collection(collection_name)
            cursor = collection.find(query, projection)

            if sort:
                cursor = cursor.sort(sort)
            if skip:
                cursor = cursor.skip(skip)
            if limit:
                cursor = cursor.limit(limit)

            documents = list(cursor)

            # 转换ObjectId为字符串
            for doc in documents:
                doc['_id'] = str(doc['_id'])

            return documents

        except OperationFailure as e:
            logger.error(f"❌ 查询操作失败: {e}")
            raise

    def update_one(self, collection_name: str, query: Dict[str, Any],
                   update: Dict[str, Any], upsert: bool = False) -> int:
        """
        更新单个文档
        :param collection_name: 集合名称
        :param query: 查询条件
        :param update: 更新数据
        :param upsert: 如果不存在是否插入
        :return: 修改的文档数量
        """
        try:
            # 添加更新时间
            if '$set' in update:
                update['$set']['updated_at'] = datetime.utcnow()
            else:
                update['$set'] = {'updated_at': datetime.utcnow()}

            collection = self.get_collection(collection_name)
            result = collection.update_one(query, update, upsert=upsert)
            logger.debug(f"🔄 更新 {collection_name} 中 {result.modified_count} 个文档")
            return result.modified_count

        except OperationFailure as e:
            logger.error(f"❌ 更新操作失败: {e}")
            raise

    def update_many(self, collection_name: str, query: Dict[str, Any],
                    update: Dict[str, Any]) -> int:
        """
        批量更新文档
        :param collection_name: 集合名称
        :param query: 查询条件
        :param update: 更新数据
        :return: 修改的文档数量
        """
        try:
            # 添加更新时间
            if '$set' in update:
                update['$set']['updated_at'] = datetime.utcnow()
            else:
                update['$set'] = {'updated_at': datetime.utcnow()}

            collection = self.get_collection(collection_name)
            result = collection.update_many(query, update)
            logger.debug(f"🔄 批量更新 {collection_name} 中 {result.modified_count} 个文档")
            return result.modified_count

        except OperationFailure as e:
            logger.error(f"❌ 批量更新操作失败: {e}")
            raise

    def delete_one(self, collection_name: str, query: Dict[str, Any]) -> int:
        """
        删除单个文档
        :param collection_name: 集合名称
        :param query: 查询条件
        :return: 删除的文档数量
        """
        try:
            collection = self.get_collection(collection_name)
            result = collection.delete_one(query)
            logger.debug(f"🗑️  删除 {collection_name} 中 {result.deleted_count} 个文档")
            return result.deleted_count

        except OperationFailure as e:
            logger.error(f"❌ 删除操作失败: {e}")
            raise

    def delete_many(self, collection_name: str, query: Dict[str, Any]) -> int:
        """
        批量删除文档
        :param collection_name: 集合名称
        :param query: 查询条件
        :return: 删除的文档数量
        """
        try:
            collection = self.get_collection(collection_name)
            result = collection.delete_many(query)
            logger.debug(f"🗑️  批量删除 {collection_name} 中 {result.deleted_count} 个文档")
            return result.deleted_count

        except OperationFailure as e:
            logger.error(f"❌ 批量删除操作失败: {e}")
            raise

    def count_documents(self, collection_name: str, query: Dict[str, Any]) -> int:
        """
        统计文档数量
        :param collection_name: 集合名称
        :param query: 查询条件
        :return: 文档数量
        """
        try:
            collection = self.get_collection(collection_name)
            count = collection.count_documents(query)
            return count

        except OperationFailure as e:
            logger.error(f"❌ 统计操作失败: {e}")
            raise

    # ==================== 索引操作 ====================

    def create_index(self, collection_name: str, keys: Union[str, List[tuple]],
                     unique: bool = False, ttl_seconds: Optional[int] = None) -> str:
        """
        创建索引
        :param collection_name: 集合名称
        :param keys: 索引字段，可以是字符串或元组列表
        :param unique: 是否唯一索引
        :param ttl_seconds: TTL过期时间（秒）
        :return: 索引名称
        """
        try:
            collection = self.get_collection(collection_name)

            index_options = {'unique': unique}
            if ttl_seconds:
                index_options['expireAfterSeconds'] = ttl_seconds

            index_name = collection.create_index(keys, **index_options)
            logger.info(f"📊 创建索引 {index_name} 在 {collection_name}")
            return index_name

        except OperationFailure as e:
            logger.error(f"❌ 创建索引失败: {e}")
            raise

    def drop_index(self, collection_name: str, index_name: str):
        """
        删除索引
        :param collection_name: 集合名称
        :param index_name: 索引名称
        """
        try:
            collection = self.get_collection(collection_name)
            collection.drop_index(index_name)
            logger.info(f"📊 删除索引 {index_name} 从 {collection_name}")

        except OperationFailure as e:
            logger.error(f"❌ 删除索引失败: {e}")
            raise

    def list_indexes(self, collection_name: str) -> List[Dict[str, Any]]:
        """
        列出集合的所有索引
        :param collection_name: 集合名称
        :return: 索引列表
        """
        try:
            collection = self.get_collection(collection_name)
            indexes = collection.list_indexes()
            return list(indexes)

        except OperationFailure as e:
            logger.error(f"❌ 列出索引失败: {e}")
            raise

    # ==================== 聚合操作 ====================

    def aggregate(self, collection_name: str, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        执行聚合查询
        :param collection_name: 集合名称
        :param pipeline: 聚合管道
        :return: 聚合结果
        """
        try:
            collection = self.get_collection(collection_name)
            results = list(collection.aggregate(pipeline))

            # 转换ObjectId为字符串
            for doc in results:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])

            return results

        except OperationFailure as e:
            logger.error(f"❌ 聚合操作失败: {e}")
            raise

    # ==================== 事务操作 ====================

    @contextmanager
    def session(self):
        """
        上下文管理器：创建会话用于事务操作
        """
        if self._client is None:
            self._connect()

        session = self._client.start_session()
        try:
            yield session
        finally:
            session.end_session()

    def transaction(self, callback, **kwargs):
        """
        执行事务
        :param callback: 事务回调函数
        :param kwargs: 传递给回调函数的参数
        """
        with self.session() as session:
            with session.start_transaction():
                callback(session=session, **kwargs)


class AsyncMongoUtils:
    """
    MongoDB异步工具类
    提供异步的MongoDB操作接口
    """

    _instance: Optional['AsyncMongoUtils'] = None
    _client: Optional[AsyncIOMotorClient] = None
    _db: Optional[AsyncIOMotorDatabase] = None

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化MongoDB连接"""
        if self._client is None:
            self._connect()

    def _connect(self):
        """建立MongoDB连接"""
        try:
            self._client = AsyncIOMotorClient(
                MongoConfig.MONGO_URI,
                maxPoolSize=MongoConfig.MAX_POOL_SIZE,
                minPoolSize=MongoConfig.MIN_POOL_SIZE,
                maxIdleTimeMS=MongoConfig.MAX_IDLE_TIME_MS,
                serverSelectionTimeoutMS=MongoConfig.SERVER_SELECTION_TIMEOUT_MS,
                connectTimeoutMS=MongoConfig.CONNECT_TIMEOUT_MS
            )
            self._db = self._client[MongoConfig.MONGO_DATABASE]
            logger.info(f"✅ MongoDB异步连接成功: {MongoConfig.MONGO_DATABASE}")

        except ConnectionFailure as e:
            logger.error(f"❌ MongoDB异步连接失败: {e}")
            raise

    async def get_collection(self, collection_name: str) -> AsyncIOMotorCollection:
        """获取集合对象"""
        if self._db is None:
            self._connect()
        return self._db[collection_name]

    async def get_database(self) -> AsyncIOMotorDatabase:
        """获取数据库对象"""
        if self._db is None:
            self._connect()
        return self._db

    async def close(self):
        """关闭连接"""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            logger.info("🔌 MongoDB异步连接已关闭")

    # ==================== CRUD 操作 ====================

    async def insert_one(self, collection_name: str, document: Dict[str, Any]) -> str:
        """
        插入单个文档
        :param collection_name: 集合名称
        :param document: 文档数据
        :return: 插入的文档ID
        """
        try:
            # 添加创建时间
            if 'created_at' not in document:
                document['created_at'] = datetime.utcnow()

            collection = await self.get_collection(collection_name)
            result = await collection.insert_one(document)
            logger.debug(f"📝 插入文档到 {collection_name}: {result.inserted_id}")
            return str(result.inserted_id)

        except DuplicateKeyError as e:
            logger.warning(f"⚠️ 重复键错误: {e}")
            raise
        except OperationFailure as e:
            logger.error(f"❌ 插入操作失败: {e}")
            raise

    async def insert_many(self, collection_name: str, documents: List[Dict[str, Any]]) -> List[str]:
        """
        批量插入文档
        :param collection_name: 集合名称
        :param documents: 文档列表
        :return: 插入的文档ID列表
        """
        try:
            # 添加创建时间
            for doc in documents:
                if 'created_at' not in doc:
                    doc['created_at'] = datetime.utcnow()

            collection = await self.get_collection(collection_name)
            result = await collection.insert_many(documents)
            logger.debug(f"📝 批量插入 {len(result.inserted_ids)} 个文档到 {collection_name}")
            return [str(oid) for oid in result.inserted_ids]

        except OperationFailure as e:
            logger.error(f"❌ 批量插入操作失败: {e}")
            raise

    async def find_one(self, collection_name: str, query: Dict[str, Any],
                       projection: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        查询单个文档
        :param collection_name: 集合名称
        :param query: 查询条件
        :param projection: 投影字段
        :return: 文档数据或None
        """
        try:
            collection = await self.get_collection(collection_name)
            document = await collection.find_one(query, projection)

            if document:
                # 转换ObjectId为字符串
                document['_id'] = str(document['_id'])

            return document

        except OperationFailure as e:
            logger.error(f"❌ 查询操作失败: {e}")
            raise

    async def find_many(self, collection_name: str, query: Dict[str, Any],
                        projection: Optional[Dict[str, Any]] = None,
                        sort: Optional[List[tuple]] = None,
                        limit: int = 0,
                        skip: int = 0) -> List[Dict[str, Any]]:
        """
        查询多个文档
        :param collection_name: 集合名称
        :param query: 查询条件
        :param projection: 投影字段
        :param sort: 排序字段，例如 [('field', 1), ('field2', -1)]
        :param limit: 限制返回数量
        :param skip: 跳过数量
        :return: 文档列表
        """
        try:
            collection = await self.get_collection(collection_name)
            cursor = collection.find(query, projection)

            if sort:
                cursor = cursor.sort(sort)
            if skip:
                cursor = cursor.skip(skip)
            if limit:
                cursor = cursor.limit(limit)

            documents = await cursor.to_list(length=None)

            # 转换ObjectId为字符串
            for doc in documents:
                doc['_id'] = str(doc['_id'])

            return documents

        except OperationFailure as e:
            logger.error(f"❌ 查询操作失败: {e}")
            raise

    async def update_one(self, collection_name: str, query: Dict[str, Any],
                         update: Dict[str, Any], upsert: bool = False) -> int:
        """
        更新单个文档
        :param collection_name: 集合名称
        :param query: 查询条件
        :param update: 更新数据
        :param upsert: 如果不存在是否插入
        :return: 修改的文档数量
        """
        try:
            # 添加更新时间
            if '$set' in update:
                update['$set']['updated_at'] = datetime.utcnow()
            else:
                update['$set'] = {'updated_at': datetime.utcnow()}

            collection = await self.get_collection(collection_name)
            result = await collection.update_one(query, update, upsert=upsert)
            logger.debug(f"🔄 更新 {collection_name} 中 {result.modified_count} 个文档")
            return result.modified_count

        except OperationFailure as e:
            logger.error(f"❌ 更新操作失败: {e}")
            raise

    async def update_many(self, collection_name: str, query: Dict[str, Any],
                          update: Dict[str, Any]) -> int:
        """
        批量更新文档
        :param collection_name: 集合名称
        :param query: 查询条件
        :param update: 更新数据
        :return: 修改的文档数量
        """
        try:
            # 添加更新时间
            if '$set' in update:
                update['$set']['updated_at'] = datetime.utcnow()
            else:
                update['$set'] = {'updated_at': datetime.utcnow()}

            collection = await self.get_collection(collection_name)
            result = await collection.update_many(query, update)
            logger.debug(f"🔄 批量更新 {collection_name} 中 {result.modified_count} 个文档")
            return result.modified_count

        except OperationFailure as e:
            logger.error(f"❌ 批量更新操作失败: {e}")
            raise

    async def delete_one(self, collection_name: str, query: Dict[str, Any]) -> int:
        """
        删除单个文档
        :param collection_name: 集合名称
        :param query: 查询条件
        :return: 删除的文档数量
        """
        try:
            collection = await self.get_collection(collection_name)
            result = await collection.delete_one(query)
            logger.debug(f"🗑️  删除 {collection_name} 中 {result.deleted_count} 个文档")
            return result.deleted_count

        except OperationFailure as e:
            logger.error(f"❌ 删除操作失败: {e}")
            raise

    async def delete_many(self, collection_name: str, query: Dict[str, Any]) -> int:
        """
        批量删除文档
        :param collection_name: 集合名称
        :param query: 查询条件
        :return: 删除的文档数量
        """
        try:
            collection = await self.get_collection(collection_name)
            result = await collection.delete_many(query)
            logger.debug(f"🗑️  批量删除 {collection_name} 中 {result.deleted_count} 个文档")
            return result.deleted_count

        except OperationFailure as e:
            logger.error(f"❌ 批量删除操作失败: {e}")
            raise

    async def count_documents(self, collection_name: str, query: Dict[str, Any]) -> int:
        """
        统计文档数量
        :param collection_name: 集合名称
        :param query: 查询条件
        :return: 文档数量
        """
        try:
            collection = await self.get_collection(collection_name)
            count = await collection.count_documents(query)
            return count

        except OperationFailure as e:
            logger.error(f"❌ 统计操作失败: {e}")
            raise

    # ==================== 索引操作 ====================

    async def create_index(self, collection_name: str, keys: Union[str, List[tuple]],
                           unique: bool = False, ttl_seconds: Optional[int] = None) -> str:
        """
        创建索引
        :param collection_name: 集合名称
        :param keys: 索引字段，可以是字符串或元组列表
        :param unique: 是否唯一索引
        :param ttl_seconds: TTL过期时间（秒）
        :return: 索引名称
        """
        try:
            collection = await self.get_collection(collection_name)

            index_options = {'unique': unique}
            if ttl_seconds:
                index_options['expireAfterSeconds'] = ttl_seconds

            index_name = await collection.create_index(keys, **index_options)
            logger.info(f"📊 创建索引 {index_name} 在 {collection_name}")
            return index_name

        except OperationFailure as e:
            logger.error(f"❌ 创建索引失败: {e}")
            raise

    async def drop_index(self, collection_name: str, index_name: str):
        """
        删除索引
        :param collection_name: 集合名称
        :param index_name: 索引名称
        """
        try:
            collection = await self.get_collection(collection_name)
            await collection.drop_index(index_name)
            logger.info(f"📊 删除索引 {index_name} 从 {collection_name}")

        except OperationFailure as e:
            logger.error(f"❌ 删除索引失败: {e}")
            raise

    async def list_indexes(self, collection_name: str) -> List[Dict[str, Any]]:
        """
        列出集合的所有索引
        :param collection_name: 集合名称
        :return: 索引列表
        """
        try:
            collection = await self.get_collection(collection_name)
            indexes = await collection.list_indexes().to_list(length=None)
            return indexes

        except OperationFailure as e:
            logger.error(f"❌ 列出索引失败: {e}")
            raise

    # ==================== 聚合操作 ====================

    async def aggregate(self, collection_name: str, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        执行聚合查询
        :param collection_name: 集合名称
        :param pipeline: 聚合管道
        :return: 聚合结果
        """
        try:
            collection = await self.get_collection(collection_name)
            results = await collection.aggregate(pipeline).to_list(length=None)

            # 转换ObjectId为字符串
            for doc in results:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])

            return results

        except OperationFailure as e:
            logger.error(f"❌ 聚合操作失败: {e}")
            raise

    # ==================== 事务操作 ====================

    @asynccontextmanager
    async def session(self):
        """
        上下文管理器：创建会话用于事务操作
        """
        if self._client is None:
            self._connect()

        session = await self._client.start_session()
        try:
            yield session
        finally:
            await session.end_session()

    async def transaction(self, callback, **kwargs):
        """
        执行事务
        :param callback: 事务回调函数
        :param kwargs: 传递给回调函数的参数
        """
        async with self.session() as session:
            async with session.start_transaction():
                await callback(session=session, **kwargs)


# ==================== 便捷函数 ====================

def get_mongo() -> MongoUtils:
    """获取同步MongoDB工具类实例"""
    return MongoUtils()


def get_async_mongo() -> AsyncMongoUtils:
    """获取异步MongoDB工具类实例"""
    return AsyncMongoUtils()

