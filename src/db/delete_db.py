from qdrant_client import AsyncQdrantClient

from src.config import settings

COLLECTION_NAME = settings.collection_name


async def delete_db():
    client = AsyncQdrantClient(url="http://qdrant:6333")
    await client.delete_collection(collection_name=COLLECTION_NAME)
