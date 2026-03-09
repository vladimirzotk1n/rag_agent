import asyncio
import json

from fastembed import SparseTextEmbedding
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ApiException
from qdrant_client.models import PointStruct

from src.config import settings
from src.logger_config import logger

COLLECTION_NAME = settings.collection_name
SPARSE_MODEL_NAME = settings.sparce_model_name
CHUNKS_PATH = settings.chunks_path
DENSE_EMBEDDING_DIM = settings.dense_embedding_dim


async def init_db():
    client = AsyncQdrantClient(url="http://qdrant:6333")

    if not await client.collection_exists(COLLECTION_NAME):
        logger.info(f"Creating database collection {COLLECTION_NAME} ")
        try:
            await client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    "dense": models.VectorParams(
                        size=DENSE_EMBEDDING_DIM,
                        distance=models.Distance.COSINE,
                    ),
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    ),
                },
            )
            logger.info(f"Database collection {COLLECTION_NAME} created")

        except ApiException as e:
            logger.critical(f"Can't create database: {e}")
            raise

    else:
        logger.info(f"Collection {COLLECTION_NAME} alrady exists")

    from src.model.inference import dense_embed

    sparse_model = SparseTextEmbedding(SPARSE_MODEL_NAME)

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        data_dict = json.load(f)

    points = []
    batch_size = 50
    inserted = 0

    logger.info("Loading data in database")
    try:
        for idx, chunk in enumerate(data_dict):
            content = chunk.get("chunk_content", "")
            if not content:
                continue

            payload = {
                "section": chunk.get("section"),
                "section_title": chunk.get("section_title"),
                "chapter": chunk.get("chapter"),
                "chapter_title": chunk.get("chapter_title"),
                "article": chunk.get("article"),
                "article_title": chunk.get("article_title"),
                "article_content": chunk.get("article_content"),
                "chunk_content": chunk.get("chunk_content"),
                "chunk_id": idx,
            }

            dense_vector = await asyncio.to_thread(dense_embed, content)
            sparse_vector = await asyncio.to_thread(
                lambda: list(sparse_model.embed([content]))[0]
            )

            point = PointStruct(
                id=idx,
                vector={
                    "dense": dense_vector,
                    "sparse": sparse_vector.as_object(),
                },
                payload=payload,
            )
            points.append(point)

            if len(points) >= batch_size:
                logger.info(f"Upserting batch of {len(points)} points at idx={idx}")
                await client.upsert(collection_name=COLLECTION_NAME, points=points)
                inserted += len(points)
                points = []

        if points:
            print(f"Upserting final batch of {len(points)} points", flush=True)
            await client.upsert(collection_name=COLLECTION_NAME, points=points)
            inserted += len(points)

        logger.info(f"Successfully loaded points number: {len(data_dict)}")
    except ApiException as e:
        logger.critical(f"Can't upload points in database: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(init_db())
