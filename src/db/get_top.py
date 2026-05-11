import asyncio
from typing import Literal

from fastembed import SparseTextEmbedding
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ApiException, ResponseHandlingException

from src.config import settings
from src.logger_config import logger
from src.model.inference import dense_embed

COLLECTION_NAME = settings.collection_name
SPARSE_MODEL_NAME = settings.sparce_model_name

sparse_model = SparseTextEmbedding(SPARSE_MODEL_NAME)


async def get_top(
    query: str,
    sparse_query: str | None = None,
    top_k: int = 5,
    bm25_top: int = 100,
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid",
) -> list[dict]:

    logger.info("Connecting to Qdrant database")
    try:
        client = AsyncQdrantClient(url=settings.qdrant_url)
    except ResponseHandlingException as e:
        logger.critical(f"Can't connect to qdrant database: {e}")
        raise

    logger.info(f"Getting top queries (mode={mode})")
    try:
        if mode == "bm25":
            sparse_emb = await asyncio.to_thread(
                lambda: list(sparse_model.embed(sparse_query or query))[0]
            )
            sparse_vec = models.SparseVector(
                indices=sparse_emb.indices.tolist(), values=sparse_emb.values.tolist()
            )
            results = await client.query_points(
                collection_name=COLLECTION_NAME,
                query=sparse_vec,
                using="sparse",
                limit=top_k,
                with_payload=True,
            )
            logger.info("Successfully got top queries")
            return [hit.payload for hit in results.points]

        elif mode == "dense":
            dense_vec = await asyncio.to_thread(dense_embed, query)
            results = await client.query_points(
                collection_name=COLLECTION_NAME,
                query=dense_vec,
                using="dense",
                limit=top_k,
                with_payload=True,
            )
            logger.info("Successfully got top queries")
            return [hit.payload for hit in results.points]

        else:
            sparse_emb = await asyncio.to_thread(
                lambda: list(sparse_model.embed(sparse_query or query))[0]
            )
            sparse_vec = models.SparseVector(
                indices=sparse_emb.indices.tolist(), values=sparse_emb.values.tolist()
            )
            sparse_results = await client.query_points(
                collection_name=COLLECTION_NAME,
                query=sparse_vec,
                using="sparse",
                limit=bm25_top,
                with_payload=True,
            )
            if not sparse_results.points:
                return []

            dense_vec = await asyncio.to_thread(dense_embed, query)
            candidate_ids = [p.id for p in sparse_results.points]

            dense_results = await client.query_points(
                collection_name=COLLECTION_NAME,
                query=dense_vec,
                using="dense",
                limit=top_k,
                with_payload=True,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="chunk_id", match=models.MatchAny(any=candidate_ids)
                        )
                    ]
                ),
            )
            logger.info("Successfully got top queries")
            return [hit.payload for hit in dense_results.points]

    except ApiException as e:
        logger.critical(f"Can't get top queries: {e}")
        raise
    finally:
        await client.close()
