"""Weaviate schema initialisation.

Creates the ``UserMemory`` and ``MoodPattern`` collections if they do not
already exist.  Called once during FastAPI lifespan startup.

Collections
-----------
``UserMemory``
    Stores extracted facts about a user (genre preferences, listening habits,
    biographical snippets) as 768-dimensional BGE embeddings.

``MoodPattern``
    Stores aggregated time-bucket mood statistics (energy, valence, tempo) used
    by the DJ agent to infer context-appropriate recommendations.

Both collections use HNSW with COSINE distance and ``vectorizer=none`` so the
application controls exactly what vectors are stored.
"""

import asyncio

import weaviate
import weaviate.classes as wvc
from weaviate import WeaviateClient

from backend.app.config import settings
from backend.app.observability.logging import get_logger

logger = get_logger(__name__)

VECTOR_DIM = 768  # BGE-base-en-v1.5


def get_weaviate_client() -> WeaviateClient:
    """Build a synchronous Weaviate v4 client from application settings.

    Extracts the host and port from ``settings.weaviate_url`` and connects
    without authentication (``anonymous_access=True`` is the default in
    docker-compose for local development).

    Returns:
        A connected ``WeaviateClient``.  The caller is responsible for calling
        ``client.close()`` when done.
    """
    host = settings.weaviate_url.replace("http://", "").replace("https://", "")
    host_part, _, port_str = host.partition(":")
    port = int(port_str) if port_str else 8080
    return weaviate.connect_to_custom(
        http_host=host_part,
        http_port=port,
        http_secure=False,
        grpc_host=host_part,
        grpc_port=50051,
        grpc_secure=False,
    )


def _ensure_collection(client: WeaviateClient, name: str, properties: list) -> None:
    """Create *name* in Weaviate if it does not already exist.

    Args:
        client:     An open ``WeaviateClient``.
        name:       Collection name (PascalCase, e.g. ``"UserMemory"``).
        properties: List of ``wvc.config.Property`` objects describing the schema.
    """
    if client.collections.exists(name):
        logger.debug("weaviate_collection_exists", collection=name)
        return

    client.collections.create(
        name=name,
        properties=properties,
        vectorizer_config=wvc.config.Configure.Vectorizer.none(),
        vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
            distance_metric=wvc.config.VectorDistances.COSINE,
        ),
    )
    logger.info("weaviate_collection_created", collection=name)


def init_weaviate_schema_sync() -> None:
    """Create both Weaviate collections synchronously.

    Intended to be called from ``init_weaviate_schema`` via
    ``asyncio.to_thread`` to keep the async event loop unblocked, since the
    weaviate-client v4 connection API is synchronous.

    Raises:
        weaviate.exceptions.WeaviateConnectionError: If Weaviate is unreachable.
    """
    client = get_weaviate_client()
    try:
        _ensure_collection(
            client,
            "UserMemory",
            [
                wvc.config.Property(name="user_id", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="type", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="confidence", data_type=wvc.config.DataType.NUMBER),
                wvc.config.Property(name="created_at", data_type=wvc.config.DataType.DATE),
                wvc.config.Property(name="supersedes_id", data_type=wvc.config.DataType.TEXT),
            ],
        )

        _ensure_collection(
            client,
            "MoodPattern",
            [
                wvc.config.Property(name="user_id", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="bucket", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="energy", data_type=wvc.config.DataType.NUMBER),
                wvc.config.Property(name="valence", data_type=wvc.config.DataType.NUMBER),
                wvc.config.Property(name="tempo", data_type=wvc.config.DataType.NUMBER),
                wvc.config.Property(name="label", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="sample_size", data_type=wvc.config.DataType.INT),
                wvc.config.Property(name="created_at", data_type=wvc.config.DataType.DATE),
            ],
        )
    finally:
        client.close()


async def init_weaviate_schema() -> None:
    """Async wrapper around ``init_weaviate_schema_sync``.

    Runs the synchronous Weaviate calls in a thread pool so the FastAPI
    lifespan startup does not block the event loop.
    """
    await asyncio.to_thread(init_weaviate_schema_sync)
