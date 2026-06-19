"""Health-check endpoints.

``GET /health``
    Returns liveness status of every infrastructure dependency: Postgres,
    Weaviate, and Redis.  Used by docker-compose ``healthcheck`` and load
    balancers.  Does **not** call any LLM to keep latency and cost near zero.

``GET /health/llm``
    Performs a minimal completion against the configured LLM provider.  Kept
    as a separate endpoint so normal orchestration doesn't pay the LLM
    round-trip on every health poll.
"""

import asyncio
from typing import Annotated

import weaviate
from fastapi import APIRouter, Depends
from redis.asyncio import Redis as AsyncRedis
from sqlalchemy import text

from backend.app.config import Settings
from backend.app.db.session import AsyncSessionLocal
from backend.app.dependencies import get_redis, get_settings
from backend.app.observability.logging import get_logger
from backend.app.schemas.health import HealthResponse, LLMHealthResponse

router = APIRouter(tags=["meta"])
logger = get_logger(__name__)

# =============================================================================
# Internal check helpers
# =============================================================================


async def _check_postgres() -> str:
    """Ping Postgres with a trivial query.

    Returns:
        ``"ok"`` on success, or an error string on failure.
    """
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        return "ok"
    except Exception as exc:
        logger.warning("health_postgres_failed", error=str(exc))
        return f"error: {exc}"


async def _check_weaviate(weaviate_url: str) -> str:
    """Ping the Weaviate ``/.well-known/ready`` endpoint.

    Runs in a thread pool because the weaviate-client v4 sync ``is_ready()``
    call blocks.

    Args:
        weaviate_url: Base URL of the Weaviate instance (e.g. ``http://weaviate:8080``).

    Returns:
        ``"ok"`` on success, or an error string on failure.
    """
    def _ping() -> str:
        host = weaviate_url.replace("http://", "").replace("https://", "")
        host_part, _, port_str = host.partition(":")
        port = int(port_str) if port_str else 8080
        try:
            client = weaviate.connect_to_custom(
                http_host=host_part,
                http_port=port,
                http_secure=False,
                grpc_host=host_part,
                grpc_port=50051,
                grpc_secure=False,
            )
            client.is_ready()
            client.close()
            return "ok"
        except Exception as exc:
            return f"error: {exc}"

    return await asyncio.to_thread(_ping)


async def _check_redis(redis: AsyncRedis) -> str:
    """Ping Redis.

    Args:
        redis: The app-level Redis pool (injected).

    Returns:
        ``"ok"`` on success, or an error string on failure.
    """
    try:
        await redis.ping()
        return "ok"
    except Exception as exc:
        logger.warning("health_redis_failed", error=str(exc))
        return f"error: {exc}"

# =============================================================================
# Route handlers
# =============================================================================


@router.get(
    "/health",
    summary="Check liveness of infrastructure dependencies",
    response_model=HealthResponse,
    status_code=200,
)
async def health(
    redis: Annotated[AsyncRedis, Depends(get_redis)],
    cfg: Annotated[Settings, Depends(get_settings)],
) -> HealthResponse:
    """Return liveness status of all infrastructure dependencies.

    Runs Postgres, Weaviate, and Redis checks in parallel to minimise latency.

    Args:
        redis: App-level Redis pool (injected).
        cfg:   Application settings (injected).

    Returns:
        ``HealthResponse`` with per-service status strings and active config flags.
    """
    postgres, weaviate_status, redis_status = await asyncio.gather(
        _check_postgres(),
        _check_weaviate(cfg.weaviate_url),
        _check_redis(redis),
    )

    all_ok = all(s == "ok" for s in (postgres, weaviate_status, redis_status))
    return HealthResponse(
        status="ok" if all_ok else "degraded",
        postgres=postgres,
        weaviate=weaviate_status,
        redis=redis_status,
        llm_provider=cfg.llm_provider,
        spotify_configured=cfg.spotify_configured,
    )


@router.get(
    "/health/llm",
    summary="Check configured LLM provider",
    response_model=LLMHealthResponse,
    status_code=200,
)
async def health_llm(cfg: Annotated[Settings, Depends(get_settings)]) -> LLMHealthResponse:
    """Verify the configured LLM provider returns a valid response.

    Intentionally separate from ``/health`` to avoid LLM latency and cost on
    every infrastructure poll.  Call this manually during deployment validation.

    Args:
        cfg: Application settings (injected).

    Returns:
        ``LLMHealthResponse`` with the completion status and active provider name.
    """
    try:
        from backend.app.providers.llm import get_fast_llm

        llm = get_fast_llm(cfg)
        result = llm.call([{"role": "user", "content": "Reply with the single word: ok"}])
        status = "ok" if result else "no response"
    except Exception as exc:
        logger.warning("health_llm_failed", error=str(exc))
        status = f"error: {exc}"

    return LLMHealthResponse(llm=status, provider=cfg.llm_provider)
