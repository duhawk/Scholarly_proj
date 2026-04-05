from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.auth import verify_api_key
from app.database import engine
from app.logging_config import configure_logging, get_logger
from app.middleware import RequestIDMiddleware
from app.routes import ingest, query, documents, eval as eval_route

configure_logging()
logger = get_logger(__name__)

_auth = [Depends(verify_api_key)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: warm up ML models. Shutdown: dispose DB engine."""
    logger.info("startup_begin")

    try:
        from app.retrieval.embedder import warmup as warmup_embedder
        warmup_embedder()
        logger.info("embedder_warmed_up")
    except Exception as e:
        logger.warning("embedder_warmup_failed", extra={"error": str(e)})

    try:
        from app.retrieval.reranker import warmup as warmup_reranker
        warmup_reranker()
        logger.info("reranker_warmed_up")
    except Exception as e:
        logger.warning("reranker_warmup_failed", extra={"error": str(e)})

    logger.info("startup_complete")
    yield

    await engine.dispose()
    logger.info("shutdown_complete")


app = FastAPI(
    title="Scholarly — Academic RAG Pipeline",
    description="Production-grade RAG system for academic PDF question answering",
    version="1.0.0",
    lifespan=lifespan,
)

# Request-ID tracking + latency logging on every request
app.add_middleware(RequestIDMiddleware)

# Prometheus metrics at /metrics (unauthenticated — typically scraped internally)
Instrumentator().instrument(app).expose(app, include_in_schema=False)

# Auth applied per-router so /health and /metrics stay open
app.include_router(ingest.router, dependencies=_auth)
app.include_router(query.router, dependencies=_auth)
app.include_router(documents.router, dependencies=_auth)
app.include_router(eval_route.router, dependencies=_auth)


@app.get("/health", include_in_schema=False)
async def health():
    return {"status": "ok"}
