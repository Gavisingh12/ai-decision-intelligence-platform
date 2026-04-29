"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from cachetools import TTLCache
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.api.routes import assistant, auth, classification, data, forecasting, health, metrics, rag, tasks
from backend.core.config import get_settings
from backend.core.logging import configure_logging
from backend.db.session import init_db
from backend.services.persistence import hub_persistence_service
from backend.services.storage import ensure_app_directories


configure_logging()
logger = logging.getLogger(__name__)
settings = get_settings()
request_window: TTLCache[str, int] = TTLCache(maxsize=4096, ttl=60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize runtime dependencies at startup."""

    del app
    ensure_app_directories()
    hub_persistence_service.restore_runtime_data()
    init_db()
    logger.info("Application initialized.")
    yield


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Multi-modal decision intelligence platform with forecasting, RAG, and explainability.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Add request timing and structured logging."""

    if request.url.path.startswith(settings.api_v1_prefix) and not request.url.path.endswith("/health"):
        client_host = request.client.host if request.client else "unknown"
        request_key = f"{client_host}:{request.url.path.split('?')[0]}"
        request_count = request_window.get(request_key, 0) + 1
        request_window[request_key] = request_count
        if request_count > settings.request_limit_per_minute:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Too many requests in a short time. Please wait a moment and try again."
                },
            )

    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info("%s %s -> %s (%sms)", request.method, request.url.path, response.status_code, duration_ms)
    response.headers["X-Process-Time-Ms"] = str(duration_ms)
    return response


@app.exception_handler(ValueError)
async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
    """Return clean JSON for uncaught value errors."""

    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Return a normalized validation error response."""

    return JSONResponse(status_code=422, content={"detail": exc.errors()})


app.include_router(health.router, prefix=settings.api_v1_prefix)
app.include_router(auth.router, prefix=settings.api_v1_prefix)
app.include_router(data.router, prefix=settings.api_v1_prefix)
app.include_router(forecasting.router, prefix=settings.api_v1_prefix)
app.include_router(classification.router, prefix=settings.api_v1_prefix)
app.include_router(rag.router, prefix=settings.api_v1_prefix)
app.include_router(assistant.router, prefix=settings.api_v1_prefix)
app.include_router(metrics.router, prefix=settings.api_v1_prefix)
app.include_router(tasks.router, prefix=settings.api_v1_prefix)

static_dir = Path(__file__).resolve().parents[1] / "frontend"


@app.get("/static/config.js", include_in_schema=False)
async def serve_frontend_config() -> Response:
    """Serve lightweight frontend runtime configuration."""

    api_base_url = settings.public_api_base_url or ""
    payload = (
        "window.DECISION_ASSISTANT_CONFIG = window.DECISION_ASSISTANT_CONFIG || "
        f"{{ apiBaseUrl: {api_base_url!r} }};"
    )
    return Response(content=payload, media_type="application/javascript")


app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", include_in_schema=False)
async def serve_frontend() -> FileResponse:
    """Serve the single-page frontend."""

    return FileResponse(static_dir / "index.html")
