from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from backend.config import get_settings
from backend.db.pool import create_pool, close_pool
from backend.db.health import run_all_checks
from backend.api.ingest import router as ingest_router
from backend.api.query import router as query_router
from backend.api.pipelines import router as pipelines_router
from backend.api.compare import router as compare_router
from backend.api.finetune import router as finetune_router

settings = get_settings()


def setup_telemetry():
    provider = TracerProvider()
    exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    setup_telemetry()
    await create_pool(dsn=settings.asyncpg_dsn)
    yield
    # Shutdown
    await close_pool()


app = FastAPI(
    title="NeuroFlow API",
    description="Multi-Modal LLM Orchestration Platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FastAPIInstrumentor.instrument_app(app)

# Include routers
app.include_router(ingest_router, prefix="/api/v1", tags=["ingestion"])
app.include_router(query_router, prefix="/api/v1", tags=["query"])
app.include_router(pipelines_router, prefix="/api/v1", tags=["pipelines"])
app.include_router(compare_router, prefix="/api/v1", tags=["compare"])
app.include_router(finetune_router, prefix="/api/v1", tags=["finetune"])


@app.get("/health")
async def health():
    checks = await run_all_checks()
    status = "ok" if checks["all_ok"] else "degraded"

    # Include circuit breaker states if available
    try:
        from backend.resilience.circuit_breaker import CircuitBreaker
        cb_states = CircuitBreaker.get_all_states()
    except Exception:
        cb_states = {}

    any_open = any(v.get("state") == "OPEN" for v in cb_states.values())
    if any_open:
        status = "degraded"

    return {
        "status": status,
        "checks": {
            "postgres": checks["postgres"],
            "redis": checks["redis"],
            "mlflow": checks["mlflow"],
            "circuit_breakers": cb_states,
        },
    }


@app.get("/metrics")
async def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
