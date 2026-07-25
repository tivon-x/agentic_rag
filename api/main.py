from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.db.database import init_db
from api.routers.chat import router as chat_router
from api.routers.corpus import router as corpus_router
from api.routers.health import router as health_router
from api.routers.indexing import router as indexing_router
from core.settings import AppSettings, load_settings


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_app_settings() -> AppSettings:
    return load_settings(base_dir=PROJECT_ROOT, env_file=PROJECT_ROOT / ".env")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_app_settings()
    await init_db(settings)
    app.state.settings = settings
    yield


app = FastAPI(
    title="Agentic RAG API",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health_router, prefix="/api")
app.include_router(corpus_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(indexing_router, prefix="/api")
