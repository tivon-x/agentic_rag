from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.db.database import init_db
from api.routers.chat import router as chat_router
from api.routers.corpus import router as corpus_router
from api.routers.health import router as health_router
from api.routers.indexing import router as indexing_router
from api.routers.papers import router as papers_router
from api.routers.search import router as search_router
from api.services.index_worker import IndexWorker
from core.settings import AppSettings, load_settings
from indexing.index_versions import reconcile_active_pointer


def get_app_settings() -> AppSettings:
    return load_settings()


def create_app(settings: AppSettings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        resolved_settings = settings or get_app_settings()
        await init_db(resolved_settings)
        worker = IndexWorker(resolved_settings)
        application.state.settings = resolved_settings
        application.state.index_worker = worker
        if resolved_settings.index_write_mode == "versioned":
            await asyncio.to_thread(reconcile_active_pointer, resolved_settings)
            await worker.start()
        try:
            yield
        finally:
            await worker.stop()

    application = FastAPI(
        title="Agentic RAG API",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(health_router, prefix="/api")
    application.include_router(corpus_router, prefix="/api")
    application.include_router(chat_router, prefix="/api")
    application.include_router(indexing_router, prefix="/api")
    application.include_router(papers_router, prefix="/api")
    application.include_router(search_router, prefix="/api")
    return application


app = create_app()
