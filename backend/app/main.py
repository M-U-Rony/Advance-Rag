"""FastAPI application factory; also serves the built React frontend."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api.routes import router
from app.config import Settings
from app.runtime import build_pipeline


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI app (pass ``settings`` in tests to avoid reading the environment)."""
    settings = settings or Settings()
    app = FastAPI(
        title="Advance RAG",
        version=__version__,
        description="Advanced RAG over PDFs with a fully inspectable pipeline.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.pipeline = build_pipeline(settings)
    app.include_router(router)

    index = settings.static_dir / "index.html"
    if index.exists():
        app.mount("/assets", StaticFiles(directory=settings.static_dir / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            candidate = (settings.static_dir / path).resolve()
            if path and candidate.is_file() and settings.static_dir.resolve() in candidate.parents:
                return FileResponse(candidate)
            return FileResponse(index)

    return app


app = create_app
