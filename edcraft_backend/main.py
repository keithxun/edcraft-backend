import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.sessions import SessionMiddleware

from edcraft_backend.config import settings
from edcraft_backend.database import close_db
from edcraft_backend.dependencies import get_ai_authoring_service
from edcraft_backend.exceptions import EdCraftBaseException
from edcraft_backend.routers import (
    ai_authoring,
    ai_authoring_worker,
    assessment_templates,
    assessments,
    auth,
    collaborators,
    folders,
    input_generator,
    jobs,
    provider_connections,
    question_bank,
    question_generation,
    question_template_banks,
    question_templates,
    questions,
    users,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan events."""
    authoring_service = get_ai_authoring_service()
    await authoring_service.startup()
    try:
        yield
    finally:
        await authoring_service.shutdown()
        await close_db()


app = FastAPI(
    title=settings.app.name,
    description="API for EdCraft Backend",
    version=settings.app.version,
    lifespan=lifespan,
)


@app.exception_handler(EdCraftBaseException)
async def edcraft_exception_handler(
    request: Request, exc: EdCraftBaseException
) -> JSONResponse:
    """Handle custom EdCraft exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message, "error_type": exc.__class__.__name__},
    )


# SessionMiddleware required for OAuth (Authlib stores state in session)
app.add_middleware(
    SessionMiddleware,
    secret_key=settings.session_secret,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors.origins,
    allow_credentials=settings.cors.allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers (prefix and tags are defined in each router file)
app.include_router(input_generator.router)
app.include_router(question_generation.router)
app.include_router(jobs.router)
app.include_router(ai_authoring.router)
app.include_router(ai_authoring_worker.router)
app.include_router(provider_connections.router)
app.include_router(users.router)
app.include_router(auth.router)
app.include_router(folders.router)
app.include_router(questions.router)
app.include_router(assessments.router)
app.include_router(question_templates.router)
app.include_router(assessment_templates.router)
app.include_router(question_bank.router)
app.include_router(question_template_banks.router)
app.include_router(collaborators.router)


@app.get("/")
async def index() -> dict[str, str]:
    return {"message": "Edcraft API"}


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "environment": settings.app.env.value,
        "version": settings.app.version,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.server.host,
        port=settings.server.port,
        log_level=settings.log_level,
    )
