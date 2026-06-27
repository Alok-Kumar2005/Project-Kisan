from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
import uvicorn
from contextlib import asynccontextmanager

from src.backend.routers import auth, chat, user
from src.backend.core.config import settings
from src.backend.core.auth import verify_token


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting Project-Kisan Backend...")

    # ------------------------------------------------------------------
    # 1. Neon Postgres — create/verify all relational tables
    # ------------------------------------------------------------------
    try:
        from src.database.database import init_db
        print("Initialising Neon Postgres tables via init_db()...")
        await init_db()
        print("Neon Postgres tables ready.")
    except RuntimeError as e:
        # NEON_API not set — hard stop is intentional (requirement 9.6)
        print(f"FATAL: {e}")
        raise
    except Exception as e:
        print(f"Error during Neon Postgres init: {e}")
        raise

    # ------------------------------------------------------------------
    # 2. LangGraph components — AsyncPostgresSaver + AsyncPostgresStore
    # ------------------------------------------------------------------
    try:
        from src.ai_component.graph.graph import get_saver, get_store, get_graph
        print("Setting up LangGraph AsyncPostgresSaver (short-term memory)...")
        await get_saver()
        print("Setting up LangGraph AsyncPostgresStore (long-term memory)...")
        await get_store()
        print("Initialising LangGraph compiled graph...")
        await get_graph()
        print("LangGraph components ready.")
    except Exception as e:
        print(f"Error initialising LangGraph components: {e}")
        # Non-fatal: server can still serve auth + user routes;
        # chat endpoints will fail gracefully at request time.

    print("Project-Kisan Backend startup complete.")
    yield

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------
    print("Shutting down Project-Kisan Backend...")
    try:
        from src.ai_component.graph.graph import cleanup_database
        await cleanup_database()
        print("LangGraph connections cleaned up.")
    except Exception as e:
        print(f"Error during shutdown cleanup: {e}")


app = FastAPI(
    title="Project-Kisan Backend",
    description=(
        "Agentic farming assistant — FastAPI backend with Neon Postgres, "
        "Qdrant Cloud, and LangGraph."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# Routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(
    chat.router,
    prefix="/api/v1/chat",
    tags=["Chat"],
    dependencies=[Depends(verify_token)],
)
app.include_router(
    user.router,
    prefix="/api/v1/user",
    tags=["User"],
    dependencies=[Depends(verify_token)],
)


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "version": "2.0.0"}


# ---------------------------------------------------------------------------
# Static frontend — must be mounted LAST so it never shadows API routes
# ---------------------------------------------------------------------------
import os
from fastapi.staticfiles import StaticFiles

frontend_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'frontend')
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run(
        "src.backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
