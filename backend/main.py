import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
import uvicorn
from contextlib import asynccontextmanager

from backend.routers import auth, chat, user
from backend.core.config import settings
from backend.core.auth import verify_token
from Database.database import create_tables


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting AI Companion Backend...")
    
    # Initialize user database tables
    create_tables()
    
    # Initialize the async graph and SQLite saver
    try:
        from src.ai_component.graph.graph import initialize_database, get_async_graph
        print("Initializing chat history database...")
        await initialize_database()
        print("Initializing async graph...")
        await get_async_graph()
        print("Database and graph initialization completed successfully!")
    except Exception as e:
        print(f"Error initializing chat components: {e}")
        # Continue anyway - the system can still work for auth and user management
    
    yield
    
    print("Shutting down AI Companion Backend...")
    
    # Cleanup database connections
    try:
        from src.ai_component.graph.graph import cleanup_database
        await cleanup_database()
        print("Database connections cleaned up successfully!")
    except Exception as e:
        print(f"Error during cleanup: {e}")


app = FastAPI(
    title="AI Companion Backend",
    description="Backend service for AI Companion with authentication and chat capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(
    chat.router, 
    prefix="/api/v1/chat", 
    tags=["Chat"], 
    dependencies=[Depends(verify_token)]
)
app.include_router(
    user.router, 
    prefix="/api/v1/user", 
    tags=["User"], 
    dependencies=[Depends(verify_token)]
)


@app.get("/")
async def root():
    return {"message": "AI Companion Backend is running!"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}


# Admin endpoint for database statistics (optional)
@app.get("/admin/db-stats")
async def get_database_stats(current_user: dict = Depends(verify_token)):
    """Get database statistics (admin only or you can add role-based access)"""
    try:
        from backend.utils.db_utils import chat_history_manager
        stats = await chat_history_manager.get_database_stats()
        return stats
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving database statistics: {str(e)}"
        )


# Admin endpoint for cleanup (optional)
@app.post("/admin/cleanup-threads")
async def cleanup_old_threads(
    days_old: int = 30,
    current_user: dict = Depends(verify_token)
):
    """Clean up threads older than specified days"""
    try:
        from backend.utils.db_utils import chat_history_manager
        deleted_count = await chat_history_manager.cleanup_old_threads(days_old)
        return {
            "message": f"Successfully deleted {deleted_count} old thread checkpoints",
            "days_old": days_old
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error cleaning up threads: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )