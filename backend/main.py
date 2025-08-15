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
    create_tables()  
    yield
    print("Shutting down AI Companion Backend...")


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


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )