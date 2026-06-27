import os
from typing import List

from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    # -----------------------------------------------------------------------
    # API
    # -----------------------------------------------------------------------
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Project-Kisan Backend"

    # -----------------------------------------------------------------------
    # Security
    # -----------------------------------------------------------------------
    SECRET_KEY: str = os.getenv(
        "SECRET_KEY",
        "dev-secret-key-change-in-production-min-32-chars-required",
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15   # spec requirement 6.3
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7      # spec requirement 6.3

    # -----------------------------------------------------------------------
    # CORS
    # -----------------------------------------------------------------------
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
    ]

    # -----------------------------------------------------------------------
    # Cloud databases (requirements 9.1, 9.4, 9.5, 9.6)
    # -----------------------------------------------------------------------
    # Neon Postgres — must use postgresql+asyncpg:// scheme
    NEON_API: str = os.getenv("NEON_API", "")

    # Qdrant Cloud
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API: str = os.getenv("QDRANT_API", "")

    # -----------------------------------------------------------------------
    # File Upload
    # -----------------------------------------------------------------------
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10 MB
    ALLOWED_FILE_TYPES: List[str] = [
        "image/jpeg", "image/png", "audio/wav", "audio/mp3",
    ]

    # -----------------------------------------------------------------------
    # AI Service
    # -----------------------------------------------------------------------
    DEFAULT_THREAD_EXPIRY: int = 3600  # seconds

    class Config:
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.SECRET_KEY == "dev-secret-key-change-in-production-min-32-chars-required":
            print(
                "⚠️  WARNING: Using default SECRET_KEY. "
                "Set SECRET_KEY in .env for production!"
            )
        # Validate required cloud vars at settings-load time so the error
        # surface is clear.  database.py also raises RuntimeError on import,
        # but surfacing it here gives a friendlier startup message.
        missing = [
            v for v in ("NEON_API", "QDRANT_URL", "QDRANT_API")
            if not getattr(self, v)
        ]
        if missing:
            print(
                f"⚠️  WARNING: Required environment variable(s) not set: "
                f"{', '.join(missing)}. "
                "The platform will fail to start until these are provided."
            )


settings = Settings()
