"""
Cloud database wiring — async SQLAlchemy on Neon Postgres.

All persistence (user accounts, chats, farmer locations, LangGraph
checkpoints/store) routes through a single NEON_API connection string.
No SQLite.  No local fallback.
"""

import os
import uuid
from typing import Optional, List, Dict, Any

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import select, update, delete, text
from sqlalchemy.exc import IntegrityError

from src.ai_component.logger import logging

# ---------------------------------------------------------------------------
# Engine / session factory
# ---------------------------------------------------------------------------

_raw_neon = os.getenv("NEON_API")
if not _raw_neon:
    raise RuntimeError(
        "NEON_API environment variable is not set. "
        "Set it to a Neon Postgres connection string."
    )

def _make_asyncpg_url(raw: str) -> str:
    """
    Convert a plain Neon connection string to one asyncpg (SQLAlchemy) accepts.

    Neon's default string looks like:
      postgresql://user:pass@host/db?sslmode=require&channel_binding=require

    asyncpg does NOT understand:
      - the 'postgresql://' scheme  → needs 'postgresql+asyncpg://'
      - 'channel_binding' param     → must be removed
      - 'sslmode'                   → use connect_args instead

    We strip all query params and pass ssl=True via connect_args.
    """
    from urllib.parse import urlparse, urlunparse
    # Ensure scheme is recognised by SQLAlchemy's asyncpg dialect
    url = raw.replace("postgresql://", "postgresql+asyncpg://", 1)
    # Strip query string — asyncpg params are set via connect_args below
    parsed = urlparse(url)
    clean = urlunparse(parsed._replace(query=""))
    return clean

NEON_API = _make_asyncpg_url(_raw_neon)

engine = create_async_engine(
    NEON_API,
    pool_pre_ping=True,
    echo=False,
    connect_args={"ssl": True},   # Neon always requires TLS
)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def init_db() -> None:
    """Create all tables on Neon Postgres if they do not already exist."""
    from src.database.models import Base  # local import avoids circular deps
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logging.info("Neon Postgres tables initialised (create_all)")


# ---------------------------------------------------------------------------
# Helper: derive a psycopg2 URL for sync operations (passlib legacy, etc.)
# ---------------------------------------------------------------------------
SYNC_DB_URL: str = NEON_API.replace("postgresql+asyncpg://", "postgresql+psycopg2://")


# ---------------------------------------------------------------------------
# UserDatabase
# ---------------------------------------------------------------------------

class UserDatabase:
    """Async CRUD for the `users` table."""

    async def create_user(self, user_data: Dict[str, Any]):
        """Create a new user, hashing the password before storing."""
        from src.database.models import User

        required_fields = ["unique_name", "phone_number", "password"]
        for field in required_fields:
            if not user_data.get(field):
                raise ValueError(f"Required field '{field}' is missing or empty")

        user = User(
            unique_name=user_data["unique_name"].strip().lower(),
            phone_number=user_data["phone_number"].strip(),
            hashed_password=User.hash_password(user_data["password"]),
            full_name=(user_data.get("full_name") or user_data.get("name") or "").strip() or None,
            age=int(user_data["age"]) if user_data.get("age") else None,
            resident=(user_data.get("resident") or "").strip() or None,
            city=(user_data.get("city") or "").strip() or None,
            district=(user_data.get("district") or "").strip() or None,
            state=(user_data.get("state") or "").strip() or None,
            country=(user_data.get("country") or "").strip() or None,
            latitude=user_data.get("latitude"),
            longitude=user_data.get("longitude"),
        )

        async with AsyncSessionLocal() as session:
            try:
                session.add(user)
                await session.commit()
                await session.refresh(user)
                logging.info(f"User created: {user.unique_name}")
                return user
            except IntegrityError:
                await session.rollback()
                logging.error(
                    f"IntegrityError creating user '{user_data.get('unique_name')}'"
                )
                return None
            except Exception as e:
                await session.rollback()
                logging.error(f"Error creating user: {e}")
                return None

    async def get_user_by_unique_name(self, unique_name: str) -> Optional[Dict[str, Any]]:
        """Return user dict (including hashed_password) or None."""
        from src.database.models import User

        async with AsyncSessionLocal() as session:
            try:
                result = await session.execute(
                    select(User).where(User.unique_name == unique_name.lower().strip())
                )
                user = result.scalar_one_or_none()
                if user:
                    d = user.to_dict()
                    d["hashed_password"] = user.hashed_password
                    d["password_hash"] = user.hashed_password  # backwards compat alias
                    return d
                return None
            except Exception as e:
                logging.error(f"Error get_user_by_unique_name: {e}")
                return None

    async def get_user_by_phone(self, phone_number: str) -> Optional[Dict[str, Any]]:
        """Return user dict (including hashed_password) or None."""
        from src.database.models import User

        async with AsyncSessionLocal() as session:
            try:
                result = await session.execute(
                    select(User).where(User.phone_number == phone_number.strip())
                )
                user = result.scalar_one_or_none()
                if user:
                    d = user.to_dict()
                    d["hashed_password"] = user.hashed_password
                    d["password_hash"] = user.hashed_password
                    return d
                return None
            except Exception as e:
                logging.error(f"Error get_user_by_phone: {e}")
                return None

    async def get_user_by_id(self, user_id: int):
        """Return User ORM object or None."""
        from src.database.models import User

        async with AsyncSessionLocal() as session:
            try:
                result = await session.execute(
                    select(User).where(User.id == user_id)
                )
                return result.scalar_one_or_none()
            except Exception as e:
                logging.error(f"Error get_user_by_id: {e}")
                return None

    async def update_user(self, unique_name: str, update_data: Dict[str, Any]):
        """Update allowed user fields; re-hash password if provided."""
        from src.database.models import User

        async with AsyncSessionLocal() as session:
            try:
                result = await session.execute(
                    select(User).where(User.unique_name == unique_name.lower().strip())
                )
                user = result.scalar_one_or_none()
                if not user:
                    logging.warning(f"User not found for update: {unique_name}")
                    return None

                if update_data.get("password"):
                    user.hashed_password = User.hash_password(update_data["password"])

                allowed_fields = [
                    "full_name", "name", "age", "phone_number",
                    "resident", "city", "district", "state", "country",
                    "latitude", "longitude",
                ]
                for field in allowed_fields:
                    if field in update_data:
                        # map legacy 'name' key to 'full_name'
                        col = "full_name" if field == "name" else field
                        val = update_data[field]
                        setattr(user, col, val.strip() if isinstance(val, str) and val else val or None)

                await session.commit()
                await session.refresh(user)
                logging.info(f"User updated: {user.unique_name}")
                return user
            except Exception as e:
                await session.rollback()
                logging.error(f"Error updating user: {e}")
                return None

    async def delete_user(self, unique_name: str) -> bool:
        """Delete user by unique_name."""
        from src.database.models import User

        async with AsyncSessionLocal() as session:
            try:
                result = await session.execute(
                    delete(User).where(User.unique_name == unique_name.lower().strip())
                )
                await session.commit()
                deleted = result.rowcount > 0
                if deleted:
                    logging.info(f"User deleted: {unique_name}")
                return deleted
            except Exception as e:
                await session.rollback()
                logging.error(f"Error deleting user: {e}")
                return False

    async def user_exists(self, unique_name: str) -> bool:
        """Return True if unique_name exists."""
        from src.database.models import User

        async with AsyncSessionLocal() as session:
            try:
                result = await session.execute(
                    select(User.id).where(User.unique_name == unique_name.lower().strip())
                )
                return result.scalar_one_or_none() is not None
            except Exception as e:
                logging.error(f"Error user_exists: {e}")
                return False


# ---------------------------------------------------------------------------
# ChatDatabase
# ---------------------------------------------------------------------------

class ChatDatabase:
    """Async CRUD for the `chats` table."""

    async def create_chat(self, user_id: int):
        """Insert a new chat row and return the Chat object."""
        from src.database.models import Chat

        thread_id = f"user_{user_id}_{uuid.uuid4().hex[:8]}"
        chat = Chat(user_id=user_id, thread_id=thread_id)

        async with AsyncSessionLocal() as session:
            try:
                session.add(chat)
                await session.commit()
                await session.refresh(chat)
                logging.info(f"Chat created: {thread_id}")
                return chat
            except Exception as e:
                await session.rollback()
                logging.error(f"Error creating chat: {e}")
                return None

    async def list_chats(self, user_id: int, limit: int = 50, offset: int = 0) -> List:
        """Return chats for a user ordered by updated_at DESC with pagination."""
        from src.database.models import Chat

        async with AsyncSessionLocal() as session:
            try:
                result = await session.execute(
                    select(Chat)
                    .where(Chat.user_id == user_id)
                    .order_by(Chat.updated_at.desc())
                    .limit(limit)
                    .offset(offset)
                )
                return result.scalars().all()
            except Exception as e:
                logging.error(f"Error list_chats: {e}")
                return []

    async def count_chats(self, user_id: int) -> int:
        """Return total number of chats for a user."""
        from src.database.models import Chat
        from sqlalchemy import func

        async with AsyncSessionLocal() as session:
            try:
                result = await session.execute(
                    select(func.count()).select_from(Chat).where(Chat.user_id == user_id)
                )
                return result.scalar_one() or 0
            except Exception as e:
                logging.error(f"Error count_chats: {e}")
                return 0

    async def get_chat(self, thread_id: str):
        """Return a single Chat by thread_id or None."""
        from src.database.models import Chat

        async with AsyncSessionLocal() as session:
            try:
                result = await session.execute(
                    select(Chat).where(Chat.thread_id == thread_id)
                )
                return result.scalar_one_or_none()
            except Exception as e:
                logging.error(f"Error get_chat: {e}")
                return None

    async def set_chat_name(self, thread_id: str, first_message: str) -> None:
        """Assign chat name from first 50 non-whitespace chars of first message."""
        from src.database.models import Chat

        name = (first_message.strip() or "New Chat")[:50]
        async with AsyncSessionLocal() as session:
            try:
                await session.execute(
                    update(Chat).where(Chat.thread_id == thread_id).values(name=name)
                )
                await session.commit()
            except Exception as e:
                await session.rollback()
                logging.error(f"Error set_chat_name: {e}")

    async def increment_message_count(self, thread_id: str) -> None:
        """Atomically increment message_count and bump updated_at."""
        from src.database.models import Chat
        from sqlalchemy import func

        async with AsyncSessionLocal() as session:
            try:
                await session.execute(
                    update(Chat)
                    .where(Chat.thread_id == thread_id)
                    .values(
                        message_count=Chat.message_count + 1,
                        updated_at=func.now(),
                    )
                )
                await session.commit()
            except Exception as e:
                await session.rollback()
                logging.error(f"Error increment_message_count: {e}")

    async def get_message_count(self, thread_id: str) -> int:
        """Return message_count for thread, or 0 if not found."""
        from src.database.models import Chat

        async with AsyncSessionLocal() as session:
            try:
                result = await session.execute(
                    select(Chat.message_count).where(Chat.thread_id == thread_id)
                )
                val = result.scalar_one_or_none()
                return val if val is not None else 0
            except Exception as e:
                logging.error(f"Error get_message_count: {e}")
                return 0

    async def delete_chat(self, thread_id: str, user_id: int) -> bool:
        """Delete chat owned by user_id; return True if deleted."""
        from src.database.models import Chat

        async with AsyncSessionLocal() as session:
            try:
                result = await session.execute(
                    delete(Chat).where(
                        Chat.thread_id == thread_id,
                        Chat.user_id == user_id,
                    )
                )
                await session.commit()
                return result.rowcount > 0
            except Exception as e:
                await session.rollback()
                logging.error(f"Error delete_chat: {e}")
                return False


# ---------------------------------------------------------------------------
# FarmerLocationDatabase
# ---------------------------------------------------------------------------

class FarmerLocationDatabase:
    """Async CRUD for the `farmer_locations` table (one row per user)."""

    async def upsert(
        self,
        user_id: int,
        phone_number: str,
        latitude: Optional[float],
        longitude: Optional[float],
        district: Optional[str] = None,
        state: Optional[str] = None,
        country: Optional[str] = None,
    ) -> None:
        """Insert or replace the farmer's location record."""
        from src.database.models import FarmerLocation

        async with AsyncSessionLocal() as session:
            try:
                result = await session.execute(
                    select(FarmerLocation).where(FarmerLocation.user_id == user_id)
                )
                loc = result.scalar_one_or_none()

                if loc is None:
                    loc = FarmerLocation(user_id=user_id)
                    session.add(loc)

                loc.phone_number = phone_number
                loc.latitude = latitude
                loc.longitude = longitude
                loc.district = district
                loc.state = state
                loc.country = country

                await session.commit()
                logging.info(f"FarmerLocation upserted for user_id={user_id}")
            except Exception as e:
                await session.rollback()
                logging.error(f"Error upserting FarmerLocation: {e}")

    async def search_nearby(
        self,
        lat: float,
        lng: float,
        radius_km: float,
        exclude_user_id: int,
    ) -> List[Dict[str, Any]]:
        """
        Return farmers within radius_km using the haversine formula.
        Excludes the requesting user and users without coordinates.
        Results ordered by ascending distance, limited to 100.
        """
        haversine_sql = text("""
            SELECT
                fl.user_id,
                fl.phone_number,
                fl.district,
                fl.state,
                fl.country,
                fl.latitude,
                fl.longitude,
                (6371 * acos(
                    LEAST(1.0,
                        cos(radians(:lat)) * cos(radians(fl.latitude)) *
                        cos(radians(fl.longitude) - radians(:lng)) +
                        sin(radians(:lat)) * sin(radians(fl.latitude))
                    )
                )) AS distance_km
            FROM farmer_locations fl
            WHERE fl.user_id != :exclude_id
              AND fl.latitude  IS NOT NULL
              AND fl.longitude IS NOT NULL
            ORDER BY distance_km ASC
            LIMIT 100
        """)

        async with AsyncSessionLocal() as session:
            try:
                rows = await session.execute(
                    haversine_sql,
                    {"lat": lat, "lng": lng, "exclude_id": exclude_user_id},
                )
                results = []
                for row in rows.mappings():
                    if row["distance_km"] <= radius_km:
                        results.append(dict(row))
                return results
            except Exception as e:
                logging.error(f"Error search_nearby: {e}")
                return []


# ---------------------------------------------------------------------------
# Module-level singletons (convenience)
# ---------------------------------------------------------------------------
user_db = UserDatabase()
chat_db = ChatDatabase()
farmer_location_db = FarmerLocationDatabase()
