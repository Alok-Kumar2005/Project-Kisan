from sqlalchemy import (
    Column, Integer, String, DateTime, Float, ForeignKey, Index
)
from sqlalchemy.orm import declarative_base, relationship
import bcrypt
from datetime import datetime

Base = declarative_base()


class User(Base):
    """User model for storing user account data."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # Identity / login
    unique_name = Column(String(50), unique=True, nullable=False, index=True,
                         comment="Qdrant collection name + JWT sub")
    phone_number = Column(String(20), unique=True, nullable=False,
                          comment="Login credential — E.164 format")

    # Authentication
    hashed_password = Column(String(128), nullable=False, comment="bcrypt hash")

    # Profile
    full_name = Column(String(100), nullable=True, comment="Full name of user")
    age = Column(Integer, nullable=True, comment="Age of the user")
    resident = Column(String(200), nullable=True, comment="Residential address")
    city = Column(String(100), nullable=True, comment="City of user")
    district = Column(String(100), nullable=True, comment="District of user")
    state = Column(String(100), nullable=True, comment="State of user")
    country = Column(String(100), nullable=True, comment="Country of user")

    # Location coordinates (for farmer directory)
    latitude = Column(Float, nullable=True, comment="GPS latitude")
    longitude = Column(Float, nullable=True, comment="GPS longitude")

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, comment="Record creation timestamp")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow,
                        comment="Record update timestamp")

    # Relationships
    chats = relationship("Chat", back_populates="user", cascade="all, delete-orphan")
    farmer_location = relationship("FarmerLocation", back_populates="user",
                                   uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, unique_name='{self.unique_name}')>"

    @staticmethod
    def _truncate_password(password: str) -> bytes:
        """Encode and truncate password to 72 bytes (bcrypt hard limit)."""
        return password.encode("utf-8")[:72]

    @staticmethod
    def hash_password(password: str) -> str:
        """Generate a bcrypt hash for the password."""
        return bcrypt.hashpw(
            User._truncate_password(password),
            bcrypt.gensalt()
        ).decode("utf-8")

    def verify_password(self, password: str) -> bool:
        """Verify a plaintext password against the stored hash."""
        return bcrypt.checkpw(
            User._truncate_password(password),
            self.hashed_password.encode("utf-8")
        )

    def to_dict(self):
        """Convert user object to dictionary, excluding sensitive fields."""
        return {
            "id": self.id,
            "unique_name": self.unique_name,
            "phone_number": self.phone_number,
            "full_name": self.full_name,
            "age": self.age,
            "resident": self.resident,
            "city": self.city,
            "district": self.district,
            "state": self.state,
            "country": self.country,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Chat(Base):
    """Chat session metadata — one row per conversation thread."""
    __tablename__ = "chats"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    thread_id = Column(String(64), unique=True, nullable=False,
                       comment="Format: user_{user_id}_{uuid8}")
    name = Column(String(50), nullable=True,
                  comment="Set from first 50 chars of first message")
    message_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    user = relationship("User", back_populates="chats")

    __table_args__ = (
        Index("idx_chats_user_id", "user_id"),
    )

    def __repr__(self):
        return f"<Chat(id={self.id}, thread_id='{self.thread_id}', user_id={self.user_id})>"

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "thread_id": self.thread_id,
            "name": self.name,
            "message_count": self.message_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class FarmerLocation(Base):
    """One authoritative location record per farmer — upserted on profile create/update."""
    __tablename__ = "farmer_locations"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"),
                     unique=True, nullable=False,
                     comment="One row per user — UNIQUE enforced")
    phone_number = Column(String(20), nullable=False)
    district = Column(String(100), nullable=True)
    state = Column(String(100), nullable=True)
    country = Column(String(100), nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationship
    user = relationship("User", back_populates="farmer_location")

    def __repr__(self):
        return f"<FarmerLocation(user_id={self.user_id}, lat={self.latitude}, lng={self.longitude})>"

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "phone_number": self.phone_number,
            "district": self.district,
            "state": self.state,
            "country": self.country,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
