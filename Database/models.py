from sqlalchemy import Column, Integer , String , DateTime , create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from passlib.context import CryptContext
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class User(Base):
    """User model for storing user data"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(100), nullable=False, comment="Full name of user")
    unique_name = Column(String(50), unique=True, nullable=False, index=True, comment="Unique name")
    age = Column(Integer, nullable=False, comment="Age of the user")
    phone_number = Column(String(15), nullable=False, comment="Phone number of user")

    # authentication
    password_hash = Column(String(128), nullable=False, comment="Hashed password of user")

    # address related data
    resident = Column(String(200), nullable=True, comment="Residential address")
    city = Column(String(100), nullable=True, comment="City of user")
    district = Column(String(100), nullable=False, comment="District of user")
    state = Column(String(100), nullable=False, comment="State of user")
    country = Column(String(100), nullable=False, comment="Country of user")

    # metadata
    created_at = Column(DateTime, default=datetime.utcnow, comment="Record creation timestamp")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="Record update timestamp")

    def __repr__(self):
        return f"<User(id={self.id}, unique_name='{self.unique_name}', name='{self.name}')>"

    @staticmethod
    def hash_password(password: str) -> str:
        """Generate a bcrypt hash for the password."""
        return pwd_context.hash(password)

    def verify_password(self, password: str) -> bool:
        """Verify a plaintext password against the stored hash."""
        return pwd_context.verify(password, self.password_hash)

    def to_dict(self):
        """Convert user object to dictionary, excluding sensitive fields."""
        return {
            'id': self.id,
            'name': self.name,
            'unique_name': self.unique_name,
            'age': self.age,
            'phone_number': self.phone_number,
            'resident': self.resident,
            'city': self.city,
            'district': self.district,
            'state': self.state,
            'country': self.country,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(
    DATABASE_URL,
    echo=os.getenv("DB_ECHO", "False").lower() == "true"
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables():
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Yield a database session and ensure it is closed after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
