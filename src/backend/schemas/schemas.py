from pydantic import BaseModel, Field, validator
from typing import Optional, Literal, Union, List
from datetime import datetime


# ---------------------------------------------------------------------------
# Authentication Schemas
# ---------------------------------------------------------------------------

class UserRegister(BaseModel):
    # Required identity fields
    unique_name: str = Field(..., min_length=3, max_length=50)
    phone_number: str = Field(..., min_length=8, max_length=20,
                              description="E.164 format: +<country_code><number>")
    password: str = Field(..., min_length=8, max_length=128)

    # Profile — optional at registration (can be filled later)
    full_name: Optional[str] = Field(None, max_length=100, alias="name")
    age: Optional[int] = Field(None, ge=0, le=150)
    resident: Optional[str] = Field(None, max_length=200)
    city: Optional[str] = Field(None, max_length=100)
    district: Optional[str] = Field(None, max_length=100)
    state: Optional[str] = Field(None, max_length=100)
    country: Optional[str] = Field(None, max_length=100)
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    model_config = {"populate_by_name": True}

    @validator("unique_name")
    def validate_unique_name(cls, v):
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "Unique name can only contain letters, numbers, hyphens, and underscores"
            )
        return v.lower()

    @validator("phone_number")
    def validate_phone_number(cls, v):
        # Basic E.164 check: + followed by 8-15 digits
        import re
        if not re.match(r"^\+\d{8,15}$", v):
            raise ValueError(
                "Phone number must be in E.164 format: a leading '+' followed by 8–15 digits"
            )
        return v


class UserLogin(BaseModel):
    """Login by phone_number (E.164 format) and password."""
    phone_number: str = Field(..., description="E.164 format: +<digits>")
    password: str

    @validator("phone_number")
    def validate_phone_number(cls, v):
        import re
        if not re.match(r"^\+\d{8,15}$", v):
            raise ValueError(
                "Phone number must be in E.164 format: a leading '+' followed by 8–15 digits"
            )
        return v


class UserResponse(BaseModel):
    id: int
    unique_name: str
    phone_number: str
    full_name: Optional[str] = None
    age: Optional[int] = None
    resident: Optional[str] = None
    city: Optional[str] = None
    district: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    model_config = {"populate_by_name": True}


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class RefreshTokenRequest(BaseModel):
    refresh_token: str


# ---------------------------------------------------------------------------
# User Update
# ---------------------------------------------------------------------------

class UserUpdate(BaseModel):
    full_name: Optional[str] = Field(None, max_length=100, alias="name")
    age: Optional[int] = Field(None, ge=0, le=150)
    phone_number: Optional[str] = Field(None, min_length=8, max_length=20)
    resident: Optional[str] = Field(None, max_length=200)
    city: Optional[str] = Field(None, max_length=100)
    district: Optional[str] = Field(None, max_length=100)
    state: Optional[str] = Field(None, max_length=100)
    country: Optional[str] = Field(None, max_length=100)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    password: Optional[str] = Field(None, min_length=8, max_length=128)

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Chat Schemas
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    workflow: Optional[Literal[
        "GeneralNode",
        "DiseaseNode",
        "WeatherNode",
        "MandiNode",
        "GovSchemeNode",
        "CarbonFootprintNode",
    ]] = "GeneralNode"
    thread_id: Optional[str] = None
    stream: bool = True


class ChatResponse(BaseModel):
    message: str
    message_type: Literal["text", "error"] = "text"
    thread_id: str
    workflow_used: str
    timestamp: datetime


class ChatStreamChunk(BaseModel):
    content: str
    chunk_type: Literal["text", "tool_call", "tool_result", "final"] = "text"
    thread_id: str
    timestamp: datetime


class MediaResponse(BaseModel):
    content: Union[str, bytes]
    media_type: Literal["text", "image", "voice"]
    thread_id: str
    timestamp: datetime


# ---------------------------------------------------------------------------
# Error Schemas
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    detail: str
    status_code: int
    timestamp: datetime = Field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Thread / Chat Session Management
# ---------------------------------------------------------------------------

class ThreadCreate(BaseModel):
    thread_name: Optional[str] = Field(None, max_length=100)


class ThreadResponse(BaseModel):
    thread_id: str
    thread_name: Optional[str] = None
    created_at: datetime
    message_count: int = 0
