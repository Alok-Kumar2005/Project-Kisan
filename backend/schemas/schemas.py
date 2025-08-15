from pydantic import BaseModel, Field, validator
from typing import Optional, Literal, Union, List
from datetime import datetime


# Authentication Schemas
class UserRegister(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    unique_name: str = Field(..., min_length=3, max_length=50)
    age: int = Field(..., ge=13, le=120)
    phone_number: str = Field(..., min_length=10, max_length=15)
    password: str = Field(..., min_length=6, max_length=128)
    resident: Optional[str] = Field(None, max_length=200)
    city: Optional[str] = Field(None, max_length=100)
    district: str = Field(..., min_length=1, max_length=100)
    state: str = Field(..., min_length=1, max_length=100)
    country: str = Field(..., min_length=1, max_length=100)
    
    @validator('unique_name')
    def validate_unique_name(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Unique name can only contain letters, numbers, hyphens and underscores')
        return v.lower()
    
    @validator('phone_number')
    def validate_phone_number(cls, v):
        # Basic phone number validation
        cleaned = v.replace('+', '').replace('-', '').replace(' ', '')
        if not cleaned.isdigit():
            raise ValueError('Phone number must contain only digits, +, -, and spaces')
        return v


class UserLogin(BaseModel):
    unique_name: str
    password: str


class UserResponse(BaseModel):
    id: int
    name: str
    unique_name: str
    age: int
    phone_number: str
    resident: Optional[str]
    city: Optional[str]
    district: str
    state: str
    country: str
    created_at: Optional[datetime]
    updated_at: Optional[datetime]


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class RefreshTokenRequest(BaseModel):
    refresh_token: str


# Chat Schemas  
class ChatMessage(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    workflow: Optional[Literal[
        "GeneralNode", 
        "DiseaseNode", 
        "WeatherNode", 
        "MandiNode", 
        "GovSchemeNode", 
        "CarbonFootprintNode"
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


# User Update Schema
class UserUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    age: Optional[int] = Field(None, ge=13, le=120)
    phone_number: Optional[str] = Field(None, min_length=10, max_length=15)
    resident: Optional[str] = Field(None, max_length=200)
    city: Optional[str] = Field(None, max_length=100)
    district: Optional[str] = Field(None, min_length=1, max_length=100)
    state: Optional[str] = Field(None, min_length=1, max_length=100)
    country: Optional[str] = Field(None, min_length=1, max_length=100)
    password: Optional[str] = Field(None, min_length=6, max_length=128)


# Error Schemas
class ErrorResponse(BaseModel):
    detail: str
    status_code: int
    timestamp: datetime = Field(default_factory=datetime.now)


# Thread Management
class ThreadCreate(BaseModel):
    thread_name: Optional[str] = Field(None, max_length=100)


class ThreadResponse(BaseModel):
    thread_id: str
    thread_name: Optional[str]
    created_at: datetime
    message_count: int = 0