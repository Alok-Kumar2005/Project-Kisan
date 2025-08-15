import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import HTTPAuthorizationCredentials
from datetime import timedelta

from backend.schemas.schemas import (
    UserRegister, UserLogin, UserResponse, 
    TokenResponse, RefreshTokenRequest
)
from backend.core.config import settings
from backend.core.auth import (
    create_access_token, create_refresh_token, 
    verify_refresh_token, security
)
from Database.database import user_db
from Database.models import User

router = APIRouter()


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserRegister):
    """Register a new user"""
    # Check if user already exists
    if user_db.user_exists(user_data.unique_name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this unique name already exists"
        )
    
    # Create user
    user = user_db.create_user(user_data.model_dump())
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to create user. Please check your data and try again."
        )
    
    # Create tokens
    token_data = {"sub": user.unique_name, "user_id": user.id}
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)
    
    user_response = UserResponse(**user.to_dict())
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=user_response
    )


@router.post("/login", response_model=TokenResponse)
async def login_user(login_data: UserLogin):
    """Login user"""
    # Get user from database
    user_dict = user_db.get_user_by_unique_name(login_data.unique_name)
    if not user_dict:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Get the actual User object from database for password verification
    user_obj = user_db.get_user_by_id(user_dict['id'])
    if not user_obj:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Verify password using the actual User object
    if not user_obj.verify_password(login_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Create tokens
    token_data = {"sub": user_obj.unique_name, "user_id": user_obj.id}
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)
    
    user_response = UserResponse(**user_dict)
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=user_response
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_access_token(refresh_data: RefreshTokenRequest):
    """Refresh access token using refresh token"""
    user = verify_refresh_token(refresh_data.refresh_token)
    
    # Create new tokens
    token_data = {"sub": user["unique_name"], "user_id": user["id"]}
    access_token = create_access_token(token_data)
    new_refresh_token = create_refresh_token(token_data)
    
    user_response = UserResponse(**user)
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=user_response
    )


@router.post("/logout")
async def logout_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Logout user (client should remove tokens)"""
    return {"message": "Successfully logged out"}