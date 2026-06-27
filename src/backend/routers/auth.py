from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import HTTPAuthorizationCredentials
from datetime import timedelta

from src.backend.schemas.schemas import (
    UserRegister, UserLogin, UserResponse,
    TokenResponse, RefreshTokenRequest
)
from src.backend.core.config import settings
from src.backend.core.auth import (
    create_access_token, create_refresh_token,
    verify_refresh_token, security
)
from src.database.database import user_db, farmer_location_db

router = APIRouter()


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserRegister):
    """Register a new user."""
    # Check unique_name uniqueness
    if await user_db.user_exists(user_data.unique_name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this unique name already exists",
        )

    # Check phone_number uniqueness (requirement 6.2)
    if await user_db.get_user_by_phone(user_data.phone_number):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Phone number already registered",
        )

    user = await user_db.create_user(user_data.model_dump())
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to create user. Please check your data and try again.",
        )

    # Upsert farmer location record if lat/lng provided (requirement 5.1, 5.5)
    if user_data.latitude is not None and user_data.longitude is not None:
        await farmer_location_db.upsert(
            user_id=user.id,
            phone_number=user.phone_number,
            latitude=user_data.latitude,
            longitude=user_data.longitude,
            district=user_data.district,
            state=user_data.state,
            country=user_data.country,
        )

    token_data = {"sub": user.unique_name, "user_id": user.id}
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    user_response = UserResponse(**user.to_dict())

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=user_response,
    )


@router.post("/login", response_model=TokenResponse)
async def login_user(login_data: UserLogin):
    """Login user by phone_number (E.164) and password."""
    # Look up by phone number (requirement 6.3, 6.4)
    user_dict = await user_db.get_user_by_phone(login_data.phone_number)
    if not user_dict:
        # Return identical error regardless of whether phone is unregistered
        # or password is wrong — requirement 6.4
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    user_obj = await user_db.get_user_by_id(user_dict["id"])
    if not user_obj:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    if not user_obj.verify_password(login_data.password):
        # Same error message as "phone not found" — requirement 6.4
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    # JWT sub = unique_name (NOT phone_number) — downstream uses unique_name
    # as Qdrant collection name and user identifier (requirement 6.3)
    token_data = {"sub": user_obj.unique_name, "user_id": user_obj.id}
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)

    user_response = UserResponse(**user_dict)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=user_response,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_access_token(refresh_data: RefreshTokenRequest):
    """Refresh access token using a valid refresh token."""
    user = await verify_refresh_token(refresh_data.refresh_token)

    token_data = {"sub": user["unique_name"], "user_id": user["id"]}
    access_token = create_access_token(token_data)
    new_refresh_token = create_refresh_token(token_data)

    user_response = UserResponse(**user)

    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=user_response,
    )


@router.post("/logout")
async def logout_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Logout (client removes tokens from storage)."""
    return {"message": "Successfully logged out"}
