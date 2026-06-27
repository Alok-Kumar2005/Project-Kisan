from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, status, Depends, Query

from src.backend.schemas.schemas import UserResponse, UserUpdate
from src.backend.core.auth import verify_token
from src.database.database import user_db, farmer_location_db

router = APIRouter()


@router.get("/profile", response_model=UserResponse)
async def get_user_profile(current_user: Dict[str, Any] = Depends(verify_token)):
    """Get current user's profile."""
    return UserResponse(**current_user)


@router.put("/profile", response_model=UserResponse)
async def update_user_profile(
    user_update: UserUpdate,
    current_user: Dict[str, Any] = Depends(verify_token),
):
    """Update current user's profile."""
    update_data = {k: v for k, v in user_update.model_dump().items() if v is not None}
    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No data provided for update",
        )

    updated_user = await user_db.update_user(current_user["unique_name"], update_data)
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to update user profile",
        )

    # Upsert farmer location record whenever lat/lng is present after update (requirement 5.5)
    # Use updated values if provided, otherwise fall back to what's now stored on the user
    updated_dict = updated_user.to_dict()
    lat = update_data.get("latitude") if "latitude" in update_data else updated_dict.get("latitude")
    lng = update_data.get("longitude") if "longitude" in update_data else updated_dict.get("longitude")
    if lat is not None and lng is not None:
        phone = update_data.get("phone_number") or current_user.get("phone_number")
        await farmer_location_db.upsert(
            user_id=current_user["id"],
            phone_number=phone,
            latitude=lat,
            longitude=lng,
            district=update_data.get("district") or updated_dict.get("district"),
            state=update_data.get("state") or updated_dict.get("state"),
            country=update_data.get("country") or updated_dict.get("country"),
        )

    return UserResponse(**updated_dict)


@router.delete("/profile")
async def delete_user_account(current_user: Dict[str, Any] = Depends(verify_token)):
    """Delete current user's account."""
    success = await user_db.delete_user(current_user["unique_name"])
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to delete user account",
        )
    return {"message": "Account deleted successfully"}


@router.get("/nearby")
async def get_nearby_users(
    radius_km: float = Query(50.0, ge=1, le=500, description="Search radius in kilometres"),
    current_user: Dict[str, Any] = Depends(verify_token),
):
    """
    Return farmers within radius_km of the current user using haversine SQL.
    Requires the current user to have latitude/longitude set.
    """
    lat = current_user.get("latitude")
    lng = current_user.get("longitude")

    if lat is None or lng is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Your location is not set. Update your profile with latitude and longitude first.",
        )

    results = await farmer_location_db.search_nearby(
        lat=lat,
        lng=lng,
        radius_km=radius_km,
        exclude_user_id=current_user["id"],
    )
    return results
