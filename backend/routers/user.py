import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, status, Depends, Query

from backend.schemas.schemas import UserResponse, UserUpdate
from backend.core.auth import verify_token
from Database.database import user_db

router = APIRouter()


@router.get("/profile", response_model=UserResponse)
async def get_user_profile(current_user: Dict[str, Any] = Depends(verify_token)):
    """Get current user's profile"""
    return UserResponse(**current_user)


@router.put("/profile", response_model=UserResponse)
async def update_user_profile(
    user_update: UserUpdate,
    current_user: Dict[str, Any] = Depends(verify_token)
):
    """Update current user's profile"""
    # Filter out None values
    update_data = {k: v for k, v in user_update.model_dump().items() if v is not None}
    
    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No data provided for update"
        )
    
    updated_user = user_db.update_user(current_user["unique_name"], update_data)
    if not updated_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to update user profile"
        )
    
    return UserResponse(**updated_user.to_dict())


@router.delete("/profile")
async def delete_user_account(current_user: Dict[str, Any] = Depends(verify_token)):
    """Delete current user's account"""
    success = user_db.delete_user(current_user["unique_name"])
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to delete user account"
        )
    
    return {"message": "Account deleted successfully"}


@router.get("/search", response_model=List[UserResponse])
async def search_users_by_location(
    district: str = Query(None, description="District to search in"),
    state: str = Query(None, description="State to search in"),
    country: str = Query(None, description="Country to search in"),
    current_user: Dict[str, Any] = Depends(verify_token)
):
    """Search users by location parameters"""
    if not any([district, state, country]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one location parameter (district, state, country) is required"
        )
    
    users = user_db.search_users_by_location(district=district, state=state, country=country)
    
    # Convert to UserResponse and exclude current user
    user_responses = []
    for user in users:
        if user.unique_name != current_user["unique_name"]:
            user_responses.append(UserResponse(**user.to_dict()))
    
    return user_responses


@router.get("/nearby", response_model=List[UserResponse])
async def get_nearby_users(
    current_user: Dict[str, Any] = Depends(verify_token),
    include_district: bool = Query(True, description="Include users from same district"),
    include_state: bool = Query(False, description="Include users from same state")
):
    """Get users from the same location as current user"""
    search_params = {}
    
    if include_district:
        search_params["district"] = current_user.get("district")
        search_params["state"] = current_user.get("state")
        search_params["country"] = current_user.get("country")
    elif include_state:
        search_params["state"] = current_user.get("state")
        search_params["country"] = current_user.get("country")
    else:
        search_params["country"] = current_user.get("country")
    
    users = user_db.search_users_by_location(**search_params)
    
    # Convert to UserResponse and exclude current user
    user_responses = []
    for user in users:
        if user.unique_name != current_user["unique_name"]:
            user_responses.append(UserResponse(**user.to_dict()))
    
    return user_responses