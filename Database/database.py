import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import Optional, List, Dict, Any
from Database.models import User , SessionLocal , create_tables
from src.ai_component.logger import logging 


class UserDatabase:
    """Database operation for user model"""
    def __init__(self):
        create_tables()

    def create_user(self, user_data: Dict[str, Any]) -> Optional[User]:
        """Create a new user, hashing the password before storing."""
        db = SessionLocal()
        try:
            logging.info("Creating new user")
            required_fields = [
                'name', 'unique_name', 'age', 'phone_number',
                'district', 'state', 'country', 'password'
            ]
            # validate required fields
            for field in required_fields:
                if field not in user_data or not user_data[field]:
                    raise ValueError(f"Required field '{field}' is missing or empty")

            # prepare user object with hashed password
            user = User(
                name=user_data['name'].strip(),
                unique_name=user_data['unique_name'].strip().lower(),
                age=int(user_data['age']),
                phone_number=user_data['phone_number'].strip(),
                password_hash=User.hash_password(user_data['password']),
                resident=user_data.get('resident', '').strip() or None,
                city=user_data.get('city', '').strip() or None,
                district=user_data['district'].strip(),
                state=user_data['state'].strip(),
                country=user_data['country'].strip(),
            )

            db.add(user)
            db.commit()
            db.refresh(user)
            logging.info(f"User created successfully: {user.unique_name}")
            return user

        except IntegrityError:
            db.rollback()
            logging.error(
                f"User with unique_name '{user_data.get('unique_name')}' already exists"
            )
            return None
        except ValueError as e:
            db.rollback()
            logging.error(f"Validation error: {str(e)}")
            return None
        except Exception as e:
            db.rollback()
            logging.error(f"Error creating user: {str(e)}")
            return None
        finally:
            db.close()

    def get_user_by_unique_name(self, unique_name: str) -> Optional[Dict[str, Any]]:
        """Get user by unique_name and return as dictionary"""
        db = SessionLocal()
        try:
            user = (
                db.query(User).filter(User.unique_name == unique_name.lower().strip()).first()
            )
            if user:
                # Include password_hash in the returned dictionary for authentication
                user_dict = user.to_dict()
                user_dict['password_hash'] = user.password_hash
                return user_dict
            return None
        except Exception as e:
            logging.error(f"Error in getting user by unique_name: {str(e)}")
            return None
        finally:
            db.close()

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get the user by ID and return User object"""
        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            return user
        except Exception as e:
            logging.error(f"Error in getting user by id: {str(e)}")
            return None
        finally:
            db.close()

    def update_user(self, unique_name: str, update_data: Dict[str, Any]) -> Optional[User]:
        """
        Update user information; if password provided, re-hash and update hash.
        """
        db = SessionLocal()
        try:
            user = (
                db.query(User)
                .filter(User.unique_name == unique_name.lower().strip())
                .first()
            )
            if not user:
                logging.warning(f"User not found: {unique_name}")
                return None

            # handle password update
            if 'password' in update_data and update_data['password']:
                user.password_hash = User.hash_password(update_data['password'])

            # Update allowed fields
            allowed_fields = [
                'name', 'age', 'phone_number',
                'resident', 'city', 'district', 'state', 'country'
            ]
            for field, value in update_data.items():
                if field in allowed_fields and hasattr(user, field):
                    if field in ['resident', 'city'] and not value:
                        setattr(user, field, None)
                    else:
                        setattr(
                            user,
                            field,
                            value.strip() if isinstance(value, str) else value,
                        )

            db.commit()
            db.refresh(user)
            logging.info(f"User updated successfully: {user.unique_name}")
            return user

        except Exception as e:
            db.rollback()
            logging.error(f"Error updating user: {str(e)}")
            return None
        finally:
            db.close()

    def delete_user(self, unique_name: str) -> bool:
        """Delete user by unique_name"""
        db = SessionLocal()
        try:
            user = (
                db.query(User)
                .filter(User.unique_name == unique_name.lower().strip())
                .first()
            )
            if not user:
                logging.warning(f"User not found for deletion: {unique_name}")
                return False

            db.delete(user)
            db.commit()
            logging.info(f"User deleted successfully: {unique_name}")
            return True

        except Exception as e:
            db.rollback()
            logging.error(f"Error deleting user: {str(e)}")
            return False
        finally:
            db.close()

    def get_all_users(self, limit: int = 100, offset: int = 0) -> List[User]:
        """Get all users with pagination"""
        db = SessionLocal()
        try:
            return db.query(User).offset(offset).limit(limit).all()
        except Exception as e:
            logging.error(f"Error fetching all users: {str(e)}")
            return []
        finally:
            db.close()

    def search_users_by_location(self,district: str = None,state: str = None,country: str = None) -> List[User]:
        """Search users by location parameters"""
        db = SessionLocal()
        try:
            query = db.query(User)
            if country:
                query = query.filter(User.country.ilike(f"%{country}%"))
            if state:
                query = query.filter(User.state.ilike(f"%{state}%"))
            if district:
                query = query.filter(User.district.ilike(f"%{district}%"))
            return query.all()
        except Exception as e:
            logging.error(f"Error searching users by location: {str(e)}")
            return []
        finally:
            db.close()

    def user_exists(self, unique_name: str) -> bool:
        """Check if user exists by unique_name"""
        db = SessionLocal()
        try:
            exists = (
                db.query(User)
                .filter(User.unique_name == unique_name.lower().strip())
                .first()
                is not None
            )
            return exists
        except Exception as e:
            logging.error(f"Error checking user existence: {str(e)}")
            return False
        finally:
            db.close()


user_db = UserDatabase()

if __name__ == "__main__":
    test_data = {
        'name': 'Ankit badsen',
        'unique_name': 'ankit123',
        'age': 21,
        'phone_number': '+919234656408',
        'password': 'ankit123',
        'resident': 'Knkar bhag',
        'city': 'Patna',
        'district': 'Patna',
        'state': 'Bihar',
        'country': 'India'
    }

    user = user_db.create_user(test_data)
    if user:
        print("User created successfully:", user.to_dict())
    else:
        print("Failed to create user. Check logs for details.")

    print("Get user by unique name: ")
    data = user_db.get_user_by_unique_name('ankit123')
    print(data)