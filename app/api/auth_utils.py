import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from jose import jwt
from passlib.context import CryptContext
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.api.models import User
from app.commmon_utils.schemas import TokenData
from fastapi.security import OAuth2PasswordBearer

SECRET_KEY = os.environ["SECRET_KEY"]
from typing import Annotated  # Import the Annotated class from the typing module

from app.database import get_db  # Import the get_db function from the app.database module

ALGORITHM = "HS256"

bcrypt_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Dependencies
db_dependency = Annotated[Session, Depends(get_db)]
oauth2bearer = OAuth2PasswordBearer(tokenUrl="auth/token")

def authenticate_user(username: str, password: str, db) -> Optional[User]:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return False
    if not bcrypt_context.verify(password, user.hashed_password):
        return False
    return user

def create_access_token(token_data: TokenData) -> str:
    encode = token_data.dict()
    expires = datetime.utcnow() + token_data.expires_delta
    encode.update({"exp": expires})
    encode.pop("expires_delta", None)
    return jwt.encode(encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: Annotated[str, Depends(oauth2bearer)]) -> dict:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("username")
        user_id: int = payload.get("user_id")
        role: str = payload.get("role")
        has_access_sentiment: bool = payload.get("has_access_sentiment")
        has_access_emotion: bool = payload.get("has_access_emotion")

        if username is None or user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate user",
            )
        return {
            "username": username,
            "id": user_id,
            "role": role,
            "has_access_sentiment": has_access_sentiment,
            "has_access_emotion": has_access_emotion,
        }
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate user"
        )