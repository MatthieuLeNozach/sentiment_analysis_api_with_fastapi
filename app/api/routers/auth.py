from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import ValidationError
from sqlalchemy.orm import Session

from api.models import User
from api.auth_utils import (
    authenticate_user,
    bcrypt_context,
    create_access_token,
    db_dependency,
    get_current_user,
)
from api.common_utils.schemas import CreateUser, Token, TokenData

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/create", status_code=status.HTTP_201_CREATED)
async def create_user(db: db_dependency, create_user_request: CreateUser) -> None:
    try:
        CreateUser.parse_obj(create_user_request.dict())
        password = create_user_request.password
        create_user_data = create_user_request.dict(exclude={"password"})
        create_user_data.update(
            {
                "role": "user",
                "is_active": True,
                "has_access_sentiment": False,
                "has_access_emotion": False,
            }
        )
        create_user_model = User(**create_user_data)
        create_user_model.hashed_password = bcrypt_context.hash(password)

        db.add(create_user_model)
        db.commit()

    except ValidationError as e:
        print(f"Validation Error: {e}")
        return {"detail": str(e)}

@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: db_dependency
) -> dict:
    user = authenticate_user(form_data.username, form_data.password, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate user"
        )

    token_data = TokenData(
        username=user.username,
        user_id=user.id,
        role=user.role,
        has_access_sentiment=user.has_access_sentiment,
        has_access_emotion=user.has_access_emotion,
        expires_delta=timedelta(minutes=20),
    )
    token = create_access_token(token_data)
    return {"access_token": token, "token_type": "bearer"}