import os
from datetime import datetime, timedelta
from common_utils.schemas import TokenData
from jose import jwt


SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"


def create_access_token(token_data: TokenData) -> str:
    encode = token_data.dict()
    expires = datetime.utcnow() + token_data.expires_delta
    encode.update({"exp": expires})
    encode.pop("expires_delta", None)
    return jwt.encode(encode, SECRET_KEY, algorithm=ALGORITHM)