
from project.database import Base
from fastapi_users.db import SQLAlchemyBaseAccessTokenTableUUID, SQLAlchemyAccessTokenDatabase
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends


from project.database import get_async_session


class AccessToken(SQLAlchemyBaseAccessTokenTableUUID, Base):  
    pass


async def get_access_token_db(
    session: AsyncSession = Depends(get_async_session),
):  
    yield SQLAlchemyAccessTokenDatabase(session, AccessToken)