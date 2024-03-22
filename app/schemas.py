# file: app/schemas.py

from datetime import timedelta, datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr, SecretStr, Field


class BaseUser(BaseModel):
    username: EmailStr
    
    class Config:
        from_attributes = True



class CreateUser(BaseUser):
    first_name: str
    last_name: str
    password: str
    country: Optional[str] = None
    
    
class CreateAdmin(BaseUser):
    first_name: str
    last_name: str
    password: str 
    country: Optional[str] = None

    role: str = "admin"  
    is_active: bool = True  
    has_access_sentiment: bool = True  
    has_access_emotion: bool = True  


class UserInfo(BaseUser):
    first_name: str
    last_name: str
    role: str
    country: Optional[str] = None

    
    
class ReadUser(UserInfo):
    id: int
    is_active: bool
    has_access_sentiment: bool
    has_access_emotion: bool



    
class ChangeUserAccessRights(BaseModel):
    is_active: bool
    has_access_sentiment: bool
    has_access_emotion: bool
    

class UserVerification(BaseModel):
    password: str
    new_password: str = Field(min_length=4)




    
class Token(BaseModel):
    access_token: str
    token_type: str
    
class TokenData(BaseModel):
    username: str
    user_id: int
    role: str
    has_access_sentiment: bool
    has_access_emotion: bool
    expires_delta: timedelta



class PredictionInput(BaseModel):
    text: str

class PredictionOutputSentiment(BaseModel):
    sentiment_0: float
    sentiment_1: float
    sentiment_2: float
    sentiment_3: float
    sentiment_4: float
    
from typing import List, Tuple

class PredictionOutputEmotion(BaseModel):
    emotions: List[Tuple[str, float]]


class ServiceCallCreate(BaseModel):
    service_version: str
    success: bool
    owner_id: int
    request_time: datetime
    completion_time: datetime
    duration: float
    
    class Config:
        from_attributes = True
        

class ServiceCallRead(ServiceCallCreate):
    id: int


### UNUSED ###
class CreateUserRequest(BaseUser):
    first_name: str
    last_name: str
    password: str
    country: Optional[str] = None