# file: app/schemas.py
"""Module: schemas.py.

This module defines the Pydantic schemas used
for data validation and serialization in the application.
The schemas serve as a layer between the API requests/responses and the database models, ensuring
that the data exchanged through the API conforms to the defined structure and constraints.

The module includes the following main categories of schemas:

User Schemas:

BaseUser: Defines the common fields for user-related schemas, such as username.
CreateUser: Represents the schema for creating a new regular user.
CreateAdmin: Represents the schema for creating a new admin user.
UserInfo: Defines the fields for displaying user information.
ReadUser: Represents the schema for reading user details, including access rights.
ChangeUserAccessRights: Defines the fields for changing user access rights.
UserVerification: Represents the schema for user password verification and update.
Token Schemas:

Token: Represents the schema for the access token returned upon successful authentication.
TokenData: Defines the fields for the data embedded in the access token.
Prediction Schemas:

PredictionInput: Represents the schema for the input data required for making predictions.
PredictionOutputSentiment / PredictionOutputEmotion: Represents the schema for the output data
returned from the prediction endpoints.
Service Call Schemas:

ServiceCallCreate: Represents the schema for creating a new service call record.
ServiceCallRead: Represents the schema for reading service call details.
The schemas are defined using Pydantic's BaseModel class, which allows for defining the structure
and data types of the fields. The schemas can include various field types, such as str, int,
bool, datetime, and Optional for optional fields.

The schemas are used in the API endpoints to validate and serialize the incoming request data and
to define the structure of the API responses. They help ensure data integrity and provide clear
documentation of the expected data format for the API consumers.

Note: The Config class within some schemas, such as BaseUser and ServiceCallCreate, is used
to specify additional configuration options for the schema, such as allowing the creation of schema
instances from arbitrary attribute values.
"""

# pylint: disable=c0115
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from pydantic import BaseModel, EmailStr, Field


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
