# file: app/routers/ml_service_roberta_emotion.py
"""Module: ml_service_roberta_emotion.py.

This module provides API endpoints for an emotion analysis service using the RoBERTa model.
It uses the FastAPI framework to define routes and handle HTTP requests.
The module includes the following main components:

- Dependencies:
  - `get_db`: Dependency function to get a database session.
  - `get_current_user`: Dependency function to authenticate and retrieve the current user.
  - `get_model_roberta_emotion`: Dependency function
  to lazy-load the RoBERTa emotion analysis model.

- Routes:
  - `/healthcheck`: Endpoint to check the health status of the emotion analysis service.
  - `/predict`: Endpoint to make predictions using the loaded RoBERTa model.

The module integrates with the `RobertaEmotionAnalyzer` class
from the `ml_models.roberta_emotion` module to
load and use the RoBERTa model for emotion analysis.
It uses the `ServiceCall`, `PredictionInput`, and
`PredictionOutputEmotion` models and schemas for data validation and storage.

Authentication and authorization are handled using the `get_current_user` dependency, which verifies
the user's access rights to the emotion analysis service endpoints.

The module follows a modular structure and uses dependency injection to promote code reusability and
maintainability.
"""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..database import get_db
from ..ml_models.roberta_emotion import RobertaEmotionAnalyzer
from ..models import ServiceCall
from ..schemas import PredictionInput, PredictionOutputEmotion, ServiceCallCreate
from .auth import get_current_user

router = APIRouter(prefix="/mlservice/emotion", tags=["mlservice/emotion"])

roberta_emotion_analyzer = RobertaEmotionAnalyzer()


############### DEPENDENCIES ###############
async def get_model_roberta_emotion(
    model: RobertaEmotionAnalyzer = Depends(),
) -> RobertaEmotionAnalyzer:
    """Dependency function to lazy-load the RoBERTa emotion analysis model.

    Args:
    ----
        model (RobertaEmotionAnalyzer, optional): The RoBERTa emotion analysis model instance.
            Defaults to Depends().

    Returns:
    -------
        RobertaEmotionAnalyzer: The loaded RoBERTa emotion analysis model.

    """
    if not model.loaded:
        await model.load_model()
    return model


# pylint: disable=c0103
db_dependency = Annotated[Session, Depends(get_db)]
user_dependency = Annotated[dict, Depends(get_current_user)]
roberta_emotion_dependency = Annotated[RobertaEmotionAnalyzer, Depends(get_model_roberta_emotion)]
# pylint: enable=c0103


############### ROUTES ###############
@router.get("/healthcheck", status_code=status.HTTP_200_OK)
async def check_service_emotion(user: user_dependency, db: db_dependency) -> dict:
    """Check the health status of the emotion analysis service.

    Args:
    ----
        user (user_dependency): The authenticated user making the request.
        db (db_dependency): The database session.

    Raises:
    ------
        HTTPException: If the user is not authenticated or does not have access to the service.

    Returns:
    -------
        dict: A dictionary indicating the health status of the service.

    """
    if user is None or not user.get("has_access_emotion"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return {"status": "healthy"}


# Define your routes here
@router.post("/predict", status_code=status.HTTP_200_OK)
async def make_prediction_emotion(
    prediction_input: PredictionInput,
    user: user_dependency,
    db: db_dependency,
    model: roberta_emotion_dependency,
) -> PredictionOutputEmotion:
    """Make predictions using the loaded RoBERTa model.

    Args:
    ----
        prediction_input (PredictionInput): The input data for the prediction.
        user (user_dependency): The authenticated user making the request.
        db (db_dependency): The database session.
        model (roberta_emotion_dependency): The loaded RoBERTa emotion analysis model.

    Raises:
    ------
        HTTPException: If the user is not authenticated or does not have access to the service.

    Returns:
    -------
        PredictionOutputEmotion: The prediction output from the RoBERTa model.

    """
    if user is None or not user.get("has_access_emotion"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

    request_time = datetime.now()
    prediction_output = await model.predict(prediction_input)
    completion_time = datetime.now()
    duration = (completion_time - request_time).total_seconds()

    service_call_data = ServiceCallCreate(
        service_version="emotion",
        success=True,
        owner_id=user["id"],
        request_time=request_time,
        completion_time=completion_time,
        duration=duration,
    )
    service_call = ServiceCall(**service_call_data.dict())
    db.add(service_call)
    db.commit()
    db.refresh(service_call)

    return prediction_output
