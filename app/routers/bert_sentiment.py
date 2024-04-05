# file: app/routers/ml_service_bert_sentiment.py
"""
Module: ml_service_bert_sentiment.py

This module provides API endpoints for a sentiment analysis service using the BERT model. 
It uses the FastAPI framework to define routes and handle HTTP requests. 
he module includes the following main components:

- Dependencies:
  - `get_db`: Dependency function to get a database session.
  - `get_current_user`: Dependency function to authenticate and retrieve the current user.
  - `get_model_bert_sentiment`: Dependency function to lazy-load the BERT sentiment analysis model.

- Helper Functions:
  - `make_prediction_helper`: Helper function to make predictions 
     using the loaded BERT model and handle service call logging.
  - `interpret_results`: Helper function to interpret the raw prediction output 
     and return a more user-friendly result.

- Routes:
  - `/healthcheck`: Endpoint to check the health status of the sentiment analysis service.
  - `/predict/raw`: Endpoint to make raw predictions using the loaded BERT model.
  - `/predict/interpreted`: Endpoint to make predictions and return interpreted results.

The module integrates with the `BERTSentimentAnalyzer` class 
from the `ml_models.bert_sentiment` module to
load and use the BERT model for sentiment analysis. 
It uses the `ServiceCall`, `PredictionInput`, and `PredictionOutputSentiment`
models and schemas for data validation and storage.

Authentication and authorization are handled using the `get_current_user` dependency, which verifies
the user's access rights to the sentiment analysis service endpoints.

The module follows a modular structure and uses dependency injection 
to promote code reusability and maintainability.
"""
from datetime import datetime
from typing import Annotated
from fastapi import Depends, status, HTTPException, APIRouter
from sqlalchemy.orm import Session
import numpy as np


from ..database import get_db
from ..models import ServiceCall
from ..schemas import (
    PredictionInput,
    PredictionOutput,
    PredictionOutputSentiment,
    ServiceCallCreate,
)
from ..ml_models.bert_sentiment import BERTSentimentAnalyzer
from .auth import get_current_user

router = APIRouter(prefix="/mlservice/sentiment", tags=["mlservice/sentiment"])

bert_sentiment_analyzer = BERTSentimentAnalyzer()


############### DEPENDENCIES ###############
async def get_model_bert_sentiment(
    model: BERTSentimentAnalyzer = Depends(),
) -> BERTSentimentAnalyzer:
    """
    Dependency function to lazy-load the ML model.

    Args:
        model: The BERTSentimentAnalyzer instance to be loaded.

    Returns:
        The loaded BERTSentimentAnalyzer instance.
    """
    if not model.loaded:
        await model.load_model()
    return model


# pylint: disable=c0103
db_dependency = Annotated[Session, Depends(get_db)]
user_dependency = Annotated[dict, Depends(get_current_user)]
bert_sentiment_dependency = Annotated[
    BERTSentimentAnalyzer, Depends(get_model_bert_sentiment)
]  # pylint: enable=c0103


############### HELPERS ###############
async def make_prediction_helper(
    prediction_input: PredictionInput,
    user: user_dependency,
    db: db_dependency,
    model: bert_sentiment_dependency,
) -> PredictionOutput:
    """
    Helper function to make predictions using the loaded BERT model and handle service call logging.

    Args:
        prediction_input (PredictionInput): The input data for the prediction.
        user (user_dependency): The authenticated user making the request.
        db (db_dependency): The database session.
        model (bert_sentiment_dependency): The loaded BERT sentiment analysis model.

    Raises:
        HTTPException: If the user is not authenticated or does not have access to the service.

    Returns:
        PredictionOutputSentiment: The prediction output from the BERT model.
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User is None"
        )

    if not user.get("has_access_sentiment"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User does not have access to the service",
        )

    request_time = datetime.now()
    prediction_output = await model.predict(prediction_input)
    completion_time = datetime.now()
    duration = (completion_time - request_time).total_seconds()

    service_call_data = ServiceCallCreate(
        service_version="sentiment",
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


def interpret_results(prediction_output: PredictionOutputSentiment) -> dict:
    """
    Interprets the raw prediction output and returns a user-friendly result.

    Args:
        prediction_output (PredictionOutputSentiment):
        The raw prediction output from the BERT model.

    Returns:
        dict: A dictionary containing the predicted sentiment label and probability.
    """
    sentiment_labels = {
        0: "Very Negative",
        1: "Negative",
        2: "Neutral",
        3: "Positive",
        4: "Very Positive",
    }
    sentiment_scores = [
        prediction_output.sentiment_0,
        prediction_output.sentiment_1,
        prediction_output.sentiment_2,
        prediction_output.sentiment_3,
        prediction_output.sentiment_4,
    ]
    pred_label_idx = np.argmax(sentiment_scores)
    pred_label = sentiment_labels[pred_label_idx]
    pred_probability = sentiment_scores[pred_label_idx]
    return {"predicted_label": pred_label, "predicted_probability": pred_probability}


############### ROUTES ###############
@router.get("/healthcheck", status_code=status.HTTP_200_OK)
async def check_service_bert_sentiment(
    user: user_dependency, db: db_dependency
) -> dict:
    """
    Endpoint to check the health status of the sentiment analysis service.

    Args:
        user (user_dependency): The authenticated user making the request.
        db (db_dependency): The database session.

    Raises:
        HTTPException: If the user is not authenticated or does not have access to the service.

    Returns:
        dict: A dictionary indicating the health status of the service.
    """
    if user is None or not user.get("has_access_bert_sentiment"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return {"status": "healthy"}


@router.post("/predict/raw", status_code=status.HTTP_200_OK)
async def make_prediction_raw(
    prediction_input: PredictionInput,
    user: user_dependency,
    db: db_dependency,
    model: bert_sentiment_dependency,
) -> PredictionOutput:
    """
    Endpoint to make raw predictions using the loaded BERT model.

    Args:
        prediction_input (PredictionInput): The input data for the prediction.
        user (user_dependency): The authenticated user making the request.
        db (db_dependency): The database session.
        model (bert_sentiment_dependency): The loaded BERT sentiment analysis model.

    Returns:
        PredictionOutputSentiment: The raw prediction output from the BERT model.
    """

    prediction_output = await make_prediction_helper(prediction_input, user, db, model)
    return prediction_output


@router.post("/predict/interpreted", status_code=status.HTTP_200_OK)
async def make_prediction_interpreted(
    prediction_input: PredictionInput,
    user: user_dependency,
    db: db_dependency,
    model: bert_sentiment_dependency,
) -> dict:
    """
    Endpoint to make predictions and return interpreted results.

    Args:
        prediction_input (PredictionInput): The input data for the prediction.
        user (user_dependency): The authenticated user making the request.
        db (db_dependency): The database session.
        model (bert_sentiment_dependency): The loaded BERT sentiment analysis model.

    Returns:
        dict: A dictionary containing the predicted sentiment label and probability.
    """
    prediction_output = await make_prediction_helper(prediction_input, user, db, model)
    interpreted_results = interpret_results(prediction_output)
    return interpreted_results
