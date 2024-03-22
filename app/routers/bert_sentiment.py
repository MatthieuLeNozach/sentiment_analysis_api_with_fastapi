# file: app/routers/ml_service_bert_sentiment.py

from datetime import datetime
from typing import Annotated
from fastapi import Depends, status, HTTPException, APIRouter
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import User, ServiceCall
from ..schemas import PredictionInput, PredictionOutputSentiment, ServiceCallCreate
from ..ml_models.bert_sentiment import BERTSentimentAnalyzer
from .auth import get_current_user

router = APIRouter(prefix='/mlservice/sentiment', tags=['mlservice/sentiment'])

bert_sentiment_analyzer = BERTSentimentAnalyzer()


############### DEPENDENCIES ###############
async def get_model_bert_sentiment(model: BERTSentimentAnalyzer = Depends()) -> BERTSentimentAnalyzer:
    """Dependency to lazy-load a ML model"""
    if not model.loaded:
        await model.load_model()
    return model


db_dependency = Annotated[Session, Depends(get_db)]
user_dependency = Annotated[dict, Depends(get_current_user)]
bert_sentiment_dependency = Annotated[BERTSentimentAnalyzer, Depends(get_model_bert_sentiment)]

############### HELPERS ###############
async def make_prediction_helper(prediction_input: PredictionInput, user: user_dependency, db: db_dependency, model: bert_sentiment_dependency):
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User is None")
    
    if not user.get('has_access_sentiment'):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User does not have access to the service")
    
    request_time = datetime.now()
    prediction_output = await model.predict(prediction_input)
    completion_time = datetime.now()
    duration = (completion_time - request_time).total_seconds()
    
    service_call_data = ServiceCallCreate(
        service_version='sentiment',
        success=True,
        owner_id=user['id'],
        request_time=request_time,
        completion_time=completion_time,
        duration=duration
    )
    service_call = ServiceCall(**service_call_data.dict())
    db.add(service_call)
    db.commit()
    db.refresh(service_call)
    
    return prediction_output


def interpret_results(prediction_output: PredictionOutputSentiment) -> dict:
    import numpy as np
    sentiment_labels = {
        0: 'Very Negative',
        1: 'Negative',
        2: 'Neutral',
        3: 'Positive',
        4: 'Very Positive'
    }
    sentiment_scores = [
        prediction_output.sentiment_0,
        prediction_output.sentiment_1,
        prediction_output.sentiment_2,
        prediction_output.sentiment_3,
        prediction_output.sentiment_4,
    ]
    pred_label_idx =  np.argmax(sentiment_scores)
    pred_label = sentiment_labels[pred_label_idx]
    pred_probability = sentiment_scores[pred_label_idx]
    return {
        'predicted_label': pred_label,
        'predicted_probability': pred_probability
    }
    




############### ROUTES ###############
@router.get('/healthcheck', status_code=status.HTTP_200_OK)
async def check_service_bert_sentiment(user: user_dependency, db: db_dependency) -> dict:
    if user is None or not user.get('has_access_bert_sentiment'):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return {'status': 'healthy'}



@router.post('/predict/raw', status_code=status.HTTP_200_OK)
async def make_prediction_raw(prediction_input: PredictionInput, user: user_dependency, db: db_dependency, model: bert_sentiment_dependency):
    prediction_output = await make_prediction_helper(prediction_input, user, db, model)
    return prediction_output


@router.post('/predict/interpreted', status_code=status.HTTP_200_OK)
async def make_prediction_interpreted(prediction_input: PredictionInput, user: user_dependency, db: db_dependency, model: bert_sentiment_dependency):
    prediction_output = await make_prediction_helper(prediction_input, user, db, model)
    interpreted_results = interpret_results(prediction_output)
    return interpreted_results