# file: app/routers/ml_service_roberta_emotion.py

from datetime import datetime
from typing import Annotated
from fastapi import Depends, status, HTTPException, APIRouter
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import User, ServiceCall
from ..schemas import PredictionInput, PredictionOutputSentiment, ServiceCallCreate
from ..ml_models.roberta_emotion import RobertaEmotionAnalyzer
from .auth import get_current_user

router = APIRouter(prefix='/mlservice/emotion', tags=['mlservice/emotion'])

roberta_emotion_analyzer = RobertaEmotionAnalyzer()


############### DEPENDENCIES ###############
async def get_model_roberta_emotion(model: RobertaEmotionAnalyzer = Depends()) -> RobertaEmotionAnalyzer:
    """Dependency to lazy-load a ML model"""
    if not model.loaded:
        await model.load_model()
    return model




db_dependency = Annotated[Session, Depends(get_db)]
user_dependency = Annotated[dict, Depends(get_current_user)]
roberta_emotion_dependency = Annotated[RobertaEmotionAnalyzer, Depends(get_model_roberta_emotion)]




############### ROUTES ###############
@router.get('/healthcheck', status_code=status.HTTP_200_OK)
async def check_service_emotion(user: user_dependency, db: db_dependency) -> dict:
    if user is None or not user.get('has_access_emotion'):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return {'status': 'healthy'}

# Define your routes here
@router.post('/predict', status_code=status.HTTP_200_OK)
async def make_prediction_emotion(prediction_input: PredictionInput, user: user_dependency, db: db_dependency, model: roberta_emotion_dependency):
    if user is None or not user.get('has_access_emotion'):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    
    request_time = datetime.now()
    prediction_output = await model.predict(prediction_input)
    completion_time = datetime.now()
    duration = (completion_time - request_time).total_seconds()
    
    service_call_data = ServiceCallCreate(
        service_version='emotion',
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