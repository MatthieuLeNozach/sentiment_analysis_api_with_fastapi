# file: app/main.py

import os
from dotenv import load_dotenv
from fastapi import FastAPI

from .ml_models.bert_sentiment import BERTSentimentAnalyzer
from .ml_models.roberta_emotion import RobertaEmotionAnalyzer
from .models import Base
from .database import engine, SessionLocal
from .routers import auth, admin, bert_sentiment, users, roberta_emotion 
from .devtools import create_superuser, remove_superuser

load_dotenv(override=True) # loads environment variables from the .environment folder

############## ML MODELS ###############
bert_model = BERTSentimentAnalyzer()
roberta_model = RobertaEmotionAnalyzer()

############### API ###############
app = FastAPI()
app.include_router(auth.router) 
app.include_router(admin.router)
app.include_router(users.router)
app.include_router(bert_sentiment.router)
app.include_router(roberta_emotion.router)


############### DATABASE ###############
Base.metadata.create_all(bind=engine)


############### LIFESPAN ###############
@app.on_event('startup')
async def startup_event():
    if os.getenv('CREATE_SUPERUSER', 'False').lower() in ['true', '1', 'yes']:
        db = SessionLocal()
        create_superuser(db)
        db.close()
        
    await bert_sentiment.bert_sentiment_analyzer.load_model()
    await roberta_emotion.roberta_emotion_analyzer.load_model()
        
        
        
@app.on_event('shutdown')
async def shutdown_event():
    if os.getenv('CREATE_SUPERUSER', 'False').lower() in ['true', '1', 'yes']:
        db = SessionLocal()
        remove_superuser(db)
        db.close()



############### ROUTES ###############
@app.get('/healthcheck')
def get_healthcheck():
    return {'status': 'healthy'}

