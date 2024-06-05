from celery import Celery
from ml_models.bert_sentiment import  BERTSentimentAnalyzer
from ml_models.roberta_emotion import RobertaEmotionAnalyzer
from app_utils.schemas import (
    PredictionInput, 
    PredictionOutputEmotion, 
    PredictionOutputSentiment
)



celery = Celery(
    "worker",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/0"
)

bert_sentiment_analyzer = BERTSentimentAnalyzer()
roberta_emotion_analyzer = RobertaEmotionAnalyzer()

@celery.task
def predict_sentiment(data):
    if not bert_sentiment_analyzer.loaded:
        bert_sentiment_analyzer.load_model()
    prediction_input = PredictionInput(text=data['text'])
    prediction_output = bert_sentiment_analyzer.predict(prediction_input)
    prediction_output = roberta_emotion_analyzer.predict(prediction_input)
    return PredictionOutputSentiment(**prediction_output)

@celery.task
def predict_emotion(data):
    if not roberta_emotion_analyzer.loaded:
        roberta_emotion_analyzer.load_model()
    prediction_input = PredictionInput(text=data['text'])
    prediction_output = roberta_emotion_analyzer.predict(prediction_input)
    return PredictionOutputEmotion(**prediction_output)