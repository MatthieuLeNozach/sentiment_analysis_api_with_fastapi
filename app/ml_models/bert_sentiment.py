# File: ml_models/bert_sentiment.py
# Machine Learning Models

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..schemas import PredictionInput, PredictionOutputSentiment
from .nlp_utils import TextPreprocessor

# BERT OneHot 5-class sentiment classifier 
model_loc = 'nlptown/bert-base-multilingual-uncased-sentiment'

class BERTSentimentAnalyzer(nn.Module):
    def __init__(self):
        super().__init__()
        self.loaded = False
        self.preprocessor = TextPreprocessor()
        self.tokenizer = None
        self.model = None

    async def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(model_loc)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_loc)
        self.loaded = True

    async def predict(self, input: PredictionInput) -> PredictionOutputSentiment:
        if not self.loaded:
            await self.load_model()
        
        preprocessed_text = self.preprocessor(input.text)
        tokens = self.tokenizer.encode(preprocessed_text, return_tensors='pt')
        result = self.model(tokens)
        logits = result.logits[0]
        normalized_scores = F.softmax(logits, dim=-1).tolist()
        
        return PredictionOutputSentiment(
            sentiment_0=normalized_scores[0],
            sentiment_1=normalized_scores[1],
            sentiment_2=normalized_scores[2],
            sentiment_3=normalized_scores[3],
            sentiment_4=normalized_scores[4]
        )