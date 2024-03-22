# File: ml_models/roberta_emotion.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..schemas import PredictionInput, PredictionOutputEmotion
from .nlp_utils import TextPreprocessor


# BERT OneHot 5-class sentiment classifier 
model_loc = 'SamLowe/roberta-base-go_emotions'

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

class RobertaEmotionAnalyzer(nn.Module):
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

    async def predict(self, input: PredictionInput) -> PredictionOutputEmotion:
        if not self.loaded:
            await self.load_model()

        preprocessed_text = self.preprocessor(input.text)
        tokens = self.tokenizer(preprocessed_text, return_tensors='pt')
        logits = self.model(**tokens).logits

        scores = F.softmax(logits, dim=-1).squeeze()
        top_emotions_indices = torch.topk(scores, k=5).indices
        top_emotions = [(self.model.config.id2label[idx.item()], scores[idx].item()) for idx in top_emotions_indices]

        return PredictionOutputEmotion(emotions=top_emotions)