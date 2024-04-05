# File: ml_models/roberta_emotion.py
"""
Module: roberta_emotion.py

This module contains the RobertaEmotionAnalyzer class, which is a PyTorch module for performing
emotion analysis using the RoBERTa (Robustly Optimized BERT Pretraining Approach) model.

The RobertaEmotionAnalyzer class loads a pre-trained RoBERTa model and tokenizer for emotion
analysis. It preprocesses the input text using the TextPreprocessor module from the nlp_utils
module and then passes the tokenized text through the RoBERTa model to obtain emotion scores.

The emotion scores are normalized using the softmax function, and the top 5 emotions with their
corresponding scores are returned as a PredictionOutputEmotion object.

Dependencies:
- transformers: A library for natural language processing with pre-trained models.
- torch: The PyTorch library for building and training neural networks.
- torch.nn: The PyTorch module for building neural network layers and modules.
- torch.nn.functional: The PyTorch module for functional operations used in neural networks.
- schemas: A module containing data schemas for input and output predictions.
- nlp_utils: A module containing utility functions for natural language processing.

Constants:
- MODEL_LOC: The location or identifier of the pre-trained RoBERTa model for emotion analysis.
"""
# pylint: disable=w0223
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..schemas import PredictionInput, PredictionOutputEmotion
from .nlp_utils import TextPreprocessor


# BERT OneHot 5-class sentiment classifier
MODEL_LOC = "SamLowe/roberta-base-go_emotions"


class RobertaEmotionAnalyzer(nn.Module):
    """
    A PyTorch module for performing emotion analysis using the RoBERTa model.

    This class loads a pre-trained RoBERTa model and tokenizer for emotion analysis.
    It preprocesses the input text using the TextPreprocessor module and then passes
    the tokenized text through the RoBERTa model to obtain emotion scores.

    Attributes:
        loaded (bool): Indicates whether the model and tokenizer are loaded.
        preprocessor (TextPreprocessor):
        An instance of the TextPreprocessor module for text preprocessing.
        tokenizer (AutoTokenizer): The tokenizer for the RoBERTa model.
        model (AutoModelForSequenceClassification):
        The pre-trained RoBERTa model for emotion analysis.
    """

    def __init__(self) -> None:
        super().__init__()
        self.loaded = False
        self.preprocessor = TextPreprocessor()
        self.tokenizer = None
        self.model = None

    async def load_model(self) -> None:
        """
        Loads the pre-trained RoBERTa model and tokenizer.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_LOC)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_LOC)
        self.loaded = True

    async def predict(
        self, prediction_input: PredictionInput
    ) -> PredictionOutputEmotion:
        """
        Performs emotion analysis on the input text.

        Args:
            prediction_input (PredictionInput): The input text for emotion analysis.

        Returns:
            PredictionOutputEmotion: The top 5 emotions with their corresponding scores.
        """
        if not self.loaded:
            await self.load_model()

        preprocessed_text = self.preprocessor(prediction_input.text)
        tokens = self.tokenizer(preprocessed_text, return_tensors="pt")
        logits = self.model(**tokens).logits

        scores = F.softmax(logits, dim=-1).squeeze()
        top_emotions_indices = torch.topk(scores, k=5).indices
        top_emotions = [
            (self.model.config.id2label[idx.item()], scores[idx].item())
            for idx in top_emotions_indices
        ]

        return PredictionOutputEmotion(emotions=top_emotions)
