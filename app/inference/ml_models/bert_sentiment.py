# File: ml_models/bert_sentiment.py
"""Module: bert_sentiment.py.

This module contains the BERTSentimentAnalyzer class, which is a PyTorch module for performing
sentiment analysis using the BERT (Bidirectional Encoder Representations from Transformers) model.

The BERTSentimentAnalyzer class loads a pre-trained BERT model and tokenizer for sentiment analysis.
It preprocesses the input text using the TextPreprocessor module from the nlp_utils module and then
passes the tokenized text through the BERT model to obtain sentiment scores.

The sentiment scores are normalized using the softmax function and returned as a
PredictionOutputSentiment object, which contains the scores for each sentiment class (0 to 4).

Dependencies:
- transformers: A library for natural language processing with pre-trained models.
- torch: The PyTorch library for building and training neural networks.
- torch.nn: The PyTorch module for building neural network layers and modules.
- torch.nn.functional: The PyTorch module for functional operations used in neural networks.
- schemas: A module containing data schemas for input and output predictions.
- nlp_utils: A module containing utility functions for natural language processing.

Constants:
- MODEL_LOC: The location or identifier of the pre-trained BERT model for sentiment analysis.
"""

# pylint: disable=w0223
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app_utils.schemas import PredictionInput, PredictionOutputSentiment
from ml_models.nlp_utils import TextPreprocessor

# BERT OneHot 5-class sentiment classifier
MODEL_LOC = "nlptown/bert-base-multilingual-uncased-sentiment"


class BERTSentimentAnalyzer(nn.Module):
    """A PyTorch module for performing sentiment analysis using the BERT model.

    This class loads a pre-trained BERT model and tokenizer for sentiment analysis.
    It preprocesses the input text using the TextPreprocessor module and then passes
    the tokenized text through the BERT model to obtain sentiment scores.

    Attributes
    ----------
        loaded (bool): Indicates whether the model and tokenizer are loaded.
        preprocessor (TextPreprocessor):
        An instance of the TextPreprocessor module for text preprocessing.
        tokenizer (AutoTokenizer): The tokenizer for the BERT model.
        model (AutoModelForSequenceClassification):
        The pre-trained BERT model for sentiment analysis.

    """

    def __init__(self):
        """Initialize the BERTSentimentAnalyzer module."""
        super().__init__()
        self.loaded = False
        self.preprocessor = TextPreprocessor()
        self.tokenizer = None
        self.model = None

    async def load_model(self):
        """Load the pre-trained BERT model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_LOC)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_LOC)
        self.loaded = True

    async def predict(self, prediction_input: PredictionInput) -> PredictionOutputSentiment:
        """Perform sentiment analysis on the input text.

        Args:
        ----
            prediction_input (PredictionInput): The input text for sentiment analysis.

        Returns:
        -------
            PredictionOutputSentiment: The sentiment scores for each class (0 to 4).

        """
        if not self.loaded:
            await self.load_model()

        preprocessed_text = self.preprocessor(prediction_input.text)
        tokens = self.tokenizer.encode(preprocessed_text, return_tensors="pt")
        result = self.model(tokens)
        logits = result.logits[0]
        normalized_scores = F.softmax(logits, dim=-1).tolist()

        return PredictionOutputSentiment(
            sentiment_0=normalized_scores[0],
            sentiment_1=normalized_scores[1],
            sentiment_2=normalized_scores[2],
            sentiment_3=normalized_scores[3],
            sentiment_4=normalized_scores[4],
        )
