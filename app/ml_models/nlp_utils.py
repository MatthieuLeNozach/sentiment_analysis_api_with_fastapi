# file: ml_models/nlp_utils.py
"""Module: nlp_utils.py.

This module contains utility classes and functions for natural language processing (NLP) tasks.

The module includes the following classes:
- TextPreprocessor: A PyTorch module for preprocessing text data.

The TextPreprocessor class can be used as a component in an NLP pipeline
to clean and normalize text data before further processing or analysis.
"""

import re

import torch.nn as nn


class TextPreprocessor(nn.Module):
    """A PyTorch module for preprocessing text data.

    This class inherits from the PyTorch nn.Module class and provides a forward method
    for preprocessing text data. The preprocessing steps include:
    - Converting the text to lowercase
    - Removing HTML tags
    - Removing URLs
    - Removing non-alphanumeric characters
    - Removing extra whitespace and leading/trailing whitespace

    The preprocessed text is returned as a string.

    Attributes
    ----------
        None

    Methods
    -------
        forward(text: str) -> str:
            Preprocesses the input text and returns the cleaned text.

    """

    def __init__(self):
        """Initialize the TextPreprocessor module.

        This method is called when an instance of the TextPreprocessor class is created.
        It initializes the module by calling the __init__ method of the parent nn.Module class.

        Args:
        ----
            None

        Returns:
        -------
            None

        """
        super().__init__()

    def forward(self, text):
        """Preprocesses the input text.

        This method takes a string of text as input and applies various preprocessing steps
        to clean and normalize the text. The preprocessing steps include:
        - Converting the text to lowercase
        - Removing HTML tags using regular expressions
        - Removing URLs using regular expressions
        - Removing non-alphanumeric characters using regular expressions
        - Removing extra whitespace and leading/trailing whitespace using regular expressions

        Args:
        ----
            text (str): The input text to be preprocessed.

        Returns:
        -------
            str: The preprocessed text.

        """
        text = text.lower()
        text = re.sub(r"<[^>]*>", "", text)
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
