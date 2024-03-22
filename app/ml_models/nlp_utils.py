
import torch.nn as nn
import re

class TextPreprocessor(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, text):
        text = text.lower()
        text = re.sub(r'<[^>]*>', '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

