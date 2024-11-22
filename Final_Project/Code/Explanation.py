from transformers import BertTokenizer, BertForSequenceClassification
from torch import nn, optim
import torch

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Binary classification

def sentiment_analysis(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    return nn.functional.softmax(logits, dim=-1).numpy()

# Example usage
text = "I love machine learning, its capabilities are amazing!"
print(sentiment_analysis(text))
