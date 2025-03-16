import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

start_time = time.time()

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/ms-marco-MiniLM-L6-v2')
print(f"Model loaded in {time.time() - start_time:.2f} seconds.")

start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L6-v2')
print(f"Tokenizer loaded in {time.time() - start_time:.2f} seconds.")

start_time = time.time()
features = tokenizer(
    ['How many people live in Berlin?', 'How many people live in Berlin?'],
    ['Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.',
     'New York City is famous for the Metropolitan Museum of Art.'],
    padding=True, truncation=True, return_tensors="pt"
)
print(f"Tokenization done in {time.time() - start_time:.2f} seconds.")

model.eval()
with torch.no_grad():
    start_time = time.time()
    scores = model(**features).logits
    print(f"Inference completed in {time.time() - start_time:.2f} seconds.")

print("Scores:", scores)
