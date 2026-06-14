from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from dotenv import load_dotenv
import pandas as pd

data = load_dataset('json', data_files='dataset.jsonl')
dataset = data['train']

train_examples = []
for row in dataset:
    train_examples.append(InputExample(texts=[row['input'], row['output']]))

model = SentenceTransformer('all-MiniLM-L6-v2')
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=100,
    output_path="movie-search"
)

model = SentenceTransformer("movie-search") 
embeddings = model.encode(dataset['input'])
dataset = dataset.add_column("embeddings", embeddings.tolist())
dataset.add_faiss_index(column="embeddings")

dataset.save_faiss_index("embeddings", "index.faiss")
dataset.drop_index("embeddings")
dataset.save_to_disk("dataset")
