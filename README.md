---
title: FilmSeek
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.45.1"
app_file: app.py
pinned: false
---

# FilmSeek

Describe what you want to watch and get a movie recommendation.

---

## Demo

## Demo
[FilmSeek on Hugging Face Spaces](https://huggingface.co/spaces/celaphee/filmseek)

---

## How it works

`loaddataset.py` loads the TMDB database (~2500 films filtered by rating ≥6.5 and popularity ≥1000 votes) and uses a local LLM (Ollama / Mistral) to generate 7 descriptions per film in different styles (plot, vibe, emotion, slang, etc.). 5% of the dataset was manually verified. This gives us a diverse training set where each film is represented from multiple angles.

`setup.py` fine-tunes the `all-MiniLM-L6-v2` sentence-transformer model on these input/output pairs using `MultipleNegativesRankingLoss`. The fine-tuned model then encodes all descriptions into embedding vectors — dense numerical representations that capture the semantic meaning of each description. These embeddings are indexed with FAISS, which enables fast approximate nearest-neighbor search across the entire dataset.

At query time, `app.py` encodes the user's input into an embedding using the same fine-tuned model, searches the FAISS index for the 50 closest descriptions, groups results by film and keeps the best match per film, then displays the top 5 recommendations with poster, year and rating.

---

## Stack

- `sentence-transformers` — all-MiniLM-L6-v2 fine-tuned on generated data
- `FAISS` — fast semantic search over embeddings
- `Ollama / Mistral` — local LLM for dataset generation
- `Streamlit` — user interface

---

## Architecture

```
├── app.py
├── loaddataset.py
├── README.md
├── requirements.txt
└── setup.py
```

---

## Local launch

```bash
pip install -r requirements.txt
python loaddataset.py
python setup.py
streamlit run app.py
```

> Note: `loaddataset.py` requires Ollama running locally with the Mistral model.

---

## Future improvements

- Quantitative evaluation (MRR, NDCG)
- Larger dataset coverage
- Multilingual embeddings
