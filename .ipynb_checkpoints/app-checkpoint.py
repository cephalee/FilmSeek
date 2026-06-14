from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
import streamlit as st
import pandas as pd

@st.cache_resource
def load_everything():
    dataset = load_from_disk("dataset")
    dataset.load_faiss_index("embeddings", "index.faiss")
    model = SentenceTransformer('movie-search')
    return dataset, model

dataset, model = load_everything()

st.title("Movie Finder")
st.header("Describe the movie you want to watch")
query = st.text_input("Label", placeholder="A kid finding is a sorcerer")

with st.sidebar:
    st.header("Filters")
    year_max = st.slider("Before year", 1950, 2024, 2024)
    vote_min = st.slider("Minimum score", 0.0, 10.0, 0.0)

if query:
    st.write("Result")
    suggestion_embedding = model.encode([query])
    scores, samples = dataset.get_nearest_examples(
        "embeddings", suggestion_embedding, k=50 #if a movie appear multiple time
    )
    samples_df = pd.DataFrame.from_dict(samples)
    samples_df["scores"] = scores
    
    result = samples_df.groupby("title").agg({"scores": "min", "year": "first",
                                              "poster_url": "first", "vote": "first"}).reset_index()

    result = result[result["year"] <= year_max]
    result = result[result["vote"] >= vote_min]
    result = result.sort_values("scores", ascending=True).head(5)
    
    for _, row in result.iterrows():
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(row["poster_url"], width=120)
        with col2:
            st.subheader(row["title"])
            st.write(f"{int(row['year'])}")
            st.write(f"⭐ {row['vote']}")