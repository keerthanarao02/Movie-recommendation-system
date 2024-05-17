import pickle
import os
import re
import random
import numpy as np
import pandas as pd
import streamlit as st
from youtube_search import YoutubeSearch
import faiss

# Load the embeddings dictionary
@st.cache_data()
def load_embed_dict():
    with open("trailer/embed_dict.pkl", "rb") as file:
        return pickle.load(file)

data_base = load_embed_dict()

st.title('Movie Trailer Recommendation')
st.text("Approximate nearest neighbours using only visual features (no metadata!)")
st.text("Embeddings extracted from a custom video transformer encoder.")

@st.cache_data()
def build_faiss_index(data_base):
    embeddings = np.stack([data_base[i]["embedding"] for i in range(len(data_base))])
    index = faiss.IndexFlatL2(4096)  # Using L2 (Euclidean) distance
    index.add(embeddings)
    return index

@st.cache_data()
def load_data():
    return pd.DataFrame.from_dict(data_base, orient="index")

def faiss_processor(index, random_choice=False, id_n=0):
    if random_choice:
        id_n = random.randrange(len(data_base))
    distances, indices = index.search(np.array([data_base[id_n]["embedding"]]), 10)
    recall = {}
    for i, x in enumerate(indices[0]):
        recall[i] = {
            "name": os.path.basename(os.path.normpath(data_base[x]["path"])), 
            "actual": data_base[x]["actual"], 
            "predicted": data_base[x]["predicted"]
        }
    return recall

def retrieve_movies(index, random_choice=False, id_n=0):
    st.subheader("10 Similar Movies")
    data = faiss_processor(index, random_choice, id_n)
    col1, col2 = st.columns(2)
    cols = [col1, col2]

    for i in range(10):
        name = re.sub(r"(\w)([A-Z])", r"\1 \2", data[i]["name"])
        cols[i % len(cols)].write(name)
        results = YoutubeSearch(name + " movie trailer", max_results=1).to_dict()
        try:
            url = f"https://www.youtube.com{results[0]['url_suffix']}"
            cols[i % len(cols)].video(url)
        except (IndexError, KeyError):
            cols[i % len(cols)].write("No video available")
        
        cols[i % len(cols)].caption(f"Actual genre: {data[i]['actual']}")
        cols[i % len(cols)].caption(f"Predicted genre: {data[i]['predicted']}")

index = build_faiss_index(data_base)
data = load_data()

data_load_state = st.text("Loading data... done!")

option = st.selectbox("Pick a trailer from the drop down", data["path"].values)
if st.button("Generate random cluster"):
    retrieve_movies(index, random_choice=True)
if st.button("Search with selected"):
    id_n = data.index[data["path"] == option].tolist()[0]
    retrieve_movies(index, random_choice=False, id_n=id_n)
