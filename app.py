import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

st.markdown("""
<style>
img {
    height: 400px !important;
    width: 300px !important;
    object-fit: cover;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title="My Project",
    page_icon="📊",
    layout="wide"
)

# --------------------
# Load dataset
# --------------------
data = pd.read_csv("movie_dataset.csv")

data = data[['Movie Name', 'Genre', 'Lead Star', 'Director']]
data = data.dropna()

# Combine features
data["combined"] = (
    data["Genre"] + " " +
    data["Lead Star"] + " " +
    data["Director"]
)

# Convert text → numbers
cv = CountVectorizer()
matrix = cv.fit_transform(data["combined"])

# Similarity
similarity = cosine_similarity(matrix)

# --------------------
# Recommendation
# --------------------
def recommend_movie(name):

    index = data[data["Movie Name"] == name].index[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    movies = []
    for i in scores[1:6]:
        movies.append(data.iloc[i[0]]["Movie Name"])

    return movies


# --------------------
# Poster path
# --------------------
def get_poster(movie):
    path = f"posters/{movie}.jpeg"
    if os.path.exists(path):
        return path
    return ""


# --------------------
# UI
# --------------------
st.title("🎬 Bollywood Movie Recommendation System")

selected_movie = st.selectbox(
    "Choose a movie:",
    data["Movie Name"]
)

if st.button("Show Recommendation"):

    movies = recommend_movie(selected_movie)

    st.subheader("Recommended Movies")

    cols = st.columns(5)

    for i, movie in enumerate(movies):

        poster = get_poster(movie)

        with cols[i]:
            if poster:
                st.image(poster, use_container_width=True)

            st.write(movie)