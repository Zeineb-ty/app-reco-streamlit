import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(page_title="Restaurant Recommender", page_icon="🍽️", layout="centered")
st.markdown("""
    <style>
    .stApp {
        background-color: #f9f9f9;
    }
    .title {
        text-align: center;
        color: #4a4a4a;
    }
    </style>
""", unsafe_allow_html=True)

st.title("\U0001F372 Restaurant Recommendation System")

@st.cache_data
def load_data():
    df = pd.read_csv("restaurants-mr.csv")
    all_reviews = []
    for i in range(10):  # les 10 premiers avis maximum
        name_col = f"reviews/{i}/name"
        rating_col = f"reviews/{i}/rating"
        if name_col in df.columns and rating_col in df.columns:
            temp = df[["title", name_col, rating_col]].dropna()
            temp.columns = ["restaurant", "username", "rating"]
            all_reviews.append(temp)
    return pd.concat(all_reviews, ignore_index=True)

# Charger les données
ratings_df = load_data()

# Pivot table user-item
user_item_matrix = ratings_df.pivot_table(index='username', columns='restaurant', values='rating')
user_similarity = cosine_similarity(user_item_matrix.fillna(0))
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

def get_recommendations(user, k=5):
    if user not in user_item_matrix.index:
        return []
    # Moyenne des similarités avec les autres utilisateurs
    sims = user_similarity_df[user].drop(user)
    similar_users = sims.sort_values(ascending=False).index

    # Prédictions pondérées
    scores = {}
    for other in similar_users:
        other_ratings = user_item_matrix.loc[other]
        similarity = sims[other]
        for restaurant, rating in other_ratings.dropna().items():
            if pd.isna(user_item_matrix.loc[user, restaurant]):
                scores.setdefault(restaurant, 0)
                scores[restaurant] += rating * similarity

    top_recos = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [r[0] for r in top_recos]

# Interface
user_input = st.selectbox("Choisissez un utilisateur :", options=sorted(user_item_matrix.index))

if st.button("Recommander des restaurants"):
    recos = get_recommendations(user_input)
    if recos:
        st.success(f"\U0001F4CC Recommandations pour *{user_input}* :")
        for r in recos:
            st.markdown(f"- **{r}**")
    else:
        st.warning("Aucune recommandation disponible pour cet utilisateur.")
