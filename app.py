# app.py
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# --- Chargement et nettoyage des données ---
@st.cache_data
def load_data():
    df = pd.read_csv("restaurants-mr.csv")
    df_clean = df[["reviews/0/name", "title", "reviews/0/rating"]].dropna()
    df_clean.columns = ["username", "restaurant", "rating"]
    df_clean["rating"] = pd.to_numeric(df_clean["rating"], errors="coerce")
    df_clean = df_clean.dropna(subset=["rating"])
    return df_clean

data = load_data()

# --- Interface Streamlit ---
st.set_page_config(page_title="Restaurant Recommender", layout="centered")
st.title(" Recommande-moi un restaurant !")

users = sorted(data["username"].unique())
selected_user = st.selectbox("Choisis ton nom :", users)

# --- Génération des recommandations ---
if selected_user:
    user_item_matrix = data.pivot_table(index="username", columns="restaurant", values="rating")
    similarity_matrix = cosine_similarity(user_item_matrix.fillna(0))
    similarity_df = pd.DataFrame(similarity_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)

    # Scores pondérés par similarité
    user_similarities = similarity_df[selected_user].drop(selected_user)
    unrated_restaurants = user_item_matrix.loc[selected_user].isna()
    weighted_scores = user_item_matrix.T.dot(user_similarities).loc[unrated_restaurants]

    top_recommendations = weighted_scores.sort_values(ascending=False).head(5)

    if not top_recommendations.empty:
        st.subheader(" Recommandations pour toi :")
        for i, (resto, score) in enumerate(top_recommendations.items(), start=1):
            st.markdown(f"**{i}. {resto}** — score estimé : `{score:.2f}`")
    else:
        st.info("Aucune recommandation possible pour ce profil (trop peu de données).")
