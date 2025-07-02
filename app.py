import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Charger les données
@st.cache_data
def load_data():
    df = pd.read_csv("restaurants-mr.csv")
    return df

df = load_data()

# Validation des colonnes
required_cols = {'username', 'restaurant', 'rating'}
if not required_cols.issubset(df.columns):
    st.error(" Le fichier doit contenir les colonnes suivantes : username, restaurant, rating")
    st.stop()

# Créer matrice utilisateur-item
user_item_matrix = df.pivot_table(index="username", columns="restaurant", values="rating")

# Moyenne utilisateur
user_mean = user_item_matrix.mean(axis=1)
centered_matrix = user_item_matrix.sub(user_mean, axis=0).fillna(0)

# Matrice de similarité cosinus
similarity = cosine_similarity(centered_matrix)
similarity_df = pd.DataFrame(similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Interface
st.set_page_config(page_title="RecoResto", layout="wide")
st.markdown("<h1 style='text-align:center; color:#2c3e50;'>🍴 RecoResto : Vos Meilleurs Restaurants !</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Découvrez les restaurants les mieux adaptés à vos goûts</p>", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.header(" Utilisateur")
    user_list = user_item_matrix.index.tolist()
    selected_user = st.selectbox("Sélectionnez un utilisateur :", user_list)

def predict(user, item):
    sims = similarity_df[user]
    item_ratings = user_item_matrix[item]

    mask = item_ratings.notna()
    sims = sims[mask]
    ratings = item_ratings[mask]

    if sims.sum() == 0:
        return 0.0

    return np.dot(sims, ratings) / sims.sum()

if selected_user:
    unrated_items = user_item_matrix.loc[selected_user][user_item_matrix.loc[selected_user].isna()].index
    predicted_scores = [(item, predict(selected_user, item)) for item in unrated_items]
    top_recommendations = sorted(predicted_scores, key=lambda x: x[1], reverse=True)[:5]

    st.subheader(f" Top 5 recommandations pour **{selected_user}** :")
    for i, (resto, score) in enumerate(top_recommendations, 1):
        st.markdown(f"**{i}. {resto}** —  Score estimé : `{round(score, 2)}`")
