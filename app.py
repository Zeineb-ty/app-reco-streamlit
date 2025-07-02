
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

# Chargement des données (simulé ici pour exemple)
@st.cache_data
def load_data():
    data = pd.read_csv("user_item_matrix.csv", index_col=0)
    return data

df = load_data()

# En-tête personnalisé
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
            padding: 2rem;
        }
        h1, h2, h3 {
            color: #1a73e8;
            text-align: center;
            font-family: 'Trebuchet MS', sans-serif;
        }
        .stSelectbox label {
            font-weight: bold;
            color: #444;
        }
        .reco-box {
            background-color: #ffffff;
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        }
    </style>
""", unsafe_allow_html=True)

st.title(" Guide Intelligent des Restaurants")
st.markdown("###  Sélectionnez un utilisateur pour découvrir ses recommandations personnalisées")

users = df.index.tolist()
selected_user = st.selectbox(" Utilisateur :", users)

# Méthode de recommandation
def get_top_recommendations(user, n=5):
    svd = TruncatedSVD(n_components=5)
    matrix_reduced = svd.fit_transform(df)
    reconstructed = svd.inverse_transform(matrix_reduced)
    preds_df = pd.DataFrame(reconstructed, index=df.index, columns=df.columns)
    top_items = preds_df.loc[user].sort_values(ascending=False).head(n)
    return top_items

st.markdown("## Suggestions Personnalisées")
recs = get_top_recommendations(selected_user)

for i, (resto, score) in enumerate(recs.items(), 1):
    st.markdown(f'''
    <div class="reco-box">
        <strong>{i}. {resto}</strong><br>
         Score estimé : <span style="color:#1a73e8;"><strong>{score:.2f}</strong></span>
    </div>
    ''', unsafe_allow_html=True)
