import pandas as pd
import numpy as np
import streamlit as st
from sklearn.decomposition import TruncatedSVD

# Configuration de la page
st.set_page_config(page_title="Recommandation de Restaurants - Mauritanie", layout="wide")

# Fonction pour charger les données
@st.cache_data
def charger_donnees():
    data = pd.read_csv("restaurants-mr.csv")
    data = data[['reviews/0/name', 'title', 'totalScore']]
    data.columns = ['utilisateur', 'restaurant', 'note']
    return data

# Fonction pour entraîner le modèle SVD
def creer_modele(df):
    mat = df.pivot_table(index='utilisateur', columns='restaurant', values='note').fillna(0)
    svd = TruncatedSVD(n_components=20, random_state=42)
    svd.fit(mat)
    predictions = svd.transform(mat) @ svd.components_
    return mat, predictions

# Titre et logo
st.markdown("##  Application de Recommandation Intelligente")
st.markdown("### Découvrez de nouveaux restaurants selon vos préférences ")

# Chargement des données
df = charger_donnees()
matrice, prediction = creer_modele(df)

# Sidebar pour la sélection utilisateur
st.sidebar.markdown("##  Paramètres Utilisateur")
utilisateur = st.sidebar.selectbox("Choisir un utilisateur :", matrice.index.tolist())

# Génération des recommandations
if utilisateur:
    idx = matrice.index.get_loc(utilisateur)
    notes = matrice.iloc[idx]
    pred_series = pd.Series(prediction[idx], index=matrice.columns)

    non_notes = notes[notes == 0]
    reco = pred_series[non_notes.index].sort_values(ascending=False).head(5)

    st.markdown("---")
    st.subheader(f" Suggestions personnalisées pour **{utilisateur}**")
    for i, (resto, score) in enumerate(reco.items(), start=1):
        with st.container():
            st.write(f"**{i}. {resto}** —  Score estimé : `{round(score, 2)}`")

# Pied de page
st.markdown("---")
st.caption("Projet MLE413 — Streamlit App | Réalisé par [Zeineb]")
ames = user_item_matrix.index.tolist()
selected_user = st.selectbox(" Sélectionnez un utilisateur :", usernames)

if selected_user:
    user_idx = user_item_matrix.index.get_loc(selected_user)
    user_ratings = user_item_matrix.iloc[user_idx]
    user_reconstructed = reconstructed_matrix[user_idx]

    # Recommandations
    unrated = user_ratings[user_ratings == 0]
    preds = pd.Series(user_reconstructed, index=user_item_matrix.columns)
    recommendations = preds[unrated.index].sort_values(ascending=False).head(5)

    st.subheader(" Top Recommandations :")
    for i, (resto, score) in enumerate(recommendations.items(), 1):
        st.write(f"{i}. **{resto}** — Note prédite : {score:.2f}")
