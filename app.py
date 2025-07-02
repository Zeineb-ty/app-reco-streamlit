import pandas as pd
import numpy as np
import streamlit as st
from sklearn.decomposition import TruncatedSVD

# Chargement des données
@st.cache_data
def load_data():
    df = pd.read_csv("restaurants-mr.csv")
    
    # Colonnes disponibles
    st.write(" Colonnes disponibles :", df.columns.tolist())

    # Vérification des colonnes attendues
    required_cols = ['reviews/0/name', 'title', 'totalScore']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Les colonnes nécessaires sont absentes du fichier. Colonnes trouvées : {df.columns.tolist()}")
        st.stop()
    
    df = df[required_cols]
    df.columns = ['name', 'restaurant', 'stars']
    return df

# Charger les données
ratings_df = load_data()

# Modèle de recommandation
def build_model(df):
    user_item_matrix = df.pivot_table(index='name', columns='restaurant', values='stars').fillna(0)
    svd = TruncatedSVD(n_components=20, random_state=42)
    svd.fit(user_item_matrix)
    reconstructed = svd.transform(user_item_matrix) @ svd.components_
    return user_item_matrix, reconstructed, svd

user_item_matrix, reconstructed_matrix, svd = build_model(ratings_df)

# Interface Streamlit
st.markdown('##  Recommandation Personnalisée')
st.subheader(" Restaurant Recommender System")

# Sélection utilisateur
usernames = user_item_matrix.index.tolist()
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
