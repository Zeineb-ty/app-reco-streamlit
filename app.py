import streamlit as st
import pandas as pd
import random

# Personnalisation du style
st.set_page_config(page_title=" Smart Reco - Restaurants MR", layout="centered")

st.markdown(
    "<h1 style='text-align: center; color: #2E8B57;'> Smart Restaurant Recommender</h1>",
    unsafe_allow_html=True
)

st.markdown("Bienvenue dans notre moteur intelligent de recommandation de restaurants en Mauritanie 🇲🇷. "
            "Choisissez votre utilisateur pour découvrir les restaurants qu’il pourrait aimer !")

# Chargement des données
@st.cache_data
def load_data():
    return pd.read_csv("restaurants-mr.csv")

df = load_data()

# Sélection utilisateur fictif
users = df['user'].dropna().unique().tolist()
user_selected = st.selectbox(" Sélectionnez un utilisateur :", users)

#  Nombre de recommandations
top_n = st.slider(" Nombre de recommandations :", 1, 10, 5)

#  Recommandation
if st.button(" Lancer la recommandation"):
    st.markdown("##  Recommandations personnalisées")
    #  Pour cette démo, on choisit aléatoirement des restaurants notés par d'autres utilisateurs
    other_recos = df[df['user'] != user_selected].sample(n=top_n, replace=True)
    st.dataframe(other_recos[['title', 'rating']].reset_index(drop=True))

#  Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center;'>Développé avec  pour MLE413</div>",
    unsafe_allow_html=True
)
