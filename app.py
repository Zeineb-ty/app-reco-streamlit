import streamlit as st
import pandas as pd
from model_utils import recommend_content_based, build_recommendation_models, get_top_recommendations

# Titre de l'application
st.title("🍽️ Système de Recommandation de Restaurants - Mauritanie")

# Charger les données (optimisé pour Streamlit)
@st.cache_data
def load_data():
    df = pd.read_csv("restaurants-mr.csv")
    return df

df = load_data()
restaurant_list = df['title'].dropna().unique().tolist()

# Sélection d’un restaurant
restaurant_selected = st.selectbox("Choisissez un restaurant :", restaurant_list)

# Nombre de recommandations à afficher
top_n = st.slider("Combien de recommandations souhaitez-vous ?", 1, 10, 5)

# Bouton pour lancer la recommandation
if st.button("Recommander"):
    st.subheader(f"🔎 Restaurants similaires à : {restaurant_selected}")
    try:
        reco_df = recommend_content_based(restaurant_selected, top_n=top_n)
        st.dataframe(reco_df)
    except Exception as e:
        st.error(f"Erreur : {e}")

# Optionnel : top 5 pour un utilisateur fictif
st.subheader("💡 Suggestions pour un utilisateur")
if st.button("Afficher top-5 utilisateur"):
    model, top_n_users = build_recommendation_models(df)
    reco = get_top_recommendations(top_n_users)
    st.text(reco)
