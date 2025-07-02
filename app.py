import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# Titre avec emoji et style
st.markdown("<h1 style='color: teal;'>🍴 Recommandation Personnalisée de Restaurants en Mauritanie</h1>", unsafe_allow_html=True)

# Chargement des données
@st.cache_data
def load_data():
    df = pd.read_csv("restaurants-mr.csv")
    df = df[['user_name', 'title', 'note']].dropna()
    return df

df = load_data()

# Liste des utilisateurs
user_list = sorted(df["user_name"].unique())

# Choix utilisateur
selected_user = st.selectbox(" Choisissez un utilisateur :", user_list)

# Bouton pour lancer la recommandation
if st.button(" Recommander des restaurants"):
    # Préparation des données
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(df[["user_name", "title", "note"]], reader)
    trainset, _ = train_test_split(data, test_size=0.2)

    model = SVD()
    model.fit(trainset)

    # Liste des restaurants non notés par l’utilisateur
    user_data = df[df["user_name"] == selected_user]
    all_items = df["title"].unique()
    rated_items = user_data["title"].unique()
    items_to_predict = [item for item in all_items if item not in rated_items]

    # Prédictions pour chaque restaurant non encore noté
    predictions = []
    for item in items_to_predict:
        pred = model.predict(selected_user, item)
        predictions.append((item, pred.est))

    # Trier par note prédite
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_5 = predictions[:5]

    st.success(f"🍽️ Top 5 recommandations pour **{selected_user}** :")
    for i, (restaurant, score) in enumerate(top_5, 1):
        st.markdown(f"**{i}. {restaurant}** — Prédiction : {score:.2f}")
