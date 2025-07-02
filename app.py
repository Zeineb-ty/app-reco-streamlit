
import streamlit as st
import pandas as pd

# Configuration de la page
st.set_page_config(
    page_title=" Restaurant Recommender - Mauritania",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Chargement des données
@st.cache_data
def load_data():
    df = pd.read_csv("restaurants-mr.csv")
    return df

df = load_data()

# En-tête stylisée
st.markdown("<h1 style='color:#FF6347; font-size: 42px;'> Discover Mauritania's Hidden Food Gems</h1>", unsafe_allow_html=True)
st.markdown("##### Explore local flavors and get personalized recommendations ")

# Liste déroulante pour choisir un restaurant
restaurant_list = df['title'].dropna().unique().tolist()
selected_restaurant = st.selectbox(" Select a restaurant you like:", restaurant_list)

# Slider pour choisir le nombre de suggestions
top_n = st.slider(" Number of similar restaurants:", 1, 10, 5)

# Recommandations
if st.button(" Show Recommendations"):
    st.markdown(f"### Similar to **{selected_restaurant}**:")
    try:
        # Basée sur la similarité de titre
        mask = df['title'].str.contains(selected_restaurant[:4], case=False, na=False)
        reco_df = df[mask].drop_duplicates('title').head(top_n)
        st.dataframe(reco_df[['title', 'address', 'rating', 'reviews']])
    except Exception as e:
        st.error(f"Oops, something went wrong: {e}")

# Pied de page
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>Made with for Mauritania</p>", unsafe_allow_html=True)
