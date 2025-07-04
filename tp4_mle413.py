# -*- coding: utf-8 -*-
"""TP4 - MLE413.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ilyYNJ4BeXM1zPpW9vWVtuMniW17QfFd

## TP4 - Système de Recommandation des Restaurants en Mauritanie
"""

# Étape 1 : Chargement et Nettoyage des Données
import streamlit as st
import pandas as pd
import numpy as np

# 1.1 Charger le fichier CSV
df = pd.read_csv("restaurants-mr.csv")
print("Dimensions initiales :", df.shape)

# 1.2 Supprimer les colonnes entièrement vides
df = df.dropna(axis=1, how='all')
print("Après suppression colonnes vides :", df.shape)

# 1.3 Afficher un aperçu des données
print("\nAperçu des données :")
st.dataframe(df.head())

# 1.4 Affichage des colonnes avec taux de valeurs manquantes
missing_percent = df.isnull().mean() * 100
print("\nColonnes avec valeurs manquantes :")
print(missing_percent[missing_percent > 0].sort_values(ascending=False))

# 1.5 Gestion des données manquantes
# Moyenne/mediane pour colonnes numériques
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    df[col] = df[col].fillna(df[col].mean())

# Mode pour colonnes catégorielles
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

# 1.6 Supprimer les doublons éventuels
df = df.drop_duplicates()
print("\nDimensions après nettoyage :", df.shape)

# 1.7 Aperçu final des données
print("\nAperçu final :")
st.dataframe(df.head())


from sklearn.preprocessing import MinMaxScaler, StandardScaler
# 2.1 Normalisation des colonnes de scores
score_cols = [col for col in df.columns if col in ['totalScore', 'reviews/0/stars']]
minmax_scaler = MinMaxScaler()
for col in score_cols:
    df[col] = minmax_scaler.fit_transform(df[[col]])

# 2.2 Standardisation des colonnes d'engagement utilisateur
engagement_cols = [col for col in df.columns if col in ['reviewsCount', 'reviews/0/reviewerNumberOfReviews']]
standard_scaler = StandardScaler()
for col in engagement_cols:
    df[col] = standard_scaler.fit_transform(df[[col]])

# 2.3 Affichage après transformation
print("\nDonnées après normalisation et standardisation :")
st.dataframe(df.head())

# ------------------------------------------------------
# Étape 3 : Prétraitement des Avis Textuels

import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Téléchargement des ressources nécessaires
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Fonction de prétraitement
def preprocess_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

if 'reviews/0/text' in df.columns:
    df['reviews/0/text_clean'] = df['reviews/0/text'].apply(preprocess_text)

print("\nAperçu du texte original et nettoyé :")
st.dataframe(df[['reviews/0/text', 'reviews/0/text_clean']].head())



# Importation
from deep_translator import GoogleTranslator
import pandas as pd

# Traduction automatique en anglais
def translate_to_english(text):
    try:
        if pd.isna(text) or text.strip() == "":
            return ""
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        return translated
    except Exception as e:
        print("Erreur de traduction:", e)
        return text  # retourne le texte original en cas d’erreur

# Application sur la colonne des avis
df['reviews/0/text_en'] = df['reviews/0/text'].apply(translate_to_english)

# Aperçu du résultat
df[['reviews/0/text', 'reviews/0/text_en']].head()

# Étape 3 : Prétraitement des Avis Textuels (avec traduction + nettoyage)

import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from deep_translator import GoogleTranslator

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    try:
        if pd.isna(text) or str(text).strip() == "":
            return ""
        text = GoogleTranslator(source='auto', target='en').translate(text)
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return " ".join(tokens)
    except Exception as e:
        return ""

# Application du traitement si la colonne existe
if 'reviews/0/text' in df.columns:
    df['reviews/0/text_clean'] = df['reviews/0/text'].apply(preprocess_text)

# Aperçu du résultat
print("\nAperçu du texte nettoyé :")
st.dataframe(df[['reviews/0/text', 'reviews/0/text_clean']].head())


# Étape 4 : Traitement de la Localisation (création de clusters de proximité)

from sklearn.cluster import KMeans

# Vérifier que les coordonnées sont présentes
if 'location/lat' in df.columns and 'location/lng' in df.columns:
    geo_data = df[['location/lat', 'location/lng']].dropna()

    kmeans = KMeans(n_clusters=5, random_state=42)
    df.loc[geo_data.index, 'region_cluster'] = kmeans.fit_predict(geo_data)

    print("\nClusters géographiques ajoutés :")
    st.dataframe(df[['title', 'city', 'region_cluster']].head())

else:
    print("Colonnes de latitude/longitude non trouvées.")

# Étape 5 : Feature Engineering
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# 5.1 Caractéristiques des Restaurants
if 'title' in df.columns:
    restaurant_stats = df.groupby('title').agg({
        'totalScore': ['mean', 'median', 'std'],
        'reviewsCount': 'mean'
    }).reset_index()

    restaurant_stats.columns = ['title', 'score_mean', 'score_median', 'score_std', 'avg_reviewsCount']

    df = pd.merge(df, restaurant_stats, on='title', how='left')
else:
    print(" Colonne 'title' introuvable pour les statistiques par restaurant.")

# 5.2 Caractéristiques Utilisateur
if 'reviews/0/reviewerNumberOfReviews' in df.columns:
    df['user_reputation'] = df['reviews/0/reviewerNumberOfReviews']
if 'reviews/0/isLocalGuide' in df.columns:
    df['is_trusted_user'] = df['reviews/0/isLocalGuide'].map({True: 1, False: 0})

# 5.3 Caractéristiques Textuelles : TF-IDF
if 'reviews/0/text_clean' in df.columns:
    tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['reviews/0/text_clean'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f'tfidf_{w}' for w in tfidf.get_feature_names_out()])
    df = pd.concat([df.reset_index(drop=True), tfidf_df], axis=1)
else:
    print(" Colonne 'reviews/0/text_clean' manquante pour TF-IDF.")

# Aperçu final
st.dataframe(df.head())


# ÉTAPE 6 - Modélisation du Système de Recommandation

#  6.1 - Content-Based Filtering (déjà partiellement fait via TF-IDF + cosine_similarity)
from sklearn.metrics.pairwise import cosine_similarity

# Recalcule la similarité s'il n'existe pas déjà
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['title'])

def recommend_content_based(title, top_n=5):
    if title not in indices:
        print("Restaurant non trouvé.")
        return pd.DataFrame()
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    restaurant_indices = [i[0] for i in sim_scores]
    return df[['title', 'city']].iloc[restaurant_indices]

print(" Recommandation par contenu pour 'Restaurant notre coin'")
display(recommend_content_based("Restaurant notre coin"))

#  6.2 - Filtrage Collaboratif (avec Surprise)
from surprise import Dataset, Reader, SVD, KNNBasic, NMF
from surprise.model_selection import train_test_split
from surprise import accuracy
import pandas as pd

# Création d'un dataset simplifié : utilisateur / restaurant / note
df_ratings = df[['reviews/0/name', 'title', 'totalScore']].dropna()
df_ratings.columns = ['user', 'restaurant', 'rating']

reader = Reader(rating_scale=(0, 1))  # totalScore a été normalisé entre 0 et 1
data = Dataset.load_from_df(df_ratings, reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

#  6.2.1 - SVD
algo_svd = SVD()
algo_svd.fit(trainset)
predictions_svd = algo_svd.test(testset)
print(" SVD RMSE :", accuracy.rmse(predictions_svd))

#  6.2.2 - KNN - User-Based
algo_knn_user = KNNBasic(sim_options={'user_based': True})
algo_knn_user.fit(trainset)
predictions_knn_user = algo_knn_user.test(testset)
print(" KNN User-Based RMSE :", accuracy.rmse(predictions_knn_user))

#  6.2.3 - KNN - Item-Based
algo_knn_item = KNNBasic(sim_options={'user_based': False})
algo_knn_item.fit(trainset)
predictions_knn_item = algo_knn_item.test(testset)
print(" KNN Item-Based RMSE :", accuracy.rmse(predictions_knn_item))

#  6.2.4 - NMF (Non-negative Matrix Factorization)
algo_nmf = NMF()
algo_nmf.fit(trainset)
predictions_nmf = algo_nmf.test(testset)
print(" NMF RMSE :", accuracy.rmse(predictions_nmf))

#  Exemple de prédiction
example_user = df_ratings['user'].iloc[0]
example_restaurant = df_ratings['restaurant'].iloc[1]
pred = algo_svd.predict(example_user, example_restaurant)
print(f"\n Note prédite par SVD pour {example_user} sur '{example_restaurant}' :", pred.est)

from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split, cross_validate, KFold
from collections import defaultdict
import pandas as pd

# Construction du dataset Surprise
reader = Reader(rating_scale=(0, 1))
df_surprise = df[['reviews/0/name', 'title', 'totalScore']]
data = Dataset.load_from_df(df_surprise, reader)

#  Split des données : 70% train, 30% test
trainset, testset = train_test_split(data, test_size=0.3)

#  Entraînement du modèle SVD
model = SVD()
model.fit(trainset)

#  Évaluation RMSE sur le test set
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
print("RMSE sur test set :", rmse)

#  Validation croisée avec SVD sur 5 plis
cv_results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

def get_top_n(predictions, n=5):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        top_n[uid] = sorted(user_ratings, key=lambda x: x[1], reverse=True)[:n]
    return top_n

top_n = get_top_n(predictions, n=5)

# Exemple d'affichage du top-5 pour un utilisateur
for uid, user_ratings in list(top_n.items())[:1]:
    print(f"Top 5 recommandations pour {uid}:")
    for iid, rating in user_ratings:
        print(f"{iid}: {rating:.3f}")

from wordcloud import WordCloud
import matplotlib.pyplot as plt

all_text = " ".join(df['reviews/0/text_clean'].dropna())
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("WordCloud des Avis Clients", fontsize=16)
plt.show()

plt.figure(figsize=(8, 4))
df['totalScore'].hist(bins=30)
plt.title("Distribution des Scores")
plt.xlabel("Score")
plt.ylabel("Nombre de restaurants")
plt.grid(True)
plt.show()


import folium

map_center = [df['location/lat'].mean(), df['location/lng'].mean()]
restaurant_map = folium.Map(location=map_center, zoom_start=12)

for _, row in df.iterrows():
    folium.Marker(
        location=[row['location/lat'], row['location/lng']],
        popup=row['title'],
        icon=folium.Icon(color="blue", icon="cutlery", prefix='fa')
    ).add_to(restaurant_map)

restaurant_map.save("restaurants_map.html")
restaurant_map

"""### Bilan Final

Nous avons construit un système de recommandation robuste appliqué aux restaurants en Mauritanie. Les étapes majeures ont inclus :

- Un prétraitement avancé des données (nettoyage, normalisation, traduction, vectorisation textuelle)
- Des recommandations basées sur le contenu (TF-IDF + Cosine Similarity)
- Des algorithmes collaboratifs (SVD, KNN, NMF) avec de bonnes performances (RMSE ~ 0.205)
- Une évaluation rigoureuse (cross-validation, RMSE, top-N ranking)
- Des visualisations claires (nuage de mots, clusters géographiques)
"""