
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Chargement des données (doit être déjà nettoyées)
df = pd.read_csv("restaurants-mr.csv")
df['text'] = df['reviews/0/name'].fillna('') + ' ' + df['title'].fillna('')

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['text'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(df.index, index=df['title'].fillna('')).drop_duplicates()

def recommend_content_based(title, top_n=5):
    idx = indices.get(title)
    if idx is None:
        raise ValueError("Restaurant non trouvé.")
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    restaurant_indices = [i[0] for i in sim_scores]
    return df[['title', 'city']].iloc[restaurant_indices]

def build_recommendation_models(df):
    # Placeholdeer dummy model, can be extended
    return None, df.sample(5)[['title', 'city']]

def get_top_recommendations(dummy_df):
    return dummy_df.reset_index(drop=True)
