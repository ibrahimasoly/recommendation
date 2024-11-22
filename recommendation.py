import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
import os

# Initialisation de Spotipy pour accéder à l'API Spotify
client_id = os.environ.get("SPOTIFY_CLIENT_ID")
client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET")


client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = Spotify(client_credentials_manager=client_credentials_manager)

# Charger le jeu de données (supposons qu'il soit au format CSV)
df = pd.read_csv("data_by_genres.csv")
data=df.drop('genres', axis=1)
feature=['mode','acousticness','danceability','duration_ms','energy','instrumentalness','liveness','loudness','speechiness','mode','tempo','valence','popularity','key']
scaler = StandardScaler()
dataScaler = scaler.fit_transform(data[feature])
model = KMeans(n_clusters=5)
df['label'] = model.fit_predict(dataScaler)

# Fonction pour chercher les genres
def search_genres_ByNameArtist(artist_name):
    results = sp.search(q=artist_name, type='artist', limit=1)
    if results['artists']['items']:
        artist = results['artists']['items'][0]
        return artist['genres']
    else:
        return None
    

# Fonction pour chercher un artiste
def search_artist_ByGenre(genre_name):
    results = sp.search(q=f"genre:{genre_name}", type="artist", limit=2)
    a=[]
    for artist in results['artists']['items']:
        a.append(artist)
    return a


def cluster_similaire(nom_genre):
    label_genre = df[df['genres']==nom_genre]['label'].values[0]
    same_genre = df[df['label']==label_genre]
    return same_genre


def recommandation_genres(nom_genre, k=5):
    # Vérifier si le genre existe dans df
    if nom_genre in df['genres'].values:
        genre_index = df[df['genres'] == nom_genre].index[0]
        same_genre = cluster_similaire(nom_genre)
        same_genre_index = same_genre.index

        similarity = cosine_similarity(dataScaler, dataScaler)
        filter_similarity = similarity[genre_index][same_genre_index]

        same_genre['similarity'] = filter_similarity
        recommend_genre = same_genre.sort_values(by='similarity', ascending=False)['genres']
        return recommend_genre[1:k+1].values


def recuperer_genres(id):
    genres = search_genres_ByNameArtist(id)
    if genres and len(genres) > 1:
        genres1 = []  # Liste pour stocker les recommandations
        genre1 = []  # Liste pour stocker les genres d'entrée

        # Obtenir les recommandations pour chaque genre
        for genre in genres:
            recommended_genres = recommandation_genres(genre, k=4)
            genres1.append([recommended_genres]) 
            genre1.append(genre)
        df_pd= pd.DataFrame(genres1)
        df_pd.dropna(inplace=True)
        d = []
        for i in np.array(df_pd).tolist():
            d.append(i[0].tolist())
    return d


def recommandation_systeme(genre_nname):
    artists=search_artist_ByGenre(genre_nname)
    return [{
            "spotify": artist['external_urls']['spotify'],
            "name": artist['name'],
            "id": artist['id'],
            "popularity": artist['popularity'],
            "images": artist['images']
        } for artist in artists]

def get_track_preview(track_name):
    # Rechercher une piste par son nom
    results = sp.search(q=track_name, type='track', limit=1)
    if results['tracks']['items']:
        track = results['tracks']['items'][0]
        preview_url = track.get('preview_url')  # URL de l'extrait audio
        track_name = track.get('name')
        artist_name = track['artists'][0]['name']
        return {"track_name": track_name, "artist_name": artist_name, "preview_url": preview_url}
    else:
        return None



st.title('Recommandation Musical')
id = st.text_input("", placeholder="Donnez le nom de votre artiste")

if st.button('Chercher'):
    recom = recuperer_genres(id)
    for i in recom:
        k = []
        for j in i:
            recommendations = recommandation_systeme([j])
            if isinstance(recommendations, list):  # Si c'est une liste
                k.extend(recommendations)
            else:  # Si c'est un objet ou une chaîne
                k.append(recommendations)

        if not k:
            st.write("")
        else:
            cols = st.columns(3)
            for col, recommendation in zip(cols, k):
                with col:
                    # Vérifier la présence d'images
                    if recommendation.get("images") and len(recommendation["images"]) > 1:
                        image_url = recommendation["images"][1]["url"]
                    elif recommendation.get("images") and len(recommendation["images"]) > 0:
                        image_url = recommendation["images"][0]["url"]
                    else:
                        image_url = "https://via.placeholder.com/150"  # Image par défaut

                    st.image(image_url)
                    st.subheader(recommendation["name"])
                    st.write(f"Popularité : {recommendation['popularity']}")
                    st.markdown(
                        f'<a href="{recommendation['spotify']}" target="_blank" style="text-decoration: none; color: green; font-size: 18px;">Écoutez {recommendation["name"]} sur Spotify</a>',
                        unsafe_allow_html=True
                    )

                    # Ajouter l'audio si disponible
                    track_info = get_track_preview(recommendation['name'])
                    if track_info and track_info['preview_url']:
                        st.audio(track_info['preview_url'], format="audio/mp3")
                    else:
                        st.write("")



