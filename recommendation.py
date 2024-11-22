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
model = KMeans(n_clusters=10,verbose=1)
df['label'] = model.fit_predict(dataScaler)

# Fonction pour chercher les genres
def search_genres_ByNameArtist(track_name, artist_name):
    # Rechercher la piste par titre et artiste
    results = sp.search(q=f"track:{track_name} artist:{artist_name}", type='track', limit=1)
    
    if results['tracks']['items']:
        # Récupérer l'artiste principal de la piste
        artist_id = results['tracks']['items'][0]['artists'][0]['id']
        
        # Récupérer les détails de l'artiste
        artist = sp.artist(artist_id)
        genres = artist.get('genres', [])
        
        if genres:
            return genres[0]  # Retourner le premier genre associé
        else:
            return "Genre non disponible"
    else:
        return "Chanson ou artiste introuvable"

    

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


def recuperer_genres(nom_artist, titre):
    genre = search_genres_ByNameArtist(nom_artist,titre)
    recommended_genres = recommandation_genres(genre, k=4)
    return recommended_genres.tolist()


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
nom_artist = st.text_input("", placeholder="Donnez le nom de votre artiste")
titre_chansons = st.text_input("", placeholder="Donnez le titre du chanson")

if st.button('Chercher') and titre_chansons and nom_artist:
    recommendation = recuperer_genres(nom_artist,titre_chansons)
    # Définir le nombre de colonnes
    num_cols = 2

    # Diviser la liste en groupes selon le nombre de colonnes
    rows = [recommendation[i:i + num_cols] for i in range(0, len(recommendation), num_cols)]
    for row in rows:
        cols = st.columns(num_cols)
        for col, item in zip(cols, row):
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



