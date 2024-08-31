import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import streamlit as st

# Load the dataset
df = pd.read_csv('spotify_songs.csv')

# Features for recommendation system
features = ['danceability', 'energy', 'key', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Preprocess the data: Normalize feature values
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])


# Function to recommend songs based on user preferences
def recommend_songs(user_input_preferences, n_recommendations=10):
    # Initialize the model
    knn = NearestNeighbors(n_neighbors=n_recommendations, metric='euclidean')

    # Fit the model on the dataset features
    knn.fit(df[features])

    # Predict the closest songs
    distances, indices = knn.kneighbors([user_input_preferences])

    # Return the recommended songs
    return df.iloc[indices[0]]


# Streamlit app interface
st.title('Spotify Song Recommender')

# Get user input for preferences
danceability = st.slider('Danceability', 0.0, 1.0, 0.5)
energy = st.slider('Energy', 0.0, 1.0, 0.5)
key = st.slider('Key', 0.0, 1.0, 0.5)
loudness = st.slider('Loudness', -60.0, 0.0, -30.0)  # Adjusting the range for loudness
speechiness = st.slider('Speechiness', 0.0, 1.0, 0.5)
acousticness = st.slider('Acousticness', 0.0, 1.0, 0.5)
instrumentalness = st.slider('Instrumentalness', 0.0, 1.0, 0.5)
liveness = st.slider('Liveness', 0.0, 1.0, 0.5)
valence = st.slider('Valence', 0.0, 1.0, 0.5)
tempo = st.slider('Tempo', 0.0, 250.0, 120.0)  # Adjusting the range for tempo

# Collect preferences into an array
user_preferences_input = np.array([danceability, energy, key, loudness, speechiness,
                                   acousticness, instrumentalness, liveness, valence, tempo])

# Get recommendations
if st.button('Recommend Songs'):
    recommendations = recommend_songs(user_preferences_input)
    st.write('Top 10 Song Recommendations:')
    st.write(recommendations[['track_name', 'track_artist', 'track_album_name', 'playlist_name']])
