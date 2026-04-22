import pandas as pd
import numpy as np
from thefuzz import process
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

def build_recommender(file_path):
    df = pd.read_csv(file_path)
    
    #clean data and remove duplicates
    df['track_name'] = df['track_name'].str.strip()
    df['artists'] = df['artists'].str.strip()
    df = df.drop_duplicates(subset=['track_name', 'artists'], keep='first')
    
    #list of the audio features
    audio_features = [
        'danceability', 'energy', 'key', 'loudness', 'mode', 
        'speechiness', 'acousticness', 'instrumentalness', 
        'liveness', 'valence', 'tempo'
    ]
    
    #drop missing values and reset index
    df = df.dropna(subset=['track_name', 'track_genre'] + audio_features).reset_index(drop=True)
    
    #ONE-HOT ENCODING GENRES
    #creates a 0 or 1 column for every single genre
    genre_dummies = pd.get_dummies(df['track_genre'], prefix='genre')
    
    #scaling the audio features
    scaler = StandardScaler()
    scaled_audio = scaler.fit_transform(df[audio_features])
    
    #combine & weight
    #multiply the genre columns by 3
    #multiply popularity by 0.5 so not too dominant
    weighted_genres = genre_dummies.values * 3.0
    
    #combining audio features and genre features into one big matrix
    final_features = np.hstack([scaled_audio, weighted_genres])
    
    #knn model
    model = NearestNeighbors(n_neighbors=20, metric='cosine')
    model.fit(final_features)
    
    return df, model, final_features

def get_recommendations(song_query, df, model, final_features, num_recommendations=5):
    #fuzzy matching to  find the song
    all_song_names = df['track_name'].tolist()
    best_match, score = process.extractOne(song_query, all_song_names)
    
    if score < 70:
        print(f"No close match found for '{song_query}'.")
        return

    #handler for multiple artists for the same song name
    possible_matches = df[df['track_name'] == best_match]
    if len(possible_matches) > 1:
        print(f"\nI found multiple artists for '{best_match}':")
        for i, artist in enumerate(possible_matches['artists'].values):
            print(f"{i}: {artist}")
        selection = int(input("Enter the number of the artist you want: "))
        song_idx = possible_matches.index[selection]
    else:
        song_idx = possible_matches.index[0]

    #get data for selected song
    query_song_features = final_features[song_idx].reshape(1, -1)
    orig_popularity = df.iloc[song_idx]['popularity']
    orig_name = df.iloc[song_idx]['track_name'].lower()

    #find neighbors (im going with 40 to ensure there is enough after filtering)
    distances, indices = model.kneighbors(query_song_features, n_neighbors=40)
    
    print(f"\nRecommendations based on '{df.iloc[song_idx]['track_name']}' by {df.iloc[song_idx]['artists']}:")
    print("-" * 60)
    
    count = 0
    for i in range(1, len(distances.flatten())):
        if count >= num_recommendations:
            break
            
        idx = indices.flatten()[i]
        rec_name = df.iloc[idx]['track_name'].lower()
        rec_popularity = df.iloc[idx]['popularity']
        
        #1st filter - skip if it's the exact same song name (different album/remix)
        if rec_name == orig_name:
            continue
            
        #2nd filter - popularity guardrail
        #only suggest songs within a 35-point popularity range to keep the vibe similar
        if abs(rec_popularity - orig_popularity) > 35:
            continue
            
        similarity = 1 - distances.flatten()[i]
        artist = df.iloc[idx]['artists']
        genre = df.iloc[idx]['track_genre']
        
        print(f"- {df.iloc[idx]['track_name']} by {artist} | Genre: {genre} ({similarity:.2%})")
        count += 1

#execution
df, knn_model, feature_matrix = build_recommender('dataset.csv')

while True:
    user_input = input("\nEnter a song name (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    get_recommendations(user_input, df, knn_model, feature_matrix)