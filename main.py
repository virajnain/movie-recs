from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import torch
import torch.nn as nn
from flask_cors import CORS
import pickle
from difflib import get_close_matches
import numpy as np
import csv

app = Flask(__name__)
CORS(app)

# Define the model class (copy from training.py, but don't train!)
class RecommenderNet(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.movie_emb = nn.Embedding(num_movies, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)
        
    def forward(self, user_indices, movie_indices):
        user_vecs = self.user_emb(user_indices)
        movie_vecs = self.movie_emb(movie_indices)
        user_b = self.user_bias(user_indices).squeeze()
        movie_b = self.movie_bias(movie_indices).squeeze()
        
        dot = (user_vecs * movie_vecs).sum(dim=1) + user_b + movie_b
        return dot

# Load datasets and mappings
movies_df = pd.read_csv('dataset/movies.csv')
all_titles = movies_df['title'].tolist()

# Load saved mappings
with open('models/mappings.pkl', 'rb') as f:
    mappings = pickle.load(f)
    movie2idx = mappings['movie2idx']
    user2idx = mappings['user2idx']
    num_users = len(user2idx)
    num_movies = len(movie2idx)

# Initialize model with EXACT training specs
model = RecommenderNet(
    num_users=num_users,
    num_movies=num_movies,
    embedding_dim=32  # Must match training.py's value
)

# Load weights (NO TRAINING HERE!)
model.load_state_dict(torch.load('models/recommender.pth'))
model.eval()  # Set to evaluation mode

def find_best_match(title):
    matches = get_close_matches(title, all_titles, n=1, cutoff=0.6)
    return matches[0] if matches else None

def create_synthetic_user_profile(liked_movie_ids, movie2idx, model):
    """
    Create a synthetic user profile by averaging embeddings of highly-rated movies
    """
    liked_movie_indices = [movie2idx[mid] for mid in liked_movie_ids]
    
    with torch.no_grad():
        # Get embeddings for liked movies
        liked_indices_tensor = torch.tensor(liked_movie_indices)
        liked_movie_embeddings = model.movie_emb(liked_indices_tensor)
        
        # Create synthetic user embedding as average of liked movie embeddings
        synthetic_user_embedding = liked_movie_embeddings.mean(dim=0)
        
        # Estimate user bias based on movie biases of liked movies
        liked_movie_biases = model.movie_bias(liked_indices_tensor).squeeze()
        synthetic_user_bias = liked_movie_biases.mean()
        
    return synthetic_user_embedding, synthetic_user_bias

def get_user_preferred_genres(liked_movie_ids, movies_df):
    """
    Extract and rank genres from user's favorite movies
    Example: If user likes ["Toy Story", "Shrek", "Finding Nemo", "The Incredibles"]
    Returns: Animation (4x), Children (4x), Comedy (2x)
    """
    liked_movies = movies_df[movies_df['movieId'].isin(liked_movie_ids)]
    
    # Count genre occurrences across all favorite movies
    genre_counts = {}
    for _, movie in liked_movies.iterrows():
        genres = movie['genres'].split('|')
        for genre in genres:
            if genre.strip() != '(no genres listed)' and genre.strip() != '':
                clean_genre = genre.strip()
                genre_counts[clean_genre] = genre_counts.get(clean_genre, 0) + 1
    
    # Sort genres by frequency (most common first)
    preferred_genres = sorted(genre_counts.keys(), key=lambda x: genre_counts[x], reverse=True)
    return preferred_genres, genre_counts

def calculate_genre_similarity(movie_genres_str, preferred_genres, genre_counts):
    """
    Calculate weighted genre similarity score for a movie
    Higher score = more genre overlap with user's favorites
    """
    if not preferred_genres or not movie_genres_str:
        return 0.0
    
    movie_genre_list = [g.strip() for g in movie_genres_str.split('|')]
    
    # Skip movies with no genres
    if '(no genres listed)' in movie_genre_list or not movie_genre_list:
        return 0.0
    
    # Calculate weighted genre overlap
    similarity_score = 0.0
    total_preferred_count = sum(genre_counts.values())
    
    for genre in movie_genre_list:
        if genre in genre_counts:
            # Weight by how frequently this genre appears in user's preferences
            weight = genre_counts[genre] / total_preferred_count
            similarity_score += weight
    
    return similarity_score

def recommend_for_user_profile(model, liked_movie_ids, movie2idx, movies_df, top_n=10):
    """
    Recommend movies using:
    - 70% model predictions (collaborative filtering)  
    - 30% genre similarity to user's favorite genres
    """
    model.eval()
    
    # Step 1: Create synthetic user profile from favorite movies
    synthetic_user_emb, synthetic_user_bias = create_synthetic_user_profile(
        liked_movie_ids, movie2idx, model
    )
    
    # Step 2: Analyze user's preferred genres
    preferred_genres, genre_counts = get_user_preferred_genres(liked_movie_ids, movies_df)
    
    # Step 3: Get candidate movies (exclude already liked ones)
    liked_movie_indices = [movie2idx[mid] for mid in liked_movie_ids]
    all_movie_indices = list(range(len(movie2idx)))
    candidate_indices = [idx for idx in all_movie_indices if idx not in liked_movie_indices]
    
    # Convert indices back to movie IDs for processing
    idx2movie = {v: k for k, v in movie2idx.items()}
    candidate_movie_ids = [idx2movie[idx] for idx in candidate_indices]
    
    with torch.no_grad():
        # Step 4: Get model predictions for all candidate movies
        candidate_indices_tensor = torch.tensor(candidate_indices)
        candidate_movie_embeddings = model.movie_emb(candidate_indices_tensor)
        candidate_movie_biases = model.movie_bias(candidate_indices_tensor).squeeze()
        
        # Compute predicted ratings: dot_product + user_bias + movie_bias
        dot_products = torch.matmul(candidate_movie_embeddings, synthetic_user_emb)
        predicted_ratings = dot_products + synthetic_user_bias + candidate_movie_biases
        
        # Step 5: Calculate genre similarity scores
        genre_scores = []
        for movie_id in candidate_movie_ids:
            movie_row = movies_df[movies_df['movieId'] == movie_id]
            if not movie_row.empty:
                movie_genres = movie_row['genres'].iloc[0]
                genre_sim = calculate_genre_similarity(movie_genres, preferred_genres, genre_counts)
                genre_scores.append(genre_sim)
            else:
                genre_scores.append(0.0)
        
        genre_scores_tensor = torch.tensor(genre_scores, dtype=torch.float32)
        
        # Step 6: Normalize both scores to [0,1] for fair combination
        if len(predicted_ratings) > 1:
            rating_min, rating_max = predicted_ratings.min(), predicted_ratings.max()
            if rating_max > rating_min:
                normalized_ratings = (predicted_ratings - rating_min) / (rating_max - rating_min)
            else:
                normalized_ratings = torch.ones_like(predicted_ratings) * 0.5
        else:
            normalized_ratings = predicted_ratings
        
        # Normalize genre scores (they're already in [0,1] but ensure consistency)
        if len(genre_scores_tensor) > 1 and genre_scores_tensor.max() > 0:
            normalized_genre_scores = genre_scores_tensor / genre_scores_tensor.max()
        else:
            normalized_genre_scores = genre_scores_tensor
            
        # Step 7: Combine scores - 70% model prediction, 30% genre similarity
        combined_scores = 0.7 * normalized_ratings + 0.3 * normalized_genre_scores
        
        # Step 8: Get top N recommendations
        _, top_indices = torch.topk(combined_scores, k=min(top_n, len(candidate_indices)))
        top_movie_indices = [candidate_indices[i.item()] for i in top_indices]
        
        # Convert back to movie IDs and get details
        recommended_movie_ids = [idx2movie[idx] for idx in top_movie_indices]
        recommendations = movies_df[movies_df['movieId'].isin(recommended_movie_ids)]
        
        # Preserve the ranking order
        recommendations = recommendations.set_index('movieId').loc[recommended_movie_ids].reset_index()
        
        return recommendations[['movieId', 'title', 'genres']], preferred_genres

@app.route("/")
def serve_index():
    return send_from_directory(".", "index.html")

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    favorite_titles = data.get('movies', [])
    
    # Convert movie titles to movie IDs
    favorite_movie_ids = []
    matched_titles = []
    unmatched_titles = []
    
    for title in favorite_titles:
        match = find_best_match(title)
        if match:
            movie_id = movies_df[movies_df['title'] == match]['movieId'].iloc[0]
            favorite_movie_ids.append(movie_id)
            matched_titles.append(match)
        else:
            unmatched_titles.append(title)
    
    if not favorite_movie_ids:
        return jsonify({
            "recommendations": [], 
            "errors": ["No matching movies found."],
            "matched_movies": [],
            "unmatched_movies": unmatched_titles,
            "preferred_genres": []
        })
    
    # Get personalized recommendations using model + genre preferences
    recommendations, preferred_genres = recommend_for_user_profile(
        model, favorite_movie_ids, movie2idx, movies_df
    )
    
    if recommendations.empty:
        return jsonify({
            "recommendations": [], 
            "errors": ["No recommendations found."],
            "matched_movies": matched_titles,
            "unmatched_movies": unmatched_titles,
            "preferred_genres": preferred_genres[:5]
        })
    
    return jsonify({
        "recommendations": recommendations.to_dict('records'), 
        "errors": [],
        "matched_movies": matched_titles,
        "unmatched_movies": unmatched_titles,
        "preferred_genres": preferred_genres[:5]  # Show top 5 preferred genres
    })

@app.route('/style.css')
def serve_css():
    return send_from_directory(".", "style.css")

# Read the movie IDs from CSV and generate options
options = []
with open("dataset/movies.csv", newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        movie_id = row['movieId']
        options.append(f'<option value="{movie_id}">')

# Now create the datalist HTML string
datalist_html = '<datalist id="movie-ids">\n' + '\n'.join(options) + '\n</datalist>'

# Full HTML example with input + datalist
html = f'''
<input list="movie-ids" name="movieId" id="movie-id-input">
{datalist_html}
'''

# Write to an HTML file or print
with open('output.html', 'w', encoding='utf-8') as out_file:
    out_file.write(html)

print("HTML file generated with datalist.")

if __name__ == "__main__":
    app.run(port=8000, debug=True)