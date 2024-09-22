import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.model_selection import train_test_split

# Load dataset
movies = pd.read_csv('movies.csv')  # Contains 'movieId', 'title', 'genres'
ratings = pd.read_csv('ratings.csv')  # Contains 'userId', 'movieId', 'rating'

# Content-Based Filtering
# 1. Feature Engineering: Create a TF-IDF matrix of movie genres
tfidf = TfidfVectorizer(stop_words='english')
movies['genres'] = movies['genres'].fillna('')

tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Compute similarity matrix using cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get movie recommendations based on content similarity
def get_content_based_recommendations(title, cosine_sim=cosine_sim):
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]

# Collaborative Filtering
# Create a user-item matrix (pivot table)
user_movie_ratings = ratings.pivot_table(index='userId', columns='movieId', values='rating')

# Fill missing values with 0 (for simplicity)
user_movie_ratings = user_movie_ratings.fillna(0)

# Compute the similarity between users using cosine similarity
user_similarity = cosine_similarity(user_movie_ratings)
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_ratings.index, columns=user_movie_ratings.index)

# Function to predict user rating for a specific movie using collaborative filtering
def predict_rating(user_id, movie_id, user_similarity, ratings_matrix):
    similar_users = user_similarity[user_id - 1]
    similar_users_ratings = ratings_matrix.iloc[:, movie_id - 1]
    
    weighted_sum = np.dot(similar_users, similar_users_ratings)
    similarity_sum = np.sum(similar_users)
    
    if similarity_sum == 0:
        return 0
    
    return weighted_sum / similarity_sum

# Hybrid Model: Combining Collaborative and Content-Based Filtering
def hybrid_recommendation(user_id, title, user_similarity, ratings_matrix, cosine_sim):
    # Get content-based recommendations
    content_recs = get_content_based_recommendations(title)
    
    # Collaborative filtering prediction
    movie_id = movies[movies['title'] == title]['movieId'].values[0]
    collab_pred = predict_rating(user_id, movie_id, user_similarity, ratings_matrix)
    
    # Combining both by giving more weight to collaborative filtering
    hybrid_score = 0.7 * collab_pred + 0.3 * cosine_sim[movies[movies['title'] == title].index[0]].mean()
    
    return hybrid_score, content_recs

# Evaluate the Hybrid Model
def evaluate_model(ratings_matrix, user_similarity):
    X_train, X_test = train_test_split(ratings, test_size=0.2, random_state=42)
    
    y_true = []
    y_pred = []
    
    for row in X_test.itertuples():
        pred = predict_rating(row.userId, row.movieId, user_similarity, ratings_matrix)
        if pred > 0:
            y_true.append(row.rating)
            y_pred.append(pred)
    
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    return rmse

# Calculate the RMSE of the hybrid model
rmse = evaluate_model(user_movie_ratings, user_similarity)
print(f"Hybrid Model RMSE: {rmse}")

# Example Usage
user_id = 1
movie_title = "Toy Story (1995)"
score, recs = hybrid_recommendation(user_id, movie_title, user_similarity, user_movie_ratings, cosine_sim)
print(f"Hybrid Recommendation Score for {movie_title}: {score}")
print(f"Content-Based Recommendations similar to {movie_title}:")
print(recs)
