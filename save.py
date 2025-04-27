# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load data
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Preprocess
user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
user_movie_matrix.fillna(0, inplace=True)

# User Similarity Matrix
user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

# Function to recommend movies for existing users
def recommend_movies(user_id, num_recommendations=5):
    if user_id not in user_movie_matrix.index:
        return recommend_for_new_user(num_recommendations)

    # Get the similarity scores for the selected user
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)

    # Weighted average of ratings by similar users
    user_ratings = user_movie_matrix.mul(similar_users, axis=0).sum(axis=0)
    user_ratings /= similar_users.sum()

    # Exclude movies already rated by the user
    recommendations = user_ratings[user_movie_matrix.loc[user_id] == 0]
    recommendations = recommendations.sort_values(ascending=False).head(num_recommendations)

    return recommendations

# Function to recommend movies for new users
def recommend_for_new_user(num_recommendations=5):
    initial_ratings = get_initial_ratings()

    # Add new user to the user-movie matrix
    new_user_id = user_movie_matrix.index.max() + 1
    new_user_ratings = pd.Series(0.0, index=user_movie_matrix.columns)
    for movie_id, rating in initial_ratings.items():
        new_user_ratings[movie_id] = float(rating)
    user_movie_matrix.loc[new_user_id] = new_user_ratings

    # Recalculate similarity matrix
    updated_similarity = cosine_similarity(user_movie_matrix)
    global user_similarity_df
    user_similarity_df = pd.DataFrame(updated_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

    # Generate recommendations using the new ratings
    return recommend_movies(new_user_id, num_recommendations)

# Function to get initial ratings from a new user
def get_initial_ratings():
    initial_movies = movies.sample(10)
    ratings = {}
    for _, row in initial_movies.iterrows():
        while True:
            try:
                rating = float(input(f"Rating for {row['title']}: "))
                if 1 <= rating <= 5:
                    ratings[row['movieId']] = rating
                    break
            except ValueError:
                pass
    return ratings

# Save the necessary data to a pickle file
model_data = {
    'movies': movies,
    'ratings': ratings,
    'user_movie_matrix': user_movie_matrix,
    'user_similarity_df': user_similarity_df
}

# Save the model data to a pickle file
with open('recommendation_model.pkl', 'wb') as file:
    pickle.dump(model_data, file)

# Test recommendation function
user_id = 2300321  # New user ID
recommended_movies_series = recommend_movies(user_id)

if isinstance(recommended_movies_series, pd.Series):
    recommended_movies_series.name = 'score'

    # Get recommended movie IDs
    recommended_movie_ids = recommended_movies_series.index.tolist()

    # Find movie titles for recommended movie IDs
    recommended_movies = movies[movies['movieId'].isin(recommended_movie_ids)][['movieId', 'title']]

    # Merge with recommendation scores for clarity
    recommended_movies = recommended_movies.merge(recommended_movies_series, left_on='movieId', right_index=True)

    # Display recommended movies with titles and scores
    print(recommended_movies[['title', 'score']])