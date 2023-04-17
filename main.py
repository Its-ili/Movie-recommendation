import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


ratings = pd.read_csv('ratings.csv')

movies = pd.read_csv('movies.csv')

# Merge the ratings and movies data
df = pd.merge(ratings, movies, on='movieId')

# Calculate the mean rating for each movie
movie_ratings_mean = df.groupby('title')['rating'].mean()

# Create a pivot table of user ratings with movies as columns and users as rows
user_ratings = df.pivot_table(index='userId', columns='title', values='rating')

# Define a function to get movie recommendations for a given movie title
def get_movie_recommendations(movie_title):
    # Get the user ratings for the given movie title
    movie_ratings = user_ratings[movie_title]
    # Calculate the similarity between the given movie and all other movies
    similarity_scores = cosine_similarity([movie_ratings], user_ratings)[0]
    # Create a dataframe of similarity scores and movie titles
    similarity_df = pd.DataFrame({'title': user_ratings.columns, 'similarity': similarity_scores})
    # Sort the dataframe by similarity scores in descending order
    similarity_df = similarity_df.sort_values('similarity', ascending=False)
    # Filter out the given movie title
    similarity_df = similarity_df[similarity_df['title'] != movie_title]
    # Get the top 5 similar movies
    top_similar_movies = similarity_df.head(5)['title']
    # Get the movie recommendations
    recommendations = movie_ratings_mean[top_similar_movies]
    return recommendations

# Example usage: Get movie recommendations for the movie "Toy Story (1995)"
movie_title = "Toy Story (1995)"
recommendations = get_movie_recommendations(movie_title)
print(f"Movie Recommendations for '{movie_title}':")
print(recommendations)
