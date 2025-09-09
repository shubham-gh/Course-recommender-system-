import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle

# Define available models
models = ["Course Similarity", "User-User Collaborative Filtering", "Item-Item Collaborative Filtering"]

# Global variables to store data
courses_df = None
ratings_df = None
similarity_matrix = None

def load_data():
    """
    Load datasets from the data directory
    """
    global courses_df, ratings_df, similarity_matrix
    
    # Check if data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Created data directory. Please add your datasets to this directory.")
        return False
    
    # Load course data
    try:
        courses_df = pd.read_csv('data/courses.csv')
    except FileNotFoundError:
        print("courses.csv not found in data directory. Creating sample data.")
        # Create sample course data
        courses_df = pd.DataFrame({
            'course_id': [f'COURSE{i:03d}' for i in range(1, 101)],
            'title': [f'Course {i}' for i in range(1, 101)],
            'description': [f'This is a description for Course {i}' for i in range(1, 101)],
            'category': [f'Category {i % 5 + 1}' for i in range(1, 101)]
        })
        courses_df.to_csv('data/courses.csv', index=False)
    
    # Load ratings data
    try:
        ratings_df = pd.read_csv('data/ratings.csv')
    except FileNotFoundError:
        print("ratings.csv not found in data directory. Creating sample data.")
        # Create sample ratings data
        user_ids = [f'USER{i:03d}' for i in range(1, 51)]
        course_ids = courses_df['course_id'].tolist()
        
        ratings_data = []
        for user_id in user_ids:
            # Each user rates 5-15 random courses
            num_ratings = np.random.randint(5, 16)
            rated_courses = np.random.choice(course_ids, num_ratings, replace=False)
            for course_id in rated_courses:
                rating = np.random.randint(1, 6)  # Rating from 1 to 5
                ratings_data.append({'user_id': user_id, 'course_id': course_id, 'rating': rating})
        
        ratings_df = pd.DataFrame(ratings_data)
        ratings_df.to_csv('data/ratings.csv', index=False)
    
    # Load or create similarity matrix
    try:
        with open('data/similarity_matrix.pkl', 'rb') as f:
            similarity_matrix = pickle.load(f)
            # Check if the similarity matrix has the correct dimensions
            if similarity_matrix.shape[0] != len(courses_df) or similarity_matrix.shape[1] != len(courses_df):
                print("Similarity matrix dimensions don't match the number of courses. Creating a new one.")
                raise ValueError("Incorrect dimensions")
    except (FileNotFoundError, ValueError):
        print("Creating a new similarity matrix.")
        # Create a similarity matrix based on course descriptions
        num_courses = len(courses_df)
        
        # Create a simple similarity matrix based on categories
        similarity_matrix = np.zeros((num_courses, num_courses))
        
        # Set diagonal to 1 (each course is most similar to itself)
        np.fill_diagonal(similarity_matrix, 1)
        
        # Set similarity between courses in the same category
        categories = courses_df['category'].unique()
        for category in categories:
            category_indices = courses_df[courses_df['category'] == category].index.tolist()
            for i in category_indices:
                for j in category_indices:
                    if i != j:
                        similarity_matrix[i, j] = 0.7  # High similarity for same category
        
        # Add some random variation
        for i in range(num_courses):
            for j in range(i+1, num_courses):
                if similarity_matrix[i, j] == 0:
                    # Random similarity between 0.1 and 0.5 for different categories
                    similarity_matrix[i, j] = np.random.uniform(0.1, 0.5)
                    similarity_matrix[j, i] = similarity_matrix[i, j]  # Make it symmetric
        
        with open('data/similarity_matrix.pkl', 'wb') as f:
            pickle.dump(similarity_matrix, f)
    
    return True

def train(model_name, params):
    """
    Train the selected model with the given parameters
    
    Args:
        model_name (str): Name of the model to train
        params (dict): Dictionary of model parameters
    
    Returns:
        bool: True if training was successful, False otherwise
    """
    global similarity_matrix
    
    if model_name == models[0]:  # Course Similarity
        # For course similarity, we don't need to train anything
        # The similarity matrix is already loaded
        return True
    
    elif model_name == models[1]:  # User-User Collaborative Filtering
        # Create user-item matrix
        user_item_matrix = ratings_df.pivot(
            index='user_id', 
            columns='course_id', 
            values='rating'
        ).fillna(0)
        
        # Calculate user similarity
        user_similarity = cosine_similarity(user_item_matrix)
        
        # Save the user similarity matrix
        with open('data/user_similarity.pkl', 'wb') as f:
            pickle.dump(user_similarity, f)
        
        return True
    
    elif model_name == models[2]:  # Item-Item Collaborative Filtering
        # Create user-item matrix
        user_item_matrix = ratings_df.pivot(
            index='user_id', 
            columns='course_id', 
            values='rating'
        ).fillna(0)
        
        # Calculate item similarity
        item_similarity = cosine_similarity(user_item_matrix.T)
        
        # Save the item similarity matrix
        with open('data/item_similarity.pkl', 'wb') as f:
            pickle.dump(item_similarity, f)
        
        return True
    
    return False

def predict(model_name, params, user_courses=None):
    """
    Generate course recommendations using the selected model
    
    Args:
        model_name (str): Name of the model to use
        params (dict): Dictionary of model parameters
        user_courses (list): List of course IDs that the user has already taken
    
    Returns:
        pd.DataFrame: DataFrame containing recommended courses
    """
    global courses_df, ratings_df, similarity_matrix
    
    if user_courses is None:
        user_courses = []
    
    if model_name == models[0]:  # Course Similarity
        # Get the top courses parameter
        top_courses = params.get('top_courses', 10)
        sim_threshold = params.get('sim_threshold', 50) / 100.0
        
        # Get course indices
        course_indices = courses_df[courses_df['course_id'].isin(user_courses)].index.tolist()
        
        if not course_indices:
            # If no courses selected, return top rated courses
            course_ratings = ratings_df.groupby('course_id')['rating'].mean().reset_index()
            course_ratings = course_ratings.sort_values('rating', ascending=False)
            recommended_courses = course_ratings.head(top_courses)
            recommended_courses = recommended_courses.merge(courses_df, on='course_id')
            return recommended_courses[['course_id', 'title', 'description', 'category', 'rating']]
        
        # Calculate average similarity scores
        avg_similarity = np.zeros(len(courses_df))
        for idx in course_indices:
            avg_similarity += similarity_matrix[idx]
        avg_similarity /= len(course_indices)
        
        # Filter out courses the user has already taken
        for idx in course_indices:
            avg_similarity[idx] = 0
        
        # Get top similar courses
        top_indices = np.argsort(avg_similarity)[::-1][:top_courses]
        top_similarities = avg_similarity[top_indices]
        
        # Filter by similarity threshold
        top_indices = top_indices[top_similarities >= sim_threshold]
        
        # Create recommendation dataframe
        recommended_courses = courses_df.iloc[top_indices].copy()
        recommended_courses['similarity'] = top_similarities[:len(top_indices)]
        
        return recommended_courses[['course_id', 'title', 'description', 'category', 'similarity']]
    
    elif model_name == models[1]:  # User-User Collaborative Filtering
        # Load user similarity matrix
        try:
            with open('data/user_similarity.pkl', 'rb') as f:
                user_similarity = pickle.load(f)
        except FileNotFoundError:
            return pd.DataFrame(columns=['course_id', 'title', 'description', 'category', 'predicted_rating'])
        
        # Create a new user with the selected courses
        new_user_id = 'NEW_USER'
        new_user_ratings = pd.DataFrame({
            'user_id': [new_user_id] * len(user_courses),
            'course_id': user_courses,
            'rating': [5] * len(user_courses)  # Assume user liked all selected courses
        })
        
        # Combine with existing ratings
        combined_ratings = pd.concat([ratings_df, new_user_ratings])
        
        # Create user-item matrix
        user_item_matrix = combined_ratings.pivot(
            index='user_id', 
            columns='course_id', 
            values='rating'
        ).fillna(0)
        
        # Get the index of the new user
        new_user_idx = user_item_matrix.index.get_loc(new_user_id)
        
        # Get top similar users
        top_users = np.argsort(user_similarity[new_user_idx])[::-1][1:11]  # Exclude the user itself
        
        # Get courses not rated by the new user
        new_user_rated = set(user_courses)
        all_courses = set(courses_df['course_id'])
        unrated_courses = list(all_courses - new_user_rated)
        
        # Predict ratings for unrated courses
        predictions = []
        for course_id in unrated_courses:
            if course_id in user_item_matrix.columns:
                # Get ratings for this course by similar users
                course_ratings = user_item_matrix.iloc[top_users][course_id]
                # Calculate weighted average
                weights = user_similarity[new_user_idx][top_users]
                predicted_rating = np.sum(course_ratings * weights) / np.sum(weights)
                predictions.append((course_id, predicted_rating))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top recommendations
        top_courses = params.get('top_courses', 10)
        top_predictions = predictions[:top_courses]
        
        # Create recommendation dataframe
        recommended_courses = []
        for course_id, predicted_rating in top_predictions:
            course_info = courses_df[courses_df['course_id'] == course_id].iloc[0]
            recommended_courses.append({
                'course_id': course_id,
                'title': course_info['title'],
                'description': course_info['description'],
                'category': course_info['category'],
                'predicted_rating': predicted_rating
            })
        
        return pd.DataFrame(recommended_courses)
    
    elif model_name == models[2]:  # Item-Item Collaborative Filtering
        # Load item similarity matrix
        try:
            with open('data/item_similarity.pkl', 'rb') as f:
                item_similarity = pickle.load(f)
        except FileNotFoundError:
            return pd.DataFrame(columns=['course_id', 'title', 'description', 'category', 'predicted_rating'])
        
        # Create a new user with the selected courses
        new_user_id = 'NEW_USER'
        new_user_ratings = pd.DataFrame({
            'user_id': [new_user_id] * len(user_courses),
            'course_id': user_courses,
            'rating': [5] * len(user_courses)  # Assume user liked all selected courses
        })
        
        # Combine with existing ratings
        combined_ratings = pd.concat([ratings_df, new_user_ratings])
        
        # Create user-item matrix
        user_item_matrix = combined_ratings.pivot(
            index='user_id', 
            columns='course_id', 
            values='rating'
        ).fillna(0)
        
        # Get courses not rated by the new user
        new_user_rated = set(user_courses)
        all_courses = set(courses_df['course_id'])
        unrated_courses = list(all_courses - new_user_rated)
        
        # Get course indices
        course_indices = {course_id: idx for idx, course_id in enumerate(courses_df['course_id'])}
        
        # Predict ratings for unrated courses
        predictions = []
        for course_id in unrated_courses:
            if course_id in user_item_matrix.columns and course_id in course_indices:
                course_idx = course_indices[course_id]
                
                # Get similarity scores with rated courses
                rated_indices = [course_indices[cid] for cid in user_courses if cid in course_indices]
                if not rated_indices:
                    continue
                
                similarities = item_similarity[course_idx, rated_indices]
                
                # Calculate weighted average
                predicted_rating = np.sum(similarities) / len(similarities)
                predictions.append((course_id, predicted_rating))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top recommendations
        top_courses = params.get('top_courses', 10)
        top_predictions = predictions[:top_courses]
        
        # Create recommendation dataframe
        recommended_courses = []
        for course_id, predicted_rating in top_predictions:
            course_info = courses_df[courses_df['course_id'] == course_id].iloc[0]
            recommended_courses.append({
                'course_id': course_id,
                'title': course_info['title'],
                'description': course_info['description'],
                'category': course_info['category'],
                'predicted_rating': predicted_rating
            })
        
        return pd.DataFrame(recommended_courses)
    
    return pd.DataFrame(columns=['course_id', 'title', 'description', 'category']) 