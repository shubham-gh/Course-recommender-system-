# Course-recommender-system-
Welcome to the Course Recommender System! This is a simple web application built with Streamlit that helps you discover new online courses tailored to your interests. Instead of endlessly scrolling through catalogs, this tool uses a few different machine learning models to give you personalized suggestions.


 ## Features
Interactive Course Selection: Easily browse and select courses you've already taken from an interactive grid.

Multiple Recommender Models: Choose from three different recommendation algorithms:

Course Similarity

User-User Collaborative Filtering

Item-Item Collaborative Filtering

Adjustable Parameters: Fine-tune your recommendations by adjusting parameters like similarity thresholds and the number of suggestions.

Data Visualization: See your results in a clean table and as a bar chart to easily compare recommendations.


## Requirements
The main libraries used are:

1. Streamlit

2. Pandas

3. Scikit-learn

4. Streamlit-AgGrid


## Project Structure
The project is split into two main files to keep the code clean and organized:

recommender_app.py: This file contains all the code for the user interface (the frontend). It uses Streamlit to create the layout, buttons, sliders, and charts.

backend.py: This is the "engine" of the application. It handles all the data loading, processing, and the logic for the recommendation models.

data/: This directory holds the course data, user ratings, and the pre-calculated similarity models (.pkl files).

# How It Works
The app uses a few different approaches to find courses you might like. When you select a model from the sidebar, you're choosing which logic to use:

The Course Similarity model looks at the content of the courses you've selected and finds others that are a close match.

The User-User model finds a small group of "digital twins"—other users who have rated courses similarly to you—and then recommends courses they loved that you haven't seen yet.

The Item-Item model works a bit differently. It looks at the courses you've selected and finds other courses that are frequently liked by the same people.
