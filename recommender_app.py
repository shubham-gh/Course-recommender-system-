import streamlit as st
import pandas as pd
import numpy as np
import backend
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode

# Set page configuration
st.set_page_config(
    page_title="Course Recommender",
    page_icon="📚",
    layout="wide"
)

# Title and description
st.title("Course Recommender System")
st.markdown("""
This application helps you discover new courses based on your preferences and past course selections.
Follow the steps in the sidebar to get personalized course recommendations.
""")

# Load data
if not backend.load_data():
    st.error("Failed to load data. Please check the data directory.")
    st.stop()

# Sidebar
st.sidebar.title("Course Recommender")
st.sidebar.markdown("Follow these steps to get personalized course recommendations:")

# Step 1: Course Selection
st.sidebar.subheader("1. Select Courses")
st.sidebar.markdown("Search and select courses you have completed or audited.")

# Create a grid for course selection
def create_course_grid():
    gb = GridOptionsBuilder.from_dataframe(backend.courses_df)
    gb.configure_pagination(paginationAutoPageSize=True)
    gb.configure_side_bar()
    gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren=True)
    gb.configure_default_column(
        resizable=True,
        filterable=True,
        sortable=True,
        editable=False
    )
    
    # Configure specific columns
    gb.configure_column("course_id", headerName="Course ID", width=100)
    gb.configure_column("title", headerName="Title", width=200)
    gb.configure_column("description", headerName="Description", width=300)
    gb.configure_column("category", headerName="Category", width=150)
    
    grid_options = gb.build()
    
    return AgGrid(
        backend.courses_df,
        gridOptions=grid_options,
        enable_enterprise_modules=True,
        update_mode="MODEL_CHANGED",
        data_return_mode="FILTERED_AND_SORTED",
        fit_columns_on_grid_load=True,
        height=400,
        width="100%",
        reload_data=False
    )

# Display the course grid
grid_response = create_course_grid()
selected_courses_df = pd.DataFrame(grid_response["selected_rows"])

# Display selected courses
if selected_courses_df.shape[0] > 0:
    st.sidebar.markdown(f"**Selected Courses:** {selected_courses_df.shape[0]}")
    st.sidebar.dataframe(selected_courses_df[["course_id", "title", "category"]], height=200)
else:
    st.sidebar.markdown("**No courses selected yet.**")

# Step 2: Model Selection
st.sidebar.subheader("2. Choose Recommender Model")
model_selection = st.sidebar.selectbox(
    "Select a recommendation model:",
    backend.models
)

# Step 3: Hyperparameters
st.sidebar.subheader("3. Configure Model Parameters")
params = {}

# Course similarity model
if model_selection == backend.models[0]:
    # Add a slide bar for selecting top courses
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=1, max_value=50,
                                    value=10, step=1)
    # Add a slide bar for choosing similarity threshold
    course_sim_threshold = st.sidebar.slider('Course Similarity Threshold %',
                                             min_value=0, max_value=100,
                                             value=50, step=5)
    params['top_courses'] = top_courses
    params['sim_threshold'] = course_sim_threshold

# User-User Collaborative Filtering
elif model_selection == backend.models[1]:
    # Add a slide bar for selecting top courses
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=1, max_value=50,
                                    value=10, step=1)
    # Add a slide bar for selecting number of similar users
    num_similar_users = st.sidebar.slider('Number of similar users',
                                          min_value=1, max_value=20,
                                          value=10, step=1)
    params['top_courses'] = top_courses
    params['num_similar_users'] = num_similar_users

# Item-Item Collaborative Filtering
elif model_selection == backend.models[2]:
    # Add a slide bar for selecting top courses
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=1, max_value=50,
                                    value=10, step=1)
    # Add a slide bar for minimum similarity threshold
    min_similarity = st.sidebar.slider('Minimum similarity threshold',
                                       min_value=0.0, max_value=1.0,
                                       value=0.1, step=0.05)
    params['top_courses'] = top_courses
    params['min_similarity'] = min_similarity

# Step 4: Training
st.sidebar.subheader("4. Training")
training_button = st.sidebar.button("Train Model")
training_text = st.sidebar.text('')

# Start training process
if training_button:
    with st.spinner("Training model..."):
        success = backend.train(model_selection, params)
        if success:
            training_text.text("Model training completed successfully!")
        else:
            training_text.text("Model training failed. Please check the console for errors.")

# Step 5: Prediction
st.sidebar.subheader("5. Get Recommendations")
pred_button = st.sidebar.button("Recommend New Courses")

# Main content area
st.subheader("Recommended Courses")

# Start prediction process
if pred_button and selected_courses_df.shape[0] > 0:
    with st.spinner("Generating recommendations..."):
        # Get selected course IDs
        selected_course_ids = selected_courses_df["course_id"].tolist()
        
        # Get recommendations
        recommendations_df = backend.predict(model_selection, params, selected_course_ids)
        
        if recommendations_df.shape[0] > 0:
            # Display recommendations
            st.dataframe(recommendations_df, height=400)
            
            # Display a bar chart of top recommendations
            if "similarity" in recommendations_df.columns:
                st.subheader("Similarity Scores")
                st.bar_chart(recommendations_df.set_index("title")["similarity"])
            elif "predicted_rating" in recommendations_df.columns:
                st.subheader("Predicted Ratings")
                st.bar_chart(recommendations_df.set_index("title")["predicted_rating"])
            elif "rating" in recommendations_df.columns:
                st.subheader("Average Ratings")
                st.bar_chart(recommendations_df.set_index("title")["rating"])
        else:
            st.warning("No recommendations found. Try selecting different courses or adjusting parameters.")
elif pred_button and selected_courses_df.shape[0] == 0:
    st.warning("Please select at least one course before getting recommendations.")
else:
    st.info("Select courses and click 'Recommend New Courses' to get personalized recommendations.")

# Footer
st.markdown("---")
st.markdown("Course Recommender System | Built with Streamlit") 