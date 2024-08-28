import os
import pandas as pd
import numpy as np
import streamlit as st
from gensim.models import LdaSeqModel
from gensim import corpora
from sklearn.metrics.pairwise import cosine_similarity
from gensim.utils import simple_preprocess

# Define paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
model_path = os.path.join(project_root, "results", "models", "dtm_lda_model")
dictionary_path = os.path.join(project_root, "results", "models", "dtm_lda_dictionary")
preprocessed_file_path = os.path.join(project_root, "data", "processed", "preprocessed_courses.csv")

@st.cache_resource
def load_model_and_data():
    model = LdaSeqModel.load(model_path)
    dictionary = corpora.Dictionary.load(dictionary_path)
    df = pd.read_csv(preprocessed_file_path)
    df['processed_text'] = df['processed_text'].str.split()
    df['last_updated'] = pd.to_datetime(df['last_updated'], format='mixed', dayfirst=True)
    return model, dictionary, df

def infer_topic_distribution(text, _model, _dictionary):
    if not text:
        return np.zeros(_model.num_topics)
    tokens = simple_preprocess(text)
    bow = _dictionary.doc2bow(tokens)
    topic_dist = _model[bow]
    
    # Handle different types of return values
    if isinstance(topic_dist, np.ndarray):
        return topic_dist
    elif isinstance(topic_dist, list):
        return np.array([prob for _, prob in topic_dist])
    elif isinstance(topic_dist, np.float64):
        # If it's a single float, create an array with this value
        return np.array([topic_dist])
    else:
        # If it's none of the above, return zeros
        print(f"Unexpected type for topic_dist: {type(topic_dist)}")
        return np.zeros(_model.num_topics)

def recommend_courses(user_history, df, model, dictionary, top_n=5):
    user_vector = np.zeros(model.num_topics)
    for course in user_history:
        course_text = df[df['title'] == course]['processed_text'].values[0]
        # Check if course_text is already a string
        if isinstance(course_text, str):
            text_to_process = course_text
        else:
            # If it's a list or another iterable, join it
            text_to_process = ' '.join(course_text) if hasattr(course_text, '__iter__') else str(course_text)
        
        course_vector = infer_topic_distribution(text_to_process, model, dictionary)
        user_vector += course_vector
    
    if len(user_history) > 0:
        user_vector /= len(user_history)
    
    recommendations = []
    for _, row in df.iterrows():
        if row['title'] in user_history:
            continue
        course_text = row['processed_text']
        # Apply the same check here
        if isinstance(course_text, str):
            text_to_process = course_text
        else:
            text_to_process = ' '.join(course_text) if hasattr(course_text, '__iter__') else str(course_text)
        
        course_vector = infer_topic_distribution(text_to_process, model, dictionary)
        similarity = cosine_similarity([user_vector], [course_vector])[0][0]
        recommendations.append((row['title'], row['url'], row['instructors'], row['description'], similarity))
    
    return sorted(recommendations, key=lambda x: x[4], reverse=True)[:top_n]

def forecast_future_interests(user_history, model, dictionary, df):
    user_vector = np.zeros(model.num_topics)
    for course in user_history:
        course_text = df[df['title'] == course]['processed_text'].values[0]
        # Check if course_text is already a string
        if isinstance(course_text, str):
            text_to_process = course_text
        else:
            # If it's a list or another iterable, join it
            text_to_process = ' '.join(course_text) if hasattr(course_text, '__iter__') else str(course_text)
        
        course_vector = infer_topic_distribution(text_to_process, model, dictionary)
        user_vector += course_vector
    
    if len(user_history) > 0:
        user_vector /= len(user_history)
    
    # Get top 5 topic indices
    future_topics = sorted(range(len(user_vector)), key=lambda i: user_vector[i], reverse=True)[:5]
    
    # Load topic interpretations
    interpretations_file = os.path.join(project_root, "results", "models", "topic_interpretations.txt")
    topic_interpretations = {}
    with open(interpretations_file, 'r') as f:
        for line in f:
            id, name, description = line.strip().split('|')
            topic_interpretations[int(id)] = name
    
    # Get topic names for the top topics
    future_interests = [topic_interpretations.get(i, f"Topic {i}") for i in future_topics]
    
    return future_interests

def trend_aware_recommendations(user_history, model, dictionary, df, top_n=5):
    user_vector = np.zeros(model.num_topics)
    for course in user_history:
        course_text = df[df['title'] == course]['processed_text'].values[0]
        if isinstance(course_text, str):
            text_to_process = course_text
        else:
            text_to_process = ' '.join(course_text) if hasattr(course_text, '__iter__') else str(course_text)
        course_vector = infer_topic_distribution(text_to_process, model, dictionary)
        user_vector += course_vector
    
    if len(user_history) > 0:
        user_vector /= len(user_history)
    
    trend_vector = np.ones(model.num_topics)
    combined_vector = (user_vector + trend_vector) / 2
    
    recommendations = []
    for _, row in df.iterrows():
        if row['title'] in user_history:
            continue
        course_text = row['processed_text']
        if isinstance(course_text, str):
            text_to_process = course_text
        else:
            text_to_process = ' '.join(course_text) if hasattr(course_text, '__iter__') else str(course_text)
        course_vector = infer_topic_distribution(text_to_process, model, dictionary)
        similarity = cosine_similarity([combined_vector], [course_vector])[0][0]
        recommendations.append((row['title'], row['url'], row['instructors'], row['description'], similarity))
    
    return sorted(recommendations, key=lambda x: x[4], reverse=True)[:top_n]

# Streamlit app
st.title("Course Recommendation System")

model, dictionary, df = load_model_and_data()

all_courses = df['title'].tolist()
user_history = st.multiselect("Select courses you've already taken:", all_courses)

if user_history:
    st.subheader("Recommended Courses:")
    recommendations = recommend_courses(user_history, df, model, dictionary)
    for title, url, instructor, description, similarity in recommendations:
        st.write(f"**Title:** {title}")
        st.write(f"**URL:** {url}")
        st.write(f"**Instructor:** {instructor}")
        st.write(f"**Description:** {description[:200]}...")
        st.write(f"**Similarity:** {similarity:.4f}")
        st.write("---")

    st.subheader("Forecasting Future Interests:")
    future_interests = forecast_future_interests(user_history, model, dictionary, df)
    if future_interests:
        st.write(f"Based on your course history, you might be interested in topics related to: {', '.join(future_interests)}")
    else:
        st.write("Unable to forecast future interests based on your current course history.")

    st.subheader("Trend-Aware Recommendations:")
    trend_recommendations = trend_aware_recommendations(user_history, model, dictionary, df)
    for title, url, instructor, description, similarity in trend_recommendations:
        st.write(f"**Title:** {title}")
        st.write(f"**URL:** {url}")
        st.write(f"**Instructor:** {instructor}")
        st.write(f"**Description:** {description[:200]}...")
        st.write(f"**Similarity:** {similarity:.4f}")
        st.write("---")
else:
    st.warning("Please select at least one course from your learning history to get recommendations.")