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
    
    # Function to safely process text
    def safe_process(x):
        if isinstance(x, str):
            return x.split()
        elif pd.isna(x):
            return []  # Return empty list for NaN values
        else:
            return str(x).split()  # Convert to string and split for other types
    
    # Apply safe processing to the processed_text column
    df['processed_text'] = df['processed_text'].apply(safe_process)
    df['last_updated'] = pd.to_datetime(df['last_updated'], format='mixed', dayfirst=True)
    
    # Debug: Print a sample of processed text
    st.write("Sample processed text:", df['processed_text'].iloc[0])
    
    return model, dictionary, df

@st.cache_data
def infer_topic_distribution(text, _model, _dictionary):
    if not text:
        return np.zeros(_model.num_topics)

    tokens = simple_preprocess(text)
    st.write(f"Tokens: {tokens}")
    
    bow = _dictionary.doc2bow(tokens)
    st.write(f"Bag of words: {bow}")
    
    if not bow:
        st.warning(f"No recognized words in the input: {text[:100]}...")  # Print first 100 characters
        return np.zeros(_model.num_topics)

    topic_dist = _model[bow]
    st.write(f"Raw topic distribution: {topic_dist}")
    
    if isinstance(topic_dist, np.ndarray):
        return topic_dist
    elif isinstance(topic_dist, list):
        return np.array([prob for _, prob in topic_dist])
    else:
        st.warning(f"Unexpected topic distribution type: {type(topic_dist)}")
        return np.zeros(_model.num_topics)

def recommend_courses(user_input, df, model, dictionary, top_n=5):
    user_vector = infer_topic_distribution(user_input, model, dictionary)
    
    st.write("User vector shape:", user_vector.shape)
    st.write("User vector:", user_vector)

    if np.all(user_vector == 0):
        st.warning("The input didn't match any known topics. Try using more specific terms related to the courses.")
        return []
    
    recommendations = []
    
    for _, row in df.iterrows():
        # The processed_text is already a list of words, so we don't need to join and split
        course_text = row['processed_text']
        
        course_vector = infer_topic_distribution(' '.join(course_text), model, dictionary)
        
        st.write(f"Course text for {row['title']}:", ' '.join(course_text[:10]))  # Print first 10 words
        st.write(f"Course vector shape for {row['title']}:", course_vector.shape)
        st.write(f"Course vector for {row['title']}:", course_vector)
        
        if np.all(course_vector == 0):
            similarity = 0
        else:
            similarity = cosine_similarity([user_vector], [course_vector])[0][0]
        recommendations.append((row['title'], row['url'], row['instructors'], row['description'], similarity))
    
    return sorted(recommendations, key=lambda x: x[4], reverse=True)[:top_n]

# Streamlit app for course recommendation
st.title("Course Recommendation System")

st.header("Find Courses Based on Your Interests")

user_input = st.text_area("Describe your interests or the type of course you're looking for:")

if user_input:
    model, dictionary, df = load_model_and_data()
    
    # Debug: Print model information
    st.write("Model Information:")
    st.write(f"Number of topics: {model.num_topics}")
    st.write(f"Number of time slices: {model.num_time_slices}")
    st.write(f"Vocabulary size: {model.vocab_len}")
    
    try:
        with st.spinner('Processing your request...'):
            recommendations = recommend_courses(user_input, df, model, dictionary)
        
        if recommendations:
            st.subheader("Recommended Courses:")
            for title, url, instructor, description, similarity in recommendations:
                st.write(f"**Title:** {title}")
                st.write(f"**URL:** {url}")
                st.write(f"**Instructor:** {instructor}")
                st.write(f"**Description:** {description[:200]}...")
                st.write(f"**Similarity:** {similarity:.4f}")
                st.write("---")
        else:
            st.info("No similar courses found. Try adjusting your input or exploring our course catalog.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Debug information:")
        st.write(f"User input: {user_input}")
        st.write(f"Model type: {type(model)}")
        st.write(f"Dictionary type: {type(dictionary)}")
        st.write(f"DataFrame shape: {df.shape}")
        st.write("Model methods:", dir(model))
else:
    st.warning("Please enter your interests or the type of course you're looking for.")