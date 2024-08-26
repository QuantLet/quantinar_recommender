import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from gensim.models import LdaSeqModel
from gensim import corpora

# Define paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
model_path = os.path.join(project_root, "results", "models", "dtm_lda_model")
dictionary_path = os.path.join(project_root, "results", "models", "dtm_lda_dictionary")
interpretations_file = os.path.join(project_root, "results", "models", "topic_interpretations.txt")
preprocessed_file_path = os.path.join(project_root, "data", "processed", "preprocessed_courses.csv")

# Load model, dictionary, and interpretations
st.title("Dynamic Topic Modeling - Topic Trends and Interpretations")

@st.cache(allow_output_mutation=True)
def load_model_and_data():
    model = LdaSeqModel.load(model_path)
    dictionary = corpora.Dictionary.load(dictionary_path)
    df = pd.read_csv(preprocessed_file_path)
    df['processed_text'] = df['processed_text'].apply(eval)
    df['last_updated'] = pd.to_datetime(df['last_updated'], format='mixed', dayfirst=True)
    
    return model, dictionary, df

@st.cache
def load_topic_interpretations(file_path):
    interpretations = {}
    with open(file_path, 'r') as f:
        for line in f:
            id, name, description = line.strip().split('|')
            interpretations[int(id)] = {'name': name, 'description': description}
    return interpretations

model, dictionary, df = load_model_and_data()
topic_interpretations = load_topic_interpretations(interpretations_file)

# Display Topic Interpretations
st.sidebar.title("Topic Interpretations")
for tid, interpretation in topic_interpretations.items():
    st.sidebar.write(f"**Topic {tid}:** {interpretation['name']}")
    st.sidebar.write(f"*Description:* {interpretation['description']}")

# Select topic to visualize trends
selected_topic = st.sidebar.selectbox("Select Topic to Visualize", options=list(topic_interpretations.keys()), format_func=lambda x: f"Topic {x}: {topic_interpretations[x]['name']}")

# Plot topic trends over time
def plot_topic_trends(model, time_slices, topic_labels, dates):
    plt.figure(figsize=(10, 6))
    topic_trends = [model.print_topic(topic=selected_topic, time=j)[0][1] for j in range(len(time_slices))]
    plt.plot(dates, topic_trends, label=topic_labels[selected_topic])
    plt.xlabel('Time')
    plt.ylabel('Topic Prominence')
    plt.title(f'Topic {selected_topic}: {topic_labels[selected_topic]} Trends Over Time')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt

df['month'] = df['last_updated'].dt.to_period('M')
time_slices = df.groupby('month').size().tolist()
dates = df.groupby('month').first()['last_updated'].tolist()

topic_labels = [f"Topic {tid}: {topic_interpretations[tid]['name']}" for tid in topic_interpretations.keys()]
trends_plt = plot_topic_trends(model, time_slices, topic_labels, dates)

st.pyplot(trends_plt)
