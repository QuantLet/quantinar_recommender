# Course Recommendation System

This project implements a dynamic topic modeling-based course recommendation system. It analyzes course data, identifies topics, and provides personalized course recommendations based on a user's learning history.

## Features

- Dynamic Topic Modeling (DTM) using LDA techniques
- Course recommendations based on user history
- Trend-aware recommendations
- Future interest forecasting
- Interactive web interface using Streamlit

## Installation

1. Clone this repository
2. Install required packages:
   ```
   pip install pandas numpy streamlit gensim scikit-learn matplotlib seaborn pyLDAvis openai
   ```
3. Set up your OpenAI API key as an environment variable:
   ```
   export OPENAI_API_KEY='your-api-key-here'
   ```

## Usage

1. Run the topic modeling script:
   ```
   python code/topic_modeling/dtm_lda_model.py
   ```
   This will generate the topic model and save the results in the `results/` directory.

2. Start the Streamlit app:
   ```
   streamlit run code/app/course_recommender.py
   ```
   This will launch the web interface for course recommendations.
