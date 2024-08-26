import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import os
import ssl

# Bypass SSL certificate verification (use with caution)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def tokenize_and_lemmatize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords.words('english')]

def preprocess_data(input_file, output_file):
    # Read the data
    df = pd.read_csv(input_file)
    
    # Clean and preprocess the 'description' column
    df['processed_text'] = df['description'].fillna('').apply(clean_text)
    df['processed_text'] = df['processed_text'].apply(tokenize_and_lemmatize)
    
    # Optionally, join tokens back into a single string for each description
    df['processed_text'] = df['processed_text'].apply(lambda x: ' '.join(x))
    
    # Save the preprocessed data
    df.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")

if __name__ == "__main__":
    # Get the absolute path of the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    input_file = os.path.join(project_root, "data", "raw", "course_data.csv")
    output_file = os.path.join(project_root, "data", "processed", "preprocessed_courses.csv")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        print("Please make sure the 'course_data.csv' file is in the correct location.")
    else:
        preprocess_data(input_file, output_file)
