import os
import pandas as pd
import numpy as np
from gensim import corpora
from gensim.models import LdaSeqModel, LdaModel, CoherenceModel
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
from matplotlib.dates import DateFormatter
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import time
from openai import OpenAI

# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# OpenAI API setup
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Ensure this environment variable is set

# Function to safely evaluate or split the text
def safely_evaluate(text):
    try:
        return eval(text) if isinstance(text, str) else []
    except (SyntaxError, NameError):
        return text.split() if isinstance(text, str) else []

# Load preprocessed data
def load_preprocessed_data(file_path):
    df = pd.read_csv(file_path)
    df['processed_text'] = df['processed_text'].apply(safely_evaluate)
    df['last_updated'] = pd.to_datetime(df['last_updated'], format='mixed', dayfirst=True)
    return df

# Create corpus and dictionary
def create_corpus_and_dictionary(texts):
    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=2, no_above=0.9)
    corpus = [dictionary.doc2bow(text) for text in texts]
    return corpus, dictionary

# Train DTM model
def train_dtm_lda_model(corpus, dictionary, time_slices, num_topics):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = LdaSeqModel(
            corpus=corpus,
            id2word=dictionary,
            time_slice=time_slices,
            num_topics=num_topics,
            passes=10,
            random_state=42,
            em_min_iter=6,
            em_max_iter=20,
            chunksize=100
        )
        if len(w) > 0:
            print("Warnings during model training:")
            for warning in w:
                print(warning.message)
    return model

# Get topic details from the model
def get_topic_details(model, dictionary, num_words=10):
    topics = []
    for i in range(model.num_topics):
        topic = model.print_topic(topic=i, time=-1)
        top_words = sorted(topic, key=lambda x: x[1], reverse=True)[:num_words]
        topics.append((i, top_words))
    return topics

# AI-based topic interpretation
def ai_interpret_topic(topic_words):
    prompt = f"Given the following top words for a topic: {', '.join([word for word, _ in topic_words])}, provide a concise name for this topic that's no more than 3-4 words long. The name should be specific and relate to a field of study or area of interest. Format your response as 'Name: [topic name]'"

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that interprets topic models."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error in AI interpretation: {e}")
        return f"Topic: {', '.join([word for word, _ in topic_words[:3]])}"

# Interpret topics with AI
def interpret_topics_with_ai(topics):
    interpretations = []
    for topic_id, word_prob in topics:
        print(f"\nTopic {topic_id}:")
        print("Top 10 words:")
        for word, prob in word_prob:
            print(f"  {word}: {prob:.4f}")

        # Get AI interpretation
        ai_interpretation = ai_interpret_topic(word_prob)
        print("\nAI Interpretation:")
        print(ai_interpretation)

        # Ask for user confirmation or modification
        user_choice = input("Accept AI interpretation? (y/n): ").lower()
        if user_choice == 'y':
            if ai_interpretation.startswith("Name: "):
                name = ai_interpretation.split("Name: ", 1)[1]
            else:
                name = ai_interpretation
            description = f"Top words: {', '.join([word for word, _ in word_prob[:5]])}"
        else:
            name = input("Enter a name for this topic: ")
            description = input("Enter a brief description of this topic: ")

        interpretations.append((topic_id, name, description))
        time.sleep(1)  # To avoid hitting API rate limits

    return interpretations

# Plot topic trends over time
def plot_topic_trends(model, time_slices, topic_labels, dates):
    plt.figure(figsize=(15, 10))
    for i in range(model.num_topics):
        topic_trends = [model.print_topic(topic=i, time=j)[0][1] for j in range(len(time_slices))]
        plt.plot(dates, topic_trends, label=topic_labels[i])

    plt.xlabel('Time')
    plt.ylabel('Topic Prominence')
    plt.title('Dynamic Topic Model: Topic Trends Over Time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.gcf().autofmt_xdate()  # Rotate and align the tick labels
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    plt.tight_layout()
    return plt

# Plot topic heatmap
def plot_topic_heatmap(model, time_slices, topic_labels, dates):
    topic_trends = np.array([[model.print_topic(topic=i, time=j)[0][1] 
                              for j in range(len(time_slices))] 
                             for i in range(model.num_topics)])

    plt.figure(figsize=(15, 10))
    sns.heatmap(topic_trends, annot=False, cmap='YlOrRd', 
                xticklabels=[d.strftime('%Y-%m') for d in dates], 
                yticklabels=topic_labels)
    plt.xlabel('Time')
    plt.ylabel('Topics')
    plt.title('Dynamic Topic Model: Topic Evolution Heatmap')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt

# Save topic interpretations
def save_topic_interpretations(interpretations, file_path):
    with open(file_path, 'w') as f:
        for id, name, description in interpretations:
            f.write(f"{id}|{name}|{description}\n")

# Train and visualize LDA model
def train_and_visualize_lda(corpus, dictionary, figure_dir, num_topics=5):
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=15, random_state=42)
    vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis_data, os.path.join(figure_dir, "lda_visualization.html"))
    print("LDA visualization saved as lda_visualization.html")
    return lda_model

# New function to compute coherence values
def compute_coherence_values(dictionary, corpus, texts, start=2, limit=20, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=15, random_state=42)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

# New function to compute perplexity
def compute_perplexity(model, corpus):
    return model.log_perplexity(corpus)

# New function to plot coherence and perplexity scores
def plot_topic_selection_metrics(topic_range, coherence_values, perplexity_values, figure_dir):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel("Number of Topics")
    ax1.set_ylabel("Coherence Score", color="blue")
    ax1.plot(topic_range, coherence_values, color="blue", marker="o")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Perplexity Score", color="red")
    ax2.plot(topic_range, perplexity_values, color="red", marker="s")
    ax2.tick_params(axis="y", labelcolor="red")

    plt.title("Topic Selection Metrics")
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, "topic_selection_metrics.png"))
    plt.close()

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    input_file = os.path.join(project_root, "data", "processed", "preprocessed_courses.csv")
    output_dir = os.path.join(project_root, "results", "models")
    figure_dir = os.path.join(project_root, "results", "figures")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    df = load_preprocessed_data(input_file)
    corpus, dictionary = create_corpus_and_dictionary(df['processed_text'])

    df['month'] = df['last_updated'].dt.to_period('M')
    time_slices = df.groupby('month').size().tolist()

    # Get the start date of each month for x-axis labels
    dates = df.groupby('month').first()['last_updated'].tolist()

    # Topic number selection
    start, limit, step = 2, 20, 3
    model_list, coherence_values = compute_coherence_values(dictionary, corpus, df['processed_text'].tolist(), start, limit, step)
    
    perplexity_values = [compute_perplexity(model, corpus) for model in model_list]
    
    topic_range = range(start, limit, step)
    plot_topic_selection_metrics(topic_range, coherence_values, perplexity_values, figure_dir)

    # Find the optimal number of topics
    optimal_num_topics = topic_range[coherence_values.index(max(coherence_values))]
    print(f"Optimal number of topics based on coherence score: {optimal_num_topics}")

    # Use the optimal number of topics for LDA and DTM
    num_topics = optimal_num_topics

    # Train Dynamic Topic Model (DTM)
    dtm_model = train_dtm_lda_model(corpus, dictionary, time_slices, num_topics)
    dtm_model.save(os.path.join(output_dir, "dtm_lda_model"))
    dictionary.save(os.path.join(output_dir, "dtm_lda_dictionary"))

    # Get detailed topic information
    topic_details = get_topic_details(dtm_model, dictionary, num_words=10)

    # Interpret topics with AI assistance
    topic_interpretations = interpret_topics_with_ai(topic_details)
    interpretations_file = os.path.join(output_dir, "topic_interpretations.txt")
    save_topic_interpretations(topic_interpretations, interpretations_file)

    # Use interpretations for labels
    topic_labels = [f"Topic {id}: {name}" for id, name, _ in topic_interpretations]

    # Plot and save topic trends with interpreted labels
    trends_plt = plot_topic_trends(dtm_model, time_slices, topic_labels, dates)
    trends_plt.savefig(os.path.join(figure_dir, "dtm_lda_topic_trends.png"))

    # Plot and save topic heatmap with interpreted labels
    heatmap_plt = plot_topic_heatmap(dtm_model, time_slices, topic_labels, dates)
    heatmap_plt.savefig(os.path.join(figure_dir, "dtm_lda_topic_heatmap.png"))

    print("Dynamic Topic Model (using LDA techniques) analysis completed. Model and results saved.")

    # Print topic interpretations
    print("\nFinal Topic Interpretations:")
    for id, name, description in topic_interpretations:
        print(f"Topic {id}: {name}")
        print(f"Description: {description}")
        print()

    # Print time slices information
    print("\nTime slices information:")
    for i, (date, count) in enumerate(zip(dates, time_slices)):
        print(f"Time slice {i}: {date.strftime('%Y-%m')}, {count} documents")

    print("\nNote: This analysis uses LDA techniques applied dynamically across time slices.")

    # Train and visualize LDA model for comparison
    lda_model = train_and_visualize_lda(corpus, dictionary, figure_dir, num_topics=num_topics)

    print("LDA visualization completed. Check the 'lda_visualization.html' file.")

if __name__ == "__main__":
    main()