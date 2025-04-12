# source/paraphrase_detection.py

from sentence_transformers import SentenceTransformer, util
import pandas as pd

# OLD
def load_model(model_name='all-MiniLM-L6-v2'):
    """Load a pre-trained Sentence-BERT model."""
    model = SentenceTransformer(model_name)
    return model

def compute_similarity(text1, text2, model):
    """Compute cosine similarity between two texts."""
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    cosine_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return cosine_sim

def compute_similarities(df, text_column):
    """Compute similarity scores for all keyphrase-title pairs."""
    model = load_model()
    similarities = []

    for idx, row in df.iterrows():
        for phrase in row['keyphrases']:
            similarity_score = compute_similarity(phrase, row[text_column], model)
            similarities.append((idx, phrase, row[text_column], similarity_score))

    return similarities

if __name__ == "__main__":
    twitter_df = pd.read_csv("data/processed/twitter_keyphrases.csv")
    reddit_df = pd.read_csv("data/processed/reddit_keyphrases.csv")

    twitter_similarities = compute_similarities(twitter_df, text_column="tweet")
    reddit_similarities = compute_similarities(reddit_df, text_column="title")

    print("Twitter Similarity Scores:", twitter_similarities[:5])
    print("Reddit Similarity Scores:", reddit_similarities[:5])
