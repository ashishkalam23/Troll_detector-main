# source/main.py

import pandas as pd
from data_preprocessing import load_data, preprocess_dataframe
from keyphrase_extraction import extract_from_dataframe
from paraphrase_detection import load_model, compute_similarity
from repetition_analysis import compute_repetition_statistics
import os
import numpy as np


# OLD
def main():
    # print(os.listdir(os.path.abspath("data")))
    twitter_df = load_data("data/twitter.csv")
    twitter_df = preprocess_dataframe(twitter_df, text_column='tweet')

    news_df = pd.read_csv("data/news.csv")
    news_df = extract_from_dataframe(news_df, text_column='article_text', top_n=5)

    model = load_model()

    similarities = []
    similarity_scores = []

    for i, keyphrase_list in enumerate(news_df['keyphrases']):
        for keyphrase in keyphrase_list:
            for j, tweet in enumerate(twitter_df['tweet']):
                similarity_score = compute_similarity(keyphrase, tweet, model)
                similarities.append((i, j, keyphrase, tweet, similarity_score))
                similarity_scores.append(similarity_score)
                #print(f"\nComparing:\n  - Keyphrase: \"{keyphrase}\"\n  - Tweet: \"{tweet}\"\n  → Similarity Score: {similarity_score:.4f}")

    avg_similarity = np.mean(similarity_scores) if similarity_scores else 0

    #print("\nComputed Similarities:")
    #for i, j, keyphrase, tweet, score in similarities:
        #print(f"News {i} & Tweet {j} - Score: {score:.4f}\n  \"{keyphrase}\"\n  \"{tweet}\"\n")
    print(f"\nAverage Similarity Score Across All Tweet to News Keyphrase Comparisons: {avg_similarity:.4f}")

    """
    sample_news_phrase = news_df['keyphrases'].iloc[0][0]  # first keyphrase from first news article
    sample_tweet = twitter_df['tweet'].iloc[0]
    similarity = compute_similarity(sample_news_phrase, sample_tweet, model)
    print("Similarity between sample news phrase and tweet:", similarity)"
    """

    tweets_list = twitter_df['tweet'].tolist()[:100]
    similar_tweets = compute_repetition_statistics(tweets_list, model, threshold=0.85)
    print("Number of similar tweet pairs:", len(similar_tweets))

if __name__ == "__main__":
    main()
