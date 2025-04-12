# source/data_preprocessing.py
# Module 1
import pandas as pd
import re
import string
import contractions

def load_data(file_path, chunk_size=50000):
    """Load dataset in chunks for large CSV processing."""
    return pd.read_csv(file_path, chunksize=chunk_size)

def clean_text(text):
    """Lowercase, expand contractions, remove punctuation, and trim spaces."""
    if pd.isna(text):
        return ""

    text = text.lower()
    text = contractions.fix(text)
    text = re.sub(r'\s+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

def preprocess_dataframe(df, text_column='text', max_length=150):
    """Apply text cleaning and filter long comments."""
    df[text_column] = df[text_column].astype(str).apply(clean_text)
    return df[df[text_column].str.len() <= max_length]

def process_large_csv(input_path, output_path, text_column):
    """Process large CSV files in chunks."""
    chunks = []
    for chunk in load_data(input_path):
        processed_chunk = preprocess_dataframe(chunk, text_column)
        chunks.append(processed_chunk)

    final_df = pd.concat(chunks, ignore_index=True)
    final_df.to_csv(output_path, index=False)
    return final_df

if __name__ == "__main__":
    process_large_csv("data/twitter.csv", "data/processed/twitter_clean.csv", text_column="tweet")
    process_large_csv("data/reddit.csv", "data/processed/reddit_clean.csv", text_column="title")


