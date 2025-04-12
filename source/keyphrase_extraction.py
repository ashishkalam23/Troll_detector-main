# source/keyphrase_extraction.py
# Module 2

from keybert import KeyBERT
import pandas as pd
import swifter

def extract_keyphrases(text, model, top_n=5):
    """Extract key phrases using KeyBERT."""
    keyphrases = model.extract_keywords(text, top_n=top_n, stop_words='english')
    return [phrase for phrase, score in keyphrases]

def extract_from_dataframe(df, text_column='text', model=None, top_n=5):
    """Extract keyphrases for every row in the DataFrame efficiently."""
    assert model is not None, "Pass a pre-loaded KeyBERT model"
    
    model = KeyBERT(model)
    try:
        import swifter
        df['keyphrases'] = df[text_column].swifter.apply(lambda x: extract_keyphrases(x, model, top_n=top_n))
    except ImportError:
        df['keyphrases'] = df[text_column].apply(lambda x: extract_keyphrases(x, model, top_n=top_n))
    
    return df

# Unused
def process_keyphrases(input_path, output_path, text_column):
    """Apply keyphrase extraction on large CSV files."""
    df = pd.read_csv(input_path)
    df = extract_from_dataframe(df, text_column)
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    process_keyphrases("data/processed/twitter_clean.csv", "data/processed/twitter_keyphrases.csv", text_column="tweet")
    process_keyphrases("data/processed/reddit_clean.csv", "data/processed/reddit_keyphrases.csv", text_column="title")

