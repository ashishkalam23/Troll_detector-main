# source/paraphrase_detection.py
# Module 3

from sentence_transformers import SentenceTransformer, util
import pandas as pd
from tqdm import tqdm
import ast
import torch

def load_model(model_name='all-MiniLM-L6-v2'):
    """Load a pre-trained Sentence-BERT model."""
    model = SentenceTransformer(model_name)
    return model

def compute_similarity_fast(
    df1, df2, model,
    text_column="text",
    ref_column="text",
    keyphrase_column="keyphrases",
    threshold=0.85,
    batch_size=32
):
    """
    FAST: Compute similarity between keyphrase sets using matrix operations (no nested loops).
    """
    similarities = []

    df1 = df1[df1[keyphrase_column].map(lambda x: isinstance(x, list) and len(x) > 0)].reset_index(drop=True)
    df2 = df2[df2[keyphrase_column].map(lambda x: isinstance(x, list) and len(x) > 0)].reset_index(drop=True)

    def encode_keyphrases(df):
        all_keyphrases = []
        row_to_keyphrases = []

        for keyphrases in df[keyphrase_column]:
            all_keyphrases.extend(keyphrases)
            row_to_keyphrases.append(len(keyphrases))

        embeddings = model.encode(
            all_keyphrases,
            convert_to_tensor=True,
            batch_size=batch_size,
            show_progress_bar=True
        )

        embeddings_per_row = []
        idx = 0
        for n in row_to_keyphrases:
            row_embeds = embeddings[idx:idx+n]
            mean_embed = torch.mean(row_embeds, dim=0)
            embeddings_per_row.append(mean_embed)
            idx += n
        return torch.stack(embeddings_per_row)

    candidate_embeddings = encode_keyphrases(df1)
    reference_embeddings = encode_keyphrases(df2)

    similarity_matrix = util.pytorch_cos_sim(reference_embeddings, candidate_embeddings)  # Shape (num_references, num_candidates)

    matches = (similarity_matrix >= threshold).nonzero(as_tuple=False)

    for match in matches:
        reference_idx, candidate_idx = match.tolist()
        similarity_score = similarity_matrix[reference_idx, candidate_idx].item()

        reference_text = df2.loc[reference_idx, ref_column]
        candidate_text = df1.loc[candidate_idx, text_column]

        similarities.append((reference_text, candidate_text, similarity_score))

    return similarities

# Unused
def compute_similarity(df1, df2, model, text_column="text", ref_column="text", keyphrase_column="keyphrases", threshold=0.85):
    """
    Compute similarity between entire keyphrase sets for tweets and Reddit titles.
    
    :param df1: First dataset (Twitter)
    :param df2: Second dataset (Reddit)
    :param model: SentenceTransformer model
    :param text_column: Column containing full text (candidate)
    :param ref_column: Column containing full text (reference)
    :param keyphrase_column: Column containing extracted keyphrases
    :param threshold: Minimum similarity score for a match
    :return: List of (reference, candidate, similarity) tuples
    """
    similarities = []
    count = 0
    print("Ref column: " + ref_column)
    print("Cand column: " + text_column)
    for candidate_idx, candidate_row in df1.iterrows():
        print(count)
        count+=1
        candidate_text = candidate_row.get(text_column, None)
        candidate_keyphrases = candidate_row.get(keyphrase_column, [])

        if not candidate_keyphrases:
            continue
        candidate_embeddings = model.encode(candidate_keyphrases, convert_to_tensor=True)
        candidate_vector = torch.mean(candidate_embeddings, dim=0)

        for reference_idx, reference_row in df2.iterrows():
            reference_text = reference_row.get(ref_column, None)
            reference_keyphrases = reference_row.get(keyphrase_column, [])

            if not reference_keyphrases:
                continue
            reference_embeddings = model.encode(reference_keyphrases, convert_to_tensor=True)
            reference_vector = torch.mean(reference_embeddings, dim=0)

            similarity_score = util.pytorch_cos_sim(candidate_vector, reference_vector).item()

            if similarity_score >= threshold:
                similarities.append((candidate_text, reference_text, similarity_score))

    return similarities

import torch
from tqdm import tqdm
from sentence_transformers import util

# Unused
def compute_similarity_fast1(
    df1, df2, model,
    text_column="text",
    ref_column="text",
    keyphrase_column="keyphrases",
    threshold=0.85,
    batch_size=32
):
    """
    Compute similarity between entire keyphrase sets for tweets and Reddit titles.
    Precompute embeddings with individual keyphrase encoding (batched).
    
    :param df1: First dataset (Twitter)
    :param df2: Second dataset (Reddit)
    :param model: SentenceTransformer model
    :param text_column: Column containing full text (candidate)
    :param ref_column: Column containing full text (reference)
    :param keyphrase_column: Column containing extracted keyphrases
    :param threshold: Minimum similarity score for a match
    :param batch_size: Batch size for encoding keyphrases
    :return: List of (candidate, reference, similarity) tuples
    """
    similarities = []
    
    # print("Preparing keyphrases...")
    df1 = df1[df1[keyphrase_column].map(lambda x: isinstance(x, list) and len(x) > 0)].reset_index(drop=True)
    df2 = df2[df2[keyphrase_column].map(lambda x: isinstance(x, list) and len(x) > 0)].reset_index(drop=True)

    def encode_keyphrases(df):
        all_keyphrases = []
        row_to_keyphrases = []

        for keyphrases in df[keyphrase_column]:
            all_keyphrases.extend(keyphrases)
            row_to_keyphrases.append(len(keyphrases))

        embeddings = model.encode(
            all_keyphrases,
            convert_to_tensor=True,
            batch_size=batch_size,
            show_progress_bar=True
        )

        embeddings_per_row = []
        idx = 0
        for n in row_to_keyphrases:
            row_embeds = embeddings[idx:idx+n]
            mean_embed = torch.mean(row_embeds, dim=0)
            embeddings_per_row.append(mean_embed)
            idx += n
        return torch.stack(embeddings_per_row)

    # print("Encoding candidate keyphrases...")
    candidate_embeddings = encode_keyphrases(df1)

    # print("Encoding reference keyphrases...")
    reference_embeddings = encode_keyphrases(df2)

    # print("Computing similarities...")
    for reference_idx in tqdm(range(len(df2)), desc="References"):
        reference_vector = reference_embeddings[reference_idx]
        reference_text = df2.loc[reference_idx, ref_column]

        for candidate_idx in range(len(df1)):
            candidate_vector = candidate_embeddings[candidate_idx]
            candidate_text = df1.loc[candidate_idx, text_column]

            similarity_score = util.pytorch_cos_sim(candidate_vector, reference_vector).item()

            if similarity_score >= threshold:
                similarities.append((reference_text, candidate_text, similarity_score))

    return similarities

if __name__ == "__main__":
    model = load_model("all-MiniLM-L6-v2")

    df1 = pd.read_csv("data/processed/twitter_keyphrases.csv")
    df2 = pd.read_csv("data/processed/reddit_keyphrases.csv")

    df1["keyphrases"] = df1["keyphrases"].apply(ast.literal_eval)
    df2["keyphrases"] = df2["keyphrases"].apply(ast.literal_eval)

    matches = compute_similarity(df1, df2, model, text_column="tweet", keyphrase_column="keyphrases", threshold=0.85)

    matches_df = pd.DataFrame(matches, columns=["tweet", "reddit_title", "similarity"])
    matches_df.to_csv("data/processed/matches.csv", index=False)

    print(f"\nSimilarity Matching Complete: {len(matches)} tweet-reddit keyphrase pairs saved to 'data/processed/matches.csv'.")

    
