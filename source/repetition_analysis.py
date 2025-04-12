#source/repetition_analysis.py
# Module 4

import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
from tqdm import tqdm
from collections import Counter
import pandas as pd

from save_pipeline_summary import save_pipeline_summary

def get_embeddings(text_list, model):
    """Convert a list of texts into embeddings."""
    embeddings = model.encode(text_list, convert_to_tensor=True)
    return embeddings

def get_match_repetitions(matches_df, repetitions_df, match_col="text", repetition_col="text"):
    """FAST: Get the number of repetitions for each paraphrased text, using Counter instead of nested loops.
    Now with tqdm progress bar!
    """
    match_repetitions = []

    repetition_texts = repetitions_df[repetition_col].dropna().astype(str).str.strip()
    repetition_counter = Counter(repetition_texts)

    # tqdm over matches
    for _, match in tqdm(matches_df.iterrows(), total=len(matches_df), desc="Counting Repetitions"):
        match_text = match.get(match_col, "")
        if isinstance(match_text, pd.Series):
            match_text = match_text.iloc[0]
        match_text = str(match_text).strip()

        if pd.isna(match_text) or match_text == "":
            continue

        count = repetition_counter.get(match_text, 0)

        if count > 0:
            match_repetitions.append((match_text, count))

    return match_repetitions

from tqdm import tqdm
from sentence_transformers import util

def compute_repetition_statistics(text_list, model, threshold=0.8, top_k=50):
    """
    FASTER: Compute repetition statistics using Top-k semantic search instead of full matrix.
    Returns list of pairs with similarity above threshold.
    """
    embeddings = get_embeddings(text_list, model)

    hits = util.semantic_search(embeddings, embeddings, top_k=top_k)
    
    similar_pairs = []
    similarity_scores = []

    for i, entry in tqdm(enumerate(hits), total=len(hits), desc="Computing Top-k Repetitions"):
        for hit in entry:
            j = hit['corpus_id']
            score = hit['score']

            if i == j:
                continue

            similarity_scores.append(score)

            if score >= threshold:
                similar_pairs.append((i, j, score, text_list[i], text_list[j]))

    return similar_pairs, similarity_scores

if __name__ == "__main__":
    CANDIDATE_COL = "body"
    REFERENCE_COL = "title"
    CANDIDATE = "r_pol_comments"
    REFERENCE = "r_pol_articles"
    keyphrased_candidate_df = pd.read_csv("data/processed/keyphrases_r_pol_comments.csv")
    keyphrased_reference_df = pd.read_csv("data/processed/keyphrases_r_pol_articles.csv")
    matches_df = pd.read_csv("data/processed/matches_r_pol_articles_title_r_pol_comments_body.csv")
    matches = list(matches_df.itertuples(index=False, name=None))
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Analyzing Text Repetitiveness...")
    # Example) CAND: governments new economic policy is already sparking discussions economy policy REPLICATED BY
    #      --> CAND: governments new economic policy is already sparking discussions economy, CAND: new economic policy is sparking discussions economy)
    similar_candidates, avg_candidate_sim = compute_repetition_statistics(
        keyphrased_candidate_df[CANDIDATE_COL].tolist(), model, threshold=0.8
    )
    similar_references, avg_reference_sim = compute_repetition_statistics(
        keyphrased_reference_df[REFERENCE_COL].tolist(), model, threshold=0.8
    )

    similar_candidates_df = pd.DataFrame(similar_candidates, columns=["index1", "index2", "similarity", "text1", "text2"])
    similar_references_df = pd.DataFrame(similar_references, columns=["index1", "index2", "similarity", "text1", "text2"])

    similar_candidates_df.to_csv(f"data/processed/repetitive_{CANDIDATE_COL}.csv", index=False)
    similar_references_df.to_csv(f"data/processed/repetitive_{REFERENCE_COL}.csv", index=False)

    print(f"Repetition Analysis Complete")
    #print(f"Candidate: {len(similar_candidates)} bot-like pairs found")
    #print(f"Reference: {len(similar_references)} bot-like pairs found\n")
    
    print("Analyzing Paraphrased Text Repetitiveness...")
    # Example) REF: new government policy announced SIMILAR TO
    #     --> CAND: governments new economic policy is already sparking discussions economy policy 
    #     --> 2 REPETITIONS, SEE PREVIOUS STEP EXAMPLE
    match_repetitions = get_match_repetitions(matches_df, similar_candidates_df, CANDIDATE_COL, "text1")
    match_ref_reps = get_match_repetitions(matches_df, similar_references_df, REFERENCE_COL, "text1")

    pd.DataFrame(match_repetitions, columns=[f"paraphrased_{CANDIDATE_COL}", "repetitions"]).to_csv(
        f"data/processed/repetitive_matches_{CANDIDATE}_{CANDIDATE_COL}.csv", index=False
    )
    pd.DataFrame(match_ref_reps, columns=[f"paraphrased_{REFERENCE_COL}", "repetitions"]).to_csv(
        f"data/processed/repetitive_matches_{REFERENCE}_{REFERENCE_COL}.csv", index=False
    )
    print(f"Writing pipeline summary to txt...")
    save_pipeline_summary(CANDIDATE, 
                          REFERENCE, 
                          CANDIDATE_COL, 
                          REFERENCE_COL, 
                          matches_df, 
                          matches, 
                          similar_candidates, 
                          similar_references,
                          match_repetitions,
                          match_ref_reps,
                          0.8,
                          0.8)
    print(f"Pipeline summary written.\n")
