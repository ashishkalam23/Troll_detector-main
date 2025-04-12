# source/pipeline.py

import os
from keybert import KeyBERT
import pandas as pd
from sentence_transformers import SentenceTransformer
from data_preprocessing import preprocess_dataframe
from keyphrase_extraction import extract_from_dataframe
from paraphrase_detection import compute_similarity_fast, load_model
from repetition_analysis import compute_repetition_statistics, get_match_repetitions
from save_pipeline_summary import save_pipeline_summary

def ensure_directories():
    """Ensure data/processed directory exists."""
    os.makedirs("data/processed", exist_ok=True)

import shutil

def archive_processed_files(archive_name):
    """
    Move all files from data/processed/ into data/archives/{archive_name}/.
    
    :param archive_name: Name of the archive folder to create inside data/archives
    """
    processed_dir = "data/processed"
    archive_dir = f"data/archives/{archive_name}"

    os.makedirs(archive_dir, exist_ok=True)

    for filename in os.listdir(processed_dir):
        file_path = os.path.join(processed_dir, filename)
        if os.path.isfile(file_path):
            shutil.move(file_path, os.path.join(archive_dir, filename))

    print(f"Archived all processed files to: {archive_dir}\n")

def run_pipeline(
    CANDIDATE,
    REFERENCE,
    CANDIDATE_COL,
    REFERENCE_COL,
    CANDIDATE_DIRECTORY,
    REFERENCE_DIRECTORY,
    PARAPHRASE_SIMILARITY_THRESHOLD=0.80,
    REPETITION_THRESHOLD=0.80
):
    print("Pipeline Input:")
    print(f"Candidate: {CANDIDATE}, Reference: {REFERENCE}\n")
    print(f"Cand column: {CANDIDATE_COL}, Ref column: {REFERENCE_COL}")
    print(f"Cand Directory: {CANDIDATE_DIRECTORY}, Ref Directory: {REFERENCE_DIRECTORY}")
    print(f"Paraphrase, Repetition Thresholds: {PARAPHRASE_SIMILARITY_THRESHOLD},{REPETITION_THRESHOLD}")
    print("Pipeline Starting...\n")
    ensure_directories()

    model = load_model("all-MiniLM-L6-v2")

    print("Loading reference (REF) and candidate (CAND) CSVs...")
    candidate_df = pd.read_csv(f"data/{CANDIDATE_DIRECTORY}/{CANDIDATE}.csv")
    reference_df = pd.read_csv(f"data/{REFERENCE_DIRECTORY}/{REFERENCE}.csv")
    if CANDIDATE_COL == REFERENCE_COL:
        print(f"Candidate and Reference columns are the same ({CANDIDATE_COL}). Renaming candidate column to avoid conflict.")
        new_candidate_col = CANDIDATE_COL + "2"
        candidate_df = candidate_df.rename(columns={CANDIDATE_COL: new_candidate_col})
        CANDIDATE_COL = new_candidate_col
    print("CSV Loading Complete\n")

    print("Cleaning and Preprocessing Text...") 
    # Example) REF: "New Government Policy Announced" CLEANS --> new government policy announced
    clean_candidate_df = preprocess_dataframe(candidate_df, text_column=CANDIDATE_COL)
    clean_reference_df = preprocess_dataframe(reference_df, text_column=REFERENCE_COL)
    print("Preprocessing/Cleaning Complete\n")

    print("Keyphrase Extraction Starting...")
    # Example) REF: new government policy announced EXTRACTS --> ['policy', 'government', 'new', 'announced']
    keyphrased_candidate_df = extract_from_dataframe(clean_candidate_df, text_column=CANDIDATE_COL, model=model)
    keyphrased_reference_df = extract_from_dataframe(clean_reference_df, text_column=REFERENCE_COL, model=model)

    keyphrased_candidate_df.to_csv(f"data/processed/keyphrases_{CANDIDATE}.csv", index=False)
    keyphrased_reference_df.to_csv(f"data/processed/keyphrases_{REFERENCE}.csv", index=False)
    print("Keyphrase Extraction Complete.\n")

    print("Matching Texts Via Keyphrase Similarity...")
    # Example)  REF: ['policy','government','new','announced'] SIMILAR TO CAND: ['policy','news','government','today','announced'] 
    #       --> REF: new government policy announced, CAND: the government announced a new policy today policy news
    matches = compute_similarity_fast(
        keyphrased_candidate_df,
        keyphrased_reference_df,
        model,
        text_column=CANDIDATE_COL,
        ref_column=REFERENCE_COL,
        keyphrase_column="keyphrases",
        threshold=PARAPHRASE_SIMILARITY_THRESHOLD,
        batch_size=32
    )
    matches_df = pd.DataFrame(matches, columns=[REFERENCE_COL, CANDIDATE_COL, "similarity"])
    matches_df.to_csv(f"data/processed/matches_{REFERENCE}_{REFERENCE_COL}_{CANDIDATE}_{CANDIDATE_COL}.csv", index=False)

    print(f"Similarity Matching Complete")
    #print(f"Threshold: {PARAPHRASE_SIMILARITY_THRESHOLD}, Average: {np.mean(matches_df['similarity']):.2f}")
    #print(f"Matches: {len(matches)}\n")

    print("Analyzing Text Repetitiveness...")
    # Example) CAND: governments new economic policy is already sparking discussions economy policy REPLICATED BY
    #      --> CAND: governments new economic policy is already sparking discussions economy, CAND: new economic policy is sparking discussions economy)
    similar_candidates, avg_candidate_sim = compute_repetition_statistics(
        keyphrased_candidate_df[CANDIDATE_COL].tolist(), model, threshold=REPETITION_THRESHOLD
    )
    similar_references, avg_reference_sim = compute_repetition_statistics(
        keyphrased_reference_df[REFERENCE_COL].tolist(), model, threshold=REPETITION_THRESHOLD
    )

    similar_candidates_df = pd.DataFrame(similar_candidates, columns=["index1", "index2", "similarity", CANDIDATE_COL, "duplicate"])
    similar_references_df = pd.DataFrame(similar_references, columns=["index1", "index2", "similarity", REFERENCE_COL, "duplicate"])

    similar_candidates_df.to_csv(f"data/processed/repetitive_{CANDIDATE_COL}.csv", index=False)
    similar_references_df.to_csv(f"data/processed/repetitive_{REFERENCE_COL}.csv", index=False)

    print(f"Repetition Analysis Complete")
    #print(f"Candidate: {len(similar_candidates)} bot-like pairs found")
    #print(f"Reference: {len(similar_references)} bot-like pairs found\n")
    
    print("Analyzing Paraphrased Text Repetitiveness...")
    # Example) REF: new government policy announced SIMILAR TO
    #     --> CAND: governments new economic policy is already sparking discussions economy policy 
    #     --> 2 REPETITIONS, SEE PREVIOUS STEP EXAMPLE
    match_repetitions = get_match_repetitions(matches_df, similar_candidates_df, CANDIDATE_COL, "duplicate")
    match_ref_reps = get_match_repetitions(matches_df, similar_references_df, REFERENCE_COL, "duplicate")

    pd.DataFrame(match_repetitions, columns=[f"paraphrased_{CANDIDATE_COL}", "repetitions"]).to_csv(
        f"data/processed/repetitive_matches_{CANDIDATE}_{CANDIDATE_COL}.csv", index=False
    )
    pd.DataFrame(match_ref_reps, columns=[f"paraphrased_{REFERENCE_COL}", "repetitions"]).to_csv(
        f"data/processed/repetitive_matches_{REFERENCE}_{REFERENCE_COL}.csv", index=False
    )

    print(f"Repetition Analysis of Matched Paraphrases Complete")
    #print(f"Candidate: {len(match_repetitions)} bot-like repetitions")
    #print(f"Reference: {len(match_ref_reps)} bot-like repetitions\n")

    print(f"Writing pipeline summary to txt...")
    save_pipeline_summary(CANDIDATE, 
                          REFERENCE, 
                          CANDIDATE_COL, 
                          REFERENCE_COL, 
                          matches_df,
                          clean_candidate_df,
                          clean_reference_df,
                          similar_candidates, 
                          similar_references,
                          match_repetitions,
                          match_ref_reps,
                          PARAPHRASE_SIMILARITY_THRESHOLD,
                          REPETITION_THRESHOLD)
    print(f"Pipeline summary written.\n")
    print("Pipeline Completed Successfully!")

if __name__ == "__main__":
    run_pipeline(
        CANDIDATE="twitter",
        REFERENCE="reddit",
        CANDIDATE_COL="tweet",
        REFERENCE_COL="title",
        CANDIDATE_DIRECTORY="trivial",
        REFERENCE_DIRECTORY="trivial",
        PARAPHRASE_SIMILARITY_THRESHOLD=0.80,
        REPETITION_THRESHOLD=0.80
    )
