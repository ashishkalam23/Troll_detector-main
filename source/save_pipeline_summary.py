import numpy as np
import pandas as pd

def save_pipeline_summary(
    candidate_name,
    reference_name,
    candidate_col,
    reference_col,
    matches_df,
    candidates_df,
    references_df,
    similar_candidates,
    similar_references,
    match_repetitions,
    match_ref_reps,
    paraphrase_threshold,
    repetition_threshold
):
    """
    Save the pipeline run summary to a text file.
    Now tracks full set sizes, percentages, and includes Light + Detailed sections.
    """

    summary_path = f"data/processed/summary_{candidate_name}_{reference_name}.txt"
    output_path = f"output/summary_{candidate_name}_{reference_name}.txt"

    ### --- BASE NUMBERS ---
    # Candidates
    total_candidates = candidates_df[candidate_col].dropna().str.strip().nunique()

    # References
    total_references = references_df[reference_col].dropna().str.strip().nunique()

    # Matches
    total_matches = len(matches_df)
    average_similarity = matches_df['similarity'].mean() if not matches_df.empty else 0.0

    # Unique matched candidates
    unique_matched_candidates = matches_df[candidate_col].dropna().str.strip().nunique()

    # Unique matched references
    unique_matched_references = matches_df[reference_col].dropna().str.strip().nunique()

    ### --- REPETITION ANALYSIS ---
    # Candidates internal repetition
    repetitive_texts = pd.concat([
        pd.Series([row[3] for row in similar_candidates]),  # text1
        pd.Series([row[4] for row in similar_candidates])   # text2
    ]).dropna().str.strip().unique()
    num_repetitive_candidates = len(repetitive_texts)

    # References internal repetition
    repetitive_reference_texts = pd.concat([
        pd.Series([row[3] for row in similar_references]),  # text1
        pd.Series([row[4] for row in similar_references])   # text2
    ]).dropna().str.strip().unique()
    num_repetitive_references = len(repetitive_reference_texts)

    ### --- REPETITIVE MATCHES ---
    # Candidates that matched references
    matched_candidates = matches_df[candidate_col].dropna().str.strip().unique()

    # Repetitive candidates that matched
    repetitive_candidates_matched = set(repetitive_texts) & set(matched_candidates)
    num_repetitive_candidates_matched = len(repetitive_candidates_matched)

    # Titles matched by a repetitive candidate
    repetitive_matches_df = matches_df[matches_df[candidate_col].isin(repetitive_candidates_matched)]
    titles_with_repetitive_matches = repetitive_matches_df[reference_col].dropna().str.strip().unique()
    num_titles_with_repetitive_matches = len(titles_with_repetitive_matches)

    # Match repetition analysis
    num_repetitive_candidate_matches = len(match_repetitions)
    num_repetitive_reference_matches = len(match_ref_reps)

    ### --- PERCENTAGES ---
    percent_candidates_repetitive = (num_repetitive_candidates / total_candidates * 100) if total_candidates > 0 else 0
    percent_references_repetitive = (num_repetitive_references / total_references * 100) if total_references > 0 else 0
    percent_candidates_matched = (unique_matched_candidates / total_candidates * 100) if total_candidates > 0 else 0
    percent_references_matched = (unique_matched_references / total_references * 100) if total_references > 0 else 0
    percent_repetitive_candidates_matched = (num_repetitive_candidates_matched / num_repetitive_candidates * 100) if num_repetitive_candidates > 0 else 0
    percent_titles_with_repetitive_paraphrases = (num_titles_with_repetitive_matches / unique_matched_references * 100) if unique_matched_references > 0 else 0

    ### --- BUILD SUMMARY ---
    summary_text = f"""Pipeline Summary
=================

Summary
-----------------
Candidate Dataset: {candidate_name} ({candidate_col})
Reference Dataset: {reference_name} ({reference_col})

Paraphrase Similarity (Threshold: {paraphrase_threshold})
Total Matches Found: {total_matches}
Average Match Similarity: {average_similarity:.2f}

Candidates:
- Total Candidates: {total_candidates}
- Candidates Matched to References: {unique_matched_candidates} ({percent_candidates_matched:.1f}%)
- Repetitive Candidates: {num_repetitive_candidates} ({percent_candidates_repetitive:.1f}%)

References:
- Total References: {total_references}
- References Matched by Candidates: {unique_matched_references} ({percent_references_matched:.1f}%)
- Repetitive References: {num_repetitive_references} ({percent_references_repetitive:.1f}%)

Cross-Matching Paraphrase Repetition Analysis (Threshold {repetition_threshold}):
- Repetitive Candidates that Matched References: {num_repetitive_candidates_matched} ({percent_repetitive_candidates_matched:.1f}% of repetitive candidates)
- Titles Matched by Repetitive Candidates: {num_titles_with_repetitive_matches} ({percent_titles_with_repetitive_paraphrases:.1f}% of matched titles)


Detailed Metrics
--------------------
Internal Repetitions:
- Candidate Internal Repetitions (pairs): {len(similar_candidates)}
- Reference Internal Repetitions (pairs): {len(similar_references)}

Repetitive Match Statistics:
- Repetitive Candidate Matches (after matching): {num_repetitive_candidate_matches}
- Repetitive Reference Matches (after matching): {num_repetitive_reference_matches}
"""

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(summary_text)
    print(f"Summary written to: {summary_path}, {output_path}\n")
