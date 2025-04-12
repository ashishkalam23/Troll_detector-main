# source/main.py

import os
from pipeline import run_pipeline, archive_processed_files

def ensure_directories():
    """Ensure data/processed directory exists."""
    os.makedirs("data/processed", exist_ok=True)

if __name__ == "__main__":
    print("Starting Main")

    """
    run_pipeline("candidate.csv,
    reference.csv, 
    candidate_col, 
    reference_col, 
    candidate_directory, 
    reference, 
    directory, 
    similarity threshold, 
    repetition threshold)
    """

    # TRIVIAL
    run_pipeline("twitter", "reddit", "tweet", "title", "trivial", "trivial", 0.8, 0.8)
    archive_processed_files("trivial")

    # REDDIT1
    run_pipeline("r_comments", "r_articles", "body", "title", "reddit", "reddit", 0.8, 0.8)
    archive_processed_files("R1_posts_vs_R1_comments")
    run_pipeline("r_articles", "r_articles", "title", "title", "reddit", "reddit", 0.8, 0.8)
    archive_processed_files("R1_posts_vs_R1_posts")
    run_pipeline("r_comments", "r_comments", "body", "body", "reddit", "reddit", 0.8, 0.8)
    archive_processed_files("R1_comments_vs_R1_comments")

    # REDDIT2
    run_pipeline("r_pol_comments", "r_pol_articles", "body", "title", "reddit2", "reddit2", 0.8, 0.8)
    archive_processed_files("R2_posts_vs_R2_comments")
    run_pipeline("r_pol_articles", "r_pol_articles", "title", "title", "reddit2", "reddit2", 0.8, 0.8)
    archive_processed_files("R2_posts_vs_R2_posts")
    run_pipeline("r_pol_comments", "r_pol_comments", "body", "body", "reddit2", "reddit2", 0.8, 0.8)
    archive_processed_files("R2_comments_vs_R2_comments")
    
    # REDDIT1 x REDDIT2
    run_pipeline("r_pol_articles", "r_articles", "title", "title", "reddit2", "reddit", 0.8, 0.8)
    archive_processed_files("R1_posts_vs_R2_posts")
    run_pipeline("r_pol_comments", "r_articles", "body", "title", "reddit2", "reddit", 0.8, 0.8)
    archive_processed_files("R1_posts_vs_R2_comments")
    run_pipeline("r_comments", "r_pol_articles", "body", "title", "reddit", "reddit2", 0.8, 0.8)
    archive_processed_files("R2_posts_vs_R1_comments")
    run_pipeline("r_pol_comments", "r_comments", "body", "body", "reddit2", "reddit", 0.8, 0.8)
    archive_processed_files("R1_comments_vs_R2_comments")

    # REDDIT1 x TWITTER
    run_pipeline("bots_twitter", "r_articles", "Tweet", "title", "twitter", "reddit", 0.8, 0.8)
    archive_processed_files("R1_posts_vs_botTweets")
    run_pipeline("humans_twitter", "r_articles", "Tweet", "title", "twitter", "reddit", 0.8, 0.8)
    archive_processed_files("R1_posts_vs_humanTweets")
    run_pipeline("bots_twitter", "r_comments", "Tweet", "body", "twitter", "reddit", 0.8, 0.8)
    archive_processed_files("R1_comments_vs_botTweets")
    run_pipeline("humans_twitter", "r_comments", "Tweet", "body", "twitter", "reddit", 0.8, 0.8)
    archive_processed_files("R1_comments_vs_humanTweets")

    # REDDIT2 x TWITTER
    run_pipeline("bots_twitter", "r_pol_articles", "Tweet", "title", "twitter", "reddit2", 0.8, 0.8)
    archive_processed_files("R2_posts_vs_botTweets")
    run_pipeline("humans_twitter", "r_pol_articles", "Tweet", "title", "twitter", "reddit2", 0.8, 0.8)
    archive_processed_files("R2_posts_vs_humanTweets")
    run_pipeline("bots_twitter", "r_pol_comments", "Tweet", "body", "twitter", "reddit2", 0.8, 0.8)
    archive_processed_files("R2_comments_vs_botTweets")
    run_pipeline("humans_twitter", "r_pol_comments", "Tweet", "body", "twitter", "reddit2", 0.8, 0.8)
    archive_processed_files("R2_comments_vs_humanTweets")

    # TWITTER BOT
    run_pipeline("bots_twitter", "humans_twitter", "Tweet", "Tweet", "twitter", "twitter", 0.8, 0.8)
    archive_processed_files("humanTweets_vs_botTweets")
    run_pipeline("humans_twitter", "humans_twitter", "Tweet", "Tweet", "twitter", "twitter", 0.8, 0.8)
    archive_processed_files("humanTweets_vs_humanTweets")
    run_pipeline("bots_twitter", "bots_twitter", "Tweet", "Tweet", "twitter", "twitter", 0.8, 0.8)
    archive_processed_files("botTweets_vs_botTweets")
    
    
    
