Project: Tracking Phrase Repetition and Modification in Social Media

This project analyzes how key phrases from news articles or official statements are repeated or modified in social media posts. The goal is to detect potential signs of automated or coordinated behavior (e.g., bots or trolls) by comparing semantic similarity and repetition patterns.

Folder Structure:
project/
├── data/ 
│├── twitter.csv - Pre-collected sample Twitter dataset with tweet_id, user_id, tweet text, timestamp, and hashtags. 
│├── reddit.csv - Pre-collected sample Reddit dataset with post_id, subreddit, title, text, and timestamp. 
│└── news_articles.csv - Pre-collected sample news articles dataset with article_id, source, article text, and publication date. 
├── notebooks/ 
│ ├── exploratory_data_analysis.ipynb - Notebook for initial data exploration and analysis. 
│ └── evaluation_visualizations.ipynb - Notebook to visualize evaluation metrics and model performance. 
├── src/ │ 
├── init.py │ 
├── data_preprocessing.py - Functions to load and clean the data. 
│ ├── keyphrase_extraction.py - Code to extract key phrases using KeyBERT. 
│ ├── paraphrase_detection.py - Code to compute semantic similarity using Sentence-BERT. 
│ ├── repetition_analysis.py - Functions to analyze repetition and similarity between texts. 
│ ├── evaluation.py - Functions for computing evaluation metrics like precision, recall, and F1. 
│ └── main.py - Main script that ties all modules together and runs the complete pipeline. 
├── requirements.txt - List of Python dependencies (e.g., pandas, numpy, keybert, sentence-transformers, scikit-learn). 
└── README.md - This file.

Getting Started:
Install Dependencies: Run: pip install -r requirements.txt

Data Files: Ensure the data/ folder contains the following CSV files:

twitter.csv: Contains sample tweets.
reddit.csv: Contains sample Reddit posts.
news_articles.csv: Contains sample news articles or official statements.
Run the Pipeline: Execute the main script: python src/main.py

Notebooks: Use the notebooks in the notebooks/ folder to explore data and visualize evaluation results.

File Definitions:
data_preprocessing.py: Contains functions to load CSV files, clean text (e.g., remove punctuation, normalize case), and preprocess data for further analysis.

keyphrase_extraction.py: Contains code to extract key phrases from texts using KeyBERT. Outputs a list of key phrases for each document.

paraphrase_detection.py: Uses Sentence-BERT (or a similar model) to generate embeddings for texts and compute cosine similarity to detect paraphrases.

repetition_analysis.py: Analyzes the frequency of phrase repetition and clusters similar phrases based on computed similarity scores.

evaluation.py: Contains functions to evaluate the paraphrase detection component using metrics such as precision, recall, and F1 score.

main.py: Integrates all modules and runs the complete processing pipeline, from data loading to generating output statistics.

Notes:
This project uses pre-collected datasets for demonstration purposes. You can replace these files with your own datasets if needed. The code is modular to allow for easy updates and experimentation with different models or thresholds.
