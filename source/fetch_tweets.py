import tweepy
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import os


# Unfortunately, the twitter API is useless because you can't access tweets that are more than a week old.
load_dotenv()
BEARER_TOKEN = os.getenv("BEARER_TOKEN")
client = tweepy.Client(bearer_token=BEARER_TOKEN)

def fetch_tweet(tweet_id):
    """
    Fetches tweet text given a tweet ID.
    Returns the tweet text or 'Deleted/Suspended' if not found.
    """
    try:
        tweet = client.get_tweet(tweet_id, tweet_fields=["text"])
        return tweet.data["text"] if tweet.data else "Deleted/Suspended"
    except Exception:
        return "Deleted/Suspended"

def update_csv_with_tweets(input_csv, output_csv):
    """
    Reads a CSV, fetches tweets using status_id, and adds a 'tweet' column.
    Saves the updated CSV.
    """
    df = pd.read_csv(input_csv)

    df['status_id'] = df['status_id'].astype(str)

    tqdm.pandas(desc="Fetching Tweets")
    df["tweet"] = df["status_id"].progress_apply(fetch_tweet)

    df.to_csv(output_csv, index=False)
    print(f"\nUpdated CSV saved as: {output_csv}")

if __name__ == "__main__":
    input_csv = "data/tweepFake/ToTweet.csv"
    output_csv = "data/tweepFake/trainWithhTweets.csv"

    update_csv_with_tweets(input_csv, output_csv)
