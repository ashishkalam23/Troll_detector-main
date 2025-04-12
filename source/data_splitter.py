import pandas as pd

split_col = 'label'
split_val = 1
def split_comments_and_articles(input_csv, output1_csv, output2_csv):

    df = pd.read_csv(input_csv)

    if split_col not in df.columns:
        print("Error: CSV missing split column.")
        return

    comments_df = df[df[split_col] == split_val].reset_index(drop=True)
    articles_df = df[df[split_col] != split_val].reset_index(drop=True)

    comments_df.to_csv(output1_csv, index=False)
    articles_df.to_csv(output2_csv, index=False)

    print(f"Split complete!")

if __name__ == "__main__":
    input_csv = "unsplit.csv"
    output1_csv = "split1.csv"
    output2_csv = "split2.csv"
    split_comments_and_articles(input_csv, output1_csv, output2_csv)
