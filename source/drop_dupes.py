import pandas as pd
path = "path.csv"
df = pd.read_csv(path)
df = df.drop_duplicates(subset=['body'])
df.to_csv('deduped.csv', index=False)

