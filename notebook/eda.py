import pandas as pd

df = pd.read_csv("../data/raw/akshayapatra_50000_plus_realistic_feedback_dataset.csv")
print(df.head())

# /// df.shape()
# df.columns
# df.info()
# df.describe() 
# ///

df['feedback'].head()