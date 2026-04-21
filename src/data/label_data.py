import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

# Load data
df = pd.read_csv("../../data/raw/akshayapatra_50000_plus_realistic_feedback_dataset.csv")

# Drop old label


sia = SentimentIntensityAnalyzer()

def get_label(text):
    score = sia.polarity_scores(str(text))['compound']
    
    if score >= 0.2:
        return "positive"
    elif score <= -0.2:
        return "negative"
    else:
        return "neutral"

# Apply
df['label'] = df['feedback_text'].apply(get_label)

# Save
df.to_csv("../../data/processed/labeled_data.csv", index=False)

print(df[['feedback_text','label']].head())
print(df['label'].value_counts())