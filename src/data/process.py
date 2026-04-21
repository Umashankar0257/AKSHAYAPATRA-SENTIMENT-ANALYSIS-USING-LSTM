import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Load labeled data
df = pd.read_csv("../../data/processed/labeled_data.csv")

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text)
    
    # lowercase
    text = text.lower()
    
    # remove punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # keep negations (IMPORTANT)
    words = text.split()
    words = [word for word in words if word not in stop_words or word in ['not', 'no']]
    
    return " ".join(words)

# Apply
df['clean_text'] = df['feedback_text'].apply(clean_text)

# Save
df.to_csv("../../data/processed/cleaned_data.csv", index=False)

#print(df[['feedback_text','clean_text']].head())

print(df[['clean_text','label']].head())