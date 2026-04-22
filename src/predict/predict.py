import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')

# Load model
model = load_model("../../models/lstm_sentiment_model.h5")

# Load tokenizer
with open("../../models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load new feedback file
df = pd.read_csv("../../data/new_feedback.csv")

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    words = text.split()
    words = [word for word in words if word not in stop_words or word in ['not', 'no']]
    
    return " ".join(words)

# preprocess
df["clean_text"] = df["feedback_text"].apply(clean_text)

# tokenize
seq = tokenizer.texts_to_sequences(df["clean_text"])
pad = pad_sequences(seq, maxlen=100)

# predict
predictions = model.predict(pad)
pred_classes = np.argmax(predictions, axis=1)

label_map = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

df["predicted_sentiment"] = [label_map[i] for i in pred_classes]

# save output
df.to_csv("../../data/predictions.csv", index=False)

print(df[["feedback_text", "predicted_sentiment"]].head())
print("Predictions saved successfully!")