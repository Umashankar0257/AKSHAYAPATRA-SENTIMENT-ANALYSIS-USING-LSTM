import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load cleaned data
df = pd.read_csv("../../data/processed/cleaned_data.csv")

# Features and labels
X = df['clean_text']
y = df['label']

# Convert labels to numbers
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
y = y.map(label_map)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding
X_train_pad = pad_sequences(X_train_seq, maxlen=100)
X_test_pad = pad_sequences(X_test_seq, maxlen=100)

# Save tokenizer
with open("../../models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Save processed data
import numpy as np

np.save("../../data/processed/X_train.npy", X_train_pad)
np.save("../../data/processed/X_test.npy", X_test_pad)
np.save("../../data/processed/y_train.npy", y_train)
np.save("../../data/processed/y_test.npy", y_test)

print("Tokenization complete!")
print("Shape:", X_train_pad.shape)