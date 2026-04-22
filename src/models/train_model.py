import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load processed data
X_train = np.load("../../data/processed/X_train.npy")
X_test = np.load("../../data/processed/X_test.npy")
y_train = np.load("../../data/processed/y_train.npy")
y_test = np.load("../../data/processed/y_test.npy")

# Convert labels to categorical
y_train = to_categorical(y_train, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# Build model
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=100),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# Train
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)

print(f"Test Accuracy: {accuracy*100:.2f}%")

# Save model
model.save("../../models/lstm_sentiment_model.h5")

print("Model saved successfully!")