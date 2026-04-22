import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load
model = load_model("../../models/lstm_sentiment_model.h5")

X_test = np.load("../../data/processed/X_test.npy")
y_test = np.load("../../data/processed/y_test.npy")

# Predict
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print(confusion_matrix(y_test, y_pred_classes))
print(classification_report(y_test, y_pred_classes))