import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter

# Load training data
X_train = np.load("../../data/processed/X_train.npy")
y_train = np.load("../../data/processed/y_train.npy")

print("Before balancing:")
print(Counter(y_train))

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("\nAfter balancing:")
print(Counter(y_train_balanced))

# Save balanced data
np.save("../../data/processed/X_train_balanced.npy", X_train_balanced)
np.save("../../data/processed/y_train_balanced.npy", y_train_balanced)

print("Balanced dataset saved successfully!")