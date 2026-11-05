
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load previously extracted features
data = np.load("results/extracted_features.npz")

X_train = data["X_train"]
X_val = data["X_val"]
X_test = data["X_test"]

y_train = data["y_train"]
y_val = data["y_val"]
y_test = data["y_test"]

feature_names = data["feature_names"]

# Check for invalid values
for name, X in [
    ("Train", X_train),
    ("Validation", X_val),
    ("Test", X_test)
]:
    if not np.isfinite(X).all():
        raise ValueError(
            f"Invalid values found in {name}"
        )

# Fit scaler using training data only
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Save scaled features
np.savez_compressed(
    "results/scaled_features.npz",
    X_train=X_train_scaled,
    X_val=X_val_scaled,
    X_test=X_test_scaled,
    y_train=y_train,
    y_val=y_val,
    y_test=y_test,
    feature_names=feature_names
)

# Display results
print("Feature Scaling Completed!")
print("---------------------------")

print("Training:", X_train_scaled.shape)
print("Validation:", X_val_scaled.shape)
print("Test:", X_test_scaled.shape)

print("\nBefore Scaling:")
print(X_train[0])

print("\nAfter Scaling:")
print(X_train_scaled[0])

print("\nTraining Feature Means:")
print(np.round(X_train_scaled.mean(axis=0), 4))

print("\nTraining Feature Standard Deviations:")
print(np.round(X_train_scaled.std(axis=0), 4))

print("\nSaved: results/scaled_features.npz")
