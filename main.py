
import numpy as np
import matplotlib.pyplot as plt

# Load the medical image dataset
data = np.load("data/pneumoniamnist_128.npz")

# Extract training, validation and test data
X_train = data["train_images"]
y_train = data["train_labels"].ravel()

X_val = data["val_images"]
y_val = data["val_labels"].ravel()

X_test = data["test_images"]
y_test = data["test_labels"].ravel()

# Display dataset information
print("Training images:", len(X_train))
print("Validation images:", len(X_val))
print("Test images:", len(X_test))
print("Image shape:", X_train.shape[1:])

# Display sample images from both classes
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for row in range(2):
    indices = np.where(y_train == row)[0][:5]

    for col, idx in enumerate(indices):
        
        axes[row, col].imshow(
            X_train[idx],
            cmap="gray",
            interpolation="nearest",
            vmin=0,
            vmax=255
)
        axes[row, col].set_title(
            "Normal" if row == 0 else "Pneumonia"
        )
        axes[row, col].axis("off")

plt.suptitle("PneumoniaMNIST - Sample Medical Images")
plt.tight_layout()
plt.show()
