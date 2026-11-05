
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import pairwise_distances

# Load scaled features
data = np.load("results/scaled_features.npz")

X_train = data["X_train"]
X_test = data["X_test"]

y_train = data["y_train"].ravel()
y_test = data["y_test"].ravel()

# Load quantum selected features
quantum = np.load(
    "results/quantum_selected_features.npz"
)

selected_indices = quantum["selected_indices"].astype(int)

# Load original medical images
images = np.load("data/pneumoniamnist_128.npz")

train_images = images["train_images"]
test_images = images["test_images"]

# Verify image and feature alignment
assert np.array_equal(
    y_train, images["train_labels"].ravel()
)

assert np.array_equal(
    y_test, images["test_labels"].ravel()
)

# Apply QAOA-selected features
gallery = X_train[:, selected_indices]
queries = X_test[:, selected_indices]

class_names = {
    0: "Normal",
    1: "Pneumonia"
}

fig, axes = plt.subplots(
    2, 6, figsize=(17, 7)
)

# Test one query from each class
for row, target_class in enumerate([0, 1]):

    query_index = np.where(
        y_test == target_class
    )[0][0]

    query_features = queries[query_index:query_index + 1]

    # Calculate Euclidean distances
    distances = pairwise_distances(
        query_features,
        gallery,
        metric="euclidean"
    )[0]

    # Find 5 most similar images
    top_indices = np.argsort(distances)[:5]

    # Display query image
    axes[row, 0].imshow(
        test_images[query_index],
        cmap="gray"
    )

    axes[row, 0].set_title(
        f"QUERY\n{class_names[target_class]}",
        fontweight="bold"
    )

    axes[row, 0].axis("off")

    correct = 0

    # Display retrieved images
    for j, index in enumerate(top_indices):

        retrieved_class = int(y_train[index])

        if retrieved_class == target_class:
            correct += 1

        axes[row, j + 1].imshow(
            train_images[index],
            cmap="gray"
        )

        axes[row, j + 1].set_title(
            f"Rank {j + 1}\n"
            f"{class_names[retrieved_class]}\n"
            f"Distance: {distances[index]:.3f}",
            fontsize=10
        )

        axes[row, j + 1].axis("off")

    print(
        f"{class_names[target_class]} Query "
        f"Precision@5: {correct / 5:.2f}"
    )

plt.suptitle(
    "Quantum-Based Medical CBIR - QAOA + Euclidean",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

# Save retrieval visualization
os.makedirs("results", exist_ok=True)

plt.savefig(
    "results/cbir_qaoa_top5.png",
    dpi=200,
    bbox_inches="tight"
)

print("\nVisualization saved successfully!")
print("Saved: results/cbir_qaoa_top5.png")

plt.show()
