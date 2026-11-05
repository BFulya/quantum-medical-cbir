
import os
import numpy as np
import pandas as pd

from sklearn.metrics import pairwise_distances
from sklearn.feature_selection import f_classif

# Load scaled features
data = np.load("results/scaled_features.npz")

X_train = data["X_train"]
X_test = data["X_test"]

y_train = data["y_train"].ravel()
y_test = data["y_test"].ravel()

# Load quantum-selected features
quantum = np.load(
    "results/quantum_selected_features.npz"
)

quantum_indices = quantum["selected_indices"].astype(int)

# Classical feature selection
f_scores, _ = f_classif(X_train, y_train)
f_scores = np.nan_to_num(f_scores)

classical_indices = np.argsort(f_scores)[
    -len(quantum_indices):
]

methods = {
    "All Features": np.arange(X_train.shape[1]),
    "Quantum QAOA": quantum_indices,
    "Classical F-Score": classical_indices
}

metrics = ["euclidean", "manhattan", "cosine"]

class_names = {
    0: "Normal",
    1: "Pneumonia"
}

print("Dataset Class Distribution")
print("--------------------------")

for label, name in class_names.items():
    print(
        f"{name}: "
        f"Gallery={np.sum(y_train == label)}, "
        f"Test={np.sum(y_test == label)}"
    )

results = []

for method_name, indices in methods.items():

    print(f"\nMethod: {method_name}")
    print("-" * 45)

    gallery = X_train[:, indices]
    queries = X_test[:, indices]

    for metric in metrics:

        distances = pairwise_distances(
            queries,
            gallery,
            metric=metric
        )

        top5 = np.argsort(
            distances, axis=1
        )[:, :5]

        retrieved_labels = y_train[top5]

        # Evaluate relevance using test labels
        matches = (
            retrieved_labels == y_test[:, None]
        )

        query_precision = matches.mean(axis=1)

        normal_p5 = query_precision[y_test == 0].mean()
        pneumonia_p5 = query_precision[y_test == 1].mean()

        # Equal importance for both classes
        macro_p5 = (normal_p5 + pneumonia_p5) / 2

        overall_p5 = query_precision.mean()

        results.append({
            "Method": method_name,
            "Distance": metric,
            "Normal_P5": normal_p5,
            "Pneumonia_P5": pneumonia_p5,
            "Macro_P5": macro_p5,
            "Overall_P5": overall_p5
        })

        print(f"\nDistance: {metric}")
        print(f"Normal P@5:    {normal_p5:.4f}")
        print(f"Pneumonia P@5: {pneumonia_p5:.4f}")
        print(f"Macro P@5:     {macro_p5:.4f}")
        print(f"Overall P@5:   {overall_p5:.4f}")

# Save evaluation
os.makedirs("results", exist_ok=True)

df = pd.DataFrame(results)

df.to_csv(
    "results/class_based_evaluation.csv",
    index=False
)

print("\nClass-based evaluation completed!")
print("Saved: results/class_based_evaluation.csv")
