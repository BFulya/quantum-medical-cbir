
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

feature_names = data["feature_names"]

# Load QAOA selected features
quantum_data = np.load(
    "results/quantum_selected_features.npz"
)

quantum_indices = quantum_data[
    "selected_indices"
].astype(int)

# Classical feature selection for comparison
f_scores, _ = f_classif(X_train, y_train)
f_scores = np.nan_to_num(f_scores)

classical_indices = np.argsort(
    f_scores
)[-len(quantum_indices):]

# Feature selection methods
methods = {
    "All Features": np.arange(X_train.shape[1]),
    "Quantum QAOA": quantum_indices,
    "Classical F-Score": classical_indices
}

# Distance metrics
metrics = ["euclidean", "manhattan", "cosine"]

# CBIR evaluation
def evaluate_retrieval(
    gallery,
    queries,
    gallery_labels,
    query_labels,
    metric,
    k=10
):
    distances = pairwise_distances(
        queries,
        gallery,
        metric=metric
    )

    # Retrieve top-k most similar images
    ranked_indices = np.argsort(
        distances, axis=1
    )[:, :k]

    retrieved_labels = gallery_labels[ranked_indices]

    # Relevant = same medical image category
    matches = (
        retrieved_labels == query_labels[:, None]
    ).astype(float)

    # Precision metrics
    precision_5 = matches[:, :5].mean()
    precision_10 = matches[:, :10].mean()

    # Average Precision at K
    cumulative_precision = (
        np.cumsum(matches, axis=1)
        / np.arange(1, k + 1)
    )

    relevant_total = (
        gallery_labels[None, :]
        == query_labels[:, None]
    ).sum(axis=1)

    denominators = np.maximum(
        np.minimum(relevant_total, k), 1
    )

    ap_at_k = (
        (cumulative_precision * matches).sum(axis=1)
        / denominators
    )

    map_10 = ap_at_k.mean()

    return precision_5, precision_10, map_10


results = []

print("Medical CBIR Evaluation")
print("-----------------------")

for method_name, indices in methods.items():

    gallery = X_train[:, indices]
    queries = X_test[:, indices]

    print(f"\nMethod: {method_name}")
    print("Selected features:", list(feature_names[indices]))

    for metric in metrics:

        p5, p10, map10 = evaluate_retrieval(
            gallery,
            queries,
            y_train,
            y_test,
            metric
        )

        results.append({
            "Method": method_name,
            "Distance": metric,
            "Precision@5": p5,
            "Precision@10": p10,
            "mAP@10": map10
        })

        print(
            f"{metric:12s} | "
            f"P@5: {p5:.4f} | "
            f"P@10: {p10:.4f} | "
            f"mAP@10: {map10:.4f}"
        )

# Save results
os.makedirs("results", exist_ok=True)

df = pd.DataFrame(results)

df.to_csv(
    "results/cbir_evaluation.csv",
    index=False
)

print("\nCBIR evaluation completed!")
print("Saved: results/cbir_evaluation.csv")
