
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load evaluation results
class_df = pd.read_csv(
    "results/class_based_evaluation.csv"
)

cbir_df = pd.read_csv(
    "results/cbir_evaluation.csv"
)

os.makedirs("results", exist_ok=True)

methods = [
    "All Features",
    "Quantum QAOA",
    "Classical F-Score"
]

distances = [
    "euclidean",
    "manhattan",
    "cosine"
]

# Create grouped comparison chart
def create_comparison_chart(
    df, column, title, output_file
):
    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(distances))
    width = 0.25

    for i, method in enumerate(methods):

        values = []

        for distance in distances:
            row = df[
                (df["Method"] == method) &
                (df["Distance"] == distance)
            ]

            values.append(
                row[column].iloc[0] * 100
            )

        bars = ax.bar(
            x + (i - 1) * width,
            values,
            width,
            label=method
        )

        ax.bar_label(
            bars,
            fmt="%.2f",
            padding=3,
            fontsize=9
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Euclidean", "Manhattan", "Cosine"]
    )

    ax.set_ylabel("Score (%)")
    ax.set_title(title)
    ax.set_ylim(0, 90)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    print("Saved:", output_file)
    plt.show()


# GRAPH 1: Macro Precision@5
create_comparison_chart(
    class_df,
    "Macro_P5",
    "Macro Precision@5 Comparison",
    "results/macro_precision_comparison.png"
)

# GRAPH 2: Class-based performance
euclidean_df = class_df[
    class_df["Distance"] == "euclidean"
].set_index("Method").loc[methods]

x = np.arange(len(methods))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 6))

normal_scores = (
    euclidean_df["Normal_P5"].to_numpy() * 100
)

pneumonia_scores = (
    euclidean_df["Pneumonia_P5"].to_numpy() * 100
)

bars1 = ax.bar(
    x - width / 2,
    normal_scores,
    width,
    label="Normal"
)

bars2 = ax.bar(
    x + width / 2,
    pneumonia_scores,
    width,
    label="Pneumonia"
)

ax.bar_label(bars1, fmt="%.2f", padding=3)
ax.bar_label(bars2, fmt="%.2f", padding=3)

ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylabel("Precision@5 (%)")
ax.set_title(
    "Class-Based CBIR Performance - Euclidean"
)
ax.set_ylim(0, 105)
ax.legend()
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()

plt.savefig(
    "results/class_performance_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

print(
    "Saved: results/class_performance_comparison.png"
)

plt.show()

# GRAPH 3: Mean Average Precision@10
create_comparison_chart(
    cbir_df,
    "mAP@10",
    "Mean Average Precision@10 Comparison",
    "results/map_comparison.png"
)

print("\nAll evaluation graphs created successfully!")
