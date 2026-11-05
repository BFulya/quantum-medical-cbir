
import numpy as np
from itertools import combinations
from sklearn.feature_selection import f_classif

from qiskit.primitives import StatevectorSampler
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import (
    MinimumEigenOptimizer,
    OptimizationResultStatus
)
from qiskit_optimization.minimum_eigensolvers import QAOA
from qiskit_optimization.optimizers import COBYLA

# Load scaled features
data = np.load("results/scaled_features.npz")

X_train = data["X_train"]
y_train = data["y_train"].ravel()
feature_names = data["feature_names"]

n_features = X_train.shape[1]
k = 4
alpha = 0.3

print("Quantum Feature Selection")
print("-------------------------")
print("Total features:", n_features)
print("Features to select:", k)

# Calculate feature relevance using training labels
f_scores, _ = f_classif(X_train, y_train)
f_scores = np.nan_to_num(
    f_scores, nan=0.0, posinf=0.0, neginf=0.0
)

relevance = np.log1p(np.maximum(f_scores, 0))
relevance = relevance / max(np.max(relevance), 1e-12)

# Calculate redundancy using feature correlations
correlation = np.corrcoef(X_train, rowvar=False)
redundancy = np.abs(np.nan_to_num(correlation))
np.fill_diagonal(redundancy, 0)

# Create binary optimization problem
problem = QuadraticProgram("Medical_CBIR_Feature_Selection")

for i in range(n_features):
    problem.binary_var(name=f"x{i}")

linear = {
    f"x{i}": float(relevance[i])
    for i in range(n_features)
}

quadratic = {
    (f"x{i}", f"x{j}"): -alpha * float(redundancy[i, j])
    for i in range(n_features)
    for j in range(i + 1, n_features)
}

problem.maximize(linear=linear, quadratic=quadratic)

# Select exactly four features
problem.linear_constraint(
    linear={f"x{i}": 1 for i in range(n_features)},
    sense="==",
    rhs=k,
    name="select_four_features"
)

print("\nOptimization problem created.")
print("Binary variables:", problem.get_num_binary_vars())

# Configure quantum algorithm
sampler = StatevectorSampler(
    default_shots=1024,
    seed=42
)

optimizer = COBYLA(maxiter=60)

qaoa = QAOA(
    sampler=sampler,
    optimizer=optimizer,
    reps=1,
    initial_point=[0.1, 0.3]
)

quantum_optimizer = MinimumEigenOptimizer(qaoa)

print("\nRunning QAOA simulation...")

result = quantum_optimizer.solve(problem)

# Find the best feasible solution sampled by QAOA
feasible_samples = [
    sample for sample in result.samples
    if sample.status == OptimizationResultStatus.SUCCESS
    and int(np.rint(sample.x).sum()) == k
]

if not feasible_samples:
    raise RuntimeError(
        "QAOA found no feasible selection. "
        "Please share the terminal output."
    )

best_sample = max(
    feasible_samples,
    key=lambda sample: sample.fval
)

selected_indices = np.flatnonzero(
    np.rint(best_sample.x).astype(int)
)

selected_names = feature_names[selected_indices]

# Calculate the exact optimum for comparison
def subset_score(indices):
    return (
        sum(relevance[i] for i in indices)
        - alpha * sum(
            redundancy[i, j]
            for i, j in combinations(indices, 2)
        )
    )

exact_indices = max(
    combinations(range(n_features), k),
    key=subset_score
)

print("\nQAOA Selected Features:")
for index in selected_indices:
    print(f"- {feature_names[index]}")

print("\nQAOA Objective Score:", round(best_sample.fval, 6))
print("Exact Best Score:", round(subset_score(exact_indices), 6))

# Save selected features
np.savez_compressed(
    "results/quantum_selected_features.npz",
    selected_indices=selected_indices,
    selected_names=selected_names,
    relevance=relevance,
    redundancy=redundancy
)

print("\nQuantum feature selection completed!")
print("Saved: results/quantum_selected_features.npz")
