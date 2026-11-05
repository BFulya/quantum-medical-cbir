
import os
import numpy as np
import matplotlib.pyplot as plt

from qiskit.circuit.library import qaoa_ansatz
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo

# Load feature selection results
data = np.load("results/quantum_selected_features.npz")

relevance = data["relevance"]
redundancy = data["redundancy"]
selected_names = data["selected_names"]

n_features = len(relevance)
k = 4
alpha = 0.3

# Recreate the original optimization problem
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

problem.linear_constraint(
    linear={f"x{i}": 1 for i in range(n_features)},
    sense="==",
    rhs=k,
    name="select_four_features"
)

# Convert constrained problem to QUBO
converter = QuadraticProgramToQubo()
qubo = converter.convert(problem)

# Convert QUBO to Ising Hamiltonian
cost_operator, offset = qubo.to_ising()

# Construct QAOA circuit (one repetition)
circuit = qaoa_ansatz(
    cost_operator,
    reps=1,
    insert_barriers=True
)

# Decompose to show elementary gates
detailed_circuit = circuit.decompose(reps=2)

os.makedirs("results", exist_ok=True)

# Save graphical quantum circuit
fig = detailed_circuit.draw(
    output="mpl",
    fold=20,
    scale=0.8
)

fig.savefig(
    "results/qaoa_quantum_circuit.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)

# Save text representation
with open(
    "results/qaoa_quantum_circuit.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(
        str(detailed_circuit.draw(output="text", fold=120))
    )

print("QAOA Quantum Circuit")
print("-------------------------")
print("Number of qubits:", circuit.num_qubits)
print("QAOA repetitions: 1")
print("Circuit parameters:", [str(p) for p in circuit.parameters])

print("\nSelected Features:")
for name in selected_names:
    print("-", str(name))

print("\nQuantum circuit visualization completed!")
print("Saved: results/qaoa_quantum_circuit.png")
print("Saved: results/qaoa_quantum_circuit.txt")
