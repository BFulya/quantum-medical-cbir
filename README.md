# Quantum-Based Medical Image Retrieval (CBIR)

This project implements a quantum-enhanced Content-Based Image Retrieval (CBIR) system for medical chest X-ray images using Python and Qiskit.

The system extracts eight statistical and texture-based image features and applies the Quantum Approximate Optimization Algorithm (QAOA) to select four relevant features.

Image retrieval performance is evaluated using Euclidean, Manhattan, and Cosine distance metrics.

## Key Features

- **Dataset:** PneumoniaMNIST (5,856 chest X-ray images)
- **Feature Extraction:** Intensity statistics, entropy, edge features, and GLCM texture features
- **Quantum Feature Selection:** QAOA with an 8-qubit quantum simulation
- **Dimensionality Reduction:** 8 features reduced to 4
- **Evaluation Metrics:** Precision@5, Precision@10, mAP@10, and class-based precision
- **Comparisons:** Quantum QAOA, Classical F-Score, and All Features

## Experimental Results

- Best Macro Precision@5: **73.39%** (Quantum QAOA + Euclidean)
- Best mAP@10: **71.69%** (Classical F-Score + Manhattan)
- Feature dimensionality reduction: **50%**

The results demonstrate that QAOA-based feature selection can achieve competitive retrieval performance using fewer image features. However, the quantum approach does not outperform classical methods across every evaluation metric.

All quantum experiments were performed using simulators rather than physical quantum hardware.

**Disclaimer:** This project is intended for academic research and is not a clinical diagnostic tool.
