# Quantum-Based Medical Image Retrieval (CBIR)

### Quantum Feature Selection Using Qiskit QAOA

This project implements a quantum-enhanced Content-Based Image Retrieval (CBIR) system for medical chest X-ray images using Python and Qiskit.

The main objective is to investigate the effectiveness of quantum-based feature selection in medical image retrieval by comparing QAOA with classical feature selection approaches.

## 1. Dataset

**Dataset:** PneumoniaMNIST (MedMNIST+)

- Total Images: 5,856
- Image Resolution: 128 × 128
- Classes: Normal and Pneumonia
- Training Images: 4,708
- Validation Images: 524
- Test Images: 624

**Dataset Source:** https://medmnist.com/

**Download:** https://zenodo.org/records/10519652

The dataset is not included in this repository. Download `pneumoniamnist_128.npz` and place it inside the `data` directory.

## 2. Feature Extraction

Eight statistical and texture-based features are extracted from each medical image:

1. Mean Intensity
2. Standard Deviation
3. Median Intensity
4. Shannon Entropy
5. Edge Mean
6. Edge Standard Deviation
7. GLCM Contrast
8. GLCM Homogeneity

Feature standardization is performed using StandardScaler, fitted exclusively on the training data.

## 3. Quantum Feature Selection

The Quantum Approximate Optimization Algorithm (QAOA) is implemented using Qiskit.

Feature selection is formulated as a binary optimization problem that balances feature relevance and redundancy.

**Quantum Configuration:**

- Number of Qubits: 8
- QAOA Repetitions: 1
- Quantum Simulation: StatevectorSampler
- Classical Optimizer: COBYLA
- Selected Features: 4

**QAOA Selected Features:**

- Median Intensity
- Edge Standard Deviation
- GLCM Contrast
- GLCM Homogeneity

The algorithm reduces feature dimensionality by 50%.

## 4. Content-Based Image Retrieval

Medical image retrieval is performed using the following distance metrics:

- Euclidean Distance
- Manhattan Distance
- Cosine Distance

Three approaches are compared:

- All Features (8 features)
- Quantum QAOA (4 features)
- Classical F-Score (4 features)

Test images are used as queries, while training images form the retrieval gallery.

## 5. Experimental Evaluation

Retrieval performance is evaluated using:

- Precision@5
- Precision@10
- Mean Average Precision@10 (mAP@10)
- Class-Based Precision@5
- Macro Precision@5

### Key Results

| Evaluation Metric | Best Method | Score |
|---|---|---|
| Macro Precision@5 | Quantum QAOA + Euclidean | 73.39% |
| Overall Precision@5 | Quantum QAOA + Euclidean | 78.17% |
| mAP@10 | Classical F-Score + Manhattan | 71.69% |

QAOA achieved the highest Macro Precision@5 while reducing the number of selected features from eight to four.

However, classical feature selection achieved the highest mAP@10. These experimental results do not establish a general quantum advantage.

## 6. Project Structure

The repository contains Python scripts for:

- Dataset exploration
- Feature extraction
- Feature standardization
- Quantum feature selection
- Distance-based CBIR evaluation
- Class-based evaluation
- Retrieval visualization
- Experimental comparison graphs
- QAOA quantum circuit visualization

Generated evaluation tables and visualizations are available in the `results` directory. Intermediate feature arrays and the original medical dataset are excluded from version control.

## 7. Installation and Execution

**Requirements:**

- Python 3.12
- Qiskit
- Qiskit Aer
- Qiskit Optimization
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Scikit-image

Create a Python virtual environment and install the required packages:

`python -m venv .venv`

`.\.venv\Scripts\python.exe -m pip install -r requirements.txt`

Download the dataset and place it in `data/pneumoniamnist_128.npz`.

Run the Python scripts in the following order:

1. `feature_extraction.py`
2. `preprocess_features.py`
3. `quantum_feature_selection.py`
4. `evaluate_cbir.py`
5. `class_based_evaluation.py`
6. `visualize_cbir.py`
7. `visualize_results.py`
8. `visualize_quantum_circuit.py`

On Windows, run each script using the virtual environment's Python interpreter, for example:

`.\.venv\Scripts\python.exe feature_extraction.py`

## 8. Conclusion

This study investigates the application of quantum-based feature selection to Content-Based Medical Image Retrieval.

Experimental findings demonstrate that QAOA-based feature selection can provide competitive retrieval performance with fewer image features.

The results also highlight the importance of distance metric selection, class imbalance, and evaluation methodology.

All quantum computations were performed using simulation rather than physical quantum hardware.

**Disclaimer:** This project is intended for educational and academic research purposes only. It is not designed or validated for clinical diagnosis.