
import os
import numpy as np
import pandas as pd

from skimage.filters import sobel
from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix, graycoprops

# Load the medical image dataset
data = np.load("data/pneumoniamnist_128.npz")

# Create results directory
os.makedirs("results", exist_ok=True)

# Feature names
feature_names = [
    "Mean_Intensity",
    "Standard_Deviation",
    "Median_Intensity",
    "Entropy",
    "Edge_Mean",
    "Edge_Std",
    "GLCM_Contrast",
    "GLCM_Homogeneity"
]

# Extract 8 features from a single image
def extract_features(image):

    normalized = image.astype(np.float32) / 255.0

    # Intensity features
    mean = np.mean(normalized)
    std = np.std(normalized)
    median = np.median(normalized)

    # Entropy
    entropy = shannon_entropy(image)

    # Edge features
    edges = sobel(normalized)
    edge_mean = np.mean(edges)
    edge_std = np.std(edges)

    # Texture features
    quantized = (image // 16).astype(np.uint8)

    glcm = graycomatrix(
        quantized,
        distances=[1],
        angles=[0, np.pi / 2],
        levels=16,
        symmetric=True,
        normed=True
    )

    contrast = np.mean(graycoprops(glcm, "contrast"))
    homogeneity = np.mean(graycoprops(glcm, "homogeneity"))

    return [
        mean, std, median, entropy,
        edge_mean, edge_std, contrast, homogeneity
    ]


# Process one dataset split
def process_split(split_name):

    images = data[f"{split_name}_images"]
    labels = data[f"{split_name}_labels"].ravel()

    print(f"\nProcessing {split_name} dataset...")

    features = []

    for i, image in enumerate(images):

        features.append(extract_features(image))

        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{len(images)} images")

    features = np.array(features, dtype=np.float32)

    # Save extracted features as CSV
    df = pd.DataFrame(features, columns=feature_names)
    df["Label"] = labels

    df.to_csv(
        f"results/{split_name}_features.csv",
        index=False
    )

    print(f"{split_name} feature shape: {features.shape}")

    return features, labels


# Extract features from all dataset splits
X_train, y_train = process_split("train")
X_val, y_val = process_split("val")
X_test, y_test = process_split("test")

# Save features for quantum feature selection
np.savez_compressed(
    "results/extracted_features.npz",
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    X_test=X_test,
    y_test=y_test,
    feature_names=np.array(feature_names)
)

print("\nFeature extraction completed successfully!")
print("Total features per image:", len(feature_names))
print("Total images:", len(X_train) + len(X_val) + len(X_test))
print("Files saved in results folder.")
