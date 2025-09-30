#!/usr/bin/env python3
"""
Component Retention Methods - MA2003B Multivariate Statistics Course

This script demonstrates three common methods for determining how many principal
components to retain in PCA analysis. Component retention is crucial for balancing
dimensionality reduction with information preservation.

Learning Objectives:
- Apply Kaiser criterion (eigenvalues > 1.0)
- Use cumulative variance threshold (e.g., 80%)
- Interpret scree plots for the "elbow" point
- Understand trade-offs between different retention criteria

Data: Simulated 100 observations × 5 variables with random normal distribution
Expected Output:
- Scree plot visualization
- Component counts from different methods
- Cumulative variance percentages
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# Generate reproducible random data for demonstration
# Using 5 variables to show clear component retention decisions
np.random.seed(42)
X = np.random.randn(100, 5)

print("Component Retention Analysis")
print("=" * 40)
print(f"Dataset: {X.shape[0]} observations × {X.shape[1]} variables")
print()

# Fit PCA to extract all components
pca = PCA()
pca.fit(X)

# Extract eigenvalues and cumulative variance
eigenvalues = pca.explained_variance_
cumvar = pca.explained_variance_ratio_.cumsum()

print("Eigenvalues:", np.round(eigenvalues, 3))
print("Cumulative Variance Ratios:", np.round(cumvar, 3))
print()

# Method 1: Kaiser Criterion
# Retain components with eigenvalues greater than 1.0
# Based on the idea that components should explain more variance than a single variable
n_kaiser = sum(eigenvalues > 1)
print(f"Kaiser Criterion (eigenvalues > 1.0): Retain {n_kaiser} components")
print("  Rationale: Components should explain more variance than individual variables")
print()

# Method 2: Cumulative Variance Threshold
# Retain enough components to explain a substantial portion of total variance
# Common thresholds: 70-90% depending on application
variance_threshold = 0.8
n_cumvar = np.argmax(cumvar >= variance_threshold) + 1
print(".0%")
print("  Rationale: Balance dimensionality reduction with information retention")
print()

# Method 3: Scree Plot (visual inspection)
# Plot eigenvalues and look for "elbow" where slope changes dramatically
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, "bo-", linewidth=2, markersize=8)
plt.axhline(y=1, color="red", linestyle="--", alpha=0.7, label="Kaiser criterion (y=1)")
plt.title("Scree Plot: Component Retention Decision")
plt.xlabel("Principal Component Number")
plt.ylabel("Eigenvalue")
plt.grid(True, alpha=0.3)
plt.legend()

# Add annotations for key points
for i, eigenval in enumerate(eigenvalues, 1):
    plt.annotate(
        ".2f", (i, eigenval), xytext=(5, 5), textcoords="offset points", fontsize=10
    )

# Save plot for documentation (works in non-interactive environments)
try:
    plt.savefig("scree_plot.png", dpi=150, bbox_inches="tight")
    print("Scree plot saved as 'scree_plot.png'")
except Exception as e:
    print(f"Could not save plot: {e}")

# Display plot if in interactive environment
try:
    plt.show()
except Exception:
    print("Plot display not available in this environment")

print()
print("Component Retention Summary:")
print("- Kaiser: Good default for most applications")
print(
    "- Cumulative: Application-specific (higher for prediction, lower for visualization)"
)
print("- Scree Plot: Subjective but often most insightful")
print("- Consider domain knowledge and practical constraints")
