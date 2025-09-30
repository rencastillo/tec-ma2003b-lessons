#!/usr/bin/env python3
"""
PCA Basic Example - MA2003B Multivariate Statistics Course

This script demonstrates the fundamental concepts of Principal Component Analysis (PCA)
using a simple 3x2 dataset. PCA is a dimensionality reduction technique that identifies
the principal components (directions of maximum variance) in the data.

Learning Objectives:
- Understand how PCA transforms correlated variables into uncorrelated principal components
- Interpret eigenvalues and explained variance ratios
- See how PCA rotates the coordinate system to align with data variance

Data: Simple 3 observations × 2 variables matrix to illustrate core concepts
Expected Output:
- PC1 captures most variance (horizontal spread)
- PC2 captures remaining variance (vertical spread)
- Transformed data shows uncorrelated coordinates
"""

import numpy as np
from sklearn.decomposition import PCA

# Create sample data: 3 observations with 2 correlated variables
# This represents a simple bivariate dataset where points form a diagonal pattern
X = np.array(
    [
        [5, 3],  # Point with high values on both variables
        [3, 1],  # Point with moderate values
        [1, 3],
    ]
)  # Point showing the correlation pattern

print("Original Data Matrix:")
print("Observations (rows) × Variables (columns)")
print(X)
print()

# Initialize PCA - no component limit means extract all possible components
pca = PCA()

# Fit PCA to the data and transform it to the new coordinate system
# fit_transform() centers the data and rotates it to align with principal components
X_transformed = pca.fit_transform(X)

# Extract key PCA results
eigenvalues = pca.explained_variance_  # Amount of variance explained by each PC
eigenvectors = pca.components_.T  # Directions of principal components
variance_ratio = pca.explained_variance_ratio_  # Proportion of total variance explained

print("PCA Results:")
print("=" * 50)
print(f"Eigenvalues: {eigenvalues}")
print(".3f")
print(".3f")
print()

print("Principal Component Directions (Eigenvectors):")
print("These show how original variables combine to form PCs")
print(eigenvectors)
print()

print("Transformed Data (Principal Component Scores):")
print("Original data in the new coordinate system")
print("Columns are uncorrelated with zero mean")
print(X_transformed)
print()

print("Interpretation:")
print("- PC1 captures the main diagonal trend in the data")
print("- PC2 captures the remaining perpendicular variation")
print("- The transformation decorrelates the original variables")
