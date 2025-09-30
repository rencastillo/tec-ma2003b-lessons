#!/usr/bin/env python3
"""
Factor Analysis Basic Example - MA2003B Multivariate Statistics Course

This script demonstrates the fundamental concepts of Factor Analysis (FA) using
a simple 3-variable correlation matrix. Factor Analysis models observed variables
as linear combinations of underlying latent factors plus unique error terms.

Learning Objectives:
- Understand the difference between PCA and Factor Analysis
- Interpret factor loadings as correlations between variables and factors
- Distinguish communalities (common variance) from uniqueness (unique variance)
- See how FA focuses on shared variance rather than total variance

Data: Hypothetical 3-variable correlation matrix showing moderate intercorrelations
Expected Output:
- Single factor loading for each variable
- Communalities showing proportion of variance explained by the factor
- Uniqueness showing variable-specific variance
"""

import numpy as np
from factor_analyzer import FactorAnalyzer

# Example correlation matrix representing 3 moderately correlated variables
# This could represent psychological test scores or survey items measuring similar constructs
R = np.array(
    [
        [1.00, 0.60, 0.48],  # Variable 1 correlations
        [0.60, 1.00, 0.72],  # Variable 2 correlations
        [0.48, 0.72, 1.00],
    ]
)  # Variable 3 correlations

print("Factor Analysis: Basic Single-Factor Model")
print("=" * 50)
print("Input Correlation Matrix:")
print("Variables show moderate intercorrelations (0.48-0.72)")
print(R)
print()

# Alternative approach for raw data (commented out):
# For real datasets, start with raw observations and compute correlation matrix
# X = your_data  # shape (n_samples, n_features)
# R = np.corrcoef(X.T)  # correlation matrix from raw data

# Initialize Factor Analysis with 1 factor
# rotation=None means no rotation (raw factor solution)
fa = FactorAnalyzer(n_factors=1, method="principal")

# Fit to correlation matrix (not raw data in this example)
fa.fit(R)

# Extract key results
loadings = fa.loadings_  # Correlations between variables and factor
communalities = fa.get_communalities()  # Variance explained by common factor(s)
uniqueness = fa.get_uniquenesses()  # Variable-specific variance (1 - communality)

print("Factor Analysis Results:")
print("-" * 30)
print("Factor Loadings (correlations with latent factor):")
print("Higher absolute values indicate stronger relationships")
for i, loading in enumerate(loadings.flatten(), 1):
    print(f"Variable {i}: {loading:.3f}")
print()

print("Communalities (h²):")
print("Proportion of each variable's variance explained by the common factor")
print("Range: 0 (no common variance) to 1.0 (all variance is common)")
for i, comm in enumerate(communalities, 1):
    print(f"Variable {i}: {comm:.3f}")
print()

print("Uniqueness (ψ):")
print("Variable-specific variance not explained by the common factor")
print("Includes measurement error and truly unique variance")
for i, uniq in enumerate(uniqueness, 1):
    print(f"Variable {i}: {uniq:.3f}")
print()

print("Interpretation:")
print("- All variables load positively on the single factor")
print("- Communalities show moderate shared variance (0.4-0.6)")
print("- Uniqueness varies, indicating different amounts of specific variance")
print("- Factor represents the underlying construct measured by all three variables")

print("\nNote: Install factor_analyzer with: pip install factor_analyzer")
