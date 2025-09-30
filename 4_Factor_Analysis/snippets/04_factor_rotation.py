#!/usr/bin/env python3
"""
Factor Rotation Methods - MA2003B Multivariate Statistics Course

This script demonstrates factor rotation techniques in Factor Analysis. Rotation
improves interpretability by transforming the factor solution to achieve "simple
structure" where variables load highly on few factors and factors are clearly distinct.

Learning Objectives:
- Understand the purpose of factor rotation
- Compare orthogonal (Varimax) vs oblique (Promax) rotation
- See how rotation affects factor loadings and interpretability
- Recognize when rotation improves factor structure

Data: Simulated 100 observations × 3 variables to demonstrate rotation effects
Expected Output:
- Comparison of loadings before and after rotation
- Demonstration of how rotation clarifies factor structure
"""

import numpy as np
from factor_analyzer import FactorAnalyzer

# Generate reproducible random data for demonstration
# Using 3 variables to show clear rotation effects on 2 factors
np.random.seed(42)
X = np.random.randn(100, 3)

print("Factor Rotation Comparison")
print("=" * 40)
print(f"Dataset: {X.shape[0]} observations × {X.shape[1]} variables")
print("Extracting 2 factors with different rotation methods")
print()

# Method 1: No rotation (raw factor solution)
# This gives the initial, often uninterpretable solution
fa_no_rotation = FactorAnalyzer(n_factors=2, method="principal")
fa_no_rotation.fit(X)

loadings_before = fa_no_rotation.loadings_

print("1. Unrotated Factor Solution:")
print("   Raw loadings - may be difficult to interpret")
print("   Factors are orthogonal but not necessarily meaningful")
print(loadings_before.round(3))
print()

# Method 2: Varimax rotation (orthogonal)
# Maximizes variance of squared loadings within factors
# Maintains factor independence (orthogonal factors)
fa_varimax = FactorAnalyzer(n_factors=2, rotation="varimax", method="principal")
fa_varimax.fit(X)

loadings_varimax = fa_varimax.loadings_

print("2. Varimax Rotation (Orthogonal):")
print("   Simplifies factor structure while keeping factors uncorrelated")
print("   Each variable loads highly on one factor, minimally on others")
print(loadings_varimax.round(3))
print()

# Method 3: Promax rotation (oblique)
# Oblique rotation allowing factors to correlate
# Often more realistic as psychological constructs are related
fa_promax = FactorAnalyzer(n_factors=2, rotation="promax", method="principal")
fa_promax.fit(X)

loadings_promax = fa_promax.loadings_

print("3. Promax Rotation (Oblique):")
print("   Allows factors to correlate, which is often more realistic")
print("   May provide cleaner simple structure than orthogonal rotation")
print(loadings_promax.round(3))
print()

# Additional rotation options available:
# - 'oblimin': Alternative oblique rotation
# - 'quartimax': Alternative orthogonal rotation
# - None: No rotation (default)

print("Rotation Comparison Summary:")
print("- Unrotated: Mathematically optimal but often uninterpretable")
print("- Varimax: Simplifies structure, maintains orthogonality")
print("- Promax: Allows correlation, often clearest interpretation")
print("- Choose based on theoretical expectations about factor relationships")

print("\nKey Insights:")
print("- Rotation doesn't change the overall fit, only the factor orientation")
print("- Goal is to achieve 'simple structure' for better interpretability")
print("- Orthogonal vs oblique depends on whether factors should be independent")
