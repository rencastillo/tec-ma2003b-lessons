#!/usr/bin/env python3
"""
Complete Factor Analysis Workflow - MA2003B Multivariate Statistics Course

This script demonstrates a comprehensive end-to-end Factor Analysis pipeline,
from data preparation through interpretation and comparison with PCA. This
represents the complete analytical process used in multivariate statistics.

Learning Objectives:
- Implement the full factor analysis workflow
- Apply statistical tests for factorability
- Make decisions about factor retention and rotation
- Interpret factor loadings, communalities, and variance explained
- Compare Factor Analysis results with Principal Component Analysis

Data: Simulated 100 observations × 5 variables (replace with real data)
Workflow Steps:
1. Data preparation and standardization
2. Factorability assessment (KMO, Bartlett's test)
3. Factor extraction and retention decisions
4. Factor rotation for interpretability
5. Results interpretation and validation
6. Comparison with PCA
"""

import numpy as np
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

print("Complete Factor Analysis Workflow")
print("=" * 50)
print("Following systematic steps for multivariate analysis")
print()

# ============================================================================
# Step 1: Data Loading and Preparation
# ============================================================================

# Load your actual data here (replace the simulated data)
# X = pd.read_csv('your_data.csv')  # Real data loading
# For demonstration, we use simulated multivariate normal data
np.random.seed(42)  # For reproducible results
X = np.random.randn(100, 5)  # 100 observations, 5 variables

print("Step 1: Data Preparation")
print("-" * 25)
print(f"Dataset dimensions: {X.shape[0]} observations × {X.shape[1]} variables")
print("Data type: Simulated multivariate normal (replace with real data)")
print()

# ============================================================================
# Step 2: Data Standardization
# ============================================================================

# Standardize variables to ensure equal contribution to analysis
# This is crucial when variables have different scales/units
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Step 2: Data Standardization")
print("-" * 25)
print("Variables standardized: mean ≈ 0, std ≈ 1 for all variables")
print("Ensures equal contribution regardless of original scales")
print()

# ============================================================================
# Step 3: Assess Suitability for Factor Analysis
# ============================================================================

# Test statistical assumptions before proceeding
kmo_all, kmo_model = calculate_kmo(X_scaled)
chi_square_value, p_value = calculate_bartlett_sphericity(X_scaled)

print("Step 3: Factorability Assessment")
print("-" * 25)
print("Testing whether data is suitable for factor analysis:")
print()

print("Kaiser-Meyer-Olkin (KMO) Test:")
print("  Measures sampling adequacy for each variable and overall")
print(".3f")
if kmo_model > 0.8:
    kmo_interpretation = "Excellent"
elif kmo_model > 0.7:
    kmo_interpretation = "Good"
elif kmo_model > 0.6:
    kmo_interpretation = "Acceptable"
else:
    kmo_interpretation = "Unacceptable"
print(f"  Interpretation: {kmo_interpretation} sampling adequacy")
print()

print("Bartlett's Test of Sphericity:")
print("  Tests null hypothesis: correlation matrix is identity matrix")
print(".3f")
if p_value < 0.05:
    print("  Result: Significant - variables are correlated, FA is appropriate")
else:
    print("  Result: Not significant - variables may be uncorrelated, reconsider FA")
print()

# ============================================================================
# Step 4: Determine Number of Factors to Extract
# ============================================================================

# Use PCA eigenvalues as initial guide for factor retention
pca = PCA()
pca.fit(X_scaled)
eigenvalues = pca.explained_variance_

# Kaiser criterion: retain factors with eigenvalues > 1.0
n_factors = sum(eigenvalues > 1)

print("Step 4: Factor Retention Decision")
print("-" * 25)
print("Using Kaiser criterion (eigenvalues > 1.0):")
print(f"Eigenvalues: {np.round(eigenvalues, 3)}")
print(f"Suggested number of factors: {n_factors}")
print("Rationale: Factors should explain more variance than individual variables")
print()

# ============================================================================
# Step 5: Perform Factor Analysis with Rotation
# ============================================================================

# Extract factors using Principal Axis Factoring with Varimax rotation
fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax', method='principal')
fa.fit(X_scaled)

print("Step 5: Factor Extraction and Rotation")
print("-" * 25)
print(f"Method: Principal Axis Factoring with {n_factors} factors")
print("Rotation: Varimax (orthogonal) for simple structure")
print()

# ============================================================================
# Step 6: Extract and Interpret Results
# ============================================================================

# Get key factor analysis outputs
loadings = fa.loadings_                          # Variable-factor correlations
communalities = fa.get_communalities()          # Common variance proportions
variance_explained = fa.get_factor_variance()   # Variance decomposition

print("Step 6: Results Interpretation")
print("-" * 25)
print()

print("Factor Loadings (Variable-Factor Correlations):")
print("  Values > 0.6 indicate strong factor relationships")
print("  Values > 0.3 indicate moderate relationships")
if loadings is not None:
    print(loadings.round(3))
else:
    print("  Loadings not available")
print()

print("Communalities (h² - Common Variance):")
print("  Proportion of variance explained by extracted factors")
print("  Higher values indicate better factor model fit")
print(communalities.round(3))
print()

print("Factor Variance Explained:")
print("  [0]: Sum of squared loadings (eigenvalues)")
print("  [1]: Proportional variance explained")
print("  [2]: Cumulative variance explained")
print(np.round(variance_explained, 3))
print()

# ============================================================================
# Step 7: Compare with Principal Component Analysis
# ============================================================================

# Transform data using both methods for comparison
pca_scores = pca.transform(X_scaled)[:, :n_factors]  # Keep same number of components
fa_scores = fa.transform(X_scaled)

print("Step 7: PCA vs Factor Analysis Comparison")
print("-" * 25)
print()

# Variance comparison
pca_variance = pca.explained_variance_ratio_[:n_factors].sum()
fa_variance = variance_explained[2][-1]  # Cumulative variance from FA

print("Variance Explained Comparison:")
print(".1%")
print(".1%")
print()

print("Key Differences:")
print("- PCA: Maximizes total variance, includes unique + common variance")
print("- FA: Focuses on common variance, models unique variance separately")
print("- PCA: Components are linear combinations for dimensionality reduction")
print("- FA: Factors represent latent constructs for theory testing")
print()

print("Factor Scores Shape:")
print(f"  PCA scores: {pca_scores.shape}")
print(f"  FA scores: {fa_scores.shape}")
print("  Both provide component/factor scores for further analysis")
print()

# ============================================================================
# Summary and Recommendations
# ============================================================================

print("Analysis Complete!")
print("=" * 50)
print("Summary:")
print(f"- Extracted {n_factors} factors from {X.shape[1]} variables")
print(".1%")
print(".3f")

if kmo_model > 0.6 and p_value < 0.05:
    print("- Data meets factorability requirements")
else:
    print("- Consider data quality issues or alternative methods")

print("\nNext Steps:")
print("- Examine factor loadings for theoretical interpretation")
print("- Consider oblique rotation if factors are expected to correlate")
print("- Validate factor structure with confirmatory methods")
print("- Use factor scores in subsequent analyses")
