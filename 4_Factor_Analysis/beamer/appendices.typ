// Mathematical Foundations Appendix
// Matrix algebra and statistical foundations for PCA and Factor Analysis

// Tec de Monterrey color palette
#let tec-blue = rgb("#003f7f")      // Tec de Monterrey signature blue
#let tec-light-blue = rgb("#0066cc") // Lighter blue for accents

#set page(
  width: 16cm,
  height: 9cm,
  margin: (x: 1.5cm, y: 1cm),
  numbering: "1",
)

#set document(
  title: "Mathematical Foundations - Appendix",
  author: "Dr. Juliho Castillo",
)

#set text(
  font: "Liberation Sans",
  size: 12pt,
  lang: "es",
)
#set math.equation(numbering: "(1)")

// Custom slide function with Tec blue titles
#let slide(title: none, content) = {
  pagebreak(weak: true)
  if title != none [
    #set text(size: 18pt, weight: "bold", fill: tec-blue)
    #title
    #v(0.8cm)
  ]
  content
}

// Section slide function with Tec blue
#let section-slide(title) = {
  pagebreak(weak: true)
  // Create invisible heading for table of contents after page break
  hide[#heading(level: 2, outlined: true, bookmarked: true)[#title]]
  set text(size: 20pt, weight: "bold", fill: tec-blue)
  align(center + horizon)[#title]
}

// Part slide function with Tec blue
#let part-slide(title) = {
  pagebreak(weak: true)
  set text(size: 24pt, weight: "bold", fill: tec-blue)
  align(center + horizon)[#title]
}

// Title slide with Tec colors
#align(center)[
  #v(2cm)
  #text(size: 28pt, weight: "bold", fill: tec-blue)[Mathematical Foundations]
  #v(0.6cm)
  #text(size: 16pt, fill: tec-light-blue)[Dr. Juliho Castillo]
  #v(0.4cm)
  #text(size: 14pt, fill: tec-blue)[Tecnológico de Monterrey]
  #v(0.4cm)
  #text(size: 12pt)[#datetime.today().display()]
]

// Table of contents
#slide(title: [Table of contents])[
  #outline()
]

// ============================================================================
// APPENDIX: MATHEMATICAL FOUNDATIONS
// ============================================================================

#part-slide[Appendix: Mathematical Foundations]

#section-slide[Matrix Algebra Foundations]

#slide(title: [Essential Matrix Operations for Multivariate Analysis])[
  *Matrix Transpose Properties:*
  - $(bold(A)^top)^top = bold(A)$
  - $(bold(A) + bold(B))^top = bold(A)^top + bold(B)^top$
  - $(bold(A) bold(B))^top = bold(B)^top bold(A)^top$
  - $(c bold(A))^top = c bold(A)^top$ for scalar $c$

  *Matrix Multiplication Rules:*
  - $(bold(A) bold(B)) bold(C) = bold(A) (bold(B) bold(C))$ (associativity)
  - $bold(A) (bold(B) + bold(C)) = bold(A) bold(B) + bold(A) bold(C)$ (distributivity)
  - Generally: $bold(A) bold(B) != bold(B) bold(A)$ (not commutative)

  *Trace Properties:*
  - $"tr"(bold(A)) = sum_(i=1)^n a_(i i)$ (sum of diagonal elements)
  - $"tr"(bold(A) + bold(B)) = "tr"(bold(A)) + "tr"(bold(B))$
  - $"tr"(bold(A) bold(B)) = "tr"(bold(B) bold(A))$ (cyclic property)
  - $"tr"(c bold(A)) = c "tr"(bold(A))$ for scalar $c$

  *Important for PCA/FA:* These properties ensure that variance decompositions are mathematically consistent.
]

#slide(title: [Eigenvalues and Eigenvectors: Deep Dive])[
  *Definition:* For square matrix $bold(A) in RR^(n times n)$, scalar $lambda$ and vector $bold(v) != bold(0)$ satisfy:
  $bold(A) bold(v) = lambda bold(v)$

  *Characteristic Equation:*
  $det(bold(A) - lambda bold(I)) = 0$

  *Fundamental Properties:*
  - $n times n$ matrix has exactly $n$ eigenvalues (counting multiplicities)
  - Eigenvalues can be real or complex conjugate pairs
  - Sum of eigenvalues equals trace: $sum_(i=1)^n lambda_i = "tr"(bold(A))$
  - Product of eigenvalues equals determinant: $prod_(i=1)^n lambda_i = det(bold(A))$

  *Geometric Interpretation:*
  - Eigenvector: direction unchanged by transformation
  - Eigenvalue: scaling factor in that direction
  - Eigendecomposition reveals the "natural axes" of the transformation

  *For Symmetric Matrices (Covariance/Correlation):*
  - All eigenvalues are real
  - Eigenvectors are orthogonal: $bold(v)_i^top bold(v)_j = 0$ for $i != j$
  - Can be orthonormalized: $bold(v)_i^top bold(v)_i = 1$
]

#slide(title: [Spectral Decomposition Theorem])[
  *For Symmetric Matrix $bold(A) in RR^(n times n)$:*

  *Theorem:* Every real symmetric matrix can be diagonalized by an orthogonal matrix:
  $bold(A) = bold(Q) bold(Lambda) bold(Q)^top$

  where:
  - $bold(Lambda) = "diag"(lambda_1, lambda_2, ..., lambda_n)$ (eigenvalues on diagonal)
  - $bold(Q) = [bold(q)_1 | bold(q)_2 | ... | bold(q)_n]$ (orthonormal eigenvectors)
  - $bold(Q)^top bold(Q) = bold(Q) bold(Q)^top = bold(I)$ (orthogonal matrix)

  *Equivalent Outer Product Form:*
  $bold(A) = sum_(i=1)^n lambda_i bold(q)_i bold(q)_i^top$

  *Significance for Multivariate Statistics:*
  - Covariance matrices are symmetric → always diagonalizable
  - Eigenvalues represent variance in principal directions
  - Eigenvectors define the principal directions
  - Enables dimension reduction by truncating small eigenvalues
  - Foundation for both PCA and Factor Analysis

  *Positive Semidefinite Property:*
  For covariance matrices: $lambda_i >= 0$ for all $i$ (variances cannot be negative)
]

#slide(title: [Matrix Norms and Distance Measures])[
  *Frobenius Norm (Essential for Factor Analysis):*
  $||bold(A)||_F = sqrt("tr"(bold(A)^top bold(A))) = sqrt(sum_(i=1)^m sum_(j=1)^n a_(i j)^2)$

  *Properties:*
  - $||bold(A)||_F^2 = sum_(i=1)^n sigma_i^2$ where $sigma_i$ are singular values
  - $||bold(A) bold(B)||_F <= ||bold(A)||_F ||bold(B)||_2$ (submultiplicativity)
  - $||bold(A) + bold(B)||_F <= ||bold(A)||_F + ||bold(B)||_F$ (triangle inequality)

  *Spectral Norm (2-norm):*
  $||bold(A)||_2 = sigma_"max"(bold(A)) = sqrt(lambda_"max"(bold(A)^top bold(A))}$

  *Mahalanobis Distance:*
  $d^2(bold(x), bold(mu)) = (bold(x) - bold(mu))^top bold(Sigma)^(-1) (bold(x) - bold(mu))$

  *Applications in Multivariate Analysis:*
  - Frobenius norm: measuring fit in factor analysis
  - Spectral norm: condition number analysis
  - Mahalanobis distance: outlier detection, multivariate normality tests
  - All preserve relationships under orthogonal transformations
]

#slide(title: [Matrix Calculus for Optimization])[
  *Scalar Functions of Matrices:*

  *Trace Derivatives:*
  - $frac(partial, partial bold(A)) "tr"(bold(A)) = bold(I)$
  - $frac(partial, partial bold(A)) "tr"(bold(A)^top bold(B)) = bold(B)$
  - $frac(partial, partial bold(A)) "tr"(bold(A)^top bold(A)) = 2 bold(A)$

  *Quadratic Form Derivatives:*
  - $frac(partial, partial bold(x)) bold(x)^top bold(A) bold(x) = (bold(A) + bold(A)^top) bold(x)$
  - For symmetric $bold(A)$: $frac(partial, partial bold(x)) bold(x}^top bold(A) bold(x) = 2 bold(A) bold(x)$

  *Determinant and Inverse:*
  - $frac(partial, partial bold(A)) log det(bold(A)) = (bold(A)^(-1))^top$
  - $frac(partial, partial bold(A)) bold(A)^(-1) = -bold(A)^(-1) frac(partial bold(A), partial bold(A)) bold(A)^(-1)$

  *Applications:*
  - PCA: Maximizing variance subject to orthogonality constraints
  - Factor Analysis: Maximum likelihood estimation
  - Lagrange multipliers for constrained optimization
  - Newton-Raphson methods for iterative algorithms
]

#slide(title: [Singular Value Decomposition (SVD)])[
  *Universal Matrix Decomposition:*
  For any matrix $bold(A) in RR^(m times n)$:
  $bold(A) = bold(U) bold(Sigma) bold(V)^top$

  where:
  - $bold(U) in RR^(m times m)$: left singular vectors (orthogonal)
  - $bold(Sigma) in RR^(m times n)$: diagonal matrix of singular values $sigma_i >= 0$
  - $bold(V) in RR^(n times n)$: right singular vectors (orthogonal)

  *Relationship to Eigendecomposition:*
  - $bold(A}^top bold(A) = bold(V) bold(Sigma)^top bold(Sigma) bold(V)^top$ (eigendecomposition)
  - $bold(A) bold(A)^top = bold(U) bold(Sigma) bold(Sigma}^top bold(U)^top$ (eigendecomposition)
  - Singular values: $sigma_i = sqrt(lambda_i (bold(A)^top bold(A))}$

  *Properties:*
  - Rank of $bold(A)$ = number of non-zero singular values
  - $||bold(A)||_2 = sigma_1$ (largest singular value)
  - $||bold(A)||_F = sqrt(sum_(i=1)^r sigma_i^2)$ where $r = "rank"(bold(A))$

  *Applications in Multivariate Statistics:*
  - Principal Component Analysis (PCA of data matrix)
  - Pseudoinverse computation: $bold(A)^+ = bold(V) bold(Sigma}^+ bold(U)^top$
  - Low-rank matrix approximation
  - Numerical stability in factor analysis algorithms
]

#section-slide[Statistical Foundations]

#slide(title: [Multivariate Normal Distribution])[
  *Definition:* Random vector $bold(X) in RR^p$ follows multivariate normal distribution:
  $bold(X) sim N(bold(mu), bold(Sigma))$

  *Probability Density Function:*
  $f(bold(x)) = frac(1, (2 pi)^(p/2) |bold(Sigma)|^(1/2)) exp(- frac(1, 2) (bold(x) - bold(mu))^top bold(Sigma)^(-1) (bold(x) - bold(mu)))$

  *Parameters:*
  - $bold(mu) in RR^p$: mean vector
  - $bold(Sigma) in RR^(p times p)$: covariance matrix (positive definite)

  *Key Properties:*
  - Linear combinations remain normal: $bold(a}^top bold(X) sim N(bold(a}^top bold(mu), bold(a}^top bold(Sigma) bold(a))$
  - Marginal distributions are normal: $X_i sim N(mu_i, sigma_(i i))$
  - Conditional distributions are normal (given subset of variables)

  *Standardization:*
  If $bold(X} sim N(bold(mu}, bold(Sigma))$, then $(bold(X} - bold(mu))^top bold(Sigma}^(-1) (bold(X) - bold(mu)) sim chi^2_p$

  *Importance for Factor Analysis:*
  - Maximum likelihood estimation assumes multivariate normality
  - Goodness-of-fit tests based on chi-square distribution
  - Confidence intervals and hypothesis tests for factor loadings
]

#slide(title: [Sample Statistics and Their Properties])[
  *Sample Mean Vector:*
  $overline(bold(x)) = frac(1, n) sum_(i=1)^n bold(x}_i = frac(1, n) bold(X}^top bold(1}_n$

  *Properties:*
  - $E[overline(bold(x}}] = bold(mu}$ (unbiased)
  - $"Cov"(overline(bold(x}}) = frac(1, n) bold(Sigma}$ (scales with sample size)
  - $sqrt(n)(overline(bold(x}} - bold(mu}) arrow.r N(bold(0}, bold(Sigma})$ (asymptotic normality)

  *Sample Covariance Matrix:*
  $bold(S} = frac(1, n-1) sum_(i=1)^n (bold(x}_i - overline(bold(x}})(bold(x}_i - overline(bold(x}})^top$

  *Matrix Form:*
  $bold(S} = frac(1, n-1} (bold(X} - bold(1}_n overline(bold(x}}^top)^top (bold(X} - bold(1}_n overline(bold(x}}^top)$

  *Properties:*
  - $E[bold(S}] = bold(Sigma}$ (unbiased)
  - $(n-1) bold(S} sim W_p(n-1, bold(Sigma})$ (Wishart distribution)
  - Positive semidefinite with probability 1 if $n >= p$

  *Sample Correlation Matrix:*
  $bold(R} = bold(D}^(-1/2) bold(S} bold(D}^(-1/2)$
  where $bold(D} = "diag"(s_{11}, s_{22}, ..., s_{pp})$
]

#slide(title: [Maximum Likelihood Estimation])[
  *General Principle:* Find parameters that maximize the likelihood of observing the sample data.

  *For Multivariate Normal Data:*
  $L(bold(mu}, bold(Sigma}) = prod_(i=1)^n frac(1, (2 pi)^(p/2) |bold(Sigma}|^(1/2)) exp(- frac(1, 2} (bold(x}_i - bold(mu})^top bold(Sigma}^(-1) (bold(x}_i - bold(mu}))$

  *Log-Likelihood:*
  $ell(bold(mu}, bold(Sigma}) = -frac(n p, 2) log(2 pi) - frac(n, 2} log |bold(Sigma}| - frac(1, 2} sum_(i=1)^n (bold(x}_i - bold(mu})^top bold(Sigma}^(-1) (bold(x}_i - bold(mu})$

  *ML Estimators:*
  - $hat(bold(mu}) = overline(bold(x}}$ (sample mean)
  - $hat(bold(Sigma}) = frac{1}{n} sum_(i=1)^n (bold(x}_i - overline(bold(x}})(bold(x}_i - overline(bold(x}})^top$ (ML covariance)

  *Properties:*
  - Consistent: $hat(bold(theta}) arrow.r bold(theta}$ as $n arrow.r infinity$
  - Asymptotically normal: $sqrt(n}(hat(bold(theta}) - bold(theta}) arrow.r N(bold(0}, bold(I}^(-1)(bold(theta}))$
  - Asymptotically efficient: achieves Cramér-Rao lower bound

  *Application to Factor Analysis:*
  - Estimate factor loadings $bold(Lambda}$ and unique variances $bold(Psi}$
  - Iterative algorithms (EM algorithm)
  - Model comparison via likelihood ratio tests
]

#slide(title: [Hypothesis Testing in Multivariate Analysis])[
  *Likelihood Ratio Test Principle:*
  $Lambda = frac{L(hat(bold(theta}}_0)}{L(hat(bold(theta}))} = frac{"max likelihood under " H_0}{"max likelihood unrestricted"}$

  *Test Statistic:*
  $-2 log Lambda sim chi^2_nu$ (asymptotically)
  where $nu$ = difference in number of parameters

  *Bartlett's Test of Sphericity:*
  - $H_0$: $bold(Sigma} = sigma^2 bold(I}$ (variables uncorrelated, equal variances)
  - Test statistic: $chi^2 = -(n - 1 - frac{2p + 5}{6}) log |bold(R}|$
  - Degrees of freedom: $frac{p(p-1)}{2}$

  *Goodness-of-Fit in Factor Analysis:*
  - $H_0$: $bold(Sigma} = bold(Lambda} bold(Lambda}^top + bold(Psi}$ (model fits)
  - Test statistic: $chi^2 = (n-1)[log |hat(bold(Sigma}}| - log |bold(S}| + "tr"(bold(S} hat(bold(Sigma}}^(-1)) - p]$
  - Degrees of freedom: $frac{p(p-1)}{2} - pk - p$ where $k$ = number of factors

  *Multiple Testing Considerations:*
  - Bonferroni correction for multiple comparisons
  - False Discovery Rate (FDR) control
  - Family-wise error rate control
]

#slide(title: [Asymptotic Theory and Large Sample Properties])[
  *Central Limit Theorem for Multivariate Data:*
  If $bold(X}_1, bold(X}_2, ..., bold(X}_n$ are i.i.d. with $E[bold(X}_i] = bold(mu}$ and $"Cov"(bold(X}_i) = bold(Sigma}$:
  $sqrt(n}(overline(bold(X}} - bold(mu}) arrow.r^d N(bold(0}, bold(Sigma})$

  *Delta Method:* For differentiable function $g$:
  $sqrt(n}(g(overline(bold(X}}) - g(bold(mu})) arrow.r^d N(bold(0}, bold(D} bold(Sigma} bold(D}^top)$
  where $bold(D} = nabla g(bold(mu})$ (gradient matrix)

  *Slutsky's Theorem:* If $bold(X}_n arrow.r^d bold(X}$ and $bold(Y}_n arrow.r^p bold(c}$:
  - $bold(X}_n + bold(Y}_n arrow.r^d bold(X} + bold(c}$
  - $bold(Y}_n^top bold(X}_n arrow.r^d bold(c}^top bold(X}$

  *Continuous Mapping Theorem:* If $bold(X}_n arrow.r^d bold(X}$ and $g$ is continuous:
  $g(bold(X}_n) arrow.r^d g(bold(X})$

  *Applications:*
  - Confidence intervals for eigenvalues and eigenvectors
  - Standard errors for factor loadings
  - Bootstrap methods for complex statistics
  - Robustness analysis under model misspecification
]

#slide(title: [Information Theory and Model Selection])[
  *Kullback-Leibler Divergence:*
  $D_"KL"(P || Q) = integral p(bold(x}) log frac{p(bold(x})}{q(bold(x})} d bold(x}$

  *Measures "distance" between probability distributions*
  - $D_"KL"(P || Q) >= 0$ with equality if and only if $P = Q$
  - Not symmetric: $D_"KL"(P || Q) != D_"KL"(Q || P)$

  *Akaike Information Criterion (AIC):*
  $"AIC" = -2 ell(hat(bold(theta}}) + 2k$
  where $k$ = number of parameters

  *Bayesian Information Criterion (BIC):*
  $"BIC" = -2 ell(hat(bold(theta}}) + k log(n)$

  *Model Selection Strategy:*
  1. Fit multiple models with different numbers of factors
  2. Compare AIC/BIC values
  3. Choose model with lowest criterion value
  4. Validate with cross-validation or hold-out data

  *Factor Analysis Application:*
  - Compare models with $k = 1, 2, 3, ...$ factors
  - Balance model fit vs. complexity
  - BIC typically favors more parsimonious models than AIC
  - Consider interpretability alongside statistical criteria

  *Information Matrix:*
  $bold(I}(bold(theta}) = -E[frac{partial^2 ell}{partial bold(theta} partial bold(theta}^top}]$
  - Provides asymptotic covariance: $"Cov"(hat(bold(theta}}) approx bold(I}^(-1)(bold(theta})$
  - Used for standard errors and confidence intervals
]

#slide(title: [Computational Considerations and Numerical Stability])[
  *Condition Number:*
  $kappa(bold(A}) = frac{sigma_"max"}sigma_"min"$ (ratio of largest to smallest singular value)

  *Numerical Issues:*
  - $kappa(bold(A}) > 10^{12}$ indicates near-singularity
  - Small eigenvalues lead to unstable inverse calculations
  - Rounding errors accumulate in iterative algorithms

  *Stable Algorithms:*
  1. *QR Decomposition* instead of normal equations: $bold(A} = bold(Q} bold(R}$
  2. *SVD* for pseudoinverses: $bold(A}^+ = bold(V} bold(Sigma}^+ bold(U}^top$
  3. *Cholsky Decomposition* for positive definite matrices: $bold(A} = bold(L} bold(L}^top$

  *Regularization Techniques:*
  - Ridge regularization: $bold(A} + lambda bold(I}$ for small $lambda > 0$
  - Shrinkage estimators for covariance matrices
  - Robust correlation estimation (e.g., Spearman rank correlation)

  *Convergence Criteria:*
  - Relative change in parameters: $frac{||bold(theta}_{k+1} - bold(theta}_k||}{||bold(theta}_k||} < epsilon$
  - Change in log-likelihood: $|ell_{k+1} - ell_k| < epsilon$
  - Gradient norm: $||nabla ell|| < epsilon$

  *Practical Guidelines:*
  - Use double precision arithmetic
  - Monitor condition numbers
  - Check convergence from multiple starting points
  - Validate results with alternative algorithms
]

#slide(title: [Summary: Mathematical Foundations for Factor Analysis])[
  *Essential Matrix Theory:*
  - Spectral decomposition enables eigenvalue methods
  - SVD provides numerical stability and generality
  - Matrix norms quantify approximation quality

  *Statistical Theory:*
  - Multivariate normality justifies ML estimation
  - Asymptotic theory provides inference framework
  - Information criteria guide model selection

  *Computational Aspects:*
  - Numerical stability requires careful algorithm design
  - Condition numbers indicate potential problems
  - Regularization prevents overfitting and instability

  *Integration in Practice:*
  - Theory guides algorithm development
  - Computation enables practical applications
  - Statistics provides inference and validation

  *Key Insight:* The mathematical foundations ensure that Factor Analysis results are not just empirically useful, but theoretically sound and computationally reliable. Understanding these foundations helps practitioners make informed decisions about model specification, estimation methods, and result interpretation.

  *For Further Study:*
  - Matrix Analysis (Horn & Johnson)
  - Multivariate Statistical Analysis (Anderson)
  - Numerical Linear Algebra (Trefethen & Bau)
  - Factor Analysis (Harman)
]