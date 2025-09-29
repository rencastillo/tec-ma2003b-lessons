// Factor Analysis Presentation
// Focused on FA theory, theoretical comparison, and practical examples

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
  title: "Factor Analysis",
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
  #text(size: 28pt, weight: "bold", fill: tec-blue)[Factor Analysis]
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
// PART I: FACTOR ANALYSIS THEORY
// ============================================================================

#part-slide[Part I: Factor Analysis Theory]

#section-slide[Factor Analysis Theory]

#slide(title: [Understanding Factor Analysis: The Detective Story])[
  *Imagine you're a detective investigating what makes students perform well in school.*

  You observe their grades in Math, Science, Literature, and History. But you suspect there are *hidden influences* behind these grades - things like "intelligence," "motivation," and "study habits" that you can't directly measure.

  *What Factor Analysis does:*
  - Looks for these *hidden factors* that explain why variables move together
  - Separates what's *common* (shared patterns) from what's *unique* (individual noise)
  - Tells you how much each hidden factor influences each observed variable

  *The key insight:*
  If Math and Science grades are highly correlated, there might be a hidden "quantitative ability" factor. If all subjects correlate, there might be a "general intelligence" factor.

  *Why this matters:*
  - Helps understand the *underlying structure* of phenomena
  - Validates *theoretical models* (like intelligence theory)
  - Separates *signal from noise* in measurements
]

#slide(title: [What is Factor Analysis?])[
  - A statistical method for modeling relationships among *observed variables* using *latent factors*.
  - It uses a smaller number of _unobserved variables_, known as *common factors*.
  - *Key Distinction from PCA*: Explicitly models measurement error and unique variance
  - Often used to discover and validate underlying theoretical constructs
]

#slide(title: [Why Factor Analysis Works: Simple Logic])[
  *The Factor Model:* Each variable = Common factors + Unique part
  $ X_i = lambda_(i 1) F_1 + lambda_(i 2) F_2 + ... + U_i $

  *What this means:*
  - Common factors (F) affect multiple variables
  - Unique parts (U) affect only one variable
  - Loadings (λ) tell us how much each factor affects each variable

  *Key Insight:* If two variables share the same factors, they will be correlated!

  *Simple Example:*
  - Math and Science both depend on "Analytical Ability" → they correlate
  - But each also has unique aspects (Math has geometry, Science has memorization)

  *The Magic Formula:*
  $ bold(Sigma) = bold(Lambda) bold(Lambda)^top + bold(Psi) $
  - $bold(Lambda) bold(Lambda)^top$: correlations due to shared factors
  - $bold(Psi)$: unique variance for each variable

  *Bottom Line:* Observable correlations come from shared hidden factors!
]

#slide(title: [Factor Analysis Model: Mathematical Foundation])[
  *The General Factor Model for Variable i:*
  $ X_i = mu_i + sum_(j=1)^k lambda_(i j) F_j + U_i quad "for" i = 1, 2, ..., p $

  *Centered Factor Model (Standard Practice):*
  $ X_i - mu_i = sum_(j=1)^k lambda_(i j) F_j + U_i quad "for" i = 1, 2, ..., p $

  where:
  - $X_i$: Observed variable $i$
  - $mu_i$: Mean of variable $i$ (estimated from data: $mu_i = overline(x)_i = frac(1, n) sum_(l=1)^n X_(i l)$)
  - $F_j$: Common factor $j$ with $F_j tilde N(0,1)$, independent
  - $lambda_(i j)$: Factor loading of variable $i$ on factor $j$
  - $U_i$: Unique factor for variable $i$ with $U_i tilde N(0, psi_i^2)$
  - $k < p$: Number of common factors

  *Profound Significance of This Model:*
  - *Theoretical Foundation*: This equation captures the belief that observed phenomena are driven by *underlying latent causes*
  - *Parsimony Principle*: A few common factors ($k << p$) can explain most relationships among many variables
  - *Measurement Theory*: Separates systematic variance (common factors) from random measurement error (unique factors)
  - *Scientific Discovery*: Enables identification of unobservable constructs (like intelligence, personality dimensions)
  - *Practical Relevance*: Reduces complex multivariate data to interpretable, meaningful dimensions for decision-making

  *Matrix Form (for centered data):*
  $ bold(X)_c = bold(Lambda) bold(F) + bold(U) $

  where:
  - $bold(X)_c = bold(X) - bold(1)_n overline(bold(x))^top in RR^(n times p)$ (centered data matrix)
  - $overline(bold(x)) = (overline(x)_1, overline(x)_2, dots, overline(x)_p)^top$ (sample mean vector)
  - $bold(Lambda) in RR^(p times k)$ (factor loadings matrix)
  - $bold(F) in RR^(n times k)$ (factor scores matrix)
  - $bold(U) in RR^(n times p)$ (unique factors matrix)

  *Why we center:* Factor Analysis models the covariance structure, not the means
]

#slide(title: [Variance Decomposition: Communality and Uniqueness])[
  *Variance of Observed Variable i (Scalar Form):*
  $ "Var"(X_i) = "Var"(sum_(j=1)^k lambda_(i j) F_j) + "Var"(U_i) $

  Since $F_j$ are independent with unit variance and $U_i$ is independent of all $F_j$:
  $ "Var"(X_i) = sum_(j=1)^k lambda_(i j)^2 + psi_i^2 = h_i^2 + psi_i^2 $

  *Matrix Form (Variance Structure):*
  $ "Var"(bold(X)) = bold(Sigma) = bold(Lambda) bold(Lambda)^top + bold(Psi) $
  where each diagonal element: $sigma_(i i) = sum_(j=1)^k lambda_(i j)^2 + psi_i^2$

  *Fundamental Insight Behind Variance Decomposition:*
  - *Total Variance* = *Systematic Variance* + *Random Variance*
  - This decomposition is the *mathematical foundation of measurement theory*
  - *Systematic variance* ($h_i^2$) represents what the variable shares with underlying constructs
  - *Random variance* ($psi_i^2$) captures measurement error and variable-specific effects
  - *Critical for Quality Assessment*: High communality indicates reliable measurement; high uniqueness suggests measurement problems
  - *Model Evaluation*: If most variance is unique, the factor model may be inappropriate for the data

  *Communality (Scalar and Matrix Forms):*
  - *Scalar:* $h_i^2 = sum_(j=1)^k lambda_(i j)^2$ (variance explained by common factors)
  - *Matrix:* $bold(h)^2 = "diag"(bold(Lambda) bold(Lambda)^top)$ (communality vector)

  *Deeper Meaning of Communality:*
  - *Reliability Indicator*: High $h_i^2$ (close to 1) means variable is well-explained by common factors
  - *Construct Validity*: Variables measuring the same construct should have similar communalities
  - *Practical Threshold*: $h_i^2 > 0.5$ suggests variable belongs in the factor structure; $h_i^2 < 0.3$ may indicate poor fit

  *Uniqueness (Scalar and Matrix Forms):*
  - *Scalar:* $psi_i^2 = "Var"(U_i)$ (unique variance: specific + measurement error)
  - *Matrix:* $bold(Psi) = "diag"(psi_1^2, psi_2^2, dots, psi_p^2)$ (uniqueness matrix)

  *Deeper Meaning of Uniqueness:*
  - *Measurement Quality*: High $psi_i^2$ may indicate measurement problems or variable doesn't fit the factor structure
  - *Specificity vs. Error*: Includes both legitimate specific variance and random measurement error
  - *Model Diagnostics*: Variables with $psi_i^2 > 0.7$ should be examined for potential removal or model revision

  *Total Variance Decomposition:*
  For standardized variables: $"Var"(X_i) = 1 = h_i^2 + psi_i^2$

  *Critical Insight*: This decomposition enables *quantitative assessment of measurement quality* and guides decisions about variable retention and model adequacy.
]

#slide(title: [Algorithm: Communality and Uniqueness Calculation])[
  *Input:* Factor loading matrix $bold(Lambda) in RR^(p times k)$

  *Output:* Communalities $bold(h)^2$, uniquenesses $bold(psi)^2$

  1. *Initialize Arrays*
     - $bold(h)^2 = $ zero vector of length $p$
     - $bold(psi)^2 = $ zero vector of length $p$

  2. *Compute Communalities*
     - *for* $i = 1$ to $p$ *do*
       - $h_i^2 = sum_(j=1)^k lambda_(i j)^2$ (sum of squared loadings for variable $i$)

  3. *Compute Uniquenesses*
     - *for* $i = 1$ to $p$ *do*
       - $psi_i^2 = 1 - h_i^2$ (for standardized variables)
       - *if* $psi_i^2 < 0$ *then* $psi_i^2 = 0.005$ (Heywood case correction)

  4. *Validation Check*
     - *for* $i = 1$ to $p$ *do*
       - *assert* $h_i^2 + psi_i^2 = 1$ (variance decomposition property)
       - *assert* $0 <= h_i^2 <= 1$ and $0 <= psi_i^2 <= 1$ (valid proportions)
]

#slide(title: [Factor Extraction Methods: Mathematical Approaches])[
  *1. Principal Axis Factoring (PAF):*
  - Initialize communality estimates: $h_i^2 = R_(i i) - 1/R^(-1)_(i i)$ (squared multiple correlation)
  - Form reduced correlation matrix: $bold(R)^* = bold(R) - "diag"(psi_1^2, ..., psi_p^2)$
  - Eigendecomposition: $bold(R)^* = bold(V) bold(Lambda) bold(V)^top$
  - Factor loadings: $bold(Lambda) = bold(V)_k sqrt(bold(Lambda)_k)$ (first $k$ factors)
  - Iterate until convergence: Update $h_i^2 = sum_(j=1)^k lambda_(i j)^2$

  *2. Maximum Likelihood (ML) Factoring:*
  - Assumes multivariate normality: $bold(X) tilde N(bold(0), bold(Sigma))$
  - Covariance structure: $bold(Sigma) = bold(Lambda) bold(Lambda)^top + bold(Psi)$
  - Minimize: $F_"ML" = "tr"(bold(S) bold(Sigma)^(-1)) - ln|bold(Sigma)^(-1)| - p$
  - Provides $chi^2$ goodness-of-fit test and confidence intervals
]

#slide(title: [Algorithm: Factor Analysis with Principal Axis Factoring])[
  *Input:* Data matrix $bold(X) in RR^(n times p)$ (n observations, p variables), number of factors $k$
  *Output:* Factor loadings $bold(Lambda)$, uniquenesses $bold(Psi)$, factor scores $hat(bold(F))$

  1. *Data Preprocessing and Suitability Assessment*
     - *Standardize variables:* $bold(Z) = (bold(X) - bold(1)_n overline(bold(x))^top) bold(D)^(-1/2)$
       where $bold(D) = "diag"(s_1^2, dots, s_p^2)$ (variable variances)
     - *Compute correlation matrix:* $bold(R) = frac(1, n-1) bold(Z)^top bold(Z)$
     - *Test data suitability for factoring:*
       - Bartlett's sphericity test: $H_0$: $bold(R) = bold(I)$ (variables uncorrelated)
       - Kaiser-Meyer-Olkin (KMO) measure: assess sampling adequacy

  2. *Initialize Communality Estimates*
     - *Purpose:* Estimate how much variance each variable shares with factors
     - *for* each variable $i = 1$ to $p$:
       - *Compute initial estimate:* $h_i^2 = 1 - frac(1, (bold(R)^(-1))_(i i))$
         (squared multiple correlation with other variables)
       - *Alternative:* Use maximum absolute correlation: $h_i^2 = max_j |r_(i j)|$ for $j != i$
       - *Interpretation:* Proportion of variable i's variance explained by common factors

  3. *Principal Axis Factoring Iteration*
     - *Iterative refinement:* Improve communality estimates until convergence
     - *repeat*:
       - *Form reduced correlation matrix:* $bold(R)^* = bold(R) - "diag"(1-h_1^2, dots, 1-h_p^2)$
         (Replace diagonal with communalities instead of 1's)
       - *Eigenvalue decomposition:* $bold(R)^* = bold(V) bold(Lambda) bold(V)^top$
       - *Extract k factors:* $bold(L) = bold(V)_k bold(Lambda)_k^(1/2)$ (first k columns and eigenvalues)
       - *Update communalities:* $h_i^2 = sum_(j=1)^k l_(i j)^2$ for each $i = 1, dots, p$
         (Sum of squared loadings for variable i)
     - *until* $max_i |h_i^2_"new" - h_i^2_"old"| < epsilon$ (convergence achieved)

  4. *Factor Rotation* (Optional but Recommended)
     - *Purpose:* Achieve simpler, more interpretable factor structure
     - *Apply rotation method:*
       - Varimax (orthogonal): maintains factor independence, $bold(Lambda)^* = bold(L) bold(T)$
       - Promax (oblique): allows correlated factors for more flexibility
     - *Result:* Rotated loadings $bold(Lambda)^*$ with cleaner interpretation

  5. *Factor Score Estimation*
     - *Estimate individual factor scores for each observation*
     - *Regression method:* $hat(bold(F)) = bold(Z) bold(Lambda) (bold(Lambda)^top bold(Lambda))^(-1)$
       (Simple, but scores may be correlated even with orthogonal factors)
     - *Bartlett method:* $hat(bold(F)) = bold(Z) bold(Lambda) (bold(Lambda)^top bold(Psi)^(-1) bold(Lambda))^(-1) bold(Lambda)^top bold(Psi)^(-1)$
       (More complex, but preserves factor orthogonality if assumed)
]

#slide(title: [Factor Analysis: Simple Numerical Example])[
  *Given Data:* 3 variables, 1 factor model
  $ bold(R) = mat(1.00, 0.60, 0.48; 0.60, 1.00, 0.72; 0.48, 0.72, 1.00) $

  *Step 1: Initial communality estimates (SMC)*
  - Compute $bold(R)^(-1)$ and extract diagonal elements
  - $h_1^2 = 1 - frac(1, (bold(R)^(-1))_(11)) = 1 - frac(1, 2.78) = 0.64$
  - $h_2^2 = 1 - frac(1, (bold(R)^(-1))_(22)) = 1 - frac(1, 3.57) = 0.72$
  - $h_3^2 = 1 - frac(1, (bold(R)^(-1))_(33)) = 1 - frac(1, 2.17) = 0.54$

  *Step 2: Form reduced correlation matrix*
  $ bold(R)^* = mat(0.64, 0.60, 0.48; 0.60, 0.72, 0.72; 0.48, 0.72, 0.54) $

  *Step 3: Eigenvalue decomposition*
  - Largest eigenvalue: $lambda_1 = 1.84$
  - Corresponding eigenvector: $bold(v)_1 = mat(0.53; 0.67; 0.52)$

  *Step 4: Factor loadings*
  $ bold(L) = bold(v)_1 sqrt(lambda_1) = mat(0.53; 0.67; 0.52) times 1.36 = mat(0.72; 0.91; 0.71) $

  *Pointwise Form:*
  $ lambda_(i 1) = v_(i 1) sqrt(lambda_1) quad "for" i = 1, 2, 3 $

  *Step 5: Updated communalities*
  $ h_1^2 = 0.72^2 = 0.52, quad h_2^2 = 0.91^2 = 0.83, quad h_3^2 = 0.71^2 = 0.50 $

  *Final Model:* $bold(Sigma) = bold(L) bold(L)^top + bold(Psi)$
  $ = mat(0.72; 0.91; 0.71) mat(0.72, 0.91, 0.71) + mat(0.48, 0, 0; 0, 0.17, 0; 0, 0, 0.50) $

  *Pointwise Form:*
  $ sigma_(i j) = lambda_(i 1) lambda_(j 1) + psi_i^2 delta_(i j) $
]

#slide(title: [Python Implementation: Factor Analysis Example])[
  ```python
  import numpy as np
  from factor_analyzer import FactorAnalyzer
  from sklearn.datasets import make_spd_matrix

  # Create correlation matrix (our example)
  R = np.array([[1.00, 0.60, 0.48],
                [0.60, 1.00, 0.72],
                [0.48, 0.72, 1.00]])

  # For real data, you'd start with raw data:
  # X = your_data  # shape (n_samples, n_features)
  # R = np.corrcoef(X.T)  # correlation matrix

  # Perform Factor Analysis
  fa = FactorAnalyzer(n_factors=1, rotation=None)
  fa.fit(R)  # For correlation matrix
  # fa.fit(X)  # For raw data

  # Get results
  loadings = fa.loadings_
  communalities = fa.get_communalities()
  uniqueness = fa.get_uniquenesses()

  print(f"Factor loadings:\n{loadings}")
  print(f"Communalities: {communalities}")
  print(f"Uniqueness: {uniqueness}")
  ```

  *Note: Install with: pip install factor_analyzer*
]

#slide(title: [From Detective Work to Algorithm: Factor Analysis])[
  *Now let's turn our detective story into a systematic investigation!*

  Remember, Factor Analysis is like being a detective looking for hidden influences. The algorithm is our systematic method for uncovering these hidden factors.

  *What makes FA algorithm different from PCA?*
  - *Iterative process*: Like refining a theory through multiple rounds of evidence
  - *Two unknowns*: We don't know the factors OR how they influence variables
  - *Modeling errors*: Accounts for measurement noise (like witness reliability)
  - *Maximum Likelihood*: Finds the most probable explanation for what we observe

  *The detective's toolkit:*
  1. *Start with a guess* (initial hypothesis)
  2. *Estimate hidden factors* (what would the suspects be doing?)
  3. *Update our theory* (revise how factors influence variables)
  4. *Repeat until story makes sense* (convergence)
  5. *Test the theory* (does it explain the data well?)

  This is more complex than PCA, but gives us *richer insights* about underlying structure!
]

#slide(title: [Algorithm: Maximum Likelihood Factor Analysis])[
  *Input:* Data matrix $bold(X) in RR^(n times p)$, number of factors $k$, tolerance $epsilon$

  *Output:* ML factor loadings $bold(Lambda)$, uniquenesses $bold(Psi)$, model fit statistics

  1. *Start with Initial Guess*
     - Make first estimate of factor loadings
     - Use simple method (like PCA) as starting point

  2. *Iterative Improvement* (EM Algorithm)
     - Step E: Estimate what the hidden factors would be
     - Step M: Update loadings based on those estimates
     - Repeat until no more improvement
     - *Why this works:* Each step makes the model fit better

  3. *Check How Well It Fits*
     - Does our model explain the correlations well?
     - Statistical tests tell us if the fit is good enough
     - *Good fit:* Model explains most observed relationships

  4. *Get Final Results*
     - Factor loadings (how factors affect variables)
     - Confidence intervals (how certain are we?)
     - *Software does the complex calculations automatically*
]

#slide(title: [Making Sense of the Results: Factor Rotation])[
  *You've found the hidden factors, but they're still confusing! Now what?*

  Imagine you've identified two hidden factors in student performance, but:
  - Factor 1 loads on Math (0.6), Science (0.7), Literature (0.5), History (0.4)
  - Factor 2 loads on Math (0.4), Science (0.3), Literature (0.6), History (0.7)

  *This is messy!* What do these factors actually represent?

  *Factor Rotation is like adjusting the camera angle* after taking the photo:
  - We want *simple structure*: each factor should clearly represent something
  - Goal: High loadings for relevant variables, low for irrelevant ones
  - Like rotating a map so north points up - same information, clearer presentation

  *Two types of rotation:*
  - *Orthogonal* (factors stay independent): Clean, simple interpretation
  - *Oblique* (factors can correlate): More realistic, factors can be related

  This is where art meets science in factor analysis!
]

#slide(title: [Factor Rotation: Mathematical Transformation])[
  *Purpose:* Transform initial factor loadings $bold(Lambda)$ to rotated loadings $bold(Lambda)^* = bold(Lambda) bold(T)$ where $bold(T)$ is transformation matrix.

  *Profound Significance of Factor Loadings ($lambda_(i j)$):*
  - *Conceptual Meaning*: Each loading represents the *strength of relationship* between observed variable $i$ and latent factor $j$
  - *Statistical Interpretation*: Correlation between variable and factor (in standardized form)
  - *Practical Meaning*: How much a 1-unit change in the factor affects the observed variable
  - *Measurement Perspective*: Quality indicator - high loadings suggest variable is a good indicator of the construct

  *Orthogonal Rotation (Varimax):*
  - Constraint: $bold(T)^top bold(T) = bold(I)$ (orthogonal transformation)
  - Objective: Maximize variance of squared loadings within each factor
  - Varimax criterion: $V = frac(1, p) sum_(j=1)^k [sum_(i=1)^p (lambda_(i j)^*)^4 - frac(1, p)(sum_(i=1)^p (lambda_(i j)^*)^2)^2]$
  - Seeks "simple structure": high loadings for few variables, low for others

  *Why Rotation Matters*: Initial loadings may be mathematically optimal but practically uninterpretable. Rotation preserves statistical properties while achieving *psychological/theoretical interpretability*.

  *Oblique Rotation (Promax):*
  - Allows factor correlations: $bold(Phi) = "Corr"(bold(F))$ may be non-diagonal
  - Pattern matrix $bold(P)$: direct effects (regression coefficients)
  - Structure matrix $bold(S) = bold(P) bold(Phi)$: correlations with factors
  - Relationship: $bold(Sigma) = bold(P) bold(Phi) bold(P)^top + bold(Psi)^2$
]

#slide(title: [Algorithm: Varimax Rotation for Simple Structure])[
  *Input:* Initial factor loadings $bold(Lambda) in RR^(p times k)$, convergence tolerance $epsilon$

  *Output:* Rotated loadings $bold(Lambda)^*$, rotation matrix $bold(T)$

  1. *Initialize Rotation Matrix*
     - $bold(T) = bold(I)_k$ (identity matrix)
     - $bold(Lambda)^* = bold(Lambda)$ (initial loadings)

  2. *Rotate for Simple Structure*
     - *Goal:* Make each factor load highly on few variables
     - Algorithm finds the best rotation angle automatically
     - *Process:*
       - Compare pairs of factors
       - Rotate them to maximize simple structure
       - Repeat until no more improvement

  3. *Check the Results*
     - Verify factors are still independent (orthogonal)
     - Loadings should be clearer: high or low, not medium
]

#slide(title: [Varimax Rotation: Simple Numerical Example])[
  *Given Factor Loadings:* 3 variables, 2 factors (before rotation)
  $ bold(Lambda) = mat(0.71, 0.45; 0.89, 0.32; 0.67, -0.58) $

  *Before Rotation:* Mixed loadings (hard to interpret)
  - Variable 1: loads on both factors (0.71, 0.45)
  - Variable 2: loads on both factors (0.89, 0.32)
  - Variable 3: loads on both factors (0.67, -0.58)

  *After Rotation:* Clearer structure
  $ bold(Lambda)^* = mat(0.78, 0.33; 0.93, 0.16; 0.56, -0.67) $
  - Variable 1: mainly Factor 1 (0.78 vs 0.33)
  - Variable 2: mainly Factor 1 (0.93 vs 0.16)
  - Variable 3: mainly Factor 2 (0.56 vs -0.67)

  *Interpretation:*
  - Factor 1: Variables 1 & 2 (maybe "Verbal ability")
  - Factor 2: Variable 3 (maybe "Spatial ability")

  *Software handles the complex rotation calculations automatically!*
]

#slide(title: [Python Implementation: Factor Rotation])[
  ```python
  import numpy as np
  from factor_analyzer import FactorAnalyzer

  # Simulate data that would produce our example loadings
  np.random.seed(42)
  X = np.random.randn(100, 3)

  # Apply Factor Analysis with rotation
  fa_no_rotation = FactorAnalyzer(n_factors=2, rotation=None)
  fa_varimax = FactorAnalyzer(n_factors=2, rotation='varimax')

  fa_no_rotation.fit(X)
  fa_varimax.fit(X)

  # Compare loadings before and after rotation
  loadings_before = fa_no_rotation.loadings_
  loadings_after = fa_varimax.loadings_

  print("Before rotation:")
  print(loadings_before)
  print("\nAfter Varimax rotation:")
  print(loadings_after)

  # Other rotation options: 'promax', 'oblimin', 'quartimax'
  fa_promax = FactorAnalyzer(n_factors=2, rotation='promax')
  fa_promax.fit(X)
  print("\nAfter Promax rotation:")
  print(fa_promax.loadings_)
  ```
]

// ============================================================================
// PART II: THEORETICAL COMPARISON
// ============================================================================

#part-slide[Part II: Theoretical Comparison]

#section-slide[Theoretical Comparison: PCA vs Factor Analysis]

#slide(title: [Conceptual Differences])[
  #align(center)[
    #table(
      columns: (1fr, 1fr),
      [*Principal Component Analysis*], [*Factor Analysis*],
      [Dimensionality reduction], [Latent variable modeling],
      [Components are linear combinations of all variables], [Factors are hypothetical constructs],
      [Explains total variance], [Explains common variance only],
      [No measurement error model], [Explicitly models unique variance],
      [Descriptive technique], [Statistical model with assumptions],
    )
  ]
]

#slide(title: [Mathematical Perspective])[
  *PCA Approach:*
  - Components are exact linear combinations: $"PC"_j = sum_(i=1)^p w_(i j) X_i$
  - Maximizes variance explained: $max "Var"("PC"_j)$ subject to orthogonality
  - All variance (including noise) is retained in the full solution

  #v(12pt)
  *Factor Analysis Approach:*
  - Models observed variables: $X_i = sum_(j=1)^k lambda_(i j) F_j + U_i$
  - Separates common factors from unique variance
  - Estimates factor loadings and unique variances simultaneously
]

#slide(title: [When Theory Guides Method Selection])[
  *Choose PCA when:*
  - Goal is data compression or visualization
  - Want to capture maximum variance with minimal components
  - No strong assumptions about underlying constructs

  #v(12pt)
  *Choose Factor Analysis when:*
  - Testing specific theories about latent constructs
  - Need to separate measurement error from true factors
  - Interpretability of factors is primary concern
  - Building models for prediction or explanation
]

#slide(title: [Quick Decision Guide: PCA vs Factor Analysis])[
  *Use PCA when:*
  - You want to reduce dimensions for visualization
  - You want to compress data (remove redundancy)
  - You don't have specific theory about hidden factors
  - You want the mathematically optimal solution

  *Use Factor Analysis when:*
  - You want to test a theory (like "intelligence" causes test performance)
  - You want to understand underlying causes
  - You want to separate "signal" from "noise"
  - You care more about interpretation than data compression

  *Quick Rules:*
  - Psychology/Education → Usually Factor Analysis
  - Data compression/Machine learning → Usually PCA
  - Exploratory analysis → Try both and compare!
]

#slide(title: [Practical Checklist for Your Analysis])[
  *Before You Start:*
  1. Do variables measure similar things? (similar units?)
  2. Do you have enough data? (at least 100 observations)
  3. Are variables correlated? (if not, both methods will fail)

  *Choosing the Method:*
  4. Do you have a theory about hidden factors? → Factor Analysis
  5. Just want to reduce dimensions? → PCA
  6. Want to understand causes? → Factor Analysis

  *Interpreting Results:*
  7. PCA: How much variance explained by first few components?
  8. FA: Do factor loadings make sense theoretically?
  9. Rotation: Does it make interpretation clearer?

  *Red Flags:*
  - Eigenvalues all very similar → No clear structure
  - Factors don't make theoretical sense → Reconsider approach
  - Too many factors needed → Maybe not suitable for these methods
]

#slide(title: [Complete Python Workflow Example])[
  ```python
  import numpy as np
  import pandas as pd
  from sklearn.decomposition import PCA
  from sklearn.preprocessing import StandardScaler
  from factor_analyzer import FactorAnalyzer
  from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
  import matplotlib.pyplot as plt

  # 1. Load and prepare data
  # X = pd.read_csv('your_data.csv')  # Your actual data
  X = np.random.randn(100, 5)  # Simulated data for demo

  # 2. Standardize if needed
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  # 3. Check suitability for factor analysis
  kmo_all, kmo_model = calculate_kmo(X_scaled)
  chi_square_value, p_value = calculate_bartlett_sphericity(X_scaled)

  print(f"KMO: {kmo_model:.3f} (>0.6 is good)")
  print(f"Bartlett's test p-value: {p_value:.3f} (<0.05 is good)")

  # 4. Determine number of factors
  pca = PCA()
  pca.fit(X_scaled)
  eigenvalues = pca.explained_variance_
  n_factors = sum(eigenvalues > 1)  # Kaiser criterion

  print(f"Suggested factors: {n_factors}")

  # 5. Perform Factor Analysis with rotation
  fa = FactorAnalyzer(n_factors=n_factors, rotation='varimax')
  fa.fit(X_scaled)

  # 6. Get and interpret results
  loadings = fa.loadings_
  communalities = fa.get_communalities()
  variance_explained = fa.get_factor_variance()

  print(f"Factor loadings:\n{loadings}")
  print(f"Communalities: {communalities}")
  print(f"Variance explained: {variance_explained[1]}")  # Proportional variance

  # 7. Compare with PCA
  pca_result = pca.transform(X_scaled)
  fa_scores = fa.transform(X_scaled)

  print("\nPCA vs FA comparison completed!")
  ```

  *This workflow covers the complete analysis pipeline!*
]

#slide(title: [Covariance Structure and Model Identification])[
  *Implied Covariance Matrix:*
  $bold(Sigma) = bold(Lambda) bold(Lambda)^top + bold(Psi)$

  where $bold(Psi) = "diag"(psi_1^2, psi_2^2, ..., psi_p^2)$ is the uniqueness matrix.

  *Pointwise Form:*
  $sigma_(i j) = sum_(l=1)^k lambda_(i l) lambda_(j l) + psi_i^2 delta_(i j)$
  where $delta_(i j) = 1$ if $i = j$, $0$ otherwise.

  *Model Parameters:*
  - Factor loadings: $k times p$ parameters in $bold(Lambda)$
  - Unique variances: $p$ parameters in $bold(Psi)$
  - Total: $k p + p$ parameters to estimate

  *Identification Constraints:*
  - Sample covariance matrix has $p(p+1)/2$ unique elements
  - For identification: $k p + p <= p(p+1)/2$
  - Simplifies to: $k <= frac(p(p-1)/2 - p, p) = frac(p-1, 2) - 1$
  - Additional constraint: Fix factor scale (unit variance) or loading scale

  *Degrees of Freedom:* $"df" = p(p+1)/2 - (k p + p - k)$
]

#slide(title: [Algorithm: Factor Analysis Suitability Tests])[
  *Input:* Correlation matrix $bold(R) in RR^(p times p)$, significance level $alpha$

  *Output:* KMO measure, Bartlett test statistic, suitability decision

  1. *Kaiser-Meyer-Olkin (KMO) Test*
     - Compute anti-image correlation matrix: $bold(A) = -"diag"(bold(R)^(-1))^(-1) bold(R)^(-1) "diag"(bold(R)^(-1))^(-1)$
     - *for* $i,j = 1$ to $p$ *do*
       - $a_(i j) = A_(i j) / sqrt(A_(i i) A_(j j))$ (off-diagonal anti-image correlations)
     - Compute KMO: $"KMO" = frac(sum_(i != j) R_(i j)^2, sum_(i != j) R_(i j)^2 + sum_(i != j) a_(i j)^2)$

  2. *Interpret KMO Value*
     - *if* KMO $>= 0.9$ *then* "marvelous"
     - *else if* KMO $>= 0.8$ *then* "meritorious"
     - *else if* KMO $>= 0.7$ *then* "middling"
     - *else if* KMO $>= 0.6$ *then* "mediocre"
     - *else if* KMO $>= 0.5$ *then* "miserable"
     - *else* "unacceptable for FA"

  3. *Bartlett's Test of Sphericity*
     - $chi^2 = -(n - 1 - frac(2p + 5, 6)) ln(|bold(R)|)$
     - Degrees of freedom: $"df" = frac(p(p-1), 2)$
     - *if* $chi^2 >$ critical value at $alpha$ *then* reject sphericity (good for FA)

  4. *Final Decision*
     - *if* KMO $>= 0.6$ *and* Bartlett significant *then* "proceed with FA"
     - *else* "reconsider data or variables"
]

#slide(title: [FA vs PCA — Quick comparison])[
  - PCA: descriptive, decomposes total variance via $S = V Lambda V^T$, components are orthogonal
  - FA: model-based, explains common variance; explicitly models uniqueness ($U_i$)
  - When to use: PCA for compression/visualization; FA when modeling latent constructs & measurement error
]
