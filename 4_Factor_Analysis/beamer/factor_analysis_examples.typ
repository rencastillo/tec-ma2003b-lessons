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
  title: "Factor Analysis: Examples and Applications",
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
  #text(size: 28pt, weight: "bold", fill: tec-blue)[Factor Analysis: Examples and Applications]
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
// PART III: FACTOR ANALYSIS EXAMPLES
// ============================================================================


#section-slide[Example 1B: Educational Assessment Factor Analysis]

#slide(title: [Educational Assessment: Dataset Overview])[
  *Research Question:* What are the underlying cognitive abilities measured by standardized tests?

  *Dataset:* 500 high school students, 8 standardized test scores
  - Math Reasoning (MR), Math Computation (MC)
  - Reading Comprehension (RC), Vocabulary (V)
  - Spatial Visualization (SV), Pattern Recognition (PR)
  - Writing Skills (WS), Logical Reasoning (LR)

  *Theoretical Expectation:* Three latent abilities
  - *Quantitative Ability*: Math tests should load highly
  - *Verbal Ability*: Language tests should load highly
  - *Spatial Ability*: Visual-spatial tests should load highly

  *Factor Analysis Focus:*
  - Model latent cognitive constructs
  - Separate true ability from measurement error
  - Test theoretical structure of intelligence
]

#slide(title: [FA Model Setup and Suitability Tests])[
  *Data Suitability Assessment:*
  - KMO Measure: 0.84 (meritorious)
  - Bartlett's Test: $chi^2 = 1247.3$, $p < 0.001$ (sphericity rejected)
  - All individual KMO values > 0.7 (adequate sampling)

  *Factor Analysis Model:*
  ```
  X_i = λ_i1 × F_1 + λ_i2 × F_2 + λ_i3 × F_3 + U_i
  ```
  - 3 common factors (based on Kaiser criterion and theory)
  - Principal Axis Factoring with Varimax rotation
  - 8 unique factors (measurement error + specific abilities)

  *Why FA over PCA here:*
  - Testing specific theory about cognitive structure
  - Need to model measurement error in test scores
  - Want interpretable factors representing abilities
  - Focus on common variance shared across tests
]

#slide(title: [Factor Loadings Matrix (After Varimax Rotation)])[
  #align(center)[
    #table(
      columns: (2fr, 1fr, 1fr, 1fr, 1fr),
      [*Test*], [*Factor 1*], [*Factor 2*], [*Factor 3*], [*Communality*],
      [Math Reasoning], [**.82**], [.31], [.15], [.78],
      [Math Computation], [**.89**], [.22], [.08], [.84],
      [Reading Comprehension], [.24], [**.91**], [.19], [.92],
      [Vocabulary], [.19], [**.85**], [.23], [.81],
      [Spatial Visualization], [.18], [.21], [**.87**], [.83],
      [Pattern Recognition], [.25], [.15], [**.79**], [.71],
      [Writing Skills], [.31], [**.73**], [.28], [.71],
      [Logical Reasoning], [**.64**], [.45], [.33], [.72],
    )
  ]

  *Factor Interpretation:*
  - *Factor 1*: Quantitative Ability (Math tests + Logic)
  - *Factor 2*: Verbal Ability (Language tests)
  - *Factor 3*: Spatial Ability (Visual-spatial tests)

  *Key FA Insights:*
  - High communalities (0.71-0.92) indicate good factor representation
  - Clear simple structure after rotation
  - Logical Reasoning loads on both Quantitative and Verbal (as expected)
]

#slide(title: [Uniqueness and Model Diagnostics])[
  *Uniqueness Analysis:*
  #align(center)[
    #table(
      columns: (2fr, 1fr, 1fr),
      [*Test*], [*Uniqueness*], [*Interpretation*],
      [Math Reasoning], [.22], [Low measurement error],
      [Math Computation], [.16], [Very reliable],
      [Reading Comprehension], [.08], [Excellent reliability],
      [Vocabulary], [.19], [Good reliability],
      [Spatial Visualization], [.17], [Good reliability],
      [Pattern Recognition], [.29], [Some specific variance],
      [Writing Skills], [.29], [Some measurement error],
      [Logical Reasoning], [.28], [Complex construct],
    )
  ]

  *Model Fit Assessment:*
  - Total variance explained: 76.3% (common factors only)
  - Average communality: 0.79 (excellent)
  - All uniqueness values < 0.3 (acceptable measurement quality)
  - Residual correlations mostly < 0.05 (good model fit)

  *FA vs PCA Comparison:*
  - FA explains less total variance (76% vs 85%) but focuses on reliable variance
  - FA provides cleaner factor interpretation due to rotation
  - FA separates measurement error, PCA includes it in components
]

#section-slide[Example 2B: European Stock Markets Factor Analysis]

#slide(title: [Stock Markets: Dataset and Research Question])[
  *Research Question:* What are the underlying economic factors driving European stock market movements?

  *Dataset:* Daily returns (252 trading days) for 8 major indices
  - FTSE 100 (UK), DAX (Germany), CAC 40 (France), IBEX 35 (Spain)
  - AEX (Netherlands), BEL 20 (Belgium), ATX (Austria), PSI 20 (Portugal)

  *Economic Theory:* Common factors might represent
  - *Market-wide sentiment* (systematic risk affecting all markets)
  - *Regional economic integration* (EU economic policies)
  - *Currency effects* (Euro vs non-Euro countries)

  *Factor Analysis Approach:*
  - Model common economic forces across markets
  - Separate systematic risk from country-specific effects
  - Identify diversification opportunities (unique variances)
  - Test economic theory about market integration
]

#slide(title: [FA Results: Market Factor Structure])[
  *Model Specifications:*
  - Principal Axis Factoring, 2 factors retained (Kaiser + Scree test)
  - Promax rotation (allows correlated factors - more realistic for economics)

  *Rotated Factor Pattern Matrix:*
  #align(center)[
    #table(
      columns: (2fr, 1fr, 1fr, 1fr),
      [*Market Index*], [*Factor 1*], [*Factor 2*], [*Communality*],
      [FTSE 100 (UK)], [**.73**], [.21], [.58],
      [DAX (Germany)], [**.91**], [-.08], [.84],
      [CAC 40 (France)], [**.88**], [.15], [.80],
      [IBEX 35 (Spain)], [**.67**], [**.52**], [.72],
      [AEX (Netherlands)], [**.85**], [.22], [.77],
      [BEL 20 (Belgium)], [**.79**], [.31], [.72],
      [ATX (Austria)], [**.58**], [**.48**], [.57],
      [PSI 20 (Portugal)], [.43], [**.71**], [.69],
    )
  ]

  *Factor Correlation:* $r = 0.34$ (moderate positive correlation)

  *Economic Interpretation:*
  - *Factor 1*: Core EU Market Integration (Germany, France, Netherlands)
  - *Factor 2*: Peripheral Market Dynamics (Spain, Portugal, Austria)
]

#slide(title: [Economic Insights from FA Model])[
  *Unique Variance Analysis (Risk Diversification):*
  - UK (FTSE): 42% unique variance (Brexit effects, Sterling currency)
  - Austria (ATX): 43% unique variance (smaller economy, local factors)
  - Germany (DAX): 16% unique variance (highly integrated with EU)

  *Practical Investment Implications:*
  1. *Systematic Risk*: ~65% average communality means most risk is systematic
  2. *Diversification*: UK and Austria offer best diversification benefits
  3. *Core vs Periphery*: Two-factor structure confirms economic theory
  4. *Currency Effects*: UK's unique variance partly reflects non-Euro status

  *Model Validation:*
  - Factor scores correlate with EU economic indicators
  - Crisis periods show increased factor loadings (contagion effect)
  - Model explains 69% of total market variance
  - Residuals show minimal autocorrelation (good model fit)

  *FA vs PCA for Finance:*
  - FA focuses on systematic risk (relevant for portfolio theory)
  - PCA would mix systematic and idiosyncratic risks
  - FA provides better economic interpretation of market structure
]

#section-slide[Example 3B: Kuiper Belt Objects Factor Analysis]

#slide(title: [Kuiper Belt Objects: Astronomical Dataset])[
  *Research Question:* What are the fundamental physical processes that shaped Kuiper Belt Object (KBO) characteristics?

  *Dataset:* 347 Kuiper Belt Objects with 7 orbital/physical parameters
  - Semi-major axis (a), Eccentricity (e), Inclination (i)
  - Perihelion distance (q), Aphelion distance (Q)
  - Absolute magnitude (H), Estimated diameter (D)

  *Astronomical Theory:* Two main formation processes
  - *Primordial Disk Structure*: Original solar nebula properties
  - *Dynamical Evolution*: Gravitational perturbations over 4.5 billion years

  *Factor Analysis Goals:*
  - Identify latent physical processes from observable parameters
  - Separate primordial signals from evolutionary effects
  - Model measurement uncertainties in astronomical observations
  - Test theoretical models of outer solar system formation
]

#slide(title: [FA Model: Cosmic Factor Structure])[
  *Preprocessing and Suitability:*
  - Log-transformed skewed variables (a, D)
  - Standardized all parameters (different units/scales)
  - KMO = 0.73 (middling), Bartlett $p < 0.001$ (suitable for FA)

  *Factor Analysis Results (Varimax Rotation):*
  #align(center)[
    #table(
      columns: (2fr, 1fr, 1fr, 1fr),
      [*Orbital Parameter*], [*Factor 1*], [*Factor 2*], [*Communality*],
      [Semi-major axis (log a)], [**.89**], [.12], [.81],
      [Aphelion distance (log Q)], [**.94**], [.08], [.89],
      [Perihelion distance (q)], [**.76**], [-.35], [.70],
      [Eccentricity (e)], [.23], [**.87**], [.81],
      [Inclination (i)], [-.08], [**.82**], [.68],
      [Absolute magnitude (H)], [-.67], [.31], [.55],
      [Diameter (log D)], [.71], [-.28], [.58],
    )
  ]

  *Astrophysical Interpretation:*
  - *Factor 1*: Distance/Size Relationship (primordial disk structure)
  - *Factor 2*: Dynamical Excitation (scattering events, orbital evolution)
]

#slide(title: [Astronomical Insights and Model Implications])[
  *Factor 1 - Primordial Disk Structure (62% of common variance):*
  - Larger objects found at greater distances (mass segregation)
  - Reflects original solar nebula density gradient
  - Size-distance correlation preserved over 4.5 Gyr

  *Factor 2 - Dynamical Evolution (31% of common variance):*
  - High eccentricity and inclination cluster together
  - Indicates gravitational scattering by giant planets
  - Separates "cold" vs "hot" KBO populations

  *Unique Variance Analysis:*
  - Absolute magnitude (H): 45% unique (observational bias effects)
  - Perihelion distance: 30% unique (specific orbital resonances)
  - Most orbital elements well-explained by two factors

  *Scientific Validation:*
  - Factor structure matches theoretical predictions
  - Cold Classical KBOs score low on Factor 2
  - Scattered Disk Objects score high on both factors
  - Results consistent with Nice Model of solar system evolution

  *Why FA over PCA for Astronomy:*
  - Models observational uncertainties explicitly
  - Tests specific theoretical hypotheses about formation
  - Separates physical processes from measurement noise
  - Provides meaningful astrophysical factor interpretation
]

#section-slide[Example 4B: Hospital Health Outcomes Factor Analysis]

#slide(title: [Hospital Quality: Healthcare Dataset])[
  *Research Question:* What are the underlying dimensions of hospital quality that affect patient outcomes?

  *Dataset:* 285 hospitals with 9 quality indicators
  - 30-day mortality rates (Heart Attack, Heart Failure, Pneumonia)
  - 30-day readmission rates (same 3 conditions)
  - Patient satisfaction scores (Communication, Responsiveness, Pain Management)

  *Healthcare Theory:* Quality dimensions might include
  - *Clinical Excellence*: Medical competency, protocols, outcomes
  - *Patient Experience*: Communication, comfort, service quality
  - *System Efficiency*: Care coordination, discharge planning

  *Factor Analysis Applications:*
  - Identify core quality constructs for hospital evaluation
  - Separate clinical performance from patient satisfaction
  - Model measurement error in quality metrics
  - Guide quality improvement initiatives and resource allocation
]

#slide(title: [FA Results: Healthcare Quality Factors])[
  *Model Configuration:*
  - Maximum Likelihood estimation (for significance tests)
  - 2 factors retained (eigenvalue > 1, theoretical fit)
  - Promax rotation (quality dimensions may correlate)

  *Factor Pattern Matrix:*
  #align(center)[
    #table(
      columns: (2fr, 1fr, 1fr, 1fr),
      [*Quality Indicator*], [*Factor 1*], [*Factor 2*], [*Communality*],
      [Heart Attack Mortality], [**.78**], [-.12], [.62],
      [Heart Failure Mortality], [**.71**], [.08], [.51],
      [Pneumonia Mortality], [**.69**], [.15], [.50],
      [Heart Attack Readmission], [**.83**], [-.05], [.69],
      [Heart Failure Readmission], [**.77**], [.22], [.64],
      [Pneumonia Readmission], [**.65**], [.31], [.52],
      [Communication Score], [-.21], [**.89**], [.84],
      [Responsiveness Score], [-.15], [**.82**], [.69],
      [Pain Management Score], [.08], [**.74**], [.55],
    )
  ]

  *Factor Correlation:* $r = -0.41$ (better clinical outcomes associated with better patient experience)

  *Healthcare Quality Interpretation:*
  - *Factor 1*: Clinical Performance (mortality and readmission rates)
  - *Factor 2*: Patient Experience (satisfaction and communication)
]

#slide(title: [Healthcare Policy Implications])[
  *Clinical Performance Factor Analysis:*
  - High communalities (0.50-0.69) suggest reliable quality measurement
  - Hospitals strong in one clinical area tend to excel in others
  - Readmission rates more reliable indicators than mortality (higher loadings)

  *Patient Experience Insights:*
  - Communication most important (λ = 0.89) for patient satisfaction
  - All satisfaction measures highly correlated (common service quality)
  - 55-84% of satisfaction variance explained by common factor

  *Unique Variance Components:*
  - Heart Failure Mortality: 49% unique (condition-specific factors)
  - Pain Management: 45% unique (specialty-specific protocols)
  - Other indicators: 31-48% unique variance

  *Quality Improvement Strategies:*
  1. *Systemic Approach*: Factors are correlated - improve both simultaneously
  2. *Clinical Focus*: Target readmission reduction (high factor loadings)
  3. *Patient Experience*: Prioritize communication training
  4. *Measurement*: Consider composite scores based on factor weights

  *FA Advantages for Healthcare:*
  - Models measurement error in quality metrics
  - Identifies underlying quality constructs
  - Supports evidence-based quality improvement
  - Enables fair hospital comparisons accounting for measurement uncertainty
]

#slide(title: [Summary: Factor Analysis Across Domains])[
  *Common FA Benefits Across All Examples:*

  1. *Theoretical Validation*: Tests specific theories about latent constructs
     - Education: Cognitive ability structure
     - Finance: Economic integration theory
     - Astronomy: Solar system formation models
     - Healthcare: Quality dimension theory

  2. *Measurement Error Modeling*: Separates signal from noise
     - Explicit uniqueness estimates
     - Better reliability assessment
     - More accurate factor interpretation

  3. *Interpretable Results*: Rotation achieves simple structure
     - Clear factor loadings patterns
     - Meaningful construct identification
     - Actionable insights for practitioners

  *When Factor Analysis Excels:*
  - Theory-driven research questions
  - Need to model measurement error
  - Focus on common variance and latent constructs
  - Interpretation more important than data compression

  *Key Insight*: Factor Analysis transforms correlation patterns into theoretically meaningful latent constructs, making it invaluable for scientific understanding and practical decision-making across diverse fields.
]