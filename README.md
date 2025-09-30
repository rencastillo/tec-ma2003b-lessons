# Course Lessons Directory

This directory contains all course materials for MA2003B - Análisis Multivariado, with a focus on modern presentation systems using Typst.

## 📚 Documentation

### Quick Start
- **[TYPST_CHEATSHEET.md](TYPST_CHEATSHEET.md)** - Essential Typst commands and syntax reference
- **[TYPST_GUIDE.md](TYPST_GUIDE.md)** - Comprehensive guide to Typst for course development

### Templates
- **[Course Presentation Template](shared/templates/course_presentation_template.typ)** - Ready-to-use presentation template

## 📖 Current Lessons

### Chapter 4: Factor Analysis ✅ Complete

- **Location**: `4_Factor_Analysis/`
- **Presentation**: `presentation/` (Typst format)
- **Examples**: `examples/` (4 domain-specific examples)
- **Snippets**: `snippets/` (5 interactive Jupyter notebooks)
- **Notes**: `notes/` (study guides and additional materials)
- **Status**: Fully migrated from LaTeX to Typst + Interactive notebooks
- **Topics**: PCA, Factor Analysis, 4 comprehensive examples, method comparisons, interactive learning

### Chapter 5: Discriminant Analysis ✅ Complete

- **Location**: `5_Discriminant_Analysis/`
- **Presentation**: `presentation/` (Typst format)
- **Examples**: `examples/` (3 domain-specific examples)
- **Status**: Complete with presentation and examples
- **Topics**: Linear/Quadratic Discriminant Analysis, classification methods

## 🚀 Getting Started with Typst

### 1. Installation Check

```bash
# Verify Typst is installed
~/.local/bin/typst --version
```

### 2. Create New Presentation

```bash
# Copy template
cp lessons/shared/templates/course_presentation_template.typ my_presentation.typ

# Edit content (use VS Code with Typst extensions)
code my_presentation.typ

# Compile
typst compile my_presentation.typ
```

### 3. Development Workflow

```bash
# Watch mode (auto-compile on changes)
typst watch my_presentation.typ

# Open PDF viewer (will auto-refresh)
evince my_presentation.pdf &
```

## 📋 Migration Status

### Completed ✅

- [x] **Factor Analysis** - Full presentation (419 slides → Typst)
- [x] **Performance improvement** - 47x faster compilation (9.4s → 0.2s)
- [x] **Template system** - Reusable course presentation template
- [x] **Documentation** - Complete guides and cheat sheets

### Future Chapters (To be developed)

- [ ] **Multivariate Regression**
- [ ] **Discriminant Analysis**
- [ ] **Cluster Analysis**
- [ ] **Advanced Topics**## 🎯 Key Benefits Achieved

| Aspect | LaTeX | Typst | Improvement |
|--------|--------|--------|-------------|
| **Compilation Speed** | 9.4 seconds | 0.2 seconds | 47x faster |
| **Error Messages** | Cryptic | Clear & helpful | Much better |
| **Syntax** | Complex | Clean & readable | Easier maintenance |
| **Learning Curve** | Steep | Gentle | Faster onboarding |
| **File Size** | 838KB | 103KB | Smaller output |

## 📁 Directory Structure

```
lessons/
├── README.md                          # This overview
├── requirements.txt                   # Python dependencies for examples
├── TYPST_GUIDE.md                     # Complete Typst guide
├── TYPST_CHEATSHEET.md               # Quick reference
├── shared/
│   └── templates/
│       └── course_presentation_template.typ
├── 4_Factor_Analysis/                 # ✅ Complete
│   ├── README.md                      # Chapter-specific docs
│   ├── presentation/
│   │   └── factor_analysis_presentation.typ    # Main Typst file
│   ├── examples/                      # 4 domain examples (education, finance, astronomy, healthcare)
│   │   ├── educational_example/       # PCA vs FA comparison
│   │   ├── invest_example/            # European stock markets
│   │   ├── kuiper_example/            # Kuiper Belt objects
│   │   ├── hospitals_example/         # Hospital quality assessment
│   │   └── EXAMPLES_OVERVIEW.md       # Comparative guide
│   ├── snippets/                      # Interactive Jupyter notebooks
│   │   ├── 01_pca_basic_example.ipynb # Basic PCA concepts
│   │   ├── 02_component_retention.ipynb # Component selection methods
│   │   ├── 03_factor_analysis_basic.ipynb # Basic factor analysis
│   │   ├── 04_factor_rotation.ipynb   # Rotation techniques
│   │   ├── 05_complete_workflow.ipynb # End-to-end workflow
│   │   ├── README.md                  # Notebook usage guide
│   │   ├── TESTING_RESULTS.md         # Test results
│   │   └── test_all_snippets.py       # Validation script
│   └── notes/                         # Study guides and materials
└── 5_Discriminant_Analysis/           # ✅ Complete
    ├── README.md                      # Chapter-specific docs
    ├── presentation/
    │   └── discriminant_analysis_presentation.typ
    ├── examples/                      # 3 domain examples (marketing, quality control, sports)
    └── notes/                         # Study guides
```

## 🔧 VS Code Setup (Recommended)

1. **Install Extensions**:
   - "Typst LSP" - Language server support
   - "Typst Preview" - Live preview

2. **Keybinding**: Ctrl+Shift+P → "Typst Preview"

3. **Workflow**:
   - Edit `.typ` file in VS Code
   - See live preview in sidebar
   - PDF updates automatically

## 💡 Best Practices

### File Organization

- Use the template for new presentations
- Keep presentation-specific assets in same directory
- Preserve LaTeX files in `legacy_` folders during migration

### Content Development

- Start with template structure
- Use `#slide(title: [Title])[content]` for regular slides
- Use `#section-slide[Title]` for section dividers
- Test compilation frequently during development

### Collaboration

- Typst files are plain text (version control friendly)
- Share `.typ` files and generated PDFs
- Use consistent formatting and functions

## 📞 Getting Help

- **Quick answers**: See [TYPST_CHEATSHEET.md](TYPST_CHEATSHEET.md)
- **Detailed guide**: See [TYPST_GUIDE.md](TYPST_GUIDE.md)
- **Template example**: `shared/templates/course_presentation_template.typ`
- **Working example**: `4_Factor_Analysis/presentation/factor_analysis_presentation.typ`

## 🎉 Success Story

The Factor Analysis chapter demonstrates the full potential of migrating to Typst:

- **419 slides** successfully migrated
- **All mathematical content** properly rendered
- **47x faster compilation**
- **Professional output** maintained
- **Legacy preserved** for future reference

This establishes Typst as the recommended system for future course development.
