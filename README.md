# Lab 2: Introduction to Data Science
**Student ID**: 23120067

## 📋 Project Overview

This project implements a complete pipeline for:
1. **Hierarchical LaTeX Parsing**: Convert raw LaTeX sources into structured JSON format
2. **Reference Matching**: Use machine learning to match BibTeX entries with arXiv references

## 🏗️ Project Structure
```
23120067/
├── src/
│   ├── parser/              # LaTeX parsing modules
│   ├── matcher/             # ML matching modules
│   ├── utils/               # Utility functions
│   ├── config.py            # Configuration
│   ├── main_parser.py       # Run parser
│   └── main_matcher.py      # Run ML pipeline
├── notebooks/               # Jupyter notebooks (optional)
├── tests/                   # Unit tests (optional)
├── requirements.txt
├── README.md
└── Report.pdf
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r src/requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Download SpaCy model (if using SpaCy)
python -m spacy download en_core_web_sm
```

### 2. Run Parser
```bash
# Parse all publications
python src/main_parser.py \
    --input-dir ./23120067 \
    --output-dir ./output

# Parse single publication
python src/main_parser.py \
    --input-dir ./23120067/2504-13946 \
    --output-dir ./output/2504-13946
```

**Expected Output**:
- `hierarchy.json`: Hierarchical structure
- `refs.bib`: Unified BibTeX file
- Logs in console

### 3. Run ML Pipeline
```bash
# Train and evaluate
python src/main_matcher.py \
    --data-dir ./23120067 \
    --output-dir ./output \
    --model-type ranker

# Options:
#   --model-type: 'classifier' or 'ranker' (default: ranker)
```

**Expected Output**:
- Trained model: `output/model.cbm`
- Predictions: `<pub-id>/pred.json` for each test publication
- MRR score in console

## 📊 Data Format

### Input Structure (from Lab 1)
```
23120067/
├── 2504-13946/
│   ├── metadata.json
│   ├── references.json
│   └── tex/
│       ├── 2504-13946v1/
│       │   ├── main.tex
│       │   └── ...
│       └── 2504-13946v2/
│           └── ...
└── ...
```

### Output Structure
```
23120067/
├── 2504-13946/
│   ├── hierarchy.json    ← NEW
│   ├── refs.bib          ← NEW
│   ├── metadata.json
│   ├── references.json
│   └── pred.json         ← NEW (if used for ML)
└── ...
```

## 🔧 Configuration

Edit `src/config.py` to customize:
- Student ID
- Model parameters
- Feature thresholds
- Data split ratios

Example:
```python
STUDENT_ID = "23120067"
MODEL_TYPE = 'ranker'
CATBOOST_PARAMS = {
    'iterations': 200,
    'learning_rate': 0.1,
    'depth': 6
}
```

## 📈 Features Used

### Traditional Features
- Title similarity (Jaccard, Levenshtein, TF-IDF cosine)
- Author matching (overlap ratio, first/last author)
- Year difference
- Text embeddings (optional)

### Hierarchy-Based Features (NEW!)
- Citation frequency
- Citation sections (intro/methods/results)
- Citation depth in hierarchy
- Proximity to figures/tables
- Co-citation patterns

## 🎯 Model Performance

| Metric | Score |
|--------|-------|
| MRR    | 0.XXX |
| Hit@1  | X.XX% |
| Hit@3  | X.XX% |
| Hit@5  | X.XX% |

*(Fill in after running evaluation)*

## 🐛 Troubleshooting

### Issue 1: TexSoup parsing errors
**Solution**: Check for malformed LaTeX commands. Add error handling in `hierarchy_builder.py`

### Issue 2: Memory error during feature extraction
**Solution**: Process publications in batches. Reduce TF-IDF max_features in `config.py`

### Issue 3: CatBoost installation fails
**Solution**: 
```bash
# Try with conda
conda install -c conda-forge catboost

# Or build from source
pip install catboost --no-binary catboost
```

## 📚 Key Implementation Details

### 1. Reference Deduplication with Citation Renaming
The deduplicator finds duplicate references and automatically renames all `\cite{}` commands:
```python
# Before
\cite{lipton2018interpretability}
\cite{lipton2018mythos}

# After (both refer to same entry)
\cite{lipton2018mythos}
\cite{lipton2018mythos}
```

### 2. Itemize as Branching Structure
Itemize blocks are parsed as hierarchical elements:
```latex
\begin{itemize}
    \item First point
    \item Second point
\end{itemize}
```

Becomes:
```
itemize-block-1 (parent)
  ├── item-1 (child)
  └── item-2 (child)
```

### 3. m×n Pairs Generation
For each publication with m BibTeX entries and n candidates, we create m×n pairs:
```python
# Example: 10 BibTeX × 50 candidates = 500 pairs
pairs = [
    (bibtex_1, candidate_1),
    (bibtex_1, candidate_2),
    ...
    (bibtex_10, candidate_50)
]
```

## 🎥 Demonstration Video

**Link**: [YouTube Video](https://youtube.com/...)

**Contents**:
- Environment setup
- Running parser
- Running ML pipeline
- Results visualization

**Duration**: 4-5 minutes

## 📝 Report

See `Report.pdf` for detailed explanation of:
- Implementation approach
- Feature engineering rationale
- Model selection justification
- Results analysis
- Statistics and insights

## 📧 Contact

**Student**: [Your Name]  
**Student ID**: 23120067  
**Email**: [your-email]

For questions about this implementation, please contact the instructor:
- **Huỳnh Lâm Hải Đăng**: hlhdang@fit.hcmus.edu.vn

## 🙏 Acknowledgments

- LaTeX parsing: TexSoup library
- Machine Learning: CatBoost
- Text processing: NLTK, scikit-learn
- Data from: arXiv, Semantic Scholar API

## 📄 License

This project is for educational purposes only.