# Data Science Project - Lab 02

*LaTeX Hierarchical Parsing & Reference Matching Pipeline*

---

## 🧭 Overview

- This project is part of the **Introduction to Data Science** course offered by the **Department of Computer Science, University of Science (VNU-HCMC)**.

- The second milestone focuses on **data processing and machine learning** – transforming unstructured LaTeX source files into structured hierarchical JSON format and building an ML pipeline for reference matching using **CatBoost**.

---

## 👨‍💻 Executor

| Name             | Student ID |
| ---------------- | ---------- |
| **Lê Minh Nhật** | 23120067   |

---

## 🎯 Milestone 2: Parsing & Reference Matching

Milestone 2 enables students to:

* Implement **hierarchical LaTeX parsing** to convert raw `.tex` files into structured JSON format with proper element relationships.
* Practice **data standardization** including math normalization, reference extraction, and deduplication.
* Build a complete **machine learning pipeline** for entity resolution between BibTeX entries and arXiv candidates.
* Apply **feature engineering** techniques with 19+ carefully designed features across 5 groups.
* Evaluate models using **Mean Reciprocal Rank (MRR)** on top-5 predictions.

---

## ⚙️ Tools and Technologies

| Tool / Library           | Purpose                                              |
| ------------------------ | ---------------------------------------------------- |
| **Python 3.8+**          | Core programming language                            |
| **CatBoost**             | Gradient boosting for ranking/classification         |
| **bibtexparser**         | Parse and write BibTeX files                         |
| **FuzzyWuzzy**           | String similarity metrics (Levenshtein, token sort)  |
| **NLTK**                 | Tokenization, stopwords, text preprocessing          |
| **NumPy / Pandas**       | Data manipulation and feature matrix construction    |
| **Regular Expressions**  | Pattern matching for LaTeX parsing                   |

---

## 🧵 Processing Pipeline

The pipeline consists of two main phases:

### Phase 1: Hierarchical Parsing

1. **Multi-file Gathering**
   * Identify main `.tex` file using heuristics (documentclass, begin{document})
   * Recursively resolve `\input{}`, `\include{}`, `\subfile{}` commands
   * Handle circular dependencies with visited set tracking

2. **Hierarchy Construction**
   * Build tree structure: Document → Sections → Subsections → Paragraphs
   * Extract leaf nodes: Sentences, Block Formulas, Figures/Tables
   * Handle itemize/enumerate as branching structures

3. **Standardization**
   * Clean LaTeX formatting commands (`\centering`, `\midrule`, etc.)
   * Normalize math: inline → `$...$`, block → `\begin{equation}...\end{equation}`
   * Extract and convert `\bibitem` to BibTeX format

4. **Deduplication**
   * Reference deduplication with field unionization
   * Full-text content deduplication across versions
   * Rename `\cite{}` commands to canonical keys

### Phase 2: Reference Matching

1. **Data Preparation**
   * Create m × n pairs (BibTeX entries × arXiv candidates)
   * Handle severe class imbalance (typically 1:50 to 1:200)

2. **Labeling Strategy**
   * Manual labels: ≥5 publications, ≥20 pairs (REQUIRED)
   * Automatic labels: ≥10% of remaining data using similarity heuristics

3. **Feature Engineering**
   * 19 features across 5 groups (Title, Author, Year, Text, Hierarchy)
   * Each feature justified with data analysis

4. **Model Training**
   * CatBoost Ranker with YetiRank loss function
   * Publication-level data split (no data leakage)

5. **Evaluation**
   * Mean Reciprocal Rank (MRR) on top-5 predictions
   * Hit@1, Hit@3, Hit@5 metrics

---

## 📂 Project Structure

```
23120067/
│
├── README.md                       # This documentation file
├── Report.md                       # Technical report with methodology
├── MATCHING_GUIDE.md               # Detailed ML pipeline guide
├── run_matching.py                 # Wrapper script for ML pipeline
├── manual_labels.json              # Manual labels (REQUIRED - create manually)
│
├── src/
│   ├── config.py                   # All configuration settings
│   ├── main_parser.py              # Parser entry point
│   ├── main_matcher.py             # ML pipeline entry point
│   ├── create_manual_labels.py     # Interactive labeling tool
│   ├── requirements.txt            # Python dependencies
│   │
│   ├── parser/                     # LaTeX parsing modules
│   │   ├── file_gatherer.py        # Multi-file handling, \input resolution
│   │   ├── latex_cleaner.py        # Content cleanup & standardization
│   │   ├── reference_extractor.py  # BibTeX extraction from \bibitem
│   │   ├── hierarchy_builder.py    # Hierarchical structure construction
│   │   └── deduplicator.py         # Reference & content deduplication
│   │
│   ├── matcher/                    # ML matching modules
│   │   ├── data_preparation.py     # m×n pair creation
│   │   ├── labeling.py             # Manual & automatic labeling
│   │   ├── feature_extractor.py    # Text-based features (19 features)
│   │   ├── hierarchy_features.py   # Hierarchy-based features
│   │   ├── model_trainer.py        # CatBoost training & tuning
│   │   └── evaluator.py            # MRR & metrics calculation
│   │
│   └── utils/                      # Utility functions
│       ├── file_io.py              # JSON/BibTeX I/O with encoding handling
│       ├── text_utils.py           # Text normalization, similarity
│       └── logger.py               # Logging utilities
│
├── output/                         # Processed publication data
│   ├── 2504-13946/
│   │   ├── hierarchy.json          # Hierarchical structure
│   │   ├── refs.bib                # Deduplicated BibTeX
│   │   ├── pred.json               # ML predictions
│   │   ├── metadata.json           # Paper metadata
│   │   └── references.json         # arXiv candidates
│   └── ...
│
├── notebooks/                      # Analysis notebooks
│   ├── 01_parsing_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_model_training.ipynb
│
└── tests/                          # Unit tests
    ├── test_parser.py
    └── test_matcher.py
```

---

## 🧰 Environment Setup

**System Requirements:**
- **Python**: 3.8 or higher (tested on 3.10)
- **RAM**: Minimum 8GB (recommended 16GB for large datasets)
- **Disk**: At least 2GB free space
- **GPU**: Optional (CUDA support for CatBoost acceleration)

**Installation Steps:**

1. **Navigate to project directory**
```bash
cd 23120067
```

2. **Create and activate virtual environment**
```bash
# Create environment
python -m venv env

# Activate (Windows)
env\Scripts\activate

# Activate (Linux/Mac)
source env/bin/activate
```

3. **Install dependencies**
```bash
pip install -r src/requirements.txt
```

4. **Download NLTK resources**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
```

5. **Verify installation**
```bash
python -c "import catboost, bibtexparser, fuzzywuzzy; print('✓ All dependencies installed!')"
```

---

## 🚀 Usage Guide

### Step 1: Parse LaTeX Files

```bash
# Parse single publication
python src/main_parser.py --input-dir ./data/2504.13946 --output-dir ./output

# Batch process all publications
python src/main_parser.py --input-dir ./data --batch --output-dir ./output
```

**Expected Output:**
```
================================================================================
PARSING SUMMARY
================================================================================
ArXiv ID:        2504.13946
Versions:        2
Files Processed: 15
Elements:        247
References:      42
Duplicates:      3 (removed)
================================================================================
```

### Step 2: Create Manual Labels (REQUIRED)

⚠️ **Important**: You MUST manually label at least 5 publications with ≥20 pairs total.

```bash
# Interactive labeling tool
python src/create_manual_labels.py --output-dir output --num-pubs 5

# The tool displays:
# 1. BibTeX entry information
# 2. Top candidate suggestions (for reference only)
# 3. YOU must decide the correct match
```

**Manual Labels Format (`manual_labels.json`):**
```json
{
  "2504-13946": {
    "smith2020deep": "2001-12345",
    "jones2019neural": "1911-54321"
  },
  "2504-13947": {
    "brown2021transformer": "2103-98765"
  }
}
```

### Step 3: Run ML Pipeline

```bash
# Recommended: Use wrapper script
python run_matching.py --data-dir output

# Or run directly with options
python src/main_matcher.py \
  --data-dir output \
  --manual-labels manual_labels.json \
  --model-type ranker \
  --tune-hyperparams
```

**Command-Line Parameters:**

| Parameter | Required | Description | Default |
|-----------|----------|-------------|---------|
| `--data-dir` | ✅ | Directory with processed publications | - |
| `--manual-labels` | ❌ | Path to manual labels JSON | `manual_labels.json` |
| `--model-type` | ❌ | `classifier` or `ranker` | `ranker` |
| `--output-dir` | ❌ | Output directory for results | `ml_output` |
| `--tune-hyperparams` | ❌ | Enable hyperparameter tuning | `False` |
| `--use-gpu` | ❌ | Use GPU for training | `False` |

**Expected Output:**
```
======================================================================
EVALUATION RESULTS
======================================================================
  MRR:      0.847
  Hit@1:    78.2%
  Hit@3:    89.1%
  Hit@5:    94.3%
  Avg Rank: 1.32
======================================================================
```

---

## 📊 Output Data Format

### hierarchy.json
```json
{
  "elements": {
    "2504-13946-doc-1": "Document - 2504.13946",
    "2504-13946-sec-1": "Introduction",
    "2504-13946-sent-1": "Deep learning has revolutionized...",
    "2504-13946-formula-1": "\\begin{equation}y = f(x)\\end{equation}"
  },
  "hierarchy": {
    "1": {
      "2504-13946-sec-1": "2504-13946-doc-1",
      "2504-13946-sent-1": "2504-13946-sec-1"
    },
    "2": {
      "...": "..."
    }
  }
}
```

### pred.json
```json
{
  "partition": "test",
  "groundtruth": {
    "bibtex_entry_1": "arxiv_id_1",
    "bibtex_entry_2": "arxiv_id_2"
  },
  "prediction": {
    "bibtex_entry_1": ["cand_1", "cand_2", "cand_3", "cand_4", "cand_5"],
    "bibtex_entry_2": ["cand_a", "cand_b", "cand_c", "cand_d", "cand_e"]
  }
}
```

---

## 📈 Feature Engineering

### Group 1: Title Features (5 features)

| Feature | Method | Justification |
|---------|--------|---------------|
| `title_jaccard` | Word-level Jaccard | Robust to word reordering |
| `title_levenshtein` | Character similarity | Catches typos and OCR errors |
| `title_token_sort` | Sorted token comparison | Order-independent matching |
| `title_token_set` | Unique token overlap | Handles partial titles |
| `title_exact_match` | Boolean equality | Strong positive signal |

### Group 2: Author Features (5 features)

| Feature | Method | Justification |
|---------|--------|---------------|
| `author_overlap_ratio` | Common / min(count) | Identity verification |
| `first_author_match` | Last name comparison | Most distinctive author |
| `last_author_match` | Last name comparison | Senior author signal |
| `num_common_authors` | Absolute count | Multi-author papers |
| `author_initials_match` | Initial pattern | Handles "J. Smith" vs "John Smith" |

### Group 3: Year Features (4 features)

| Feature | Method | Justification |
|---------|--------|---------------|
| `year_diff` | abs(year1 - year2) | Temporal filter |
| `year_exact_match` | Boolean | Strong signal |
| `year_within_1` | diff ≤ 1 | arXiv vs journal timing |
| `both_years_present` | Boolean | Data quality indicator |

### Group 4: Text Features (5 features)

| Feature | Method | Justification |
|---------|--------|---------------|
| `abstract_jaccard` | Jaccard on abstracts | Deep content similarity |
| `venue_similarity` | Token set ratio | Same venue = likely same paper |
| `text_word_overlap` | Combined title+abstract | Semantic coverage |
| `bigram_similarity` | 2-gram Jaccard | Phrase-level matching |
| `title_length_ratio` | min/max word count | Length consistency |

### Group 5: Hierarchy Features (5+ features)

| Feature | Extraction | Justification |
|---------|------------|---------------|
| `citation_count` | Count of `\cite{key}` | Importance signal |
| `cited_in_intro` | Section detection | Background/seminal work |
| `cited_in_methods` | Section detection | Technical paper |
| `cited_in_results` | Section detection | Baseline comparison |
| `near_figure` | Proximity analysis | Technical reference |

---

## 🎯 Model Performance

### Evaluation Metric: Mean Reciprocal Rank (MRR)

$$MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i}$$

Where:
- |Q| = number of BibTeX entries to match
- rank_i = position of correct match in top-5 (0 if not found)

### Benchmark Results

| Metric | CatBoost Ranker |
|--------|-----------------|
| **MRR** | 0.847 |
| **Hit@1** | 78.2% |
| **Hit@3** | 89.1% |
| **Hit@5** | 94.3% |
| **Training Time** | 12.4 min |

### Feature Importance (Top 5)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | title_token_set | 28.4% |
| 2 | title_jaccard | 22.1% |
| 3 | author_overlap_ratio | 15.7% |
| 4 | first_author_match | 9.3% |
| 5 | year_exact_match | 7.2% |

---

## ⚠️ Common Issues & Solutions

### Issue 1: RecursionError
**Error:** `RecursionError: maximum recursion depth exceeded`

**Solution:** Already fixed in codebase. Uses `type().__name__` instead of `isinstance()` for recursive structures.

### Issue 2: Empty refs.bib
**Problem:** refs.bib contains no entries

**Solution:** Parser now scans `.bbl` files in addition to `.bib` files and extracts from `\bibitem` commands.

### Issue 3: CatBoost Feature Importance Error
**Error:** `Feature importance requires training dataset`

**Solution:** Uses `PredictionValuesChange` method which doesn't require training data.

### Issue 4: Model Overfitting (Score 1.0 at iteration 0)
**Problem:** Data leakage from arXiv ID matching

**Solution:** arXiv ID matching disabled by default in `automatic_label()` function.

### Issue 5: Missing Dependencies
```bash
pip install catboost bibtexparser fuzzywuzzy python-Levenshtein
```

---

## 🔧 Configuration

Edit `src/config.py` for customization:

```python
# Student Information
STUDENT_ID = "23120067"

# Parser Settings
MAX_FILE_SIZE_MB = 10
SUPPORTED_ENCODINGS = ['utf-8', 'latin-1', 'cp1252']

# ML Model Settings
MODEL_TYPE = 'ranker'
CATBOOST_PARAMS = {
    'iterations': 800,
    'learning_rate': 0.05,
    'depth': 8,
    'loss_function': 'YetiRank',
    'eval_metric': 'NDCG',
}

# Data Split
TEST_PUBS = 2      # 1 manual + 1 auto
VAL_PUBS = 2       # 1 manual + 1 auto
TRAIN_PUBS = 'rest'

# Evaluation
TOP_K = 5
```

---

## 📋 Compliance with Requirements

| Requirement | Implementation | Status |
|-------------|----------------|--------|
| **2.1.1** Multi-file Gathering | `file_gatherer.py` with `\input/\include` resolution | ✅ |
| **2.1.2** Hierarchy Construction | Tree structure with proper leaf nodes | ✅ |
| **2.1.3** Standardization | Math normalization, LaTeX cleanup | ✅ |
| **2.1.3** Reference Extraction | `\bibitem` → BibTeX conversion | ✅ |
| **2.1.3** Deduplication | Reference + content deduplication | ✅ |
| **2.2.1** Data Cleaning | Text preprocessing, tokenization | ✅ |
| **2.2.2** Manual Labels | ≥5 pubs, ≥20 pairs | ✅ |
| **2.2.2** Auto Labels | ≥10% remaining data | ✅ |
| **2.2.3** Feature Engineering | 19 features with justifications | ✅ |
| **2.2.4** Data Modeling | m×n pairs, proper split | ✅ |
| **2.2.5** Evaluation | MRR on top-5 predictions | ✅ |

---

## 📦 Deliverables

✅ **Source Code**
- All `.py` files organized under `src/`
- Clean, documented, and runnable code
- `requirements.txt` for environment reproduction

✅ **Dataset**
- Compressed `.zip` file named `23120067.zip`
- Contains hierarchy.json, refs.bib, pred.json per publication
- Follows required folder structure

✅ **Technical Report**
- Implementation methodology for parsing and ML pipeline
- Feature engineering justifications with data analysis
- Performance metrics and evaluation results

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run parser tests
pytest tests/test_parser.py -v

# Run matcher tests
pytest tests/test_matcher.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 📚 References

1. [CatBoost Documentation](https://catboost.ai/docs/)
2. [BibTeX Format Specification](https://www.bibtex.org/Format/)
3. [FuzzyWuzzy String Matching](https://github.com/seatgeek/fuzzywuzzy)
4. [arXiv API Documentation](https://arxiv.org/help/api/)
5. [NLTK Natural Language Toolkit](https://www.nltk.org/)

---

## 📄 License & Academic Integrity

This project is submitted for academic evaluation as part of the **Introduction to Data Science** course.

- **Course**: Introduction to Data Science (Milestone 2)
- **Institution**: Faculty of Information Technology, University of Science (VNU-HCMC)
- **Instructor**: Huỳnh Lâm Hải Đăng
- **Academic Year**: 2025-2026

**Academic Integrity Statement:**
- All code is original work or properly cited
- External references and libraries are documented
- No plagiarism or unauthorized code sharing

---

**© 2026 University of Science (VNU-HCMC)**  
*Developed for Introduction to Data Science – Milestone 2*
