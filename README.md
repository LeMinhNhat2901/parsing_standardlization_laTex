# Lab 2: Introduction to Data Science - LaTeX Parsing & Reference Matching

**Student ID**: 23120067  
**Course**: Introduction to Data Science  
**Lab**: Lab 2 - Parsing & Standardization

---

## 📋 Project Overview

This project implements a complete data science pipeline for scientific paper analysis:

### Part 1: Hierarchical LaTeX Parsing (Requirement 2.1)
Transform raw LaTeX source files from arXiv into structured, hierarchical JSON format suitable for ML applications.

### Part 2: Reference Matching with ML (Requirement 2.2)
Build a machine learning pipeline using CatBoost to match BibTeX references with arXiv paper candidates.

⚠️ **QUAN TRỌNG về Manual Labeling:**
- Theo text2.txt Section 2.2.2: "**Manually label** references"
- Sinh viên **PHẢI TỰ TAY** label ít nhất 5 publications (≥20 pairs total)
- **KHÔNG** được sử dụng automatic matching cho manual labels!

### Key Features
- ✅ Multi-version LaTeX parsing with `\input`/`\include` resolution
- ✅ Hierarchical structure extraction (sections → sentences → formulas)
- ✅ Automatic reference deduplication with `\cite{}` renaming
- ✅ 37 engineered features across 5 groups for reference matching
- ✅ CatBoost Classifier/Ranker with hyperparameter tuning
- ✅ MRR evaluation metric for top-5 predictions
- ✅ **Interactive manual labeling tool** (tuân thủ yêu cầu 2.2.2)

---

## 🚀 Quick Start

### 1. Tạo Manual Labels (BẮT BUỘC TỰ TAY)

```bash
# Chạy interactive labeling tool
python src/create_manual_labels.py --output-dir output --num-pubs 5

# Tool sẽ hiển thị BibTeX entries và candidates
# BẠN phải tự xem xét và chọn match đúng
# Output: manual_labels.json
```

### 2. Chạy ML Pipeline

```bash
# Cách 1: Sử dụng wrapper (Khuyến nghị)
python run_matching.py --data-dir output

# Cách 2: Chạy trực tiếp
python src/main_matcher.py --data-dir output --manual-labels manual_labels.json
```

📖 **Chi tiết:** Xem [MATCHING_GUIDE.md](MATCHING_GUIDE.md)

---

## 🏗️ Project Structure

```
23120067/
├── src/
│   ├── parser/                  # LaTeX parsing modules
│   │   ├── __init__.py
│   │   ├── file_gatherer.py     # Multi-file handling, \input resolution
│   │   ├── latex_cleaner.py     # Content cleanup & standardization
│   │   ├── reference_extractor.py # BibTeX extraction from \bibitem
│   │   ├── hierarchy_builder.py # Hierarchical structure construction
│   │   └── deduplicator.py      # Reference & content deduplication
│   │
│   ├── matcher/                 # ML matching modules
│   │   ├── __init__.py
│   │   ├── data_preparation.py  # m×n pair creation
│   │   ├── labeling.py          # Manual & automatic labeling
│   │   ├── feature_extractor.py # Text-based features (37 features)
│   │   ├── hierarchy_features.py # Hierarchy-based features
│   │   ├── model_trainer.py     # CatBoost training & tuning
│   │   └── evaluator.py         # MRR & metrics calculation
│   │
│   ├── create_manual_labels.py  # ⭐ Interactive manual labeling tool
│   ├── main_matcher.py          # ⭐ ML pipeline entry point
│   └── config.py                # Configuration
│
├── run_matching.py              # Wrapper script (tự động xử lý paths)
├── manual_labels.json           # Manual labels (TỰ TAY TẠO)
├── MATCHING_GUIDE.md            # 📖 Hướng dẫn chi tiết
└── README.md                    # This file
````
│   ├── utils/                   # Utility functions
│   │   ├── __init__.py
│   │   ├── file_io.py           # JSON/BibTeX I/O
│   │   ├── text_utils.py        # Text normalization, similarity
│   │   └── logger.py            # Logging utilities
│   │
│   ├── config.py                # All configuration settings
│   ├── main_parser.py           # Parser entry point
│   ├── main_matcher.py          # ML pipeline entry point
│   └── requirements.txt         # Python dependencies
│
├── tests/                       # Unit tests
│   ├── test_parser.py           # Parser module tests
│   └── test_matcher.py          # Matcher module tests
│
├── notebooks/                   # Analysis notebooks
│   ├── 01_parsing_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_model_training.ipynb
│
├── README.md                    # This file
└── Report.pdf                   # Detailed project report
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ (tested on Python 3.10)
- Git
- 8GB+ RAM recommended

### 1. Environment Setup

```bash
# Clone repository (if applicable)
git clone https://github.com/LeMinhNhat2901/parsing_standardlization_laTex.git
cd parsing_standardlization_laTex

# Create virtual environment
python -m venv env

# Activate environment
# On Windows:
env\Scripts\activate
# On Linux/Mac:
source env/bin/activate

# Install dependencies
pip install -r src/requirements.txt

# Download required NLP resources
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
```

### 2. Run Parser Pipeline

```bash
# Parse a single publication
python src/main_parser.py --input-dir ./data/2304.12345 --output-dir ./output

# Parse all publications in batch mode
python src/main_parser.py --input-dir ./data --batch --output-dir ./output

# Additional options:
#   --verbose         Enable detailed logging
#   --no-dedup        Skip deduplication step
#   --arxiv-id ID     Specify arXiv ID manually
```

**Expected Output**:
```
================================================================================
PARSING SUMMARY
================================================================================
ArXiv ID:    2304.12345
Versions:    2
Files:       15
Elements:    247
References:  42
================================================================================
```

### 3. Run ML Pipeline

```bash
# Train and evaluate with default settings
python src/main_matcher.py --data-dir ./output --output-dir ./output_23120067

# Use ranker model (recommended)
python src/main_matcher.py --data-dir ./data --model-type ranker

# Enable hyperparameter tuning
python src/main_matcher.py --data-dir ./data --tune-hyperparams

# Use GPU acceleration (if CUDA available)
python src/main_matcher.py --data-dir ./data --use-gpu

# All options:
#   --model-type      'classifier' or 'ranker' (default: ranker)
#   --manual-labels   Path to manual labels JSON
#   --tune-hyperparams Enable hyperparameter search
#   --use-gpu         Use GPU for training
```

**Expected Output**:
```
======================================================================
EVALUATION RESULTS
======================================================================
  MRR:    0.8542
  Hit@1:  78.50%
  Hit@3:  91.20%
  Hit@5:  95.30%
  Avg Rank: 1.45
======================================================================
```

---

## 📊 Data Format

### Input Structure (from Lab 1)
```
data/
├── 2304.12345/
│   ├── metadata.json           # Paper metadata
│   ├── references.json         # Candidate arXiv references (n entries)
│   └── tex/
│       ├── 2304.12345v1/       # Version 1 LaTeX source
│       │   ├── main.tex
│       │   ├── introduction.tex
│       │   └── figures/
│       └── 2304.12345v2/       # Version 2 (if available)
│           └── ...
└── 2401.67890/
    └── ...
```

### Output Structure
```
data/
├── 2304.12345/
│   ├── hierarchy.json          # Hierarchical structure (NEW)
│   ├── refs.bib                # Deduplicated BibTeX (NEW)
│   ├── pred.json               # ML predictions (NEW)
│   ├── metadata.json           # Original
│   ├── references.json         # Original
│   └── tex/                    # Original
└── ...
```

### hierarchy.json Format
```json
{
  "arxiv_id": "2304.12345",
  "versions": ["v1", "v2"],
  "elements": {
    "2304.12345-sec-1": {
      "type": "section",
      "title": "Introduction",
      "content": "...",
      "children": ["2304.12345-sent-1", "2304.12345-sent-2"],
      "versions": ["v1", "v2"]
    },
    "2304.12345-sent-1": {
      "type": "sentence",
      "content": "Machine learning has...",
      "parent": "2304.12345-sec-1",
      "versions": ["v1", "v2"]
    },
    "2304.12345-eq-1": {
      "type": "formula",
      "content": "E = mc^2",
      "parent": "2304.12345-sec-2",
      "versions": ["v1"]
    }
  },
  "hierarchy": {
    "root": ["2304.12345-sec-1", "2304.12345-sec-2", "2304.12345-sec-3"]
  }
}
```

### pred.json Format
```json
{
  "partition": "test",
  "groundtruth": {
    "lipton2018": "1606.03490",
    "rudin2019": "1811.10154"
  },
  "prediction": {
    "lipton2018": ["1606.03490", "1705.08807", "1811.10154", "2001.09876", "1903.04562"],
    "rudin2019": ["1811.10154", "1606.03490", "1705.08807", "2001.09876", "1903.04562"]
  }
}
```

---

## 🔧 Configuration

All settings are in `src/config.py`:

```python
# Student Information
STUDENT_ID = "23120067"

# Parser Settings
MAX_FILE_SIZE_MB = 10
SUPPORTED_ENCODINGS = ['utf-8', 'latin-1', 'cp1252']

# ML Model Settings
MODEL_TYPE = 'ranker'  # 'classifier' or 'ranker'
CATBOOST_PARAMS = {
    'iterations': 500,
    'learning_rate': 0.03,
    'depth': 6,
    'l2_leaf_reg': 3.0,
}

# Data Split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Feature Settings
TITLE_SIMILARITY_THRESHOLD = 0.8
AUTHOR_OVERLAP_THRESHOLD = 0.6

# Evaluation
TOP_K = 5  # Top-k predictions for MRR
```

---

## 📈 Features Used

### Group 1: Title Features (5 features)
| Feature | Description | Justification |
|---------|-------------|---------------|
| `title_jaccard` | Word-level Jaccard similarity | Robust to word order |
| `title_levenshtein` | Character-level similarity | Catches typos |
| `title_token_sort` | Sorted token comparison | Order-independent |
| `title_token_set` | Unique token overlap | Handles duplicates |
| `title_exact_match` | Boolean exact match | Strong positive signal |

### Group 2: Author Features (5 features)
| Feature | Description | Justification |
|---------|-------------|---------------|
| `author_overlap_ratio` | Proportion of shared authors | Identity verification |
| `first_author_match` | First author last name match | Most important author |
| `last_author_match` | Last author match | Senior/corresponding |
| `num_common_authors` | Count of shared authors | Multi-author signal |
| `author_initials_match` | Initials comparison | Format variations |

### Group 3: Year Features (4 features)
| Feature | Description | Justification |
|---------|-------------|---------------|
| `year_diff` | Absolute year difference | Temporal filter |
| `year_exact_match` | Same year indicator | Strong signal |
| `year_within_1` | Within 1 year | Pre-print tolerance |
| `both_years_present` | Data quality indicator | Missing data handling |

### Group 4: Text Features (5 features)
| Feature | Description | Justification |
|---------|-------------|---------------|
| `abstract_jaccard` | Abstract similarity | Deep content match |
| `venue_similarity` | Journal/conference match | Publication venue |
| `text_word_overlap` | Combined text overlap | Semantic similarity |
| `bigram_similarity` | 2-gram comparison | Phrase matching |
| `title_length_ratio` | Title length ratio | Length consistency |

### Group 5: Hierarchy Features (8 features)
| Feature | Description | Justification |
|---------|-------------|---------------|
| `citation_count` | Number of citations | Importance signal |
| `cited_in_intro` | Cited in introduction | Background reference |
| `cited_in_methods` | Cited in methods | Technical reference |
| `cited_in_results` | Cited in results | Comparison baseline |
| `avg_citation_depth` | Average hierarchy depth | Specificity measure |
| `near_figure` | Near figure element | Technical context |
| `near_formula` | Near formula element | Mathematical context |
| `co_citation_count` | Co-cited papers count | Relatedness |

---

## 🎯 Model Performance

### Evaluation Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| **MRR** | Mean Reciprocal Rank | $\frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i}$ |
| **Hit@k** | Correct in top-k | $\frac{|\{q : rank_q \leq k\}|}{|Q|}$ |
| **Avg Rank** | Average rank of correct | $\frac{1}{|Q|} \sum_{i=1}^{|Q|} rank_i$ |

### Expected Results

| Model Type | MRR | Hit@1 | Hit@5 |
|------------|-----|-------|-------|
| Classifier | 0.75-0.85 | 65-75% | 90-95% |
| **Ranker** | **0.80-0.90** | **70-80%** | **92-98%** |

> **Note**: Actual results depend on dataset quality and size.

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_parser.py -v
pytest tests/test_matcher.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 📚 Dependencies

Core libraries:
- `catboost>=1.2` - Gradient boosting for ML
- `bibtexparser>=1.4` - BibTeX parsing
- `fuzzywuzzy>=0.18` - String matching
- `nltk>=3.8` - Text processing
- `pandas>=2.0` - Data manipulation
- `scikit-learn>=1.3` - ML utilities

See `src/requirements.txt` for complete list.

---

## 🔍 Troubleshooting

### Common Issues

**1. ImportError: No module named 'src'**
```bash
# Make sure you're in the project root
cd parsing_standardlization_laTex
# Add src to Python path
export PYTHONPATH="${PYTHONPATH}:./src"
```

**2. NLTK data not found**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

**3. CatBoost GPU error**
```bash
# Fall back to CPU
python src/main_matcher.py --data-dir ./data  # No --use-gpu flag
```

**4. Memory error on large datasets**
```python
# In config.py, reduce batch sizes:
BATCH_SIZE = 1000  # Lower this
```

---

## 📝 References

1. CatBoost Documentation: https://catboost.ai/docs/
2. BibTeX Format: https://www.bibtex.org/Format/
3. arXiv API: https://arxiv.org/help/api/

---

## 👤 Author

**Student ID**: 23120067  
**Course**: Introduction to Data Science  
**Semester**: 2024-2025

---

## 📄 License

This project is for educational purposes as part of the Introduction to Data Science course.