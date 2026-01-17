"""
Configuration for the entire project
UPDATED with optimal parameters
"""

import os

# ============================================
# PATHS
# ============================================
STUDENT_ID = "23120067"
DATA_DIR = f"../../NMKHDL/data/{STUDENT_ID}"
OUTPUT_DIR = f"../output_{STUDENT_ID}"
MODEL_DIR = f"../models_{STUDENT_ID}"

# Create directories if not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================
# PARSER SETTINGS
# ============================================

# LaTeX cleanup patterns
LATEX_CLEANUP_PATTERNS = [
    r'\\centering',
    r'\\raggedright',
    r'\\raggedleft',
    r'\[htpb\]',
    r'\[H\]',
    r'\[h!\]',
    r'\\midrule',
    r'\\toprule',
    r'\\bottomrule',
    r'\\hline',
    r'\\cmidrule',
]

# Sections to exclude from hierarchy
EXCLUDE_SECTIONS = [
    'references',
    'bibliography',
    'bibl',
]

# Sections to include (even if unnumbered)
INCLUDE_SECTIONS = [
    'acknowledgement',
    'acknowledgment',
    'appendix',
    'appendices',
]

# Hierarchy element types
ELEMENT_TYPES = {
    'document': 'doc',
    'chapter': 'chap',
    'section': 'sec',
    'subsection': 'subsec',
    'subsubsection': 'subsubsec',
    'paragraph': 'para',
    'subparagraph': 'subpara',
    'sentence': 'sent',
    'formula': 'formula',
    'equation': 'eq',
    'figure': 'fig',
    'table': 'tab',
    'itemize': 'itemize',
    'item': 'item',
    'enumerate': 'enum',
}

# ============================================
# DEDUPLICATION SETTINGS
# ============================================

# Reference deduplication
REF_TITLE_SIMILARITY_THRESHOLD = 0.95
REF_AUTHOR_OVERLAP_THRESHOLD = 0.80

# Content deduplication
CONTENT_SIMILARITY_THRESHOLD = 0.99  # Very high for exact match

# ============================================
# FEATURE ENGINEERING
# ============================================

# Text similarity thresholds
TITLE_SIMILARITY_THRESHOLD = 0.90
AUTHOR_MATCH_THRESHOLD = 0.80
YEAR_DIFFERENCE_MAX = 3

# TF-IDF settings
TFIDF_MAX_FEATURES = 500
TFIDF_MIN_DF = 1
TFIDF_MAX_DF = 0.95

# ============================================
# MODEL SETTINGS - OPTIMIZED
# ============================================

# Model type: 'classifier' or 'ranker'
MODEL_TYPE = 'ranker'  # Recommended: ranker for MRR optimization

# Use GPU if available
USE_GPU = False  # Set to True if GPU available

# CatBoost Classifier Parameters (Binary Classification)
CATBOOST_CLASSIFIER_PARAMS = {
    # Core parameters
    'iterations': 500,
    'learning_rate': 0.03,
    'depth': 6,
    
    # Loss function
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    
    # Class imbalance handling - CRITICAL!
    'auto_class_weights': 'Balanced',
    
    # Regularization
    'l2_leaf_reg': 3.0,
    'random_strength': 1.0,
    'bagging_temperature': 1.0,
    
    # Overfitting prevention
    'early_stopping_rounds': 50,
    'od_type': 'Iter',
    'od_wait': 50,
    
    # Performance
    'task_type': 'GPU' if USE_GPU else 'CPU',
    'thread_count': -1,
    
    # Reproducibility
    'random_seed': 42,
    'verbose': 50,
    
    # Other
    'use_best_model': True,
    'bootstrap_type': 'Bayesian',
}

# CatBoost Ranker Parameters (Ranking - RECOMMENDED)
CATBOOST_RANKER_PARAMS = {
    # Core parameters
    'iterations': 500,
    'learning_rate': 0.03,
    'depth': 6,
    
    # Loss function - OPTIMIZED FOR RANKING
    'loss_function': 'YetiRank',  # Best for MRR
    
    # Evaluation metrics
    'eval_metric': 'NDCG:top=5',
    'custom_metric': ['MRR:top=5', 'PrecisionAt:top=5', 'RecallAt:top=5'],
    
    # Regularization
    'l2_leaf_reg': 3.0,
    'random_strength': 1.0,
    'bagging_temperature': 1.0,
    
    # Overfitting prevention
    'early_stopping_rounds': 50,
    'od_type': 'Iter',
    'od_wait': 50,
    
    # Performance
    'task_type': 'GPU' if USE_GPU else 'CPU',
    'thread_count': -1,
    
    # Reproducibility
    'random_seed': 42,
    'verbose': 50,
    
    # Other
    'use_best_model': True,
    'bootstrap_type': 'Bayesian',
}

# Hyperparameter tuning grid
HYPERPARAM_GRID = {
    'learning_rate': [0.01, 0.03, 0.05],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [1, 3, 5],
}

# Enable hyperparameter tuning
ENABLE_HYPERPARAMETER_TUNING = False  # Set True for tuning

# ============================================
# DATA SPLIT SETTINGS
# ============================================

# Number of publications for each set
TEST_MANUAL_PUBS = 1
TEST_AUTO_PUBS = 1
VAL_MANUAL_PUBS = 1
VAL_AUTO_PUBS = 1

# Minimum required labels
MIN_MANUAL_LABELS = 20
MIN_AUTO_LABELS_PERCENT = 0.10

# ============================================
# EVALUATION SETTINGS
# ============================================

# Top-K for evaluation
TOP_K = 5

# Enable post-processing
ENABLE_POST_PROCESSING = True

# Post-processing weights
POST_PROCESS_WEIGHTS = {
    'exact_title_boost': 1.0,     # Force to top
    'year_penalty': 0.1,           # Heavy penalty if year_diff > 3
    'author_boost': 1.2,           # 20% boost
    'arxiv_id_boost': 1.0,         # Force to top
}

# ============================================
# LOGGING SETTINGS
# ============================================

LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = f'../logs_{STUDENT_ID}/lab2.log'

# Create log directory
os.makedirs(os.path.dirname(LOG_FILE) if os.path.dirname(LOG_FILE) else '../logs', exist_ok=True)

# ============================================
# MANUAL LABELS (EXAMPLE)
# ============================================
# This should be loaded from a separate file in practice
# Format: {publication_id: {bibtex_key: arxiv_id}}

MANUAL_LABELS_EXAMPLE = {
    '2504-13946': {
        'lipton2018mythos': '1606-03490',
        'rudin2019stop': '1811-10154',
        # ... more labels
    },
    # ... more publications
}

# ============================================
# LABELING SETTINGS
# ============================================

# Minimum requirements for manual labeling
MIN_MANUAL_PUBLICATIONS = 5  # At least 5 publications manually labeled
MIN_MANUAL_PAIRS = 20        # At least 20 labeled reference pairs total

# Automatic labeling requirements
AUTO_LABEL_PERCENT = 0.10    # Label at least 10% of remaining data automatically

# Automatic labeling thresholds
AUTO_LABEL_TITLE_THRESHOLD = 0.90      # Title similarity for auto-label
AUTO_LABEL_AUTHOR_THRESHOLD = 0.80     # Author overlap for auto-label
AUTO_LABEL_COMBINED_THRESHOLD = 0.85   # Combined score threshold

# ============================================
# LATEX PARSING PATTERNS
# ============================================

# Input/Include detection patterns
INPUT_PATTERNS = [
    r'\\input\{([^}]+)\}',
    r'\\include\{([^}]+)\}',
    r'\\subfile\{([^}]+)\}',
]

# Main file detection heuristics
MAIN_FILE_NAMES = [
    'main.tex',
    'paper.tex',
    'article.tex',
    'manuscript.tex',
]

MAIN_FILE_MARKERS = [
    r'\\documentclass',
    r'\\begin\{document\}',
]

# Section hierarchy commands (in order of level)
SECTION_COMMANDS = [
    'chapter',
    'section',
    'subsection',
    'subsubsection',
    'paragraph',
    'subparagraph',
]

# Unnumbered section variants (starred versions)
UNNUMBERED_SECTION_PATTERN = r'\\(chapter|section|subsection|subsubsection|paragraph|subparagraph)\*\{([^}]+)\}'

# ============================================
# MATH NORMALIZATION PATTERNS
# ============================================

# Inline math patterns to normalize to $...$
INLINE_MATH_PATTERNS = [
    (r'\\\((.+?)\\\)', r'$\1$'),          # \(...\) → $...$
    (r'\\begin\{math\}(.+?)\\end\{math\}', r'$\1$'),  # \begin{math}...\end{math}
]

# Block math patterns to normalize to \begin{equation}...\end{equation}
BLOCK_MATH_PATTERNS = [
    (r'\$\$(.+?)\$\$', r'\\begin{equation}\1\\end{equation}'),  # $$...$$ 
    (r'\\\[(.+?)\\\]', r'\\begin{equation}\1\\end{equation}'),  # \[...\]
    # Other equation environments to keep as-is:
    # align, align*, eqnarray, gather, multline, etc.
]

# Math environments that are block formulas (leaf elements)
BLOCK_FORMULA_ENVIRONMENTS = [
    'equation',
    'equation*',
    'align',
    'align*',
    'eqnarray',
    'eqnarray*',
    'gather',
    'gather*',
    'multline',
    'multline*',
    'displaymath',
]

# ============================================
# HIERARCHY SMALLEST ELEMENTS (LEAF NODES)
# ============================================

# As per requirement: Sentences, Block Formulas, and Figures (including Tables)
LEAF_ELEMENT_TYPES = [
    'sentence',      # Text split by periods
    'formula',       # Block math formulas
    'figure',        # \begin{figure}...\end{figure}
    'table',         # Tables are considered as type of Figure
]

# Higher components (branches)
BRANCH_ELEMENT_TYPES = [
    'document',
    'chapter',
    'section',
    'subsection',
    'subsubsection',
    'paragraph',
    'subparagraph',
    'itemize',       # \begin{itemize}...\end{itemize} is higher component
    'enumerate',     # \begin{enumerate}...\end{enumerate}
    'item',          # Each \item is next-level element
]

# ============================================
# FIGURE/TABLE DETECTION
# ============================================

FIGURE_ENVIRONMENTS = [
    'figure',
    'figure*',
    'subfigure',
    'wrapfigure',
]

TABLE_ENVIRONMENTS = [
    'table',
    'table*',
    'tabular',
    'tabular*',
    'longtable',
]

# ============================================
# REFERENCE EXTRACTION PATTERNS
# ============================================

# \bibitem pattern
BIBITEM_PATTERN = r'\\bibitem(?:\[([^\]]*)\])?\{([^}]+)\}'

# Citation patterns
CITE_PATTERNS = [
    r'\\cite\{([^}]+)\}',
    r'\\citep\{([^}]+)\}',
    r'\\citet\{([^}]+)\}',
    r'\\citealp\{([^}]+)\}',
    r'\\citeauthor\{([^}]+)\}',
    r'\\citeyear\{([^}]+)\}',
    r'\\citep\[([^\]]*)\]\{([^}]+)\}',
]

# BibTeX field patterns for parsing \bibitem content
BIBTEX_AUTHOR_PATTERNS = [
    r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+and\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)*)',
    r'^([A-Z][a-z]+(?:,\s*[A-Z]\.?)*(?:,?\s+and\s+[A-Z][a-z]+(?:,\s*[A-Z]\.?)*)*)',
]

BIBTEX_YEAR_PATTERNS = [
    r'\((\d{4})\)',
    r',\s*(\d{4})\.',
    r'(\d{4})\.',
]

BIBTEX_TITLE_PATTERNS = [
    r'(?:``|")([^"``]+)(?:\'\'|")',
    r'\.\s+([^.]+)\.\s+(?:In\s+|Proceedings|Journal|Conference)',
]

# ============================================
# DEDUPLICATION SETTINGS (EXTENDED)
# ============================================

# Reference deduplication strategies
REF_DEDUP_STRATEGIES = {
    'title_similarity': REF_TITLE_SIMILARITY_THRESHOLD,
    'author_overlap': REF_AUTHOR_OVERLAP_THRESHOLD,
    'year_match': True,
}

# Content deduplication (full-text match after cleanup)
CONTENT_DEDUP_NORMALIZE = {
    'lowercase': True,
    'strip_whitespace': True,
    'remove_punctuation': False,  # Keep punctuation for accuracy
}

# ============================================
# OUTPUT FILE FORMATS
# ============================================

# hierarchy.json format
HIERARCHY_JSON_SCHEMA = {
    'elements': dict,      # {element_id: content_or_title}
    'hierarchy': dict,     # {version: {child_id: parent_id}}
}

# refs.bib format (standard BibTeX)
BIBTEX_ENTRY_TYPES = [
    'article',
    'inproceedings',
    'book',
    'misc',
    'phdthesis',
    'mastersthesis',
    'techreport',
]

# metadata.json required fields
METADATA_REQUIRED_FIELDS = [
    'title',           # Paper title (string)
    'authors',         # List of author names (list of strings)
    'submission_date', # ISO format date string
    'revised_dates',   # List of revision dates (list of strings in ISO format)
]

METADATA_OPTIONAL_FIELDS = [
    'venue',           # Journal or conference name
    'doi',
    'arxiv_id',
    'abstract',
]

# references.json format
REFERENCES_JSON_REQUIRED_FIELDS = [
    'title',           # Reference paper title
    'authors',         # List of author strings
    'submission_date', # ISO format
    'semantic_scholar_id',  # Semantic Scholar ID
]

# pred.json format
PRED_JSON_SCHEMA = {
    'partition': ['train', 'valid', 'test'],
    'groundtruth': dict,   # {bibtex_key: arxiv_id}
    'prediction': dict,    # {bibtex_key: [top5_candidates]}
}

# ============================================
# ELEMENT ID GENERATION
# ============================================

# ID format: {publication_id}-{element_type}-{counter}
# Example: "2504-13946-sec-1", "2504-13946-sent-42"

ELEMENT_ID_FORMAT = "{pub_id}-{elem_type}-{counter}"

# Element type abbreviations
ELEMENT_TYPE_ABBREV = {
    'document': 'doc',
    'chapter': 'chap',
    'section': 'sec',
    'subsection': 'subsec',
    'subsubsection': 'subsubsec',
    'paragraph': 'para',
    'subparagraph': 'subpara',
    'sentence': 'sent',
    'formula': 'formula',
    'equation': 'eq',
    'figure': 'fig',
    'table': 'tab',
    'itemize': 'itemize',
    'enumerate': 'enum',
    'item': 'item',
}

# ============================================
# FEATURE ENGINEERING SETTINGS
# ============================================

# All features used in the matching pipeline
FEATURE_LIST = {
    # Title-based features
    'title_features': [
        'title_jaccard',
        'title_levenshtein',
        'title_tfidf_cosine',
        'title_exact_match',
        'title_word_overlap',
        'title_ngram_similarity',
    ],
    
    # Author-based features
    'author_features': [
        'author_overlap_ratio',
        'first_author_match',
        'last_author_match',
        'num_common_authors',
        'author_order_similarity',
    ],
    
    # Year-based features
    'year_features': [
        'year_diff',
        'year_exact_match',
        'year_within_1',
        'year_within_3',
    ],
    
    # Text-based features
    'text_features': [
        'abstract_similarity',
        'venue_match',
        'doi_match',
    ],
    
    # Hierarchy-based features (unique to this project)
    'hierarchy_features': [
        'citation_count',
        'cited_in_intro',
        'cited_in_methods',
        'cited_in_results',
        'cited_in_conclusion',
        'avg_citation_depth',
        'first_citation_depth',
        'near_figure',
        'near_table',
        'near_formula',
        'co_citation_count',
    ],
}

# Features that are categorical
CATEGORICAL_FEATURES = [
    'title_exact_match',
    'first_author_match',
    'last_author_match',
    'year_exact_match',
    'year_within_1',
    'year_within_3',
    'venue_match',
    'doi_match',
    'cited_in_intro',
    'cited_in_methods',
    'cited_in_results',
    'cited_in_conclusion',
    'near_figure',
    'near_table',
    'near_formula',
]

# ============================================
# STRING SIMILARITY METHODS
# ============================================

SIMILARITY_METHODS = {
    'jaccard': {
        'description': 'Word-level overlap (set intersection / union)',
        'robust_to': 'word order changes',
    },
    'levenshtein': {
        'description': 'Character-level edit distance ratio',
        'robust_to': 'typos, formatting differences',
    },
    'tfidf_cosine': {
        'description': 'TF-IDF weighted cosine similarity',
        'robust_to': 'common words, captures importance',
    },
    'fuzzy_ratio': {
        'description': 'Fuzzy string matching score',
        'robust_to': 'partial matches, reordering',
    },
}

# ============================================
# MODEL LOSS FUNCTIONS
# ============================================

# Available loss functions for ranking
RANKING_LOSS_FUNCTIONS = {
    'YetiRank': 'Optimized for NDCG/MRR',
    'YetiRankPairwise': 'Pairwise ranking loss',
    'PairLogit': 'Pairwise logistic loss',
    'PairLogitPairwise': 'Pairwise logistic with sampling',
    'QuerySoftMax': 'Softmax over queries',
}

# Loss function selection based on data characteristics
RECOMMENDED_LOSS = {
    'small_data': 'YetiRank',          # < 1000 pairs
    'medium_data': 'YetiRankPairwise', # 1000-10000 pairs
    'large_data': 'PairLogitPairwise', # > 10000 pairs
}

# ============================================
# CROSS-VALIDATION SETTINGS
# ============================================

CV_FOLDS = 5
CV_STRATIFIED = True  # Stratify by label
CV_SHUFFLE = True
CV_RANDOM_STATE = 42

# ============================================
# EARLY STOPPING SETTINGS
# ============================================

EARLY_STOPPING = {
    'enabled': True,
    'rounds': 50,
    'metric': 'MRR:top=5',
    'verbose': True,
}

# ============================================
# CLASS IMBALANCE HANDLING
# ============================================

# For m×n pairs: m positive, m×(n-1) negative
# Typical imbalance ratio: 1:50 or worse

CLASS_IMBALANCE_STRATEGIES = {
    'auto_class_weights': 'Balanced',  # CatBoost auto-balancing
    'sample_weights': None,            # Custom sample weights
    'oversampling': False,             # SMOTE or similar
    'undersampling': False,            # Random undersampling
}

# ============================================
# POST-PROCESSING SETTINGS
# ============================================

POST_PROCESSING = {
    'enabled': True,
    'strategies': {
        'exact_title_boost': {
            'enabled': True,
            'weight': 1.0,  # Force to top
            'description': 'If exact title match, move to rank 1',
        },
        'year_penalty': {
            'enabled': True,
            'max_diff': 3,
            'penalty': 0.1,
            'description': 'Penalize if year difference > 3',
        },
        'author_boost': {
            'enabled': True,
            'weight': 1.2,
            'description': '20% boost for first author match',
        },
        'arxiv_id_pattern': {
            'enabled': True,
            'weight': 1.0,
            'description': 'Extract arXiv ID from BibTeX if present',
            'pattern': r'arXiv[:\s]*(\d{4}\.\d{4,5})',
        },
        'deduplication': {
            'enabled': True,
            'description': 'Remove duplicate predictions',
        },
    },
}

# ============================================
# STATISTICS TRACKING
# ============================================

STATISTICS = {
    # Parsing statistics
    'parsing': {
        'total_publications': 0,
        'successful_parses': 0,
        'failed_parses': 0,
        'total_elements': 0,
        'total_references': 0,
    },
    
    # Matching statistics
    'matching': {
        'total_pairs': 0,
        'positive_pairs': 0,
        'negative_pairs': 0,
        'manual_labels': 0,
        'auto_labels': 0,
    },
    
    # Evaluation statistics
    'evaluation': {
        'train_mrr': 0.0,
        'val_mrr': 0.0,
        'test_mrr': 0.0,
        'hit_at_1': 0.0,
        'hit_at_3': 0.0,
        'hit_at_5': 0.0,
    },
}

# ============================================
# STOP WORDS FOR TEXT PROCESSING
# ============================================

# Stop words to remove during text cleaning
STOP_WORDS = [
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
    'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
    'dare', 'ought', 'used', 'this', 'that', 'these', 'those', 'i', 'we',
    'you', 'he', 'she', 'it', 'they', 'what', 'which', 'who', 'whom',
]

# Author name stop words
AUTHOR_STOP_WORDS = [
    'et', 'al', 'and', 'others', 'et al.', 'et al',
]

# ============================================
# TOKENIZATION SETTINGS
# ============================================

TOKENIZATION = {
    'title': {
        'lowercase': True,
        'remove_punctuation': True,
        'remove_stopwords': False,  # Keep for title matching
        'stemming': False,
    },
    'author': {
        'lowercase': True,
        'split_by': [',', ' and ', ';'],
        'normalize_names': True,
    },
    'abstract': {
        'lowercase': True,
        'remove_punctuation': True,
        'remove_stopwords': True,
        'stemming': True,
    },
}

# ============================================
# ARXIV ID PATTERNS
# ============================================

# ArXiv ID formats
ARXIV_ID_PATTERNS = [
    r'(\d{4})\.(\d{4,5})',           # New format: 1606.03490
    r'(\d{4})-(\d{4,5})',            # Folder format: 1606-03490
    r'arXiv:(\d{4}\.\d{4,5})',       # With prefix: arXiv:1606.03490
    r'arxiv\.org/abs/(\d{4}\.\d{4,5})',  # URL format
]

# Convert between formats
def normalize_arxiv_id(arxiv_id):
    """Convert arXiv ID to standard format: YYMM-NNNNN"""
    import re
    for pattern in ARXIV_ID_PATTERNS:
        match = re.search(pattern, str(arxiv_id))
        if match:
            if len(match.groups()) == 2:
                return f"{match.group(1)}-{match.group(2)}"
            else:
                parts = match.group(1).split('.')
                return f"{parts[0]}-{parts[1]}"
    return arxiv_id

# ============================================
# DATE HANDLING
# ============================================

DATE_FORMATS = [
    '%Y-%m-%d',          # ISO format: 2024-01-15
    '%Y-%m-%dT%H:%M:%S', # ISO with time
    '%Y/%m/%d',          # Slash format
    '%d %B %Y',          # Full month: 15 January 2024
    '%B %d, %Y',         # US format: January 15, 2024
    '%Y',                # Year only
]

# ============================================
# VALIDATION CHECKS
# ============================================

def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check minimum requirements
    if MIN_MANUAL_PUBLICATIONS < 5:
        errors.append("MIN_MANUAL_PUBLICATIONS must be at least 5")
    
    if MIN_MANUAL_PAIRS < 20:
        errors.append("MIN_MANUAL_PAIRS must be at least 20")
    
    if AUTO_LABEL_PERCENT < 0.10:
        errors.append("AUTO_LABEL_PERCENT must be at least 0.10 (10%)")
    
    # Check thresholds
    if not 0 <= REF_TITLE_SIMILARITY_THRESHOLD <= 1:
        errors.append("REF_TITLE_SIMILARITY_THRESHOLD must be between 0 and 1")
    
    if not 0 <= REF_AUTHOR_OVERLAP_THRESHOLD <= 1:
        errors.append("REF_AUTHOR_OVERLAP_THRESHOLD must be between 0 and 1")
    
    # Check TOP_K
    if TOP_K != 5:
        errors.append("TOP_K must be 5 as per requirement")
    
    # Check data split
    if TEST_MANUAL_PUBS != 1 or TEST_AUTO_PUBS != 1:
        errors.append("Test set must have exactly 1 manual and 1 auto publication")
    
    if VAL_MANUAL_PUBS != 1 or VAL_AUTO_PUBS != 1:
        errors.append("Validation set must have exactly 1 manual and 1 auto publication")
    
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(errors))
    
    return True

# ============================================
# HELPER FUNCTIONS
# ============================================

def get_publication_dir(publication_id):
    """Get path to publication directory"""
    return os.path.join(DATA_DIR, publication_id)

def get_hierarchy_path(publication_id):
    """Get path to hierarchy.json"""
    return os.path.join(get_publication_dir(publication_id), 'hierarchy.json')

def get_refs_bib_path(publication_id):
    """Get path to refs.bib"""
    return os.path.join(get_publication_dir(publication_id), 'refs.bib')

def get_metadata_path(publication_id):
    """Get path to metadata.json"""
    return os.path.join(get_publication_dir(publication_id), 'metadata.json')

def get_references_path(publication_id):
    """Get path to references.json"""
    return os.path.join(get_publication_dir(publication_id), 'references.json')

def get_pred_path(publication_id):
    """Get path to pred.json"""
    return os.path.join(get_publication_dir(publication_id), 'pred.json')

def get_model_path(model_name='catboost_model'):
    """Get path to save/load model"""
    return os.path.join(MODEL_DIR, f'{model_name}.cbm')

def get_feature_importance_path():
    """Get path to feature importance file"""
    return os.path.join(OUTPUT_DIR, 'feature_importance.json')

# ============================================
# RUNTIME CONFIGURATION
# ============================================

# These can be modified at runtime
RUNTIME_CONFIG = {
    'verbose': True,
    'debug': False,
    'dry_run': False,
    'parallel_processing': True,
    'num_workers': -1,  # -1 = use all cores
}

"""
Configuration file for ML pipeline
All parameters in one place for easy adjustment
"""

# ============================================================================
# DATA SPLIT CONFIGURATION (Requirement 2.2.4)
# ============================================================================

# Test set: 1 manual + 1 auto publication
TEST_MANUAL_PUBS = 1
TEST_AUTO_PUBS = 1

# Validation set: 1 manual + 1 auto publication  
VAL_MANUAL_PUBS = 1
VAL_AUTO_PUBS = 1

# Training set: All remaining publications (automatic)

# ============================================================================
# LABELING CONFIGURATION (Requirement 2.2.2)
# ============================================================================

# Manual labeling requirements
MIN_MANUAL_PUBLICATIONS = 5
MIN_MANUAL_PAIRS = 20

# Automatic labeling requirements
AUTO_LABEL_PERCENTAGE = 0.1  # 10% of non-manual data

# Automatic labeling thresholds
TITLE_SIMILARITY_THRESHOLD = 90  # 0-100
AUTHOR_OVERLAP_THRESHOLD = 70    # 0-100
YEAR_TOLERANCE = 1               # years

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Model type: 'classifier' or 'ranker'
MODEL_TYPE = 'ranker'  # 'ranker' recommended for ranking task

# GPU usage
USE_GPU = False  # Set to True if GPU available

# ============================================================================
# FEATURE ENGINEERING (Requirement 2.2.3)
# ============================================================================

# Feature groups enabled
ENABLE_TITLE_FEATURES = True      # 5 features
ENABLE_AUTHOR_FEATURES = True     # 5 features
ENABLE_YEAR_FEATURES = True       # 4 features
ENABLE_TEXT_FEATURES = True       # 5 features
ENABLE_HIERARCHY_FEATURES = True  # 18 features

# TF-IDF configuration
TFIDF_MAX_FEATURES = 500
TFIDF_NGRAM_RANGE = (1, 2)

# ============================================================================
# MODEL TRAINING
# ============================================================================

# CatBoost parameters
ITERATIONS = 500
LEARNING_RATE = 0.03
DEPTH = 6
L2_LEAF_REG = 3.0

# Early stopping
EARLY_STOPPING_ROUNDS = 50

# Class imbalance handling
AUTO_CLASS_WEIGHTS = True  # 'Balanced' for classifier

# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================

# Enable hyperparameter tuning
ENABLE_HYPERPARAMETER_TUNING = False  # Set to True to enable

# Parameter grid for tuning
PARAM_GRID = {
    'learning_rate': [0.01, 0.03, 0.05],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [1, 3, 5],
}

# ============================================================================
# EVALUATION (Requirement 2.2.5)
# ============================================================================

# Top-K candidates to return
TOP_K = 5

# Evaluation metrics
PRIMARY_METRIC = 'MRR'  # Mean Reciprocal Rank
ADDITIONAL_METRICS = ['Hit@1', 'Hit@3', 'Hit@5', 'Avg Rank']

# ============================================================================
# POST-PROCESSING
# ============================================================================

# Enable post-processing of predictions
ENABLE_POST_PROCESSING = True

# Post-processing strategies
POST_PROCESS_EXACT_TITLE_MATCH = True
POST_PROCESS_YEAR_FILTER = True
POST_PROCESS_FIRST_AUTHOR_BOOST = True
POST_PROCESS_ARXIV_ID_MATCH = True

# Year difference penalty threshold
MAX_YEAR_DIFFERENCE = 3  # Penalize if > 3 years apart

# ============================================================================
# OUTPUT PATHS
# ============================================================================

OUTPUT_DIR = 'output'
MODEL_DIR = 'models'
PLOTS_DIR = 'plots'
LOGS_DIR = 'logs'

# ============================================================================
# FILE NAMES
# ============================================================================

REFS_BIB_FILENAME = 'refs.bib'
REFERENCES_JSON_FILENAME = 'references.json'
HIERARCHY_JSON_FILENAME = 'hierarchy.json'
METADATA_JSON_FILENAME = 'metadata.json'
PRED_JSON_FILENAME = 'pred.json'

# ============================================================================
# VALIDATION
# ============================================================================

# Validate configuration on import
def validate_config():
    """Validate configuration parameters"""
    errors = []
    
    # Check data split
    if TEST_MANUAL_PUBS < 1 or TEST_AUTO_PUBS < 1:
        errors.append("Test set must have ≥1 manual and ≥1 auto publication (2.2.4)")
    
    if VAL_MANUAL_PUBS < 1 or VAL_AUTO_PUBS < 1:
        errors.append("Valid set must have ≥1 manual and ≥1 auto publication (2.2.4)")
    
    # Check labeling requirements
    if MIN_MANUAL_PUBLICATIONS < 5:
        errors.append("Must have ≥5 manual publications (2.2.2)")
    
    if MIN_MANUAL_PAIRS < 20:
        errors.append("Must have ≥20 manual pairs (2.2.2)")
    
    if AUTO_LABEL_PERCENTAGE < 0.1:
        errors.append("Must auto-label ≥10% of non-manual data (2.2.2)")
    
    # Check top-k
    if TOP_K != 5:
        errors.append("TOP_K must be 5 per requirement 2.2.5")
    
    # Check model type
    if MODEL_TYPE not in ['classifier', 'ranker']:
        errors.append("MODEL_TYPE must be 'classifier' or 'ranker'")
    
    if errors:
        print("\n⚠️  Configuration Warnings:")
        for error in errors:
            print(f"   - {error}")
        print()
    else:
        print("✅ Configuration validated successfully\n")
    
    return len(errors) == 0

# Validate on import
if __name__ != '__main__':
    validate_config()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_split_config():
    """Get data split configuration"""
    return {
        'test': {
            'manual_pubs': TEST_MANUAL_PUBS,
            'auto_pubs': TEST_AUTO_PUBS
        },
        'valid': {
            'manual_pubs': VAL_MANUAL_PUBS,
            'auto_pubs': VAL_AUTO_PUBS
        }
    }

def get_labeling_config():
    """Get labeling configuration"""
    return {
        'manual': {
            'min_publications': MIN_MANUAL_PUBLICATIONS,
            'min_pairs': MIN_MANUAL_PAIRS
        },
        'auto': {
            'target_percentage': AUTO_LABEL_PERCENTAGE,
            'title_threshold': TITLE_SIMILARITY_THRESHOLD,
            'author_threshold': AUTHOR_OVERLAP_THRESHOLD,
            'year_tolerance': YEAR_TOLERANCE
        }
    }

def get_model_config():
    """Get model configuration"""
    return {
        'model_type': MODEL_TYPE,
        'use_gpu': USE_GPU,
        'iterations': ITERATIONS,
        'learning_rate': LEARNING_RATE,
        'depth': DEPTH,
        'l2_leaf_reg': L2_LEAF_REG,
        'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
        'auto_class_weights': AUTO_CLASS_WEIGHTS
    }

def get_evaluation_config():
    """Get evaluation configuration"""
    return {
        'top_k': TOP_K,
        'primary_metric': PRIMARY_METRIC,
        'additional_metrics': ADDITIONAL_METRICS,
        'enable_post_processing': ENABLE_POST_PROCESSING
    }

def print_config():
    """Print all configuration"""
    print("\n" + "="*70)
    print("CONFIGURATION")
    print("="*70)
    
    print("\nData Split (2.2.4):")
    print(f"  Test:  {TEST_MANUAL_PUBS} manual + {TEST_AUTO_PUBS} auto")
    print(f"  Valid: {VAL_MANUAL_PUBS} manual + {VAL_AUTO_PUBS} auto")
    print(f"  Train: All remaining")
    
    print("\nLabeling (2.2.2):")
    print(f"  Manual: ≥{MIN_MANUAL_PUBLICATIONS} pubs, ≥{MIN_MANUAL_PAIRS} pairs")
    print(f"  Auto: ≥{AUTO_LABEL_PERCENTAGE*100}% of non-manual data")
    
    print("\nModel:")
    print(f"  Type: {MODEL_TYPE}")
    print(f"  GPU: {USE_GPU}")
    print(f"  Iterations: {ITERATIONS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    
    print("\nEvaluation (2.2.5):")
    print(f"  Metric: {PRIMARY_METRIC}")
    print(f"  Top-K: {TOP_K}")
    print(f"  Post-processing: {ENABLE_POST_PROCESSING}")
    
    print("\n" + "="*70 + "\n")

# ============================================
# VALIDATION ON IMPORT
# ============================================

# Validate configuration when module is imported
if __name__ != '__main__':
    try:
        validate_config()
    except ValueError as e:
        import warnings
        warnings.warn(str(e))