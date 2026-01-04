"""
Configuration for the entire project
"""

# Paths
STUDENT_ID = "23120067"
DATA_DIR = f"./{STUDENT_ID}"
OUTPUT_DIR = f"./output_{STUDENT_ID}"

# Parser settings
LATEX_CLEANUP_PATTERNS = [
    r'\\centering',
    r'\\raggedright',
    r'\[htpb\]',
    # ... more patterns
]

# Feature engineering
TITLE_SIMILARITY_THRESHOLD = 0.90
AUTHOR_MATCH_THRESHOLD = 0.80

# Model settings
MODEL_TYPE = 'ranker'  # 'classifier' or 'ranker'
CATBOOST_PARAMS = {
    'iterations': 200,
    'learning_rate': 0.1,
    'depth': 6,
    'random_seed': 42
}

# Data split
TEST_MANUAL_PUBS = 1
TEST_AUTO_PUBS = 1
VAL_MANUAL_PUBS = 1
VAL_AUTO_PUBS = 1

# Evaluation
TOP_K = 5