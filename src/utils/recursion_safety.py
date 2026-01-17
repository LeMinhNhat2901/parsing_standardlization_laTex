"""
RECURSION SAFETY MODULE
This module should be imported FIRST in any script to prevent RecursionError

Usage:
    # Add this at the TOP of your script, before any other imports:
    import sys
    sys.path.insert(0, 'path/to/src')
    from utils import recursion_safety  # This sets up recursion limits
    
    # Then import other modules
    import pandas as pd
    from matcher import *
"""

import sys
import traceback

# ============================================================================
# RECURSION LIMIT CONFIGURATION
# ============================================================================

# Set high recursion limit for complex LaTeX files
# Default Python limit is 1000, which is too low for:
# - pyparsing (used by bibtexparser)
# - Deeply nested LaTeX structures
# - Complex data structures with circular references
RECURSION_LIMIT = 10000

if sys.getrecursionlimit() < RECURSION_LIMIT:
    sys.setrecursionlimit(RECURSION_LIMIT)
    print(f"[RecursionSafety] Set recursion limit to {RECURSION_LIMIT}")

# ============================================================================
# PYPARSING PACKRAT DISABLE
# ============================================================================

try:
    import pyparsing
    # Packrat caching can cause RecursionError with non-optimal grammars
    # Disable it for safety
    pyparsing.ParserElement.disablePackrat()
    print("[RecursionSafety] Disabled pyparsing packrat caching")
except (ImportError, AttributeError):
    pass  # pyparsing not installed or method not available

# ============================================================================
# SAFE IMPORT WRAPPER
# ============================================================================

def safe_import(module_name):
    """
    Safely import a module with RecursionError handling
    
    Args:
        module_name: Name of module to import
        
    Returns:
        Imported module or None if import failed
    """
    try:
        return __import__(module_name)
    except RecursionError as e:
        print(f"[RecursionSafety] RecursionError importing {module_name}")
        print(f"  Current limit: {sys.getrecursionlimit()}")
        print(f"  Try increasing RECURSION_LIMIT in recursion_safety.py")
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"[RecursionSafety] Error importing {module_name}: {e}")
        return None

# ============================================================================
# FILE PROCESSING WRAPPER
# ============================================================================

def safe_process_file(func, file_path, *args, **kwargs):
    """
    Safely process a file with RecursionError handling
    
    Args:
        func: Function to call
        file_path: Path to file being processed
        *args, **kwargs: Arguments to pass to func
        
    Returns:
        Result of func, or None if RecursionError occurred
    """
    try:
        return func(file_path, *args, **kwargs)
    except RecursionError as e:
        print(f"[RecursionSafety] RecursionError processing: {file_path}")
        print(f"  This file may have:")
        print(f"  - Deeply nested environments (itemize in tabular in minipage...)")
        print(f"  - Circular \\input references (A.tex -> B.tex -> A.tex)")
        print(f"  - Very long content causing parser issues")
        print(f"  Skipping this file...")
        return None
    except Exception as e:
        print(f"[RecursionSafety] Error processing {file_path}: {e}")
        return None

# ============================================================================
# DEBUGGING UTILITIES
# ============================================================================

def check_recursion_depth():
    """
    Check current recursion depth (for debugging)
    
    Usage in a potentially deep recursive function:
        depth = check_recursion_depth()
        if depth > 500:
            print(f"Warning: Deep recursion at depth {depth}")
    """
    import traceback
    return len(traceback.extract_stack())


def log_deep_recursion(threshold=500, context=""):
    """
    Log a warning if recursion is getting deep
    
    Args:
        threshold: Depth at which to warn
        context: Description of what's being processed
    """
    depth = check_recursion_depth()
    if depth > threshold:
        print(f"[RecursionSafety] WARNING: Deep recursion ({depth}) at: {context}")
        print(f"  Stack frames: {depth}")
        print(f"  Limit: {sys.getrecursionlimit()}")


# ============================================================================
# INITIALIZATION COMPLETE
# ============================================================================

print(f"[RecursionSafety] Module loaded. Limit: {sys.getrecursionlimit()}")
