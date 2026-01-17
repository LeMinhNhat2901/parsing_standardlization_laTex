"""
SAFE TYPE CHECKING UTILITIES
Avoid RecursionError by using type().__name__ instead of isinstance()

These utilities should be used throughout the codebase when checking types
to prevent RecursionError: maximum recursion depth exceeded in __instancecheck__
"""

import json


# ============================================================================
# SAFE TYPE CHECKING FUNCTIONS
# ============================================================================

def safe_type_name(obj):
    """
    Get type name safely without using isinstance()
    
    Args:
        obj: Any object
        
    Returns:
        String type name (e.g., 'str', 'int', 'dict', 'list')
    """
    try:
        return type(obj).__name__
    except Exception:
        return 'unknown'


def is_string(obj):
    """Check if object is string without isinstance()"""
    return safe_type_name(obj) == 'str'


def is_int(obj):
    """Check if object is int (including numpy int types) without isinstance()"""
    t = safe_type_name(obj)
    return t in ('int', 'int8', 'int16', 'int32', 'int64',
                 'uint8', 'uint16', 'uint32', 'uint64')


def is_float(obj):
    """Check if object is float (including numpy float types) without isinstance()"""
    t = safe_type_name(obj)
    return t in ('float', 'float16', 'float32', 'float64')


def is_bool(obj):
    """Check if object is bool (including numpy bool) without isinstance()"""
    t = safe_type_name(obj)
    return t in ('bool', 'bool_')


def is_numeric(obj):
    """Check if object is numeric (int or float)"""
    return is_int(obj) or is_float(obj)


def is_list(obj):
    """Check if object is list or tuple without isinstance()"""
    return safe_type_name(obj) in ('list', 'tuple')


def is_dict(obj):
    """Check if object is dict without isinstance()"""
    return safe_type_name(obj) == 'dict'


def is_none(obj):
    """Check if object is None"""
    return obj is None


def is_primitive(obj):
    """
    Check if object is a primitive type (can be stored directly)
    
    Primitive types: str, int, float, bool, None
    Also includes numpy numeric types
    """
    if is_none(obj):
        return True
    
    t = safe_type_name(obj)
    return t in ('str', 'int', 'float', 'bool', 'NoneType',
                 'int8', 'int16', 'int32', 'int64',
                 'uint8', 'uint16', 'uint32', 'uint64',
                 'float16', 'float32', 'float64', 'bool_')


def is_ndarray(obj):
    """Check if object is numpy ndarray"""
    return safe_type_name(obj) == 'ndarray'


# ============================================================================
# SAFE CONVERSION FUNCTIONS
# ============================================================================

def to_primitive(obj):
    """
    Convert object to primitive type safely
    
    Args:
        obj: Any object
        
    Returns:
        Primitive value (int, float, str, bool, or 0)
    """
    if is_none(obj):
        return 0
    
    t = safe_type_name(obj)
    
    # Already primitive Python types
    if t in ('str', 'int', 'float', 'bool'):
        return obj
    
    # Numpy int types -> int
    if t in ('int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'):
        return int(obj)
    
    # Numpy float types -> float
    if t in ('float16', 'float32', 'float64'):
        return float(obj)
    
    # Numpy bool -> bool
    if t == 'bool_':
        return bool(obj)
    
    # List/tuple -> comma-separated string
    if t in ('list', 'tuple'):
        try:
            return ', '.join(str(item) for item in obj)
        except Exception:
            return str(obj)
    
    # Dict -> JSON string
    if t == 'dict':
        try:
            return json.dumps(obj)
        except Exception:
            return str(obj)
    
    # ndarray -> depends on size
    if t == 'ndarray':
        try:
            if obj.size == 1:
                return float(obj.item())
            elif obj.size <= 10:
                return ', '.join(str(x) for x in obj.flatten())
            else:
                return str(obj.shape)
        except Exception:
            return 0
    
    # Everything else -> try to convert
    try:
        return float(obj)
    except (TypeError, ValueError):
        try:
            return str(obj)
        except Exception:
            return 0


def to_float_safe(obj, default=0.0):
    """
    Convert object to float safely
    
    Args:
        obj: Any object
        default: Default value if conversion fails
        
    Returns:
        Float value or default
    """
    if is_none(obj):
        return default
    
    t = safe_type_name(obj)
    
    # Already float
    if t == 'float':
        return obj
    
    # Numeric types
    if t in ('int', 'int8', 'int16', 'int32', 'int64',
             'uint8', 'uint16', 'uint32', 'uint64',
             'float16', 'float32', 'float64'):
        return float(obj)
    
    # Bool
    if t in ('bool', 'bool_'):
        return 1.0 if obj else 0.0
    
    # String - try to parse
    if t == 'str':
        try:
            return float(obj)
        except ValueError:
            return default
    
    # Everything else
    try:
        return float(obj)
    except (TypeError, ValueError):
        return default


def to_int_safe(obj, default=0):
    """
    Convert object to int safely
    
    Args:
        obj: Any object
        default: Default value if conversion fails
        
    Returns:
        Int value or default
    """
    if is_none(obj):
        return default
    
    t = safe_type_name(obj)
    
    # Already int
    if t in ('int', 'int8', 'int16', 'int32', 'int64',
             'uint8', 'uint16', 'uint32', 'uint64'):
        return int(obj)
    
    # Float - truncate
    if t in ('float', 'float16', 'float32', 'float64'):
        return int(obj)
    
    # Bool
    if t in ('bool', 'bool_'):
        return 1 if obj else 0
    
    # String - try to parse
    if t == 'str':
        try:
            return int(float(obj))
        except ValueError:
            return default
    
    # Everything else
    try:
        return int(obj)
    except (TypeError, ValueError):
        return default


# ============================================================================
# SAFE DICT OPERATIONS
# ============================================================================

def safe_dict_copy(d, max_depth=3, current_depth=0, _seen=None):
    """
    Create safe copy of dict with only primitive values
    Prevents RecursionError with nested/circular structures
    
    Args:
        d: Dictionary to copy
        max_depth: Maximum nesting depth (default 3)
        current_depth: Current depth (internal)
        _seen: Set of seen object IDs (internal)
        
    Returns:
        Dict with primitive values only
    """
    # Initialize seen set
    if _seen is None:
        _seen = set()
    
    # Handle None
    if is_none(d):
        return {}
    
    # Not a dict
    if not is_dict(d):
        return {'value': to_primitive(d)}
    
    # Check for circular reference
    obj_id = id(d)
    if obj_id in _seen:
        return {}
    _seen.add(obj_id)
    
    # Prevent infinite recursion with depth limit
    if current_depth >= max_depth:
        return {}
    
    safe = {}
    
    try:
        for k, v in d.items():
            # Convert key to string
            key = str(k)
            
            # Handle value based on type
            if is_none(v):
                safe[key] = ''
            elif is_primitive(v):
                safe[key] = to_primitive(v)
            elif is_list(v):
                # Flatten list to string
                items = []
                for item in v:
                    if is_primitive(item):
                        items.append(str(item))
                    elif is_dict(item):
                        # Skip nested dicts in lists
                        continue
                    else:
                        items.append(str(item))
                safe[key] = ', '.join(items)
            elif is_dict(v):
                # Recursively copy nested dict (with depth limit)
                if current_depth < max_depth - 1:
                    nested = safe_dict_copy(v, max_depth, current_depth + 1, _seen.copy())
                    try:
                        safe[key] = json.dumps(nested) if nested else ''
                    except Exception:
                        safe[key] = str(nested)[:200]
                else:
                    safe[key] = str(v)[:200]
            elif is_ndarray(v):
                try:
                    if v.size <= 10:
                        safe[key] = ', '.join(str(x) for x in v.flatten())
                    else:
                        safe[key] = str(v.shape)
                except Exception:
                    safe[key] = ''
            else:
                # Unknown type - convert to primitive
                safe[key] = to_primitive(v)
    
    except Exception as e:
        print(f"Warning: Error in safe_dict_copy: {e}")
        return {}
    
    return safe


def flatten_for_dataframe(features_dict):
    """
    Flatten a features dictionary for safe DataFrame creation
    
    This ensures all values are primitive types that pandas can handle
    without triggering RecursionError
    
    Args:
        features_dict: Dictionary of features
        
    Returns:
        Dictionary with only float/int/str values
    """
    if is_none(features_dict) or not is_dict(features_dict):
        return {}
    
    clean = {}
    
    for key, value in features_dict.items():
        key_str = str(key)
        
        if is_none(value):
            clean[key_str] = 0
        elif is_numeric(value) or is_bool(value):
            clean[key_str] = to_float_safe(value, 0.0)
        elif is_string(value):
            # Try to convert string to float, else use 0
            try:
                clean[key_str] = float(value)
            except ValueError:
                clean[key_str] = 0
        elif is_ndarray(value):
            try:
                if value.size == 1:
                    clean[key_str] = float(value.item())
                else:
                    clean[key_str] = 0
            except Exception:
                clean[key_str] = 0
        else:
            # Try to convert to float, else use 0
            clean[key_str] = to_float_safe(value, 0.0)
    
    return clean


# ============================================================================
# EXPORT ALL UTILITIES
# ============================================================================

__all__ = [
    # Type checking
    'safe_type_name',
    'is_string',
    'is_int',
    'is_float',
    'is_bool',
    'is_numeric',
    'is_list',
    'is_dict',
    'is_none',
    'is_primitive',
    'is_ndarray',
    
    # Conversion
    'to_primitive',
    'to_float_safe',
    'to_int_safe',
    
    # Dict operations
    'safe_dict_copy',
    'flatten_for_dataframe',
]
