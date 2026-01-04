"""
Logging setup for the project
Provides consistent logging across all modules
"""

import logging
import os
from pathlib import Path
from datetime import datetime


def setup_logger(name, log_file=None, level=logging.INFO, console=True):
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level (default: INFO)
        console: Whether to add console handler (default: True)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name):
    """
    Get existing logger or create new one
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Create default project logger
_default_log_dir = Path('./logs')
_default_log_dir.mkdir(parents=True, exist_ok=True)
_default_log_file = _default_log_dir / f'lab2_{datetime.now().strftime("%Y%m%d")}.log'

# Initialize default logger
project_logger = setup_logger(
    'lab2_pipeline',
    log_file=str(_default_log_file),
    level=logging.INFO
)


class ProgressLogger:
    """
    Helper class for logging progress
    """
    
    def __init__(self, logger, total, desc="Processing"):
        """
        Args:
            logger: Logger instance
            total: Total number of items
            desc: Description for progress
        """
        self.logger = logger
        self.total = total
        self.desc = desc
        self.current = 0
    
    def update(self, n=1, message=None):
        """
        Update progress
        
        Args:
            n: Number of items completed
            message: Optional message to log
        """
        self.current += n
        percent = (self.current / self.total) * 100 if self.total > 0 else 0
        
        log_msg = f"{self.desc}: {self.current}/{self.total} ({percent:.1f}%)"
        if message:
            log_msg += f" - {message}"
        
        self.logger.info(log_msg)
    
    def complete(self):
        """Log completion"""
        self.logger.info(f"{self.desc}: Complete! ({self.total} items)")


def log_statistics(logger, stats_dict, title="Statistics"):
    """
    Log statistics in a formatted way
    
    Args:
        logger: Logger instance
        stats_dict: Dictionary of statistics
        title: Title for the statistics block
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"{title}")
    logger.info(f"{'='*50}")
    
    for key, value in stats_dict.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    
    logger.info(f"{'='*50}\n")