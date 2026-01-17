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
    Helper class for logging progress through pipeline steps
    """
    
    def __init__(self, total_steps=None, desc="Processing", logger=None):
        """
        Args:
            total_steps: Total number of steps (for step-based progress)
            desc: Description for progress
            logger: Logger instance (optional, uses default if not provided)
        """
        # Handle both old and new constructor signatures
        # Use type() instead of isinstance() to avoid recursion issues
        if type(total_steps).__name__ == 'Logger':
            # Old signature: ProgressLogger(logger, total, desc)
            self.logger = total_steps
            self.total = desc if type(desc).__name__ == 'int' else 0
            self.desc = "Processing"
        else:
            # New signature: ProgressLogger(total_steps, desc, logger)
            self.logger = logger or project_logger
            self.total = total_steps or 0
            self.desc = desc
        
        self.current = 0
        self.steps = {}
        self.start_time = datetime.now()
    
    def start_step(self, step_num, step_name):
        """
        Mark the start of a step
        
        Args:
            step_num: Step number
            step_name: Name of the step
        """
        self.steps[step_num] = {
            'name': step_name,
            'status': 'in_progress',
            'start_time': datetime.now()
        }
        self.logger.info(f"[Step {step_num}/{self.total}] Starting: {step_name}")
    
    def complete_step(self, step_num, message=None):
        """
        Mark a step as complete
        
        Args:
            step_num: Step number
            message: Optional completion message
        """
        if step_num in self.steps:
            self.steps[step_num]['status'] = 'complete'
            self.steps[step_num]['end_time'] = datetime.now()
            duration = self.steps[step_num]['end_time'] - self.steps[step_num]['start_time']
            step_name = self.steps[step_num]['name']
            
            msg = f"[Step {step_num}/{self.total}] ✓ Complete: {step_name} ({duration.total_seconds():.1f}s)"
            if message:
                msg += f" - {message}"
            self.logger.info(msg)
        
        self.current = step_num
    
    def skip_step(self, step_num, reason=None):
        """
        Mark a step as skipped
        
        Args:
            step_num: Step number
            reason: Reason for skipping
        """
        msg = f"[Step {step_num}/{self.total}] ⊘ Skipped"
        if reason:
            msg += f": {reason}"
        self.logger.info(msg)
        
        self.steps[step_num] = {
            'name': reason or 'Skipped',
            'status': 'skipped'
        }
    
    def update(self, n=1, message=None):
        """
        Update progress (for item-based tracking)
        
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
    
    def complete(self, message=None):
        """Log completion of entire process"""
        duration = datetime.now() - self.start_time
        msg = f"✅ {self.desc}: Complete! (Total time: {duration.total_seconds():.1f}s)"
        if message:
            msg += f" - {message}"
        self.logger.info(msg)
    
    def error(self, step_num, error_msg):
        """
        Log an error in a step
        
        Args:
            step_num: Step number
            error_msg: Error message
        """
        self.logger.error(f"[Step {step_num}/{self.total}] ✗ Error: {error_msg}")
        if step_num in self.steps:
            self.steps[step_num]['status'] = 'error'
            self.steps[step_num]['error'] = error_msg


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
        # Use type() instead of isinstance() to avoid recursion issues
        if type(value).__name__ == 'float':
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    
    logger.info(f"{'='*50}\n")