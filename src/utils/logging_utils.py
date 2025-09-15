"""
Logging configuration and utilities for the fraud detection system.
"""
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Union, Dict, Any
import json

from .config import get_config

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        log_record = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process': record.process,
            'thread': record.thread,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_record, ensure_ascii=False)

def setup_logging(
    name: str = None,
    log_level: Optional[Union[str, int]] = None,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    log_dir: str = 'logs'
) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        name: Logger name. If None, creates a root logger.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Path to the log file. If None, logs to console only.
        log_format: Log format string. If None, uses a default format.
        log_dir: Directory to store log files.
        
    Returns:
        Configured logger instance.
    """
    # Get configuration
    config = get_config()
    
    # Set default values from config if not provided
    if log_level is None:
        log_level = config.get('logging.level', 'INFO')
    
    if log_format is None:
        log_format = config.get(
            'logging.format',
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Convert string log level to logging constant
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatter
    if config.get('logging.json_format', False):
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if log_file is specified
    if log_file:
        # Create log directory if it doesn't exist
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Add date to log filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file_path = log_dir_path / f"{Path(log_file).stem}_{timestamp}{Path(log_file).suffix}"
        
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

def log_execution_time(logger: logging.Logger = None):
    """Decorator to log the execution time of a function.
    
    Args:
        logger: Logger instance. If None, creates a new logger.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    def decorator(func):
        import time
        from functools import wraps
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.info(f"Starting {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                elapsed_time = time.time() - start_time
                logger.info(
                    f"Completed {func.__name__} in {elapsed_time:.2f} seconds"
                )
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                raise
        
        return wrapper
    
    return decorator

# Create a default logger instance
default_logger = setup_logging('fraud_detection')
