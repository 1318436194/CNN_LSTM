import os
import logging
import sys
from typing import Optional


def setup_logger(
    log_dir: str = 'logs',
    log_level: int = logging.INFO,
    log_file: Optional[str] = 'training.log'
) -> logging.Logger:
    """
    Setup a logger for the project.
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level
        log_file: Log file name (if None, logs will only be printed to console)
    
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger('cnn_lstm')
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file is specified
    if log_file:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger