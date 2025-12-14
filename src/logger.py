import logging
import os
from datetime import datetime

def get_logger(name, log_dir='logs'):
    """
    Creates a logger instance that logs to both console and a file.
    
    Args:
        name (str): The name of the logger (usually __name__).
        log_dir (str): Directory where log files will be stored.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename with timestamp
    log_file = os.path.join(log_dir, f"pipeline_{datetime.now().strftime('%Y%m%d')}.log")
    
    # Create Logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        # File Handler
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Console Handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
    return logger
