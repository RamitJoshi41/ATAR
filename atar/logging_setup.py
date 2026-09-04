"""
Centralized logging setup for ATAR.
"""
import logging
import os
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger configured to output to both the console and logs/atar.log.
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "atar.log"))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger
