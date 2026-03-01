"""
Logging configuration for RAG Web Crawler
Uses loguru for structured, colored logging
"""

import sys
from pathlib import Path
from loguru import logger
from src.utils.config import config


def setup_logger():
    """
    Configure loguru logger with settings from config
    
    Features:
    - Colored console output
    - File rotation and retention
    - Structured logging with context
    - Different log levels
    """
    # Remove default handler
    logger.remove()
    
    # Get logging configuration
    log_config = config.logging
    log_level = log_config.get("level", "INFO")
    log_format = log_config.get(
        "format",
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
        "<level>{message}</level>"
    )
    
    # Console handler with colors
    logger.add(
        sys.stdout,
        format=log_format,
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )
    
    # File handler with rotation
    logs_dir = Path(config.get("paths.logs", "./logs"))
    logs_dir.mkdir(exist_ok=True, parents=True)
    
    logger.add(
        logs_dir / "rag_crawler_{time:YYYY-MM-DD}.log",
        format=log_format,
        level=log_level,
        rotation=log_config.get("rotation", "10 MB"),
        retention=log_config.get("retention", "1 week"),
        compression="zip",
        backtrace=True,
        diagnose=True,
    )
    
    logger.info(f"Logger initialized with level: {log_level}")
    return logger


# Initialize logger on import
setup_logger()


def get_logger(name: str = None):
    """
    Get a logger instance
    
    Args:
        name: Optional name for the logger (e.g., module name)
        
    Returns:
        Configured logger instance
        
    Examples:
        >>> from src.utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting crawler...")
    """
    if name:
        return logger.bind(name=name)
    return logger


if __name__ == "__main__":
    # Test logging
    test_logger = get_logger("test")
    test_logger.debug("This is a debug message")
    test_logger.info("This is an info message")
    test_logger.warning("This is a warning message")
    test_logger.error("This is an error message")
    test_logger.success("This is a success message")
