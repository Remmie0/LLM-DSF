"""
Centralized logging configuration for the thesis-code-llm project.
"""
import logging
import os
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
LOG_DIR = PROJECT_ROOT / "logging"

def ensure_log_dir():
    """Ensure the logging directory exists."""
    os.makedirs(LOG_DIR, exist_ok=True)

def get_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console_output: bool = True,
    format_string: str = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name
        log_file: Log file name (if None, only console output)
        level: Logging level
        console_output: Whether to add console handler
        format_string: Log message format string

    Returns:
        Configured logger instance
    """
    ensure_log_dir()

    # Create or get logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplication
    if logger.handlers:
        logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(format_string)

    # Add file handler if specified
    if log_file:
        file_path = LOG_DIR / log_file
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

def get_llm_response_logger() -> logging.Logger:
    """
    Get a specialized logger for LLM responses with a simpler format.

    Returns:
        Configured logger for LLM responses
    """
    ensure_log_dir()

    logger = logging.getLogger("llm_responses")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Remove existing handlers to avoid duplication
    if logger.handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(LOG_DIR / "llm_responses.log")
    file_handler.setFormatter(logging.Formatter('%(message)s\n---\n'))
    logger.addHandler(file_handler)

    return logger

def get_llm_feedback_logger() -> logging.Logger:
    """
    Get a specialized logger for LLM feedback with a simpler format.

    Returns:
        Configured logger for LLM feedback
    """
    ensure_log_dir()

    logger = logging.getLogger("llm_feedback")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Remove existing handlers to avoid duplication
    if logger.handlers:
        logger.handlers.clear()

    file_handler = logging.FileHandler(LOG_DIR / "llm_feedback.log")
    file_handler.setFormatter(logging.Formatter('%(message)s\n---\n'))
    logger.addHandler(file_handler)

    return logger

def setup_root_logger(level: int = logging.INFO):
    """Set up root logger for the entire application."""
    ensure_log_dir()

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    if root_logger.handlers:
        root_logger.handlers.clear()

    # File handler for everything
    file_handler = logging.FileHandler(LOG_DIR / "main.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    root_logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    root_logger.addHandler(console_handler)
