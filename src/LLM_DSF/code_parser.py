import re
import logging
from typing import Optional
from .central_logger import get_logger

# Configure logging
parser_logger = get_logger("code_parser", log_file="code_parser.log")

def extract_code_from_response(response: str) -> Optional[str]:
    """
    Extracts the last Python code block from a text response.
    Handles both explicitly marked Python blocks and generic code blocks.

    Args:
        text (str): Input text containing code blocks

    Returns:
        Optional[str]: The last code block found, or None if no blocks found
    """
    try:
        parser_logger.info("Starting code block extraction")

        # Try to find Python-marked code blocks first
        python_pattern = r"```python\s*(.*?)```"
        matches = list(re.finditer(python_pattern, response, re.DOTALL | re.IGNORECASE))

        # If no Python blocks found, try generic code blocks
        if not matches:
            generic_pattern = r"```\s*(.*?)```"
            matches = list(re.finditer(generic_pattern, response, re.DOTALL))

        if matches:
            code = matches[-1].group(1).strip()
            parser_logger.info("Successfully extracted last code block")
            return code

        parser_logger.warning("No code blocks found in the text")
        return None

    except Exception as e:
        parser_logger.exception("Error during code block extraction")
        return None
