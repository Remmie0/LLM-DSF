from llama_cpp import Llama
import logging
from datetime import datetime
from typing import Optional
import re
import os
from google import genai

from .central_logger import get_logger, get_llm_response_logger, get_llm_feedback_logger

# Configure logging
llm_logger = get_logger("llm", log_file="llm.log")

# Configure seperate logging for the LLM Responses and Feedback outside the main logging
responses_logger = get_llm_response_logger()
feedback_logger = get_llm_feedback_logger()

DEFAULT_GPT_SYSTEM_INSTRUCTIONS = (
    "You are a coding assistant. "
    "Return ONLY a single Python code block enclosed in ```python and ``` (no extra text). "
    "Assume pandas is already imported as pd and the dataset is already loaded as df. "
    "Do not read files from disk unless explicitly instructed."
)

def initialize_model(model_path: str, n_gpu_layers: int = 0, n_ctx: int = 32768, flash_attn: bool = True, verbose: bool = False, model_type: str = 'gguf', openai_model_name: str = 'gpt-5.2-chat-latest'):
    """Initialize the Llama model for GGUF format."""
    if model_type == 'openai':
        from openai import OpenAI
        llm_logger.info(f"Initializing OpenAI model: {openai_model_name}")
        client = OpenAI()
        llm_logger.info("OpenAI model successfully initialized.")
        return client
    
    if model_type == 'gemini':
        llm_logger.info("Gemini model selected, no local initialization required.")
        return None  # Gemini model will be handled in generate_response
    try:
        llm_logger.info(f"Initializing model from path: {model_path}")
        model = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, verbose=verbose, n_ctx=n_ctx, flash_attn=flash_attn)
        llm_logger.info("Model successfully initialized.")
        return model
    except Exception as e:
        llm_logger.exception("Failed to initialize model.")
        raise e

def generate_response(model, prompt: str, max_length: int = 32768, temperature: float = 0.6, model_type: str = 'gguf', openai_model_name: str = 'gpt-5.2-chat-latest', gemini_model_name: str = 'gemini-2.5-flash') -> str:
    """Generate code based on data, prompt, and/or feedback."""
    try:

        if model_type == 'openai':
            response = model.responses.create(
                model=openai_model_name,
                instructions=DEFAULT_GPT_SYSTEM_INSTRUCTIONS,
                input=prompt,
            )
            response = response.output_text

        elif model_type == 'gemini':
            client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
            llm_logger.info(f"Generating response using Gemini model: {gemini_model_name}")
            response = client.models.generate_content(
                model=gemini_model_name,
                contents=prompt,
            )
            response = response.text
        else:
            response = model(
                prompt,
                max_tokens=max_length,
                temperature=temperature,
            )["choices"][0]["text"]

        # Log the complete response with timestamp and metadata
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        response_log = (
            f"Timestamp: {timestamp}\n"
            f"Prompt:\n{prompt}\n\n"
            f"Response:\n{response}\n"
            f"{'='*80}"
        )
        responses_logger.info(response_log)

        llm_logger.info("Response successfully generated.")
        return response

    except Exception as e:
        llm_logger.exception("Error during response generation.")
        raise e

def prepare_prompt(prompt: str, data: str, metadata: Optional[str]) -> str:
    """Prepare full prompt with data and optional metadata."""
    if metadata:
        return f"{prompt}\n\nData Structure:\n{metadata}\n\nData Sample:\n{data}"
    return f"{prompt}\n\nData:\n{data}"

def prepare_feedback_prompt(original_prompt: str, code: Optional[str] = None, error: Optional[str] = None, feedback: Optional[str] = None, give_feedback: bool = False, received_feedback: bool = False, prev_output: Optional[str] = None, output: Optional[str] = None) -> str:
    """
    Prepare a feedback prompt based on the error or missing code. Also handles feedback scenarios by second LLM.
    """
    # No code was found in the response
    if code is None:
        feedback_logger.error("No code block found in the response.")
        feedback = (
            f"{original_prompt}\n\n"
            f"Your previous response did not contain a valid code block. "
            f"Please provide a complete solution enclosed in ```python and ``` tags."
            f"Please also to keep in mind that the import and variable loading has been done already you should not include it in your code.\n\n"
        )
        feedback_logger.info(feedback)
        return feedback
    
    # Code was found but produced an error
    elif error:
        feedback_logger.error(f"Code produced an error: {error}")
        feedback = (
            f"{original_prompt}\n\n"
            f"Your previous solution:\n```python\n{code}\n```\n\n"
            f"Error encountered:\n{error}\n\n"
            f"Please fix these issues and provide an improved solution.\n\n"
        )
        feedback_logger.info(feedback)
        return feedback
    
        # Providing feedback to the model
    elif give_feedback:
        feedback_logger.info("Incorporating provided feedback into prompt.")
        feedback = (
            f"{feedback}\n\n"
            f"{code}\n\n"
            f"OUTPUT: \n{output}"
            f"PREV OUTPUT: \n{prev_output}"
        )

        feedback_logger.info(feedback)
        return feedback
    
    # Feedback was provided to improve the code
    elif received_feedback:
        feedback_logger.info("Incorporating received feedback into prompt.")
        feedback = (
            f"{original_prompt}\n\n"
            f"Your previous solution:\n```python\n{code}\n```\n\n"
            f"Your previous output:\n```python\n{output}\n```\n\n"
            f"Expert Feedback:\n{feedback}\n\n"
            f"Please improve your solution based on this feedback.\n\n"
            f"Please also to keep in mind that the import and variable loading has been done already you should not include it in your code.\n\n"
        )
        feedback_logger.info(feedback)
        return feedback
    else:
        # Fallback to original prompt in case of other problems
        feedback_logger.info("No feedback modifications applied, returning original prompt.")
        return original_prompt
