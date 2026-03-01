import logging
from typing import Any, Optional, Tuple

from .container import run_code_in_container
from .LLM import generate_response, prepare_prompt, prepare_feedback_prompt
from .code_parser import extract_code_from_response


def run_single_inference(
    *,
    logger: logging.Logger,
    model: Any,
    task_prompt: str,
    feedback_prompt: str,
    formatted_data: str,
    metadata: Optional[str],
    complete_df: Any,
    max_length: int,
    temperature: float = 0.3,
    retry_errors: bool = True,
    use_feedback: bool = True,
    max_retries_error: int = 3,
    max_retries_feedback: int = 3,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Run one full inference cycle (generate -> extract code -> execute), optionally with:
    - bounded error retries
    - bounded expert-feedback loop

    Returns:
        (final_output, final_error, final_code, final_code_before_feedback)
    """
    error_max_attempts: int = (max_retries_error + 1) if retry_errors else 1

    final_code: Optional[str] = None
    final_code_before_feedback: Optional[str] = None
    final_output: Optional[str] = None
    final_error: Optional[str] = None

    prev_output: Optional[str] = None
    prev_error: Optional[str] = None
    prev_code: Optional[str] = None

    code: Optional[str] = None
    output: Optional[str] = None
    error: Optional[str] = None

    original_prompt: str = prepare_prompt(prompt=task_prompt, data=formatted_data, metadata=metadata)
    current_prompt: str = original_prompt

    # --------------------------
    # 1) Error-retry loop 
    # --------------------------
    for attempt_idx in range(1, error_max_attempts + 1):
        if retry_errors:
            logger.info(f"Generating response attempt {attempt_idx} out of {error_max_attempts} from the LLM.")
        else:
            logger.info("Generating response from the LLM.")

        generated_response: str = generate_response(model, current_prompt, max_length=max_length, temperature=temperature)

        logger.info("Processing generated code through parser.")
        code = extract_code_from_response(generated_response)

        if code is None:
            logger.warning("No code block found in the response.")
            if attempt_idx < error_max_attempts:
                current_prompt = prepare_feedback_prompt(original_prompt=original_prompt, code=None)
                logger.info("Retrying with feedback prompt.")
                continue

            final_output, final_error, final_code = None, "No code block found", None
            return final_output, final_error, final_code, final_code_before_feedback

        logger.info("Code parsing completed successfully.")
        final_code = code

        logger.info("Sending the generated code to the container for execution.")
        output, error = run_code_in_container(code, complete_df)

        final_output, final_error = output, error

        if error:
            logger.error(f"Error occurred during code execution: {error}")
            if attempt_idx < error_max_attempts:
                current_prompt = prepare_feedback_prompt(original_prompt, code, error)
                logger.info("Retrying with feedback prompt.")
                continue
            break

        logger.info("Code executed successfully without errors.")
        break

    # --------------------------
    # 2) Feedback loop 
    # --------------------------
    if use_feedback:
        feedback_max_attempts: int = max_retries_feedback + 1
        final_code_before_feedback = code

        for feedback_attempt in range(1, feedback_max_attempts + 1):
            logger.info(f"Generating feedback attempt {feedback_attempt} out of {feedback_max_attempts} from the LLM.")

            if feedback_attempt == 1:
                current_prompt = prepare_feedback_prompt(
                    original_prompt,
                    code,
                    feedback=feedback_prompt,
                    give_feedback=True,
                    prev_output="No previous output",
                    output=output,
                )
            else:
                current_prompt = prepare_feedback_prompt(
                    original_prompt,
                    code,
                    feedback=feedback_prompt,
                    give_feedback=True,
                    prev_output=prev_output,
                    output=output,
                )

            logger.info("Feedback prompt generated.")
            generated_feedback: str = generate_response(model, current_prompt, max_length=max_length, temperature=temperature)
            logger.info("Feedback response generated.")
            logger.info(generated_feedback)

            if "PREVIOUS OUTPUT WAS BETTER STOPPING FEEDBACK LOOP" in generated_feedback:
                logger.info("Feedback indicated previous output was better, stopping feedback loop.")
                return prev_output, prev_error, prev_code, final_code_before_feedback

            # Ask model to apply received feedback
            current_prompt = prepare_feedback_prompt(
                original_prompt,
                code,
                feedback=generated_feedback,
                received_feedback=True,
                output=output,
            )

            # One bounded attempt per feedback iteration:
            generated_response = generate_response(model, current_prompt, max_length=max_length, temperature=temperature)
            logger.info("Processing generated code through parser (post-feedback).")
            improved_code = extract_code_from_response(generated_response)

            if improved_code is None:
                logger.warning("No code block found in the post-feedback response.")
                continue

            logger.info("Sending the post-feedback code to the container for execution.")
            improved_output, improved_error = run_code_in_container(improved_code, complete_df)

            if improved_error:
                logger.error(f"Error occurred during post-feedback execution: {improved_error}")
                continue

            # Keep previous as fallback
            prev_code, prev_output, prev_error = final_code, final_output, final_error
            final_code, final_output, final_error = improved_code, improved_output, improved_error
            logger.info("Post-feedback code executed successfully.")

    return final_output, final_error, final_code, final_code_before_feedback