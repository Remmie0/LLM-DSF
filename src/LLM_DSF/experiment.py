import os
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from pathlib import Path

from .LLM import generate_response, prepare_prompt
from .code_parser import extract_code_from_response
from .container import run_code_in_container
from .central_logger import get_logger

# Configure logging
experiment_logger = get_logger("experiment", log_file="experiment.log")


def get_model_name(model_path: str) -> str:
    """Extract model name from path."""
    return Path(model_path).stem


def process_single_run(
    model: Any,
    full_prompt: str,
    temperature: float,
    df: pd.DataFrame,
    run_number: int,
    max_length: int = 32768
) -> Dict[str, Any]:
    """
    Process a single experimental run.

    Returns:
        Dict containing run results
    """
    try:
        response = generate_response(
            model,
            full_prompt,
            temperature=temperature,
            max_length=max_length
        )

        code = extract_code_from_response(response)

        if code is None:
            return {
                "run_number": run_number,
                "temperature": temperature,
                "output": None,
                "error": "No code block found in response",
                "is_correct": None,
            }

        output, error = run_code_in_container(code, df)

        return {
            "run_number": run_number,
            "temperature": temperature,
            "output": output,
            "error": error,
            "is_correct": None,
        }

    except Exception as e:
        error_msg = str(e)
        experiment_logger.error(f"Error in run {run_number} at temperature {temperature}: {error_msg}")
        return {
            "run_number": run_number,
            "temperature": temperature,
            "output": None,
            "error": error_msg,
            "is_correct": None,
        }


def save_experiment_results(
    results_df: pd.DataFrame,
    model_path: str,
    num_runs: int,
    manual_label: str = "",
    output_dir: Optional[str] = None,
) -> str:
    """
    Save experiment results to an Excel file.

    Folder logic:
      - If output_dir is provided: save directly into output_dir.
      - Else: save into experiments/temperature_experiments[/manual_label].

    Returns:
        Path to saved file
    """
    try:
        if output_dir is not None and output_dir.strip():
            results_dir = output_dir
        else:
            base_results_dir = os.path.join("experiments", "temperature_experiments")
            if manual_label:
                results_dir = os.path.join(base_results_dir, manual_label)
            else:
                results_dir = base_results_dir

        os.makedirs(results_dir, exist_ok=True)

        model_name = get_model_name(model_path)
        filename = f"{model_name}_nr_runs_{num_runs}_temperature_analysis.xlsx"
        filepath = os.path.join(results_dir, filename)

        results_df.to_excel(filepath, index=False)
        experiment_logger.info(f"Results saved to {filepath}")
        return filepath

    except Exception as e:
        experiment_logger.exception("Failed to save results")
        raise e


def run_temperature_experiment(
    model: Any,
    model_path: str,
    prompt: str,
    data: str,
    df: pd.DataFrame,
    metadata: Optional[str] = None,
    num_runs: int = 3,
    manual_label: str = "",
    output_dir: Optional[str] = None,
    temp_start: float = 0.0,
    temp_end: float = 1.0,
    temp_step: float = 0.1,
    max_length: int = 32768
) -> pd.DataFrame:
    """
    Run temperature analysis experiment.

    If output_dir is provided, results are saved there directly.
    """
    experiment_logger.info(f"Starting temperature analysis experiment with {num_runs} runs per temperature")

    try:
        full_prompt = prepare_prompt(prompt, data, metadata)

        results = []
        temperatures = np.arange(temp_start, temp_end + temp_step, temp_step)

        for temp in temperatures:
            experiment_logger.info(f"Running experiments for temperature {temp:.1f}")

            for run in range(1, num_runs + 1):
                experiment_logger.info(f"Starting run {run} for temperature {temp:.1f}")
                result = process_single_run(model, full_prompt, temp, df, run, max_length)
                results.append(result)

        results_df = pd.DataFrame(results)

        save_experiment_results(
            results_df=results_df,
            model_path=model_path,
            num_runs=num_runs,
            manual_label=manual_label,
            output_dir=output_dir,
        )

        return results_df

    except Exception as e:
        experiment_logger.exception("Experiment failed")
        raise e