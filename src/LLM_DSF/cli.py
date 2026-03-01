import argparse
from typing import Any, Mapping


def parse_arguments(defaults: Mapping[str, Any]) -> argparse.Namespace:
    """
    Parse command line arguments.

    defaults:
        Central defaults passed from main.py DEFAULT_CONFIG, so we don't hardcode
        fallback strings in multiple places.
    """
    parser = argparse.ArgumentParser(description="Run LLM code generation and execution")

    # Run mode arguments
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "temperature_experiment", "repeat_experiment", "ablation_study"],
        default=str(defaults.get("mode", "single")),
        help="Run mode: single, temperature_experiment, repeat_experiment, or ablation_study",
    )

    # Unified experiment parameters (ignored in single mode)
    parser.add_argument(
        "--number_experiment_runs",
        type=int,
        default=int(defaults.get("number_experiment_runs", 3)),
        help=(
            "Number of runs for experiments. "
            "temperature_experiment: runs per temperature; "
            "repeat_experiment: total runs; "
            "ablation_study: runs per config."
        ),
    )
    parser.add_argument(
        "--manual_label_experiment",
        type=str,
        default=str(defaults.get("manual_label_experiment", "")),
        help="Experiment label used for output folder naming (non-single modes).",
    )

    # Model and data arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(defaults.get("model_path", "")),
        help="Path to the model file either absolute or relative to the thesis-code-llm folder",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["gguf", "openai", "gemini"],
        default=str(defaults.get("model_type", "gguf")),
        help="Type of model to use: gguf, openai, or gemini",
    )
    parser.add_argument(
        "--openai_model_name",
        type=str,
        default=str(defaults.get("openai_model_name", "gpt-5.2-chat-latest")),
        help="Name of the OpenAI model to use if model_type is openai",
    )
    parser.add_argument(
        "--gemini_model_name",
        type=str,
        default=str(defaults.get("gemini_model_name", "gemini-3-flash-preview")),
        help="Name of the Gemini model to use if model_type is gemini",
    )

    parser.add_argument(
        "--data_file",
        type=str,
        default=str(defaults.get("data_file", "")),
        help="Path to the data file either absolute or relative to the thesis-code-llm folder",
    )
    parser.add_argument(
        "--data_file_type",
        type=str,
        choices=["data", "openml"],
        default=str(defaults.get("data_file_type", "data")),
        help="Type of the data file: data (CSV/JSON) or openml",
    )
    parser.add_argument(
        "--openml_task_id",
        type=int,
        default=defaults.get("openml_task_id", None),
        help="OpenML task ID (used if data_file_type is openml)",
    )

    parser.add_argument(
        "--prompt_file",
        type=str,
        default=str(defaults.get("prompt_file", "")),
        help="Path to the prompt file either absolute or relative to the thesis-code-llm folder",
    )
    parser.add_argument(
        "--feedback_file",
        type=str,
        default=str(defaults.get("feedback_file", "")),
        help="Path to the feedback file either absolute or relative to the thesis-code-llm folder",
    )

    # Inference arguments
    parser.add_argument(
        "--include_metadata",
        action="store_true",
        default=bool(defaults.get("include_metadata", True)),
        help="Whether to include metadata in the prompt",
    )
    parser.add_argument("--sample_rows", type=int, default=int(defaults.get("sample_rows", 5)))
    parser.add_argument("--random_seed", type=int, default=int(defaults.get("random_seed", 42)))
    parser.add_argument("--test_size", type=float, default=float(defaults.get("test_size", 0.2)))
    parser.add_argument("--n_ctx", type=int, default=int(defaults.get("n_ctx", 16384)))
    parser.add_argument("--n_gpu_layers", type=int, default=int(defaults.get("n_gpu_layers", -1)))
    parser.add_argument("--temperature", type=float, default=float(defaults.get("temperature", 0.2)))

    # Feedback loop arguments
    parser.add_argument(
        "--retry_errors",
        action="store_true",
        default=bool(defaults.get("retry_errors", True)),
        help="Whether to retry code generation when code fails",
    )
    parser.add_argument(
        "--use_feedback",
        action="store_true",
        default=bool(defaults.get("use_feedback", True)),
        help="Whether to run feedback loop (expert feedback) during generation",
    )
    parser.add_argument("--max_feedback_retries", type=int, default=int(defaults.get("max_feedback_retries", 3)))
    parser.add_argument("--max_error_retries", type=int, default=int(defaults.get("max_error_retries", 3)))

    return parser.parse_args()