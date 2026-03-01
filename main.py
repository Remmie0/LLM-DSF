import os
import itertools
import copy
from typing import Any, Dict

from LLM_DSF.central_logger import get_logger, setup_root_logger
from LLM_DSF.LLM import initialize_model
from LLM_DSF.input_parser import read_data_file, read_txt_file
from LLM_DSF.experiment import run_temperature_experiment

from LLM_DSF.cli import parse_arguments
from LLM_DSF.inference_runner import run_single_inference
from LLM_DSF.write_logs import write_iteration_log

# ==============================================================================
# DEFAULT CONFIG (set your preferred defaults here)
# - CLI defaults are populated from this dict.
# ==============================================================================
DEFAULT_CONFIG: Dict[str, Any] = {
    # Run mode defaults
    "mode": "single",

    # Paths 
    "model_path": "models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf",
    "data_file": "model-input/KNMI Hard/KNMI_20191231.csv",
    "prompt_file": "model-input/KNMI Hard/prompt feature models.txt",
    "feedback_file": "models/feedback prompt/prompt_5.txt",

    # Model settings
    "model_type": "gguf",
    "openai_model_name": "gpt-5.2-chat-latest",
    "gemini_model_name": "gemini-3-flash-preview",

    # Data settings
    "data_file_type": "data",
    "openml_task_id": None,

    # Unified experiment controls
    "number_experiment_runs": 3,
    "manual_label_experiment": "",

    # Inference + preprocessing defaults
    "include_metadata": True,
    "sample_rows": 5,
    "random_seed": 42,
    "test_size": 0.2,
    "n_ctx": 16384,
    "n_gpu_layers": -1,
    "temperature": 0.2,

    # Retry/feedback defaults
    "retry_errors": True,
    "use_feedback": True,
    "max_feedback_retries": 3,
    "max_error_retries": 3,

    # Logging + output structure
    "main_log_file": "main.log",
    "experiments_base_dir": "experiments",
    "experiments_subdirs": {
        "repeat_experiment": "repeat_experiments",
        "ablation_study": "ablation_study",
        "temperature_experiment": "temperature_experiments",
    },
    "default_experiment_labels": {
        "repeat_experiment": "unlabeled_run",
        "ablation_study": "ablation_run",
        "temperature_experiment": "",
    },
}

# ==============================================================================
# ABLATION STUDY CONFIGURATION
# Keys must match the argument names.
# ==============================================================================
ABLATION_PARAMS: Dict[str, list[Any]] = {
    "include_metadata": [True, False],
    "retry_errors": [True, False],
    "use_feedback": [True, False],
}

def main() -> None:
    setup_root_logger()
    logger = get_logger("main", log_file=DEFAULT_CONFIG["main_log_file"])

    args = parse_arguments(DEFAULT_CONFIG)

    try:
        model_path: str = args.model_path
        data_file_path: str = args.data_file
        prompt_file_path: str = args.prompt_file
        feedback_file_path: str = args.feedback_file

        for p in [data_file_path, prompt_file_path, feedback_file_path]:
            if not os.path.exists(p):
                logger.error(f"File not found: {p}")
                return
            logger.info(f"File path used: {p}")

        logger.info(f"Initializing the LLM model: {model_path}")
        model = initialize_model(model_path=model_path, n_gpu_layers=args.n_gpu_layers, n_ctx=args.n_ctx, verbose=False)

        logger.info(f"Reading task prompt from: {prompt_file_path}")
        task_prompt: str = read_txt_file(prompt_file_path)

        logger.info(f"Reading feedback prompt from: {feedback_file_path}")
        feedback_prompt: str = read_txt_file(feedback_file_path)

        # Preprocess data once for modes that do not change preprocessing parameters
        logger.info(f"Reading and preprocessing input data from: {data_file_path}")
        formatted_data, metadata, train_df, test_df, complete_df = read_data_file(
            data_file_path,
            include_metadata=args.include_metadata,
            sample_rows=args.sample_rows,
            random_seed=args.random_seed,
            test_size=args.test_size,
        )

        # ----------------------------------------------------------------------
        # MODE: ABLATION STUDY
        # ----------------------------------------------------------------------
        if args.mode == "ablation_study":
            base_dir = os.path.join("experiments", "ablation_study")
            label = args.manual_label_experiment or "ablation_run"
            experiment_dir = os.path.join(base_dir, label)
            os.makedirs(experiment_dir, exist_ok=True)
            logger.info(f"Starting Ablation Study. Results in: {experiment_dir}")

            keys, values = zip(*ABLATION_PARAMS.items())
            combinations = list(itertools.product(*values))
            logger.info(f"Generated {len(combinations)} combinations for ablation study.")

            master_config_path = os.path.join(experiment_dir, "master_config.txt")
            with open(master_config_path, "w", encoding="utf-8") as f:
                f.write("### ABLATION STUDY MASTER CONFIG ###\n\n")
                f.write(f"Parameters varied: {list(keys)}\n\n")
                for idx, combo in enumerate(combinations):
                    f.write(f"Config {idx}: {dict(zip(keys, combo))}\n")

            for config_idx, combo in enumerate(combinations):
                current_params = dict(zip(keys, combo))
                logger.info(f"Starting Config {config_idx}: {current_params}")

                current_args = copy.deepcopy(args)
                for k, v in current_params.items():
                    setattr(current_args, k, v)

                # Re-read data if preprocessing-related params differ
                formatted_data_cfg, metadata_cfg, train_df_cfg, test_df_cfg, complete_df_cfg = read_data_file(
                    data_file_path,
                    include_metadata=current_args.include_metadata,
                    sample_rows=current_args.sample_rows,
                    random_seed=current_args.random_seed,
                    test_size=current_args.test_size,
                )

                for run_idx in range(current_args.number_experiment_runs):
                    logger.info(
                        f"Config {config_idx} - Run {run_idx + 1}/{current_args.number_experiment_runs}"
                    )

                    output, error, final_code, code_before_feedback = run_single_inference(
                        logger=logger,
                        model=model,
                        task_prompt=task_prompt,
                        feedback_prompt=feedback_prompt,
                        formatted_data=formatted_data_cfg,
                        metadata=metadata_cfg if current_args.include_metadata else None,
                        complete_df=complete_df_cfg,
                        max_length=current_args.n_ctx,
                        temperature=current_args.temperature,
                        retry_errors=current_args.retry_errors,
                        use_feedback=current_args.use_feedback,
                        max_retries_error=current_args.max_error_retries,
                        max_retries_feedback=current_args.max_feedback_retries,
                    )

                    filepath = os.path.join(experiment_dir, f"config{config_idx}_iteration{run_idx + 1}.txt")
                    write_iteration_log(
                        filepath=filepath,
                        code_before_feedback=code_before_feedback,
                        final_code=final_code,
                        output=output,
                        error=error,
                    )

            logger.info("Ablation study completed.")
            return

        # ----------------------------------------------------------------------
        # MODE: SINGLE
        # ----------------------------------------------------------------------
        if args.mode == "single":
            logger.info("Running in single inference mode.")
            output, error, final_code, code_before_feedback = run_single_inference(
                logger=logger,
                model=model,
                task_prompt=task_prompt,
                feedback_prompt=feedback_prompt,
                formatted_data=formatted_data,
                metadata=metadata if args.include_metadata else None,
                complete_df=complete_df,
                max_length=args.n_ctx,
                temperature=args.temperature,
                retry_errors=args.retry_errors,
                use_feedback=args.use_feedback,
                max_retries_error=args.max_error_retries,
                max_retries_feedback=args.max_feedback_retries,
            )

            logger.info("Single inference completed successfully.")
            logger.info(f"Code Before Feedback:\n{code_before_feedback}")
            logger.info(f"Final Error:\n{error}")
            logger.info(f"Final Code:\n{final_code}")
            logger.info(f"Final Output:\n{output}")
            return

        # ----------------------------------------------------------------------
        # MODE: TEMPERATURE EXPERIMENT
        # ----------------------------------------------------------------------
        if args.mode == "temperature_experiment":
            base_dir = os.path.join("experiments", "temperature_experiments")
            label = args.manual_label_experiment.strip()
            experiment_dir = os.path.join(base_dir, label) if label else base_dir
            os.makedirs(experiment_dir, exist_ok=True)

            logger.info(
                f"Running in temperature experiment mode with {args.number_experiment_runs} runs per temperature."
            )
            logger.info(f"Saving temperature experiment results to: {experiment_dir}")

            _ = run_temperature_experiment(
                model=model,
                model_path=model_path,
                prompt=task_prompt,
                data=formatted_data,
                df=complete_df,
                metadata=metadata if args.include_metadata else None,
                num_runs=args.number_experiment_runs,
                manual_label="",          
                output_dir=experiment_dir, 
                max_length=args.n_ctx,
            )

            logger.info("Temperature experiment completed successfully.")
            return

        # ----------------------------------------------------------------------
        # MODE: REPEAT EXPERIMENT
        # ----------------------------------------------------------------------
        if args.mode == "repeat_experiment":
            base_dir = os.path.join("experiments", "repeat_experiments")
            label = args.manual_label_experiment or "unlabeled_run"
            experiment_dir = os.path.join(base_dir, label)
            os.makedirs(experiment_dir, exist_ok=True)

            logger.info(f"Running in repeat experiment mode with {args.number_experiment_runs} runs.")
            logger.info(f"Logging results to directory: {experiment_dir}")

            config_path = os.path.join(experiment_dir, "config.txt")
            with open(config_path, "w", encoding="utf-8") as f:
                f.write("############################################################################\n")
                f.write("### EXPERIMENT CONFIGURATION\n")
                f.write("############################################################################\n\n")
                for key, value in vars(args).items():
                    f.write(f"{key}: {value}\n")

            for i in range(args.number_experiment_runs):
                logger.info(f"Starting repeat experiment run {i + 1} out of {args.number_experiment_runs}.")

                output, error, final_code, code_before_feedback = run_single_inference(
                    logger=logger,
                    model=model,
                    task_prompt=task_prompt,
                    feedback_prompt=feedback_prompt,
                    formatted_data=formatted_data,
                    metadata=metadata if args.include_metadata else None,
                    complete_df=complete_df,
                    max_length=args.n_ctx,
                    temperature=args.temperature,
                    retry_errors=args.retry_errors,
                    use_feedback=args.use_feedback,
                    max_retries_error=args.max_error_retries,
                    max_retries_feedback=args.max_feedback_retries,
                )

                iteration_file_path = os.path.join(experiment_dir, f"iteration{i + 1}.txt")
                write_iteration_log(
                    filepath=iteration_file_path,
                    code_before_feedback=code_before_feedback,
                    final_code=final_code,
                    output=output,
                    error=error,
                )
                logger.info(f"Repeat experiment run {i + 1} completed. Logs saved to iteration{i + 1}.txt")
            return

        logger.error(f"Invalid mode selected: {args.mode}")

    except Exception as e:
        logger.exception(f"An unexpected error occurred:\n\n{e}")

    finally:
        logger.info("############################################################################\n")


if __name__ == "__main__":
    main()