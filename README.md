# LLM-DSF (Large Language Model Data Science Framework)

LLM-DSF is a research framework that uses large language models (LLMs) to generate executable Python code for data science tasks, and evaluates the generated code in an isolated execution environment. The framework was developed in the context of a master's thesis at Leiden University and is published for transparency and reuse. The thesis name is LLM-DSF: Harnessing the powers of large language models for automating data science and is written by Remco Stuij.

## Scope and goals

The primary goal of this repository is to support reproducible experimentation with LLM-driven code generation workflows for tabular data analysis and modeling tasks.

Key design elements include:

- Prompt construction from a task description, an optional schema summary, and a small data sample.
- Code extraction from model responses.
- Sandboxed execution of the generated code inside a Docker container.
- Optional retry and feedback loops for iterative improvement.
- Experiment modes for repeated runs and ablation-style configurations.
- Support for local GGUF models (via llama-cpp-python) and API-based models (OpenAI and Gemini).

## Requirements

- Python 3.12
- Conda (recommended for environment management)
- Docker Engine (required for code execution sandboxing)
- Optional: NVIDIA GPU drivers and NVIDIA Container Toolkit (for GPU-enabled Docker execution)
- Optional: a local GGUF model file (for `--model_type gguf`)
- Optional: API credentials (for `--model_type openai` or `--model_type gemini`)

For installation steps, see `INSTALLATION.md`.

## Quick start

1. Create and activate the conda environment:
   - `conda env create -f environment.yaml`
   - `conda activate pyenv-gpu`

2. Install the project as an editable package (required so `LLM_DSF` imports resolve):
   - `pip install -e .`

3. Build the Docker image used for sandboxed execution:
   - `docker build -t code_executor .`

4. Set your defaults in main.py

5. Run an example OpenML task experiment (repeat experiment mode):

```bash
python ./main.py \
  --mode repeat_experiment \
  --manual_label_repeat framework_openml_168757_creditg \
  --prompt_file "model-input/OpenML/prompt_168757.txt" \
  --number_runs_repeat 1 \
  --data_file_type openml \
  --openml_task_id 168757
```

Notes:

- In OpenML mode, the task dataset is fetched inside the execution container using the provided `openml_task_id`.
- The prompt file controls the requested task, evaluation measure, and reporting requirements. An example can be found under `model-input/Open ML Sample Prompt.txt`

## Typical workflow

1. Select a data source:
   - Local file input (`--data_file_type data` and `--data_file <path>`)
   - OpenML task input (`--data_file_type openml` and `--openml_task_id <id>`)

2. Provide a task prompt and, optionally, a feedback prompt:
   - `--prompt_file <path>`
   - `--feedback_file <path>`

3. Choose a model backend:
   - Local GGUF model (default): `--model_type gguf --model_path <path-to-gguf>`
   - OpenAI API: `--model_type openai --openai_model_name <model>`
   - Gemini API: `--model_type gemini --gemini_model_name <model>`

4. Run an experiment mode:
   - `single`: one run
   - `repeat_experiment`: multiple runs with logs per iteration
   - `temperature_experiment`: repeated runs across temperatures
   - `ablation_study`: runs across predefined parameter combinations

## Configuration and secrets

API keys must not be committed. The repository ignores `.env` by default.

To use API-based models, create a `.env` file in the repository root and set at least one of:

- `OPENAI_API_KEY` (OpenAI)
- `GEMINI_API_KEY` (Gemini)

## Repository structure

- `logging/`
  Writes all the logs at this location by default.
- `model-input/`  
  Example prompt and the folder I used for input of both data and prompts.
- `models/`  
  Location I used for the models, also has the prompts for the feedback mechanism.
- `src/LLM_DSF/`  
  Core framework modules (LLM abstraction, input parsing, code parsing, container execution, logging, experiments).
- `main.py`  
  Command line entry point for running experiments.
- `environment.yaml`  
  Conda environment definition used for the thesis experiments.
- `Dockerfile`  
  Defines the Docker image used for sandboxed execution.
- `tests/`
  Unit tests for key components.

If you reorganize directories, keep paths in `main.py` and your prompts consistent.

## Reproducibility considerations

- The framework uses explicit random seeds in preprocessing and sampling, but LLM generation is stochastic by design, therefor results can still differ.
- API-based models may change behavior over time as providers update deployments.
.

## Contributing

For development changes, a typical workflow is:

- `pip install -e .` (refresh editable install metadata)
- `pre-commit run --all-files`
- `pytest`

If you add new logic, add or update tests accordingly.

## Contact

For any questions or issues you can reach out to <remcostuij@gmail.com>.
