# Installation

This document describes how to install and run LLM-DSF on a Linux system. The steps were tested on Ubuntu 24.04. The supported route is the provided conda environment, since the framework depends on several libraries that are sensitive to platform and hardware configuration.

## Prerequisites

Required:

- Git
- Conda (Miniconda or Anaconda)
- Docker Engine

Optional, depending on your use case:

- NVIDIA GPU drivers, CUDA toolkit, and NVIDIA Container Toolkit (for GPU-enabled container execution)
- A local GGUF model file (for `--model_type gguf`)
- API keys for OpenAI or Gemini (for `--model_type openai` or `--model_type gemini`)

## 1. Clone the repository

```bash
git clone <your-public-repository-url>
cd <repository-directory>
```

## 2. Create and activate the conda environment

The repository includes `environment.yaml`.

```bash
conda env create -f environment.yaml
conda activate pyenv-gpu
```

## 3. Install the package

The command line entry point `main.py` imports modules from `LLM_DSF`, which lives under `src/`. Therefore, you should install the project as an editable package:

```bash
pip install -e .
```

## 4. Build the Docker image used for sandboxed execution

The framework executes generated code inside a container. Build the image once:

```bash
docker build -t code_executor .
```

After the image is built, the framework expects it under the name `code_executor:latest` by default.

### Docker permissions

On some systems, Docker requires root privileges unless your user is added to the `docker` group. If you add yourself to the group, you typically need to log out and log back in for the change to take effect.

## 5. GPU support inside Docker (optional)

If you want the container to access the GPU, you must install and configure the NVIDIA Container Toolkit and ensure your host drivers are compatible with your CUDA toolkit version.

Because GPU setup is system-dependent, consult the official NVIDIA and Docker documentation for your platform.

## 6. Install llama-cpp-python (optional, required for local GGUF models)

If you run local models (`--model_type gguf`), you need `llama-cpp-python`.

### CPU-only installation

If you use CPU only, you can typically install a prebuilt wheel:

```bash
pip install llama-cpp-python
```

### CUDA installation

GPU-enabled installations are hardware- and toolkit-dependent and may require compiling from source. Follow the official instructions for your target backend. A typical pattern is to set `CMAKE_ARGS` and reinstall, but the exact flags depend on your CUDA version and compute capability.

## 7. API-based models (optional)

### OpenAI

Set `OPENAI_API_KEY` in your environment. The repository ignores `.env` by default, so a common approach is to create a local `.env` file in the repository root:

```bash
OPENAI_API_KEY=...
```

Then run with:

```bash
python ./main.py --model_type openai --openai_model_name <model-name> ...
```

### Gemini

Set `GEMINI_API_KEY` similarly:

```bash
GEMINI_API_KEY=...
```

Then run with:

```bash
python ./main.py --model_type gemini --gemini_model_name <model-name> ...
```

## 8. Verify the installation

Run unit tests:

```bash
pytest
```

## 9. Example command

The following example runs a repeated experiment on an OpenML task:

```bash
python ./main.py \
  --mode repeat_experiment \
  --manual_label_repeat framework_openml_168757_creditg \
  --prompt_file "model-input/OpenML/prompt_168757.txt" \
  --number_runs_repeat 1 \
  --data_file_type openml \
  --openml_task_id 168757
```

## Troubleshooting

### Docker image not found

If container execution fails because the image does not exist, rebuild it:

```bash
docker build -t code_executor .
```

### Docker permission errors

If you see permission errors when running Docker, ensure Docker is installed and your user has permission to access the Docker daemon.

### Missing model files

The default `--model_path` points to a local GGUF file that is not expected to be included in a public repository. Provide your own model path explicitly or use an API-based backend.

### API authentication issues

If you use OpenAI or Gemini backends, verify that the relevant environment variable is set and visible in the shell where you start `python ./main.py`.

### Final notes

If you can't get the GPU support working for either llama-cpp or docker it is possible to run in CPU mode, however this large increases the runtime.
