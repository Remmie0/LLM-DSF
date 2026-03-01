import docker
from docker.types import DeviceRequest
from typing import Tuple, Optional
import pandas as pd
import tempfile
import os
import time
import textwrap
import re

from .central_logger import get_logger

# Configure logging
container_logger = get_logger("container", log_file="container.log")

_DF_READ_RE = re.compile(
    r"^\s*df\s*=\s*(pd|pandas)\.read_[a-zA-Z0-9_]+\s*\(",
    re.IGNORECASE,
)


def _sanitize_llm_code(code: str) -> str:
    """
    Make LLM code safe to embed:
    - Dedent to avoid accidental leading indentation from markdown formatting.
    - Comment out obvious 'df = pd.read_*(' --> reloads that would fail in-container.
    """
    dedented = textwrap.dedent(code).strip("\n")
    out_lines: list[str] = []

    for line in dedented.splitlines():
        if _DF_READ_RE.match(line):
            out_lines.append(f"# [stripped df reload] {line}")
        else:
            out_lines.append(line)

    return "\n".join(out_lines).rstrip() + "\n"

def run_code_in_container(
    code: str,
    df: pd.DataFrame = None,
    image_name: str = "code_executor:latest",
    timeout: int = 1800, # 30 minutes
    openml_task_id: int = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Executes Python code in a Docker container with a provided DataFrame and timeout.
    Returns (stdout, stderr) stripped, or (None, error_message) on failure.
    """
    client = docker.from_env()
    container = None
    temp_dir = None

    try:
        container_logger.info("Creating a temporary directory to store the DataFrame + script...")
        temp_dir = tempfile.mkdtemp()

        data_path = os.path.join(temp_dir, "data.parquet")
        script_path = os.path.join(temp_dir, "analysis.py")

        # Save DataFrame to parquet file (host side)
        df.to_parquet(data_path)
        container_logger.info("Saved DataFrame to temp dir as data.parquet")

        # Build a real script instead of python -c
        llm_code = _sanitize_llm_code(code)

        if df is None:
            script = (
                "import openml\n"
                "\n"
                f"openml_task_id = {openml_task_id}\n"
                "task = openml.tasks.get_task(openml_task_id)\n"
                "dataset = task.get_dataset()\n"
                "X, y, categorical_indicator, attribute_names = dataset.get_data(target=task.target_name, dataset_format='dataframe')\n"
                "# --- llm code ---\n"
                f"{llm_code}"
            )
        else:
            script = (
            "import pandas as pd\n"
            "\n"
            "# Load the preprocessed DataFrame\n"
            "df = pd.read_parquet('/app/data/data.parquet')\n"
            "\n"
            "# --- llm code ---\n"
            f"{llm_code}"
            )

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)

        container_logger.info("Wrote analysis.py into temp dir")
        container_logger.info(f"Script to be executed:\n{script}")

        # Mount point in container
        container_mount_path = "/app/data"

        # Create container with volume mounting
        # Use `python` (not python3) to avoid 'command not found' in conda envs.
        container = client.containers.create(
            image=image_name,
            command=["conda", "run", "-n", 'pyenv-gpu', "python", "/app/data/analysis.py"],
            device_requests=[DeviceRequest(count=-1, capabilities=[["gpu"]])],
            volumes={
                temp_dir: {
                    "bind": container_mount_path,
                    "mode": "ro",
                }
            },
            working_dir="/app",
        )

        container_logger.info("Starting container execution...")
        start_time = time.time()
        container.start()

        try:
            exit_status = container.wait(timeout=timeout)
            execution_time = time.time() - start_time
            container_logger.info(f"Container finished in {execution_time:.2f}s")

            if exit_status.get("StatusCode", 1) != 0:
                container_logger.warning(f"Non-zero status: {exit_status}")

            stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace").strip()
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace").strip()

            return (stdout or None), (stderr or None)

        except Exception as e:
            container_logger.error(f"Execution error: {e}")
            try:
                container.kill()
            except Exception:
                pass
            return None, f"Execution error: {e}"

    except Exception as e:
        container_logger.exception("Container execution failed")
        return None, str(e)

    finally:
        if container:
            try:
                container_logger.info("Removing the container...")
                container.remove(force=True)
            except Exception as e:
                container_logger.error(f"Error removing container: {e}")

        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception as e:
                container_logger.error(f"Error removing temporary directory: {e}")