"""
input_parser.py

Responsibilities:
- Read raw dataframe from disk
- Call data_preprocessor.process_and_split to:
    1) Clean full dataframe
    2) Split into train/test
    3) Build metadata/stats from TRAIN ONLY
- Write TRAIN and TEST back to the SAME directory as the source, with CSV extension
  and predictable suffixes: `train.csv`, `test.csv`

Input disk:
    - Data file path

Output disk:
    Train and test files in the same directory as the source data:
    - train.csv
    - test.csv

Input main:
    - Optional parameters for metadata inclusion, sampling, random seed  and test size from main

Output to main:
    - Formatted data as a pipe-delimited string
    - Optional metadata string if requested
    - Preprocessed train DataFrame with sanitized column names
"""

import pandas as pd
from typing import Tuple, Optional
from pathlib import Path
from thesis_code_llm.data_preprocessor import preprocess_data
import openml

from .central_logger import get_logger

# Configure logging for 'input_parser.log'
parser_logger = get_logger("input_parser", log_file="input_parser.log")

# Define a mapping of file extensions to pandas read functions
# This allows for easy extension to support more file types in the future
PANDAS_READERS = {
    '.csv': pd.read_csv,
    '.tsv': lambda f: pd.read_csv(f, sep='\t'), # TSV files are tab-separated values
    '.tab': lambda f: pd.read_csv(f, sep='\t'), # TAB files are tab-separated values
    '.json': pd.read_json,
    '.xlsx': pd.read_excel,
    '.xls': pd.read_excel,
    '.parquet': pd.read_parquet,
    '.feather': pd.read_feather,
    '.pkl': pd.read_pickle,
    '.h5': pd.read_hdf,
    '.orc': pd.read_orc,
}

def read_data_file(file_path: str | Path, include_metadata: bool = False, sample_rows: int = 5, random_seed: int = 42, test_size: float = 0.2, data_file_type: str = 'data', openml_task_id: int = None) -> Tuple[str, Optional[str], pd.DataFrame]:
    """
    Read and preprocess a wide range of data files supported by pandas.

    Args:
        file_path (str | Path): Path to the data file
        include_metadata (bool): Whether to include metadata in output
        sample_rows (int): Number of rows to sample
        random_seed (int): Random seed for reproducibility

    Returns:
        Tuple[str, Optional[str]]: (formatted data, metadata if requested, preprocessed DataFrame with sanitized column names)
    """
    try:
        if data_file_type == 'openml' and openml_task_id is not None:
            task = openml.tasks.get_task(openml_task_id)
            dataset = task.get_dataset()
            X, y, categorical_indicator, attribute_names = dataset.get_data(target=task.target_name)
            train_indices, test_indices = task.get_train_test_split_indices(fold=0)
            df = pd.concat([X, y], axis=1)
            sample_df = df.iloc[train_indices[:sample_rows]]

            # Preprocess and split the data
            formatted_data, metadata, train_df, test_df = preprocess_data(
                sample_df,
                include_metadata=include_metadata,
                sample_rows=sample_rows,
                random_seed=random_seed,
                test_size=test_size,
                data_file_type=data_file_type
            )
        
        elif data_file_type == 'data':
            parser_logger.info(f"Reading file: {file_path}")

            # Convert to Path object if string
            file_path = Path(file_path)

            # Retrieve the file extension
            suffix = file_path.suffix.lower()

            # Check if the file type is supported
            if suffix not in PANDAS_READERS:
                raise ValueError(f"Unsupported file type: {file_path}")

            # Read the data using the appropriate pandas function
            df = PANDAS_READERS[suffix](file_path)

            # Preprocess and split the data
            formatted_data, metadata, train_df, test_df = preprocess_data(
                df,
                include_metadata=include_metadata,
                sample_rows=sample_rows,
                random_seed=random_seed,
                test_size=test_size,
                data_file_type=data_file_type
            )

            complete_df = pd.concat([train_df, test_df], ignore_index=True)

            # Extract the folder from the file path
            folder = file_path.parent

            train_output_path = folder / "train.csv"
            test_output_path = folder / "test.csv"
            complete_output_path = folder / "complete.csv"

            # Write the train and test DataFrames to CSV files
            train_df.to_csv(train_output_path, index=False)
            test_df.to_csv(test_output_path, index=False)
            complete_df.to_csv(complete_output_path, index=False)

            parser_logger.info("File successfully read and preprocessed")
        else:
            raise ValueError("For 'openml' data_file_type, openml_task_id must be provided or a data file must be specified.")
        
        return formatted_data, metadata, train_df, test_df, complete_df

    except Exception as e:
        parser_logger.exception("Failed to read the file")
        raise e

def read_txt_file(file_path: str | Path) -> str:
    """
    Reads the task prompt from a text file.

    Args:
        file_path (str | Path): Path to the text file.

    Returns:
        str: Content of the text file as a string.
    """
    try:
        parser_logger.info(f"Reading task prompt from text file: {file_path}")
        file_path = Path(file_path)
        with open(file_path, 'r') as file:
            task_prompt = file.read().strip()
        parser_logger.info("Task prompt successfully read from file.")
        return task_prompt

    except Exception as e:
        parser_logger.exception("Failed to read the text file.")
        raise e
