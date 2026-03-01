"""
data_preprocessor.py

Responsibilities:
1) Stateless, LLM-friendly cleaning on the FULL raw dataframe
   - trim/normalize text column names
    - remove special characters

2) Split using scikit-learn (train/test)
   - No metadata/stat computation before the split.

3) Generate metadata/statistics **from TRAIN ONLY**
   - column schema
   - missing value counts
   - numeric summary (mean, std, min, max)

Input:
    - DataFrame with raw data
    - Optional parameters for metadata inclusion, sampling, and random seed

Output:
    - Formatted data as a pipe-delimited string
    - Optional metadata string if requested
    - train and test processed DataFrames with sanitized column names
"""

import pandas as pd
from pandas.api.types import is_numeric_dtype
import re
from typing import Dict, Tuple, Optional
import numpy as np

from sklearn.model_selection import train_test_split
from .central_logger import get_logger

# Configure logging
preprocessor_logger = get_logger("preprocessor", log_file="preprocessor.log")

def clean_column_name(column: str) -> str:
    """
    Clean column names by removing special characters and standardizing format.

    Args:
        column (str): Original column name

    Returns:
        str: Cleaned column name
    """
    # Convert to lowercase and replace spaces/special chars with underscore
    cleaned = re.sub(r'[^a-zA-Z0-9]+', '_', column.strip().lower())
    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')
    # Replace multiple underscores with single underscore
    cleaned = re.sub(r'_+', '_', cleaned)
    return cleaned

def get_data_metadata(df: pd.DataFrame) -> Dict:
    """
    Generate metadata about the dataset.

    Args:
        df (pd.DataFrame): Input DataFrame

    Returns:
        Dict: Dictionary containing metadata
    """
    metadata = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'column_types': {},
        'missing_values': {},
        'unique_counts': {},
        'numeric_summary': {}
    }

    for column in df.columns:
        # Get data type
        metadata['column_types'][column] = str(df[column].dtype)

        # Count missing values
        missing_count = df[column].isna().sum()
        if missing_count > 0:
            metadata['missing_values'][column] = int(missing_count)

        # Count unique values
        metadata['unique_counts'][column] = int(df[column].nunique())

        # Basic statistics for numeric columns
        if is_numeric_dtype(df[column].dtype):
            metadata['numeric_summary'][column] = {
                'min': float(df[column].min()),
                'max': float(df[column].max()),
                'mean': float(df[column].mean()),
                'median': float(df[column].median())
            }

    return metadata

def format_metadata_for_llm(metadata: Dict, sample_rows: int) -> str:
    """
    Format metadata into a string suitable for LLM prompt.

    Args:
        metadata (Dict): Metadata dictionary

    Returns:
        str: Formatted metadata string
    """
    formatted = "Dataset Metadata:\n"
    formatted += f"Total Rows: {metadata['total_rows']}\n"
    formatted += f"Sample Size: {min(sample_rows, metadata['total_rows'])} rows\n"
    formatted += f"Total Columns: {metadata['total_columns']}\n\n"

    formatted += "Column Information:\n"
    for column in metadata['column_types']:
        formatted += f"- {column}:\n"
        formatted += f"  Type: {metadata['column_types'][column]}\n"
        formatted += f"  Unique Values: {metadata['unique_counts'][column]}\n"

        if column in metadata['missing_values']:
            formatted += f"  Missing Values: {metadata['missing_values'][column]}\n"

        if column in metadata['numeric_summary']:
            stats = metadata['numeric_summary'][column]
            formatted += f"  Range: {stats['min']} to {stats['max']}\n"
            formatted += f"  Average: {stats['mean']:.2f}\n"

        formatted += "\n"

    return formatted

def preprocess_data(
    df: pd.DataFrame,
    include_metadata: bool = False,
    sample_rows: int = 5,
    random_seed: int = 42,
    test_size: float = 0.2,
    data_file_type: str = 'data'
) -> Tuple[str, Optional[str]]:
    """
    Preprocess DataFrame and optionally generate metadata.

    Args:
        df (pd.DataFrame): Input DataFrame
        include_metadata (bool): Whether to include metadata
        sample_rows (int): Number of sample rows to include

    Returns:
        Tuple[str, Optional[str]]: (formatted data, metadata if requested)
    """
    try:
        if data_file_type != 'data':
            # Convert DataFrame to pipe-delimited string
            formatted_data = df.to_csv(sep='|', index=False)

            # Generate metadata if requested
            metadata_str = None
            if include_metadata:
                metadata = get_data_metadata(df)
                metadata_str = format_metadata_for_llm(metadata, sample_rows)
            return formatted_data, metadata_str, None, None
        
        # Clean column names
        df.columns = [clean_column_name(col) for col in df.columns]

        train_df, test_df = train_test_split(
                                df,
                                test_size=test_size,
                                random_state=random_seed
                            )

        # Randomly sample rows, but if sample_rows > total rows, use all rows
        sample_size = min(sample_rows, len(train_df))

        # Set random_state for reproducibility, only sample from train set
        np.random.seed(random_seed)
        sampled_df = train_df.sample(n=sample_size, random_state=random_seed)

        # Convert DataFrame to pipe-delimited string
        formatted_data = sampled_df.to_csv(sep='|', index=False)

        # Generate metadata if requested
        metadata_str = None
        if include_metadata:
            metadata = get_data_metadata(train_df)
            metadata_str = format_metadata_for_llm(metadata, sample_rows)

        return formatted_data, metadata_str, train_df, test_df

    except Exception as e:
        preprocessor_logger.exception("Error in preprocessing data")
        raise e
