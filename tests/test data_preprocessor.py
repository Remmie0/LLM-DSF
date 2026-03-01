import pytest
import pandas as pd
from thesis_code_llm.data_preprocessor import (
    clean_column_name,
    preprocess_data,
    get_data_metadata,
    format_metadata_for_llm
)

class TestDataPreprocessor:
    def test_clean_column_name(self):
        """Test column name cleaning."""
        assert clean_column_name("Test Column!") == "test_column"
        assert clean_column_name("Multiple__Underscores") == "multiple_underscores"
        assert clean_column_name(" Leading Space") == "leading_space"

    def test_preprocess_data(self, sample_dataframe):
        """Test data preprocessing."""
        formatted_data, metadata, df = preprocess_data(
            sample_dataframe,
            include_metadata=True,
            sample_rows=2
        )
        assert isinstance(formatted_data, str)
        assert isinstance(metadata, str)
        assert isinstance(df, pd.DataFrame)

    def test_metadata_generation(self, sample_dataframe):
        """Test metadata generation."""
        metadata = get_data_metadata(sample_dataframe)
        assert isinstance(metadata, dict)
        assert 'total_rows' in metadata
        assert 'column_types' in metadata
