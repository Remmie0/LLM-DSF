import pytest
from thesis_code_llm.input_parser import read_data_file, read_txt_file
import pandas as pd

class TestInputParser:
    def test_read_file_csv(self, test_files):
        """Test reading CSV file."""
        formatted_data, metadata, df = read_data_file(
            test_files['csv'],
            include_metadata=True
        )
        assert isinstance(formatted_data, str)
        assert isinstance(metadata, str)
        assert isinstance(df, pd.DataFrame)

    def test_read_txt_file(self, test_files):
        """Test reading text file."""
        content = read_txt_file(test_files['txt'])
        assert isinstance(content, str)
        assert "Analyze" in content

    def test_unsupported_file_type(self):
        """Test handling of unsupported file types."""
        with pytest.raises(ValueError):
            read_data_file("test.unsupported")
