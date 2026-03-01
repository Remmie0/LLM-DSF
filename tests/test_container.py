import pytest
from thesis_code_llm.container import run_code_in_container

class TestContainer:
    def test_basic_execution(self, sample_dataframe):
        """Test basic code execution."""
        code = "print(df['revenue'].sum())"
        output, error = run_code_in_container(code, sample_dataframe)
        assert output == "6000"
        assert not error

    def test_execution_error(self, sample_dataframe):
        """Test handling of execution errors."""
        code = "print(undefined_variable)"
        output, error = run_code_in_container(code, sample_dataframe)
        assert error is not None
        assert "NameError" in error

    @pytest.mark.timeout(5)
    def test_timeout(self, sample_dataframe):
        """Test timeout handling."""
        code = "import time; time.sleep(3)"
        output, error = run_code_in_container(
            code,
            sample_dataframe,
            timeout=2
        )
        assert error is not None

        # Check for various timeout-related messages that might occur
        possible_error_messages = [
            "timeout",
            "timed out",
            "killed",
            "read timed out"
        ]
        error_lower = error.lower()
        assert any(msg in error_lower for msg in possible_error_messages), \
            f"Expected timeout-related error, got: {error}"
