import pytest
from thesis_code_llm.LLM import initialize_model, generate_response
import os

class TestLLM:
    def test_initialize_model_errors(self):
        """Test model initialization error handling."""
        # Test for non-existent model
        with pytest.raises(ValueError, match="Model path does not exist"):
            initialize_model("nonexistent_model.gguf")

        # Test for invalid model file
        with open("invalid.gguf", "w") as f:
            f.write("invalid model content")
        try:
            with pytest.raises(ValueError, match="Failed to load model"):
                initialize_model("invalid.gguf")
        finally:
            # Clean up test file
            if os.path.exists("invalid.gguf"):
                os.remove("invalid.gguf")

    def test_generate_response_max_length(self, mocker):
        """Test response generation respects max length."""
        mock_model = mocker.Mock()
        mock_model.return_value = {"choices": [{"text": "test response"}]}

        response = generate_response(mock_model, "test prompt", max_length=100)
        assert isinstance(response, str)
        assert len(response) > 0

    def test_generate_response_error_handling(self, mocker):
        """Test error handling in response generation."""
        mock_model = mocker.Mock()
        mock_model.side_effect = Exception("Test error")

        with pytest.raises(Exception, match="Test error"):
            generate_response(mock_model, "test prompt")
