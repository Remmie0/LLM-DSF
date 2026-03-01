import pytest
from thesis_code_llm.code_parser import extract_code_from_response

class TestCodeParser:
    def test_extract_python_code(self, sample_llm_response):
        """Test extracting Python code from response."""
        code = extract_code_from_response(sample_llm_response)
        assert code is not None
        assert "import pandas" in code
        assert "print" in code

    def test_extract_no_code(self):
        """Test handling when no code block exists."""
        response = "This is just text without any code blocks"
        code = extract_code_from_response(response)
        assert code is None

    def test_multiple_code_blocks(self):
        """Test extracting last code block when multiple exist."""
        response = '''
```python
def first():
    pass
```

```python
def second():
    pass
```'''
        code = extract_code_from_response(response)
        assert "def second()" in code
        assert "def first()" not in code
