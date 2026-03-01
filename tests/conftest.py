import pytest
import pandas as pd
import tempfile
import os
from typing import Generator, Dict, Any
from pathlib import Path

@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'revenue': [1000, 2000, 3000],
        'cost': [500, 1000, 1500],
        'product': ['A', 'B', 'C']
    })

@pytest.fixture
def temp_test_dir() -> Generator[str, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def sample_llm_response() -> str:
    """Create a sample LLM response with code block."""
    return '''Here's the analysis code for your data:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Calculate total revenue
total_revenue = df['revenue'].sum()
print(f"Total Revenue: ${total_revenue:,.2f}")

# Calculate profit margin
df['profit'] = df['revenue'] - df['cost']
df['profit_margin'] = df['profit'] / df['revenue'] * 100

# Print summary
print(f"Average Profit Margin: {df['profit_margin'].mean():.2f}%")
```
'''

@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Create mock configuration settings."""
    return {
        'model': {
            'path': 'models/test_model.gguf',
            'n_ctx': 32768,
            'n_gpu_layers': -1,
            'flash_attn': True
        },
        'data': {
            'sample_rows': 5,
            'include_metadata': True
        },
        'container': {
            'image_name': 'test_executor:latest',
            'timeout': 30
        }
    }

@pytest.fixture
def sample_metadata() -> str:
    """Create sample metadata string."""
    return """Dataset Metadata:
Total Rows: 3
Sample Size: 3 rows
Total Columns: 4

Column Information:
- date:
  Type: object
  Unique Values: 3

- revenue:
  Type: int64
  Unique Values: 3
  Range: 1000 to 3000
  Average: 2000.00

- cost:
  Type: int64
  Unique Values: 3
  Range: 500 to 1500
  Average: 1000.00

- product:
  Type: object
  Unique Values: 3"""

@pytest.fixture
def test_files(temp_test_dir) -> Dict[str, Path]:
    """Create test files in temporary directory."""
    files = {}

    # Create CSV file
    csv_path = Path(temp_test_dir) / "test_data.csv"
    pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02'],
        'value': [100, 200]
    }).to_csv(csv_path, index=False)
    files['csv'] = csv_path

    # Create text file
    txt_path = Path(temp_test_dir) / "test_prompt.txt"
    with open(txt_path, 'w') as f:
        f.write("Analyze the data and calculate summary statistics.")
    files['txt'] = txt_path

    return files
