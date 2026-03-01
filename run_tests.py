import pytest
import sys
import os
import argparse

def main():
    """Run the test suite with specified parameters."""
    parser = argparse.ArgumentParser(description='Run the test suite')
    parser.add_argument('--coverage', action='store_true', help='Run with coverage report')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--test-file', help='Run specific test file')
    parser.add_argument('--test-function', help='Run specific test function')
    args = parser.parse_args()

    # Add the project root directory to Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)

    # Build pytest arguments
    pytest_args = []

    # Add test path
    if args.test_file:
        if args.test_function:
            pytest_args.append(f'tests/{args.test_file}::{args.test_function}')
        else:
            pytest_args.append(f'tests/{args.test_file}')
    else:
        pytest_args.append('tests')

    # Add options
    if args.verbose:
        pytest_args.append('-v')

    pytest_args.extend([
        '--tb=short',  # shorter traceback format
        '-s',         # allow printing to stdout
    ])

    if args.coverage:
        pytest_args.extend([
            '--cov=.',
            '--cov-report=term-missing',
            '--cov-report=html:coverage_report'
        ])

    # Run pytest with specified arguments
    exit_code = pytest.main(pytest_args)

    return exit_code

if __name__ == '__main__':
    sys.exit(main())
