#!/usr/bin/env python3
import pytest
import sys
import os

def main():
    """Run tests with coverage reporting"""
    # Ensure we're in the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    # Run tests with coverage
    args = [
        "--verbose",
        "--cov=.",
        "--cov-report=term-missing",
        "--cov-report=html",
        "--cov-report=xml",
        "--cov-fail-under=80",
        "tests/"
    ]
    
    # Add parallel execution for faster testing
    args.insert(0, "-n")
    args.insert(1, "auto")
    
    return pytest.main(args)

if __name__ == "__main__":
    sys.exit(main()) 