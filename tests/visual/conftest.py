"""
Pytest configuration for visual module tests.

This conftest is minimal and does not import numpy to avoid import conflicts
when running tests with coverage.
"""

import pytest
from pathlib import Path


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory for export tests."""
    output = tmp_path / "output"
    output.mkdir()
    return output
