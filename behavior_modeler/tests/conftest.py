"""Pytest fixtures for behavior_modeler tests."""

import pytest
from pathlib import Path
import tempfile

from behavior_modeler.config import BehaviorModelerConfig
from behavior_modeler.store import FlowStore
from behavior_modeler.mock import MockFlowGenerator


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_behavior.db"


@pytest.fixture
def config(temp_db_path):
    """Create test configuration."""
    return BehaviorModelerConfig(db_path=temp_db_path)


@pytest.fixture
def store(config):
    """Create test flow store."""
    store = FlowStore(config)
    yield store
    store.close()


@pytest.fixture
def mock_generator():
    """Create mock flow generator with fixed seed."""
    return MockFlowGenerator(seed=42)


@pytest.fixture
def sample_sessions(mock_generator):
    """Generate sample sessions for testing."""
    return mock_generator.generate_batch(20)
