"""Pytest fixtures for behavior_modeler tests."""

import pytest
from pathlib import Path
import tempfile

from behavior_modeler.config import BehaviorModelerConfig
from behavior_modeler.store import FlowStore
from behavior_modeler.encoder import FallbackEncoder
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
def encoder():
    """Create fallback encoder for testing."""
    return FallbackEncoder()


@pytest.fixture
def mock_generator():
    """Create mock flow generator with fixed seed."""
    return MockFlowGenerator(seed=42)


@pytest.fixture
def sample_sessions(mock_generator):
    """Generate sample sessions for testing."""
    return mock_generator.generate_batch(20)


@pytest.fixture
def populated_store(store, encoder, sample_sessions):
    """Store with encoded sessions."""
    for session in sample_sessions:
        vector = encoder.encode_session(session)
        session.vector = vector
        store.save_session(session)
    return store
