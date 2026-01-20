"""
Pytest fixtures for integradio tests.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock
from pathlib import Path


@pytest.fixture
def mock_embedder():
    """Embedder that returns deterministic vectors without API calls."""
    embedder = MagicMock()
    embedder.dimension = 768

    def mock_embed(text: str) -> np.ndarray:
        # Generate deterministic vector from text hash
        np.random.seed(hash(text) % 2**32)
        return np.random.rand(768).astype(np.float32)

    def mock_embed_query(query: str) -> np.ndarray:
        np.random.seed(hash(f"query:{query}") % 2**32)
        return np.random.rand(768).astype(np.float32)

    embedder.embed.side_effect = mock_embed
    embedder.embed_query.side_effect = mock_embed_query
    embedder.embed_batch.side_effect = lambda texts: [mock_embed(t) for t in texts]

    return embedder


@pytest.fixture
def registry():
    """Fresh in-memory registry for each test."""
    from integradio.registry import ComponentRegistry
    return ComponentRegistry(db_path=None)


@pytest.fixture
def sample_metadata():
    """Sample ComponentMetadata for testing."""
    from integradio.registry import ComponentMetadata
    return ComponentMetadata(
        component_id=1,
        component_type="Textbox",
        intent="user enters search query",
        label="Search",
        elem_id="search-input",
        tags=["input", "text"],
        file_path="app.py",
        line_number=42,
    )


@pytest.fixture
def sample_vector():
    """Sample embedding vector."""
    np.random.seed(42)
    return np.random.rand(768).astype(np.float32)


@pytest.fixture
def populated_registry(registry, sample_metadata, sample_vector):
    """Registry with some components pre-registered."""
    from integradio.registry import ComponentMetadata

    # Register multiple components
    components = [
        (1, "user enters search query", "Textbox", "Search Query", ["input", "text"]),
        (2, "triggers the search", "Button", "Search", ["trigger", "action"]),
        (3, "displays search results", "Markdown", "Results", ["output", "display"]),
        (4, "shows image preview", "Image", "Preview", ["output", "media"]),
        (5, "controls result count", "Slider", "Num Results", ["input", "numeric"]),
    ]

    for comp_id, intent, comp_type, label, tags in components:
        np.random.seed(hash(intent) % 2**32)
        vector = np.random.rand(768).astype(np.float32)

        metadata = ComponentMetadata(
            component_id=comp_id,
            component_type=comp_type,
            intent=intent,
            label=label,
            tags=tags,
        )
        registry.register(comp_id, vector, metadata)

    # Add some relationships
    registry.add_relationship(2, 1, "trigger")  # Button triggers Textbox
    registry.add_relationship(1, 3, "dataflow")  # Textbox flows to Markdown
    registry.add_relationship(5, 3, "dataflow")  # Slider flows to Markdown

    return registry


@pytest.fixture
def temp_db_path(tmp_path):
    """Temporary database path for persistence tests."""
    return tmp_path / "test_registry.db"


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Temporary cache directory for embedder tests."""
    cache_dir = tmp_path / "embeddings_cache"
    cache_dir.mkdir()
    return cache_dir
