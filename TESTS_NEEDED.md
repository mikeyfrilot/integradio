# Integradio - Tests Needed

Pass this file to another Claude instance to implement tests.

## Test Environment Setup

```bash
cd F:/AI/integradio
pip install -e ".[dev]"
# Ensure Ollama is running with nomic-embed-text
ollama pull nomic-embed-text
```

---

## 1. Embedder Tests (`tests/test_embedder.py`)

### Unit Tests
- [ ] `test_embed_single_text` - Verify embedding returns numpy array of correct dimension (768)
- [ ] `test_embed_batch` - Verify batch embedding returns list of correct length
- [ ] `test_embed_query_uses_different_prefix` - Ensure search_query prefix is used for queries
- [ ] `test_cache_hit` - Embed same text twice, verify second call uses cache
- [ ] `test_cache_persistence` - With cache_dir, verify cache survives new Embedder instance
- [ ] `test_cache_key_uniqueness` - Different texts should have different cache keys

### Integration Tests (require Ollama)
- [ ] `test_ollama_connection` - Verify can connect to Ollama API
- [ ] `test_embedding_determinism` - Same text should produce same embedding
- [ ] `test_semantic_similarity` - "cat" and "dog" should be more similar than "cat" and "airplane"

### Mock Tests (no Ollama needed)
- [ ] `test_embedder_with_mock_api` - Mock httpx to test without real API

---

## 2. Registry Tests (`tests/test_registry.py`)

### Unit Tests
- [ ] `test_register_component` - Register a component, verify it's stored
- [ ] `test_register_duplicate_id` - Re-registering same ID should update
- [ ] `test_get_component` - Retrieve component by ID
- [ ] `test_get_nonexistent` - Getting non-existent ID returns None
- [ ] `test_search_returns_sorted_by_score` - Results should be sorted by similarity
- [ ] `test_search_with_type_filter` - Filter by component_type works
- [ ] `test_search_with_tags_filter` - Filter by tags works (any match)
- [ ] `test_add_relationship` - Add and retrieve relationships
- [ ] `test_get_dataflow_upstream` - Trace upstream components correctly
- [ ] `test_get_dataflow_downstream` - Trace downstream components correctly
- [ ] `test_export_graph` - Graph export has correct nodes/links structure
- [ ] `test_clear_registry` - Clear removes all components
- [ ] `test_len_and_contains` - __len__ and __contains__ work correctly

### Edge Cases
- [ ] `test_search_empty_registry` - Search on empty registry returns empty list
- [ ] `test_circular_dataflow` - Handle circular dependencies without infinite loop
- [ ] `test_fallback_search_without_hnswlib` - Search works when hnswlib not installed

---

## 3. Components Tests (`tests/test_components.py`)

### Unit Tests
- [ ] `test_semantic_wraps_component` - semantic() returns SemanticComponent
- [ ] `test_attribute_delegation` - Accessing wrapped component attributes works
- [ ] `test_explicit_intent` - Provided intent is stored
- [ ] `test_inferred_intent_from_label` - No intent uses label
- [ ] `test_auto_tags_by_type` - Textbox gets ["input", "text"] tags
- [ ] `test_custom_tags_merged` - Custom tags merge with inferred
- [ ] `test_source_location_captured` - file_path and line_number are set
- [ ] `test_get_semantic_by_id` - get_semantic() retrieves correct instance

### Integration with Gradio
- [ ] `test_semantic_in_blocks_context` - semantic() works inside gr.Blocks
- [ ] `test_event_binding_preserved` - .click(), .change() still work on semantic wrapper

---

## 4. SemanticBlocks Tests (`tests/test_blocks.py`)

### Unit Tests
- [ ] `test_blocks_creates_registry` - SemanticBlocks has registry attribute
- [ ] `test_blocks_creates_embedder` - SemanticBlocks has embedder attribute
- [ ] `test_auto_register_on_exit` - Components registered when exiting context
- [ ] `test_search_method` - blocks.search() returns SearchResult list
- [ ] `test_find_single_component` - blocks.find() returns most relevant component
- [ ] `test_trace_component` - blocks.trace() returns upstream/downstream
- [ ] `test_map_returns_graph` - blocks.map() returns nodes/links dict
- [ ] `test_describe_component` - blocks.describe() returns full metadata
- [ ] `test_summary` - blocks.summary() returns formatted string

### Dataflow Tests
- [ ] `test_click_creates_trigger_relationship` - btn.click() creates trigger link
- [ ] `test_input_output_creates_dataflow` - inputs/outputs create dataflow links

---

## 5. Introspection Tests (`tests/test_introspect.py`)

### Unit Tests
- [ ] `test_get_source_location` - Returns correct file and line
- [ ] `test_extract_component_info` - Extracts label, elem_id, type
- [ ] `test_build_intent_text` - Builds proper embedding text
- [ ] `test_infer_tags_textbox` - Textbox → ["input", "text"]
- [ ] `test_infer_tags_button` - Button → ["trigger", "action"]
- [ ] `test_infer_tags_markdown` - Markdown → ["output", "text", "display"]
- [ ] `test_extract_dataflow` - Extracts event relationships from Blocks

---

## 6. API Tests (`tests/test_api.py`)

### FastAPI Integration
- [ ] `test_search_endpoint` - GET /semantic/search returns results
- [ ] `test_search_with_filters` - type and tags query params work
- [ ] `test_component_endpoint` - GET /semantic/component/{id} returns metadata
- [ ] `test_component_not_found` - Returns 404 for invalid ID
- [ ] `test_graph_endpoint` - GET /semantic/graph returns nodes/links
- [ ] `test_trace_endpoint` - GET /semantic/trace/{id} returns dataflow
- [ ] `test_summary_endpoint` - GET /semantic/summary returns counts

---

## 7. Visualization Tests (`tests/test_viz.py`)

### Unit Tests
- [ ] `test_generate_mermaid` - Returns valid Mermaid diagram string
- [ ] `test_mermaid_node_styles` - Different types get different styles
- [ ] `test_generate_html_graph` - Returns complete HTML document
- [ ] `test_html_contains_d3` - HTML includes D3.js script
- [ ] `test_generate_ascii_graph` - Returns formatted ASCII art
- [ ] `test_ascii_empty_registry` - Handles empty registry gracefully

---

## 8. End-to-End Tests (`tests/test_e2e.py`)

### Full Workflow
- [ ] `test_full_app_workflow` - Create app, register components, search, trace
- [ ] `test_search_finds_relevant_component` - "user input" finds Textbox
- [ ] `test_dataflow_traced_correctly` - Button → Input → Output chain works

---

## Test Fixtures Needed

```python
# conftest.py

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_embedder():
    """Embedder that returns random vectors without API calls."""
    embedder = MagicMock()
    embedder.dimension = 768
    embedder.embed.return_value = np.random.rand(768).astype(np.float32)
    embedder.embed_query.return_value = np.random.rand(768).astype(np.float32)
    return embedder

@pytest.fixture
def registry(mock_embedder):
    """Fresh registry for each test."""
    from integradio.registry import ComponentRegistry
    return ComponentRegistry(db_path=None)  # In-memory

@pytest.fixture
def sample_metadata():
    """Sample ComponentMetadata for testing."""
    from integradio.registry import ComponentMetadata
    return ComponentMetadata(
        component_id=1,
        component_type="Textbox",
        intent="user enters search query",
        label="Search",
        tags=["input", "text"],
    )
```

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=integradio --cov-report=html

# Run specific test file
pytest tests/test_registry.py -v

# Run tests matching pattern
pytest tests/ -k "search" -v

# Run without Ollama (mock only)
pytest tests/ -v -m "not integration"
```

---

## Priority Order

1. **Registry tests** - Core functionality
2. **Components tests** - User-facing API
3. **Blocks tests** - Integration
4. **Embedder tests** - External dependency
5. **API tests** - Optional feature
6. **Viz tests** - Nice to have
7. **E2E tests** - Final validation
