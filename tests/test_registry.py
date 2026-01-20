"""
Tests for ComponentRegistry - HNSW vector index + SQLite metadata.

Priority 1: Core functionality tests.
"""

import pytest
import numpy as np
from integradio.registry import ComponentRegistry, ComponentMetadata, SearchResult


class TestComponentRegistration:
    """Test component registration functionality."""

    def test_register_component(self, registry, sample_metadata, sample_vector):
        """Register a component, verify it's stored."""
        registry.register(1, sample_vector, sample_metadata)

        assert 1 in registry
        assert len(registry) == 1

        retrieved = registry.get(1)
        assert retrieved is not None
        assert retrieved.component_id == 1
        assert retrieved.component_type == "Textbox"
        assert retrieved.intent == "user enters search query"

    def test_register_duplicate_id(self, registry, sample_vector):
        """Re-registering same ID should update."""
        meta1 = ComponentMetadata(
            component_id=1,
            component_type="Textbox",
            intent="original intent",
            label="First",
        )
        meta2 = ComponentMetadata(
            component_id=1,
            component_type="Button",
            intent="updated intent",
            label="Second",
        )

        registry.register(1, sample_vector, meta1)
        assert registry.get(1).intent == "original intent"

        registry.register(1, sample_vector, meta2)
        assert registry.get(1).intent == "updated intent"
        assert registry.get(1).component_type == "Button"
        assert len(registry) == 1  # Still only one component

    def test_get_component(self, populated_registry):
        """Retrieve component by ID."""
        meta = populated_registry.get(1)
        assert meta is not None
        assert meta.component_type == "Textbox"
        assert meta.label == "Search Query"

    def test_get_nonexistent(self, registry):
        """Getting non-existent ID returns None."""
        assert registry.get(999) is None
        assert 999 not in registry


class TestSearch:
    """Test semantic search functionality."""

    def test_search_returns_sorted_by_score(self, populated_registry, mock_embedder):
        """Results should be sorted by similarity (highest first)."""
        # Generate a query vector
        query_vector = mock_embedder.embed_query("search functionality")

        results = populated_registry.search(query_vector, k=5)

        assert len(results) > 0
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), "Results should be sorted by score descending"

    def test_search_with_type_filter(self, populated_registry, mock_embedder):
        """Filter by component_type works."""
        query_vector = mock_embedder.embed_query("anything")

        # Search only for Buttons
        results = populated_registry.search(query_vector, k=10, component_type="Button")

        assert len(results) > 0
        for r in results:
            assert r.metadata.component_type == "Button"

    def test_search_with_tags_filter(self, populated_registry, mock_embedder):
        """Filter by tags works (any match)."""
        query_vector = mock_embedder.embed_query("anything")

        # Search for components with "media" tag
        results = populated_registry.search(query_vector, k=10, tags=["media"])

        assert len(results) > 0
        for r in results:
            assert any(t in r.metadata.tags for t in ["media"])

    def test_search_empty_registry(self, registry, mock_embedder):
        """Search on empty registry returns empty list."""
        query_vector = mock_embedder.embed_query("anything")
        results = registry.search(query_vector, k=10)
        assert results == []

    def test_search_k_limits_results(self, populated_registry, mock_embedder):
        """Search respects k parameter."""
        query_vector = mock_embedder.embed_query("anything")

        results = populated_registry.search(query_vector, k=2)
        assert len(results) <= 2


class TestRelationships:
    """Test relationship tracking functionality."""

    def test_add_relationship(self, populated_registry):
        """Add and retrieve relationships."""
        # Relationship already exists from fixture: 2 -> 1 (trigger)
        relationships = populated_registry.get_relationships(1)

        assert "trigger" in relationships
        assert 2 in relationships["trigger"]

    def test_get_dataflow_upstream(self, populated_registry):
        """Trace upstream components correctly."""
        # Component 3 (Markdown) receives from 1 (Textbox) and 5 (Slider)
        flow = populated_registry.get_dataflow(3)

        assert "upstream" in flow
        # Both Textbox (1) and Slider (5) feed into Markdown (3)
        assert 1 in flow["upstream"] or 5 in flow["upstream"]

    def test_get_dataflow_downstream(self, populated_registry):
        """Trace downstream components correctly."""
        # Component 1 (Textbox) outputs to component 3 (Markdown)
        flow = populated_registry.get_dataflow(1)

        assert "downstream" in flow
        assert 3 in flow["downstream"]

    def test_circular_dataflow(self, registry, sample_vector):
        """Handle circular dependencies without infinite loop."""
        # Create components
        for i in range(1, 4):
            meta = ComponentMetadata(
                component_id=i,
                component_type="Textbox",
                intent=f"component {i}",
            )
            registry.register(i, sample_vector, meta)

        # Create circular: 1 -> 2 -> 3 -> 1
        registry.add_relationship(1, 2, "dataflow")
        registry.add_relationship(2, 3, "dataflow")
        registry.add_relationship(3, 1, "dataflow")

        # This should not hang
        flow = registry.get_dataflow(1)

        assert "upstream" in flow
        assert "downstream" in flow
        # Should find components but not infinite loop
        assert len(flow["upstream"]) <= 3
        assert len(flow["downstream"]) <= 3


class TestExportAndUtilities:
    """Test export and utility methods."""

    def test_export_graph(self, populated_registry):
        """Graph export has correct nodes/links structure."""
        graph = populated_registry.export_graph()

        assert "nodes" in graph
        assert "links" in graph
        assert len(graph["nodes"]) == 5  # populated_registry has 5 components

        # Check node structure
        node = graph["nodes"][0]
        assert "id" in node
        assert "type" in node
        assert "intent" in node
        assert "label" in node

        # Check link structure
        assert len(graph["links"]) > 0
        link = graph["links"][0]
        assert "source" in link
        assert "target" in link
        assert "type" in link

    def test_clear_registry(self, populated_registry):
        """Clear removes all components."""
        assert len(populated_registry) > 0

        populated_registry.clear()

        assert len(populated_registry) == 0
        assert populated_registry.get(1) is None
        assert populated_registry.all_components() == []

    def test_len_and_contains(self, populated_registry):
        """__len__ and __contains__ work correctly."""
        assert len(populated_registry) == 5
        assert 1 in populated_registry
        assert 2 in populated_registry
        assert 999 not in populated_registry

    def test_all_components(self, populated_registry):
        """all_components returns list of all metadata."""
        components = populated_registry.all_components()

        assert len(components) == 5
        assert all(isinstance(c, ComponentMetadata) for c in components)

        # Verify all expected components are present
        ids = [c.component_id for c in components]
        assert set(ids) == {1, 2, 3, 4, 5}


class TestPersistence:
    """Test database persistence."""

    def test_data_survives_reconnection(self, temp_db_path, sample_metadata, sample_vector):
        """Data persists across registry instances."""
        # Create registry and add data
        registry1 = ComponentRegistry(db_path=temp_db_path)
        registry1.register(1, sample_vector, sample_metadata)
        del registry1

        # Create new registry with same DB
        registry2 = ComponentRegistry(db_path=temp_db_path)

        # Verify data exists in SQLite (metadata dict is empty but we can query)
        # Note: The in-memory vectors and metadata won't persist, but SQLite has the data
        # For full persistence, the registry would need to reload from SQLite
        cursor = registry2._conn.execute("SELECT component_id FROM components WHERE component_id = 1")
        result = cursor.fetchone()
        assert result is not None


class TestFallbackSearch:
    """Test fallback brute-force search when hnswlib unavailable."""

    def test_fallback_search_without_hnswlib(self, sample_vector):
        """Search works when hnswlib not installed (via fallback)."""
        # Create registry and force fallback mode by setting index to None
        registry = ComponentRegistry(db_path=None)
        registry._index = None  # Simulate no hnswlib

        # Add components
        for i in range(3):
            meta = ComponentMetadata(
                component_id=i,
                component_type="Textbox",
                intent=f"component {i}",
                tags=["test"],
            )
            np.random.seed(i)
            vector = np.random.rand(768).astype(np.float32)
            registry.register(i, vector, meta)

        # Search should work via fallback
        query = np.random.rand(768).astype(np.float32)
        results = registry.search(query, k=3)

        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)


class TestComponentMetadata:
    """Test ComponentMetadata dataclass."""

    def test_to_dict(self, sample_metadata):
        """to_dict returns proper dictionary."""
        d = sample_metadata.to_dict()

        assert d["component_id"] == 1
        assert d["component_type"] == "Textbox"
        assert d["intent"] == "user enters search query"
        assert d["tags"] == ["input", "text"]

    def test_from_dict(self):
        """from_dict creates ComponentMetadata from dictionary."""
        data = {
            "component_id": 42,
            "component_type": "Button",
            "intent": "submit form",
            "label": "Submit",
            "elem_id": None,
            "tags": ["action"],
            "file_path": None,
            "line_number": None,
            "inputs_from": [],
            "outputs_to": [],
            "extra": {},
        }

        meta = ComponentMetadata.from_dict(data)

        assert meta.component_id == 42
        assert meta.component_type == "Button"
        assert meta.tags == ["action"]


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_register_with_zero_id(self, registry, sample_vector):
        """Component with ID 0 registers correctly."""
        meta = ComponentMetadata(
            component_id=0,
            component_type="Textbox",
            intent="zero id component",
        )
        registry.register(0, sample_vector, meta)

        assert 0 in registry
        assert registry.get(0).intent == "zero id component"

    def test_register_with_negative_id(self, registry, sample_vector):
        """Component with negative ID registers correctly."""
        meta = ComponentMetadata(
            component_id=-1,
            component_type="Button",
            intent="negative id component",
        )
        registry.register(-1, sample_vector, meta)

        assert -1 in registry
        assert registry.get(-1).intent == "negative id component"

    def test_register_with_large_id(self, registry, sample_vector):
        """Component with very large ID registers correctly."""
        large_id = 2**31 - 1  # Max signed 32-bit int
        meta = ComponentMetadata(
            component_id=large_id,
            component_type="Slider",
            intent="large id component",
        )
        registry.register(large_id, sample_vector, meta)

        assert large_id in registry
        assert registry.get(large_id).intent == "large id component"

    def test_empty_intent(self, registry, sample_vector):
        """Component with empty intent registers correctly."""
        meta = ComponentMetadata(
            component_id=100,
            component_type="Textbox",
            intent="",
        )
        registry.register(100, sample_vector, meta)

        assert registry.get(100).intent == ""

    def test_unicode_intent(self, registry, sample_vector):
        """Component with unicode intent registers correctly."""
        meta = ComponentMetadata(
            component_id=101,
            component_type="Textbox",
            intent="用户输入 🔍 recherche",
        )
        registry.register(101, sample_vector, meta)

        assert registry.get(101).intent == "用户输入 🔍 recherche"

    def test_empty_tags(self, registry, sample_vector):
        """Component with empty tags list works."""
        meta = ComponentMetadata(
            component_id=102,
            component_type="Textbox",
            intent="no tags",
            tags=[],
        )
        registry.register(102, sample_vector, meta)

        assert registry.get(102).tags == []

    def test_many_tags(self, registry, sample_vector):
        """Component with many tags works."""
        tags = [f"tag_{i}" for i in range(100)]
        meta = ComponentMetadata(
            component_id=103,
            component_type="Textbox",
            intent="many tags",
            tags=tags,
        )
        registry.register(103, sample_vector, meta)

        assert len(registry.get(103).tags) == 100

    def test_extra_with_complex_data(self, registry, sample_vector):
        """Extra metadata supports complex nested structures."""
        extra = {
            "nested": {"a": {"b": {"c": 1}}},
            "array": [1, 2, [3, 4]],
            "null": None,
            "bool": True,
            "unicode": "日本語",
        }
        meta = ComponentMetadata(
            component_id=104,
            component_type="Textbox",
            intent="complex extra",
            extra=extra,
        )
        registry.register(104, sample_vector, meta)

        retrieved = registry.get(104)
        assert retrieved.extra["nested"]["a"]["b"]["c"] == 1
        assert retrieved.extra["null"] is None
        assert retrieved.extra["unicode"] == "日本語"

    def test_vector_with_zeros(self, registry):
        """Zero vector is handled correctly."""
        zero_vector = np.zeros(768, dtype=np.float32)
        meta = ComponentMetadata(
            component_id=105,
            component_type="Textbox",
            intent="zero vector",
        )
        registry.register(105, zero_vector, meta)

        assert 105 in registry

    def test_vector_with_negative_values(self, registry):
        """Vector with negative values works."""
        neg_vector = -np.ones(768, dtype=np.float32)
        meta = ComponentMetadata(
            component_id=106,
            component_type="Textbox",
            intent="negative vector",
        )
        registry.register(106, neg_vector, meta)

        assert 106 in registry

    def test_search_with_k_zero(self, populated_registry, mock_embedder):
        """Search with k=0 returns empty list."""
        query_vector = mock_embedder.embed_query("anything")
        results = populated_registry.search(query_vector, k=0)
        assert results == []

    def test_search_with_k_larger_than_registry(self, populated_registry, mock_embedder):
        """Search with k larger than registry size returns all items."""
        query_vector = mock_embedder.embed_query("anything")
        results = populated_registry.search(query_vector, k=1000)
        assert len(results) == 5  # populated_registry has 5 components

    def test_search_with_nonexistent_type_filter(self, populated_registry, mock_embedder):
        """Search with filter for nonexistent type returns empty."""
        query_vector = mock_embedder.embed_query("anything")
        results = populated_registry.search(query_vector, k=10, component_type="NonexistentType")
        assert results == []

    def test_search_with_nonexistent_tag_filter(self, populated_registry, mock_embedder):
        """Search with filter for nonexistent tag returns empty."""
        query_vector = mock_embedder.embed_query("anything")
        results = populated_registry.search(query_vector, k=10, tags=["nonexistent_tag"])
        assert results == []

    def test_relationship_self_reference(self, registry, sample_vector):
        """Self-referencing relationship is stored."""
        meta = ComponentMetadata(
            component_id=200,
            component_type="Textbox",
            intent="self reference",
        )
        registry.register(200, sample_vector, meta)
        registry.add_relationship(200, 200, "dataflow")

        relationships = registry.get_relationships(200)
        assert "dataflow" in relationships
        assert 200 in relationships["dataflow"]

    def test_relationship_to_nonexistent_component(self, registry, sample_vector):
        """Relationship to nonexistent component is stored without error."""
        meta = ComponentMetadata(
            component_id=201,
            component_type="Textbox",
            intent="existing",
        )
        registry.register(201, sample_vector, meta)

        # Add relationship to non-existent component
        registry.add_relationship(201, 9999, "dataflow")

        relationships = registry.get_relationships(201)
        assert "dataflow" in relationships
        assert 9999 in relationships["dataflow"]

    def test_duplicate_relationship_ignored(self, registry, sample_vector):
        """Adding same relationship twice doesn't create duplicates."""
        for i in [301, 302]:
            meta = ComponentMetadata(
                component_id=i,
                component_type="Textbox",
                intent=f"component {i}",
            )
            registry.register(i, sample_vector, meta)

        registry.add_relationship(301, 302, "dataflow")
        registry.add_relationship(301, 302, "dataflow")  # Duplicate

        relationships = registry.get_relationships(301)
        # Should only have one occurrence
        assert relationships["dataflow"].count(302) == 1

    def test_get_dataflow_nonexistent_component(self, registry):
        """get_dataflow on nonexistent component returns empty."""
        flow = registry.get_dataflow(99999)
        assert flow["upstream"] == []
        assert flow["downstream"] == []

    def test_get_relationships_nonexistent_component(self, registry):
        """get_relationships on nonexistent component returns empty dict."""
        relationships = registry.get_relationships(99999)
        assert relationships == {}


class TestConcurrency:
    """Test thread safety and concurrent access.

    Note: SQLite has limited support for concurrent writes from multiple threads.
    The registry uses check_same_thread=False but concurrent writes to HNSW
    can still cause issues. Tests here verify read-only concurrency.
    """

    def test_concurrent_registration_sequential(self, sample_vector):
        """Sequential registration from different threads works."""
        import threading

        registry = ComponentRegistry(db_path=None)
        registered_ids = []
        lock = threading.Lock()

        def register_component(thread_id: int):
            # Each thread registers sequentially with a lock to avoid HNSW/SQLite issues
            with lock:
                for i in range(5):
                    comp_id = thread_id * 1000 + i
                    meta = ComponentMetadata(
                        component_id=comp_id,
                        component_type="Textbox",
                        intent=f"thread {thread_id} component {i}",
                    )
                    np.random.seed(comp_id)
                    vec = np.random.rand(768).astype(np.float32)
                    registry.register(comp_id, vec, meta)
                    registered_ids.append(comp_id)

        threads = [threading.Thread(target=register_component, args=(i,)) for i in range(3)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(registered_ids) == 15  # 3 threads * 5 components
        assert len(registry) == 15

    def test_concurrent_search(self, populated_registry, mock_embedder):
        """Multiple threads can search concurrently."""
        import threading

        errors = []
        results_count = []
        lock = threading.Lock()

        def search_components(thread_id: int):
            try:
                for _ in range(10):
                    query = mock_embedder.embed_query(f"search {thread_id}")
                    results = populated_registry.search(query, k=5)
                    with lock:
                        results_count.append(len(results))
            except Exception as e:
                with lock:
                    errors.append(str(e))

        threads = [threading.Thread(target=search_components, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors during concurrent search: {errors}"
        assert len(results_count) == 50
        assert all(count <= 5 for count in results_count)

    def test_concurrent_memory_reads(self, populated_registry):
        """Multiple threads can read in-memory metadata concurrently."""
        import threading

        errors = []
        results = []
        lock = threading.Lock()

        def reader(thread_id: int):
            try:
                for _ in range(20):
                    for i in range(1, 6):  # populated_registry has components 1-5
                        # Only read from in-memory structures (no SQLite)
                        meta = populated_registry.get(i)
                        in_reg = i in populated_registry
                        with lock:
                            results.append((thread_id, i, meta is not None, in_reg))
            except Exception as e:
                with lock:
                    errors.append(f"Reader {thread_id}: {e}")

        threads = [threading.Thread(target=reader, args=(i,)) for i in range(5)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors during concurrent reads: {errors}"
        # 5 threads * 20 iterations * 5 components = 500 results
        assert len(results) == 500

    def test_single_thread_read_write(self, sample_vector):
        """Single-threaded read/write operations work correctly."""
        registry = ComponentRegistry(db_path=None)

        # Pre-register some components
        for i in range(10):
            meta = ComponentMetadata(
                component_id=i,
                component_type="Textbox",
                intent=f"initial {i}",
            )
            registry.register(i, sample_vector, meta)

        # Interleave reads and writes
        for i in range(10, 20):
            # Read existing
            _ = registry.get(i - 10)
            _ = (i - 10) in registry

            # Write new
            meta = ComponentMetadata(
                component_id=i,
                component_type="Button",
                intent=f"new {i}",
            )
            np.random.seed(i)
            vec = np.random.rand(768).astype(np.float32)
            registry.register(i, vec, meta)

        assert len(registry) == 20


class TestSearchResultDataclass:
    """Test SearchResult dataclass."""

    def test_search_result_fields(self):
        """SearchResult has all expected fields."""
        meta = ComponentMetadata(
            component_id=1,
            component_type="Textbox",
            intent="test",
        )
        result = SearchResult(
            component_id=1,
            metadata=meta,
            score=0.95,
            distance=0.05,
        )

        assert result.component_id == 1
        assert result.metadata is meta
        assert result.score == 0.95
        assert result.distance == 0.05

    def test_search_result_equality(self):
        """SearchResult supports equality comparison."""
        meta = ComponentMetadata(
            component_id=1,
            component_type="Textbox",
            intent="test",
        )
        result1 = SearchResult(component_id=1, metadata=meta, score=0.9, distance=0.1)
        result2 = SearchResult(component_id=1, metadata=meta, score=0.9, distance=0.1)

        assert result1 == result2


class TestFallbackSearchEdgeCases:
    """Test fallback brute-force search edge cases."""

    def test_fallback_with_zero_vector_query(self, sample_vector):
        """Fallback search handles zero vector query."""
        registry = ComponentRegistry(db_path=None)
        registry._index = None  # Force fallback

        meta = ComponentMetadata(
            component_id=1,
            component_type="Textbox",
            intent="test",
            tags=["test"],
        )
        registry.register(1, sample_vector, meta)

        zero_query = np.zeros(768, dtype=np.float32)
        results = registry.search(zero_query, k=5)

        # Should return results without error (using epsilon for division)
        assert isinstance(results, list)

    def test_fallback_with_type_and_tag_filter(self, sample_vector):
        """Fallback search applies both type and tag filters."""
        registry = ComponentRegistry(db_path=None)
        registry._index = None  # Force fallback

        # Register components with different types and tags
        registry.register(
            1,
            sample_vector,
            ComponentMetadata(
                component_id=1,
                component_type="Textbox",
                intent="text input",
                tags=["input", "text"],
            ),
        )
        registry.register(
            2,
            sample_vector * 0.9,
            ComponentMetadata(
                component_id=2,
                component_type="Button",
                intent="submit button",
                tags=["action"],
            ),
        )
        registry.register(
            3,
            sample_vector * 0.8,
            ComponentMetadata(
                component_id=3,
                component_type="Textbox",
                intent="another text",
                tags=["input", "special"],
            ),
        )

        query = sample_vector
        results = registry.search(query, k=10, component_type="Textbox", tags=["input"])

        assert len(results) == 2
        for r in results:
            assert r.metadata.component_type == "Textbox"
            assert "input" in r.metadata.tags


class TestExportGraphEdgeCases:
    """Test export_graph edge cases."""

    def test_export_empty_registry(self, registry):
        """export_graph on empty registry returns empty structure."""
        graph = registry.export_graph()

        assert graph["nodes"] == []
        assert graph["links"] == []

    def test_export_with_no_relationships(self, registry, sample_vector):
        """export_graph works when there are no relationships."""
        meta = ComponentMetadata(
            component_id=1,
            component_type="Textbox",
            intent="lonely component",
        )
        registry.register(1, sample_vector, meta)

        graph = registry.export_graph()

        assert len(graph["nodes"]) == 1
        assert graph["links"] == []

    def test_export_node_label_fallback(self, registry, sample_vector):
        """export_graph uses elem_id or generated label when label is None."""
        # Component with no label but with elem_id
        meta1 = ComponentMetadata(
            component_id=1,
            component_type="Textbox",
            intent="test",
            label=None,
            elem_id="my-element",
        )
        # Component with no label or elem_id
        meta2 = ComponentMetadata(
            component_id=2,
            component_type="Button",
            intent="test",
            label=None,
            elem_id=None,
        )

        registry.register(1, sample_vector, meta1)
        registry.register(2, sample_vector, meta2)

        graph = registry.export_graph()

        labels = {n["id"]: n["label"] for n in graph["nodes"]}
        assert labels[1] == "my-element"
        assert labels[2] == "Button_2"


class TestClearEdgeCases:
    """Test clear method edge cases."""

    def test_clear_empty_registry(self, registry):
        """Clearing empty registry doesn't error."""
        registry.clear()
        assert len(registry) == 0

    def test_clear_reinitializes_hnsw(self, populated_registry, mock_embedder):
        """After clear, HNSW index is re-initialized and functional."""
        populated_registry.clear()

        # Register new component
        np.random.seed(999)
        vector = np.random.rand(768).astype(np.float32)
        meta = ComponentMetadata(
            component_id=999,
            component_type="Textbox",
            intent="new after clear",
        )
        populated_registry.register(999, vector, meta)

        # Search should work
        query = mock_embedder.embed_query("new")
        results = populated_registry.search(query, k=5)

        assert len(results) == 1
        assert results[0].component_id == 999
