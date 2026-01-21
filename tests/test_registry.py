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

    def test_fallback_with_type_filter_only(self, sample_vector):
        """Fallback search with only component_type filter."""
        registry = ComponentRegistry(db_path=None)
        registry._index = None  # Force fallback

        # Register components with different types
        registry.register(
            1,
            sample_vector,
            ComponentMetadata(component_id=1, component_type="Textbox", intent="a"),
        )
        registry.register(
            2,
            sample_vector * 0.9,
            ComponentMetadata(component_id=2, component_type="Button", intent="b"),
        )
        registry.register(
            3,
            sample_vector * 0.8,
            ComponentMetadata(component_id=3, component_type="Textbox", intent="c"),
        )

        query = sample_vector
        results = registry.search(query, k=10, component_type="Textbox")

        assert len(results) == 2
        for r in results:
            assert r.metadata.component_type == "Textbox"

    def test_fallback_with_tags_filter_only(self, sample_vector):
        """Fallback search with only tags filter (covers line 334)."""
        registry = ComponentRegistry(db_path=None)
        registry._index = None  # Force fallback

        # Register components with different tags but same type
        registry.register(
            1,
            sample_vector,
            ComponentMetadata(
                component_id=1,
                component_type="Textbox",
                intent="a",
                tags=["input"],
            ),
        )
        registry.register(
            2,
            sample_vector * 0.9,
            ComponentMetadata(
                component_id=2,
                component_type="Textbox",
                intent="b",
                tags=["action"],  # Different tag - should be filtered out
            ),
        )
        registry.register(
            3,
            sample_vector * 0.8,
            ComponentMetadata(
                component_id=3,
                component_type="Textbox",
                intent="c",
                tags=["input", "special"],
            ),
        )

        query = sample_vector
        # Only filter by tags, not by type
        results = registry.search(query, k=10, tags=["input"])

        assert len(results) == 2
        for r in results:
            assert "input" in r.metadata.tags

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


class TestHNSWSearchPath:
    """Test HNSW search when hnswlib is available (not using fallback)."""

    def test_hnsw_search_basic(self, sample_vector):
        """Test search uses HNSW index when available."""
        registry = ComponentRegistry(db_path=None)
        # Don't set _index to None - use real HNSW

        # Register components
        for i in range(5):
            meta = ComponentMetadata(
                component_id=i,
                component_type="Textbox" if i % 2 == 0 else "Button",
                intent=f"component {i}",
                tags=["test", f"tag_{i}"],
            )
            np.random.seed(i)
            vec = np.random.rand(768).astype(np.float32)
            registry.register(i, vec, meta)

        # Search using HNSW (not fallback)
        query = np.random.rand(768).astype(np.float32)
        results = registry.search(query, k=3)

        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)
        # Results should be sorted by score (highest first)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_hnsw_search_with_type_filter(self, sample_vector):
        """Test HNSW search with component_type filter."""
        registry = ComponentRegistry(db_path=None)

        # Register mixed components
        for i in range(6):
            comp_type = "Textbox" if i < 3 else "Button"
            meta = ComponentMetadata(
                component_id=i,
                component_type=comp_type,
                intent=f"component {i}",
            )
            np.random.seed(i)
            vec = np.random.rand(768).astype(np.float32)
            registry.register(i, vec, meta)

        query = np.random.rand(768).astype(np.float32)
        results = registry.search(query, k=10, component_type="Textbox")

        assert len(results) == 3
        for r in results:
            assert r.metadata.component_type == "Textbox"

    def test_hnsw_search_with_tags_filter(self, sample_vector):
        """Test HNSW search with tags filter."""
        registry = ComponentRegistry(db_path=None)

        # Register components with different tags
        registry.register(
            1,
            np.random.rand(768).astype(np.float32),
            ComponentMetadata(component_id=1, component_type="Textbox", intent="a", tags=["input"]),
        )
        registry.register(
            2,
            np.random.rand(768).astype(np.float32),
            ComponentMetadata(component_id=2, component_type="Button", intent="b", tags=["action"]),
        )
        registry.register(
            3,
            np.random.rand(768).astype(np.float32),
            ComponentMetadata(component_id=3, component_type="Textbox", intent="c", tags=["input", "special"]),
        )

        query = np.random.rand(768).astype(np.float32)
        results = registry.search(query, k=10, tags=["input"])

        assert len(results) == 2
        for r in results:
            assert "input" in r.metadata.tags

    def test_hnsw_search_filters_missing_metadata(self, sample_vector):
        """Test HNSW search gracefully handles labels not in metadata."""
        registry = ComponentRegistry(db_path=None)

        # Register a component
        meta = ComponentMetadata(component_id=1, component_type="Textbox", intent="test")
        vec = np.random.rand(768).astype(np.float32)
        registry.register(1, vec, meta)

        # Manually remove from metadata dict to simulate inconsistency
        registry._metadata.clear()

        query = np.random.rand(768).astype(np.float32)
        results = registry.search(query, k=5)

        # Should return empty because metadata is missing
        assert results == []

    def test_hnsw_search_k_limit_reached_early(self, sample_vector):
        """Test HNSW search stops when k results found after filtering."""
        registry = ComponentRegistry(db_path=None)

        # Register many components with same type
        for i in range(10):
            meta = ComponentMetadata(
                component_id=i,
                component_type="Textbox",
                intent=f"component {i}",
            )
            np.random.seed(i)
            vec = np.random.rand(768).astype(np.float32)
            registry.register(i, vec, meta)

        query = np.random.rand(768).astype(np.float32)
        results = registry.search(query, k=3)

        assert len(results) == 3


class TestDatabaseErrors:
    """Test database error handling paths."""

    def test_database_connection_error(self, tmp_path):
        """Test handling of database connection error."""
        from integradio.exceptions import RegistryDatabaseError

        # Try to create a database in a non-existent directory
        invalid_path = tmp_path / "nonexistent_subdir" / "another" / "test.db"

        with pytest.raises(RegistryDatabaseError) as exc_info:
            ComponentRegistry(db_path=invalid_path)

        assert "connect" in str(exc_info.value)

    def test_schema_init_error(self, tmp_path):
        """Test handling of schema initialization error."""
        from integradio.exceptions import RegistryDatabaseError
        from unittest.mock import patch, MagicMock
        import sqlite3

        # Create a mock connection that raises on executescript
        mock_conn = MagicMock()
        mock_conn.executescript.side_effect = sqlite3.Error("Schema init failed")

        with patch("sqlite3.connect", return_value=mock_conn):
            with pytest.raises(RegistryDatabaseError) as exc_info:
                ComponentRegistry(db_path=None)

        assert "schema_init" in str(exc_info.value)

    def test_registration_database_error(self, sample_vector):
        """Test registration error with rollback."""
        from integradio.exceptions import ComponentRegistrationError

        registry = ComponentRegistry(db_path=None)

        # Close the connection to force an error on next operation
        registry._conn.close()

        meta = ComponentMetadata(
            component_id=1,
            component_type="Textbox",
            intent="test",
        )

        with pytest.raises(ComponentRegistrationError) as exc_info:
            registry.register(1, sample_vector, meta)

        assert exc_info.value.component_id == 1
        # Memory state should be rolled back
        assert 1 not in registry._vectors
        assert 1 not in registry._metadata

    def test_add_relationship_database_error(self, sample_vector):
        """Test add_relationship error handling."""
        from integradio.exceptions import RegistryDatabaseError

        registry = ComponentRegistry(db_path=None)

        # Register components first
        for i in [1, 2]:
            meta = ComponentMetadata(component_id=i, component_type="Textbox", intent=f"test {i}")
            registry.register(i, sample_vector, meta)

        # Close connection to force error
        registry._conn.close()

        with pytest.raises(RegistryDatabaseError) as exc_info:
            registry.add_relationship(1, 2, "dataflow")

        assert "add_relationship" in str(exc_info.value)

    def test_get_relationships_database_error(self, sample_vector):
        """Test get_relationships error handling."""
        from integradio.exceptions import RegistryDatabaseError

        registry = ComponentRegistry(db_path=None)

        meta = ComponentMetadata(component_id=1, component_type="Textbox", intent="test")
        registry.register(1, sample_vector, meta)

        # Close connection to force error
        registry._conn.close()

        with pytest.raises(RegistryDatabaseError) as exc_info:
            registry.get_relationships(1)

        assert "get_relationships" in str(exc_info.value)

    def test_export_graph_database_error(self, sample_vector):
        """Test export_graph error handling."""
        from integradio.exceptions import RegistryDatabaseError

        registry = ComponentRegistry(db_path=None)

        meta = ComponentMetadata(component_id=1, component_type="Textbox", intent="test")
        registry.register(1, sample_vector, meta)

        # Close connection to force error
        registry._conn.close()

        with pytest.raises(RegistryDatabaseError) as exc_info:
            registry.export_graph()

        assert "export_graph" in str(exc_info.value)

    def test_clear_database_error(self, sample_vector):
        """Test clear error handling."""
        from integradio.exceptions import RegistryDatabaseError

        registry = ComponentRegistry(db_path=None)

        meta = ComponentMetadata(component_id=1, component_type="Textbox", intent="test")
        registry.register(1, sample_vector, meta)

        # Close connection to force error
        registry._conn.close()

        with pytest.raises(RegistryDatabaseError) as exc_info:
            registry.clear()

        assert "clear" in str(exc_info.value)


class TestHNSWInitialization:
    """Test HNSW index initialization and re-initialization."""

    def test_hnsw_initialization_with_custom_params(self):
        """Test HNSW is initialized with custom parameters."""
        registry = ComponentRegistry(
            db_path=None,
            dimension=512,
            max_elements=5000,
            ef_construction=100,
            M=32,
        )

        assert registry.dimension == 512
        assert registry.max_elements == 5000
        # HNSW index should be initialized
        assert registry._index is not None

    def test_registry_without_hnswlib(self):
        """Test registry works when HAS_HNSWLIB is False."""
        import integradio.registry as registry_module

        # Save original value
        original_has_hnswlib = registry_module.HAS_HNSWLIB

        try:
            # Simulate hnswlib not being available
            registry_module.HAS_HNSWLIB = False

            registry = ComponentRegistry(db_path=None)

            # _index should be None
            assert registry._index is None

            # Registry should still work via fallback
            meta = ComponentMetadata(
                component_id=1,
                component_type="Textbox",
                intent="test",
            )
            vec = np.random.rand(768).astype(np.float32)
            registry.register(1, vec, meta)

            assert 1 in registry
            results = registry.search(vec, k=1)
            assert len(results) == 1
        finally:
            # Restore original value
            registry_module.HAS_HNSWLIB = original_has_hnswlib

    def test_import_without_hnswlib_available(self):
        """Test HAS_HNSWLIB is False when hnswlib import fails (lines 26-27)."""
        import sys
        import importlib

        # Save references to clean up later
        modules_to_restore = {}
        for key in list(sys.modules.keys()):
            if "hnswlib" in key or key == "integradio.registry":
                modules_to_restore[key] = sys.modules.pop(key)

        # Set up import hook to make hnswlib fail
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "hnswlib":
                raise ImportError("No module named 'hnswlib'")
            return original_import(name, *args, **kwargs)

        try:
            builtins.__import__ = mock_import

            # Re-import the registry module to trigger the ImportError path
            # The module was removed from sys.modules, so this is a fresh import
            import integradio.registry as fresh_registry

            # Now HAS_HNSWLIB should be False
            assert fresh_registry.HAS_HNSWLIB is False
        finally:
            # Restore original import first
            builtins.__import__ = original_import

            # Remove the freshly imported module with HAS_HNSWLIB=False
            if "integradio.registry" in sys.modules:
                del sys.modules["integradio.registry"]

            # Restore original modules
            for key, module in modules_to_restore.items():
                sys.modules[key] = module

    def test_hnsw_add_items_during_registration(self, sample_vector):
        """Test HNSW add_items is called during registration."""
        registry = ComponentRegistry(db_path=None)

        meta = ComponentMetadata(
            component_id=1,
            component_type="Textbox",
            intent="test",
        )
        registry.register(1, sample_vector, meta)

        # Verify it's searchable via HNSW
        results = registry.search(sample_vector, k=1)
        assert len(results) == 1
        assert results[0].component_id == 1

    def test_clear_reinitializes_hnsw_index(self, sample_vector):
        """Test that clear() properly reinitializes HNSW index."""
        registry = ComponentRegistry(db_path=None)

        # Register and verify
        meta = ComponentMetadata(component_id=1, component_type="Textbox", intent="test")
        registry.register(1, sample_vector, meta)

        old_index = registry._index

        # Clear and verify index is recreated
        registry.clear()

        # Index should be a new object
        assert registry._index is not old_index
        assert len(registry) == 0

        # Should be able to register and search again
        registry.register(2, sample_vector, ComponentMetadata(
            component_id=2, component_type="Button", intent="new"
        ))
        results = registry.search(sample_vector, k=1)
        assert len(results) == 1
        assert results[0].component_id == 2
