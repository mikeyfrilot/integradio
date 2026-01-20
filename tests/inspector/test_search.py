"""
Tests for Search module.

Tests:
- SearchResult creation and serialization
- SearchEngine fuzzy matching
- Intent, tag, type, label search
- Semantic search fallback
- Convenience functions
"""

import pytest
from integradio.inspector.search import (
    SearchResult,
    SearchEngine,
    search_by_intent,
    search_by_tag,
    search_by_type,
    find_component,
    list_all_intents,
    list_all_tags,
    list_all_types,
)


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_create_result(self):
        """Test basic result creation."""
        result = SearchResult(
            component_id="123",
            component_type="Button",
            intent="submit form",
            score=0.85,
            match_type="intent",
            matched_text="submit form",
        )

        assert result.component_id == "123"
        assert result.component_type == "Button"
        assert result.intent == "submit form"
        assert result.score == 0.85
        assert result.match_type == "intent"

    def test_result_with_tags(self):
        """Test result with tags."""
        result = SearchResult(
            component_id="456",
            component_type="Textbox",
            intent="user input",
            score=0.75,
            match_type="tag",
            matched_text="input",
            tags=["input", "form", "text"],
        )

        assert len(result.tags) == 3
        assert "input" in result.tags

    def test_result_with_file_info(self):
        """Test result with file path and line number."""
        result = SearchResult(
            component_id="789",
            component_type="Dropdown",
            intent="select option",
            score=0.90,
            match_type="intent",
            matched_text="select option",
            file_path="/path/to/app.py",
            line_number=42,
        )

        assert result.file_path == "/path/to/app.py"
        assert result.line_number == 42

    def test_result_to_dict(self):
        """Test result serialization."""
        result = SearchResult(
            component_id="100",
            component_type="Slider",
            intent="adjust value",
            score=0.65,
            match_type="type",
            matched_text="Slider",
            tags=["control"],
            file_path="/app.py",
            line_number=10,
        )

        data = result.to_dict()

        assert data["id"] == "100"
        assert data["type"] == "Slider"
        assert data["intent"] == "adjust value"
        assert data["score"] == 0.65
        assert data["match_type"] == "type"
        assert data["matched_text"] == "Slider"
        assert data["tags"] == ["control"]
        assert data["file_path"] == "/app.py"
        assert data["line_number"] == 10

    def test_result_default_values(self):
        """Test default values for optional fields."""
        result = SearchResult(
            component_id="1",
            component_type="Button",
            intent="click",
            score=1.0,
            match_type="intent",
            matched_text="click",
        )

        assert result.tags == []
        assert result.file_path is None
        assert result.line_number is None


class TestSearchEngineFuzzyMatch:
    """Tests for SearchEngine fuzzy matching algorithm."""

    def test_exact_match(self):
        """Test exact string match."""
        engine = SearchEngine()
        score = engine._fuzzy_match("submit", "submit")
        assert score == 1.0

    def test_contains_match(self):
        """Test query contained in text."""
        engine = SearchEngine()
        score = engine._fuzzy_match("submit", "submit form data")
        assert score == 0.9

    def test_word_overlap(self):
        """Test word overlap scoring."""
        engine = SearchEngine()
        score = engine._fuzzy_match("user input", "input from user")
        # Both words match
        assert score >= 0.5

    def test_substring_match(self):
        """Test substring matching."""
        engine = SearchEngine()
        score = engine._fuzzy_match("search button", "searchbutton")
        # "search" is a substring
        assert score >= 0.3

    def test_prefix_match(self):
        """Test prefix matching."""
        engine = SearchEngine()
        score = engine._fuzzy_match("sub", "submit")
        assert score >= 0.3

    def test_partial_character_match(self):
        """Test partial character overlap."""
        engine = SearchEngine()
        score = engine._fuzzy_match("btn", "button")
        # b, t, n all appear in button
        assert score > 0.0

    def test_empty_query(self):
        """Test empty query returns 0."""
        engine = SearchEngine()
        assert engine._fuzzy_match("", "text") == 0.0

    def test_empty_text(self):
        """Test empty text returns 0."""
        engine = SearchEngine()
        assert engine._fuzzy_match("query", "") == 0.0

    def test_both_empty(self):
        """Test both empty returns 0."""
        engine = SearchEngine()
        assert engine._fuzzy_match("", "") == 0.0

    def test_no_match(self):
        """Test completely unrelated strings."""
        engine = SearchEngine()
        score = engine._fuzzy_match("xyz", "abc")
        assert score < 0.2


class TestSearchEngineConfig:
    """Tests for SearchEngine configuration."""

    def test_create_engine_no_blocks(self):
        """Test creating engine without blocks."""
        engine = SearchEngine()
        assert engine.blocks is None
        assert engine._cache == {}

    def test_create_engine_with_blocks(self):
        """Test creating engine with blocks."""
        mock_blocks = object()
        engine = SearchEngine(mock_blocks)
        assert engine.blocks is mock_blocks

    def test_search_options(self):
        """Test search method accepts all options."""
        engine = SearchEngine()

        # Should not raise
        results = engine.search(
            query="test",
            search_intents=True,
            search_tags=True,
            search_types=True,
            search_labels=True,
            max_results=10,
            min_score=0.5,
        )

        # Without real components, returns empty list
        assert isinstance(results, list)

    def test_search_semantic_fallback(self):
        """Test semantic search falls back to text search."""
        engine = SearchEngine()

        # Without embedder, should fall back to regular search
        results = engine.search_semantic("test query")
        assert isinstance(results, list)


class TestSearchEngineResultOrdering:
    """Tests for search result ordering and limits."""

    def test_results_sorted_by_score(self):
        """Test that results are sorted by score descending."""
        # Create results with different scores
        results = [
            SearchResult("1", "Button", "low", 0.3, "intent", "low"),
            SearchResult("2", "Button", "high", 0.9, "intent", "high"),
            SearchResult("3", "Button", "mid", 0.6, "intent", "mid"),
        ]

        # Sort like the engine does
        results.sort(key=lambda r: r.score, reverse=True)

        assert results[0].score == 0.9
        assert results[1].score == 0.6
        assert results[2].score == 0.3

    def test_max_results_respected(self):
        """Test max_results limits output."""
        # The engine should respect max_results parameter
        engine = SearchEngine()
        # Without components, can't test directly but verify the parameter is accepted
        results = engine.search("test", max_results=5)
        assert len(results) <= 5

    def test_min_score_filtering(self):
        """Test min_score filters low-scoring results."""
        engine = SearchEngine()
        # High min_score should filter most results
        results = engine.search("test", min_score=0.99)
        # All results should have score >= 0.99
        for result in results:
            assert result.score >= 0.99


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_search_by_intent(self):
        """Test intent-only search."""
        # Without real components, just verify it runs
        results = search_by_intent("submit")
        assert isinstance(results, list)

    def test_search_by_tag(self):
        """Test tag-only search."""
        results = search_by_tag("form")
        assert isinstance(results, list)

    def test_search_by_type(self):
        """Test type-only search."""
        results = search_by_type("Button")
        assert isinstance(results, list)

    def test_find_component_returns_none(self):
        """Test find_component returns None when no match."""
        result = find_component("nonexistent_query_xyz")
        assert result is None

    def test_find_component_returns_single(self):
        """Test find_component returns single result."""
        # It should return SearchResult or None, never a list
        result = find_component("test")
        assert result is None or isinstance(result, SearchResult)

    def test_list_all_intents(self):
        """Test listing all intents."""
        intents = list_all_intents()
        assert isinstance(intents, list)

    def test_list_all_tags(self):
        """Test listing all tags."""
        tags = list_all_tags()
        assert isinstance(tags, list)

    def test_list_all_types(self):
        """Test listing all types."""
        types = list_all_types()
        assert isinstance(types, list)


class TestSearchEdgeCases:
    """Edge case tests for search module."""

    def test_special_characters_in_query(self):
        """Test search with special characters."""
        engine = SearchEngine()
        # Should not raise
        results = engine.search("user@input")
        assert isinstance(results, list)

    def test_unicode_query(self):
        """Test search with unicode characters."""
        engine = SearchEngine()
        results = engine.search("日本語")
        assert isinstance(results, list)

    def test_very_long_query(self):
        """Test search with very long query."""
        engine = SearchEngine()
        long_query = "a" * 1000
        results = engine.search(long_query)
        assert isinstance(results, list)

    def test_whitespace_only_query(self):
        """Test search with whitespace only."""
        engine = SearchEngine()
        results = engine.search("   ")
        assert isinstance(results, list)

    def test_newlines_in_query(self):
        """Test search with newlines."""
        engine = SearchEngine()
        results = engine.search("search\nquery")
        assert isinstance(results, list)

    def test_zero_max_results(self):
        """Test max_results=0 returns empty list."""
        engine = SearchEngine()
        results = engine.search("test", max_results=0)
        assert results == []

    def test_high_min_score_filters_all(self):
        """Test very high min_score filters everything."""
        engine = SearchEngine()
        results = engine.search("test", min_score=2.0)  # Impossible score
        assert results == []

    def test_search_all_disabled(self):
        """Test search with all options disabled."""
        engine = SearchEngine()
        results = engine.search(
            "test",
            search_intents=False,
            search_tags=False,
            search_types=False,
            search_labels=False,
        )
        assert isinstance(results, list)

    def test_result_to_dict_round_trip(self):
        """Test that to_dict contains all necessary info."""
        result = SearchResult(
            component_id="test",
            component_type="Test",
            intent="test intent",
            score=0.5,
            match_type="test",
            matched_text="test",
            tags=["a", "b"],
            file_path="/test.py",
            line_number=1,
        )

        data = result.to_dict()

        # All fields should be present
        assert "id" in data
        assert "type" in data
        assert "intent" in data
        assert "score" in data
        assert "match_type" in data
        assert "matched_text" in data
        assert "tags" in data
        assert "file_path" in data
        assert "line_number" in data
