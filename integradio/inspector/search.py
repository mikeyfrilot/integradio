"""
Search Engine - Search components by intent, tags, and type.

Provides semantic search across UI components using:
- Intent matching (fuzzy text search)
- Tag filtering
- Component type filtering
- Combined queries
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum
import re


@dataclass
class SearchResult:
    """A single search result."""
    component_id: str
    component_type: str
    intent: str
    score: float  # 0.0 to 1.0 relevance score
    match_type: str  # "intent", "tag", "type", "label"
    matched_text: str  # What matched the query
    tags: list[str] = field(default_factory=list)
    file_path: str | None = None
    line_number: int | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.component_id,
            "type": self.component_type,
            "intent": self.intent,
            "score": self.score,
            "match_type": self.match_type,
            "matched_text": self.matched_text,
            "tags": self.tags,
            "file_path": self.file_path,
            "line_number": self.line_number,
        }


class SearchEngine:
    """
    Search engine for semantic components.

    Supports multiple search strategies:
    - Text matching (fuzzy)
    - Embedding-based semantic search (if embedder available)
    - Tag/type filtering
    """

    def __init__(self, blocks: Any = None):
        """
        Initialize search engine.

        Args:
            blocks: SemanticBlocks instance to search
        """
        self.blocks = blocks
        self._cache: dict[str, Any] = {}

    def search(
        self,
        query: str,
        search_intents: bool = True,
        search_tags: bool = True,
        search_types: bool = True,
        search_labels: bool = True,
        max_results: int = 20,
        min_score: float = 0.1,
    ) -> list[SearchResult]:
        """
        Search for components matching a query.

        Args:
            query: Search query string
            search_intents: Search in component intents
            search_tags: Search in component tags
            search_types: Search in component types
            search_labels: Search in component labels
            max_results: Maximum number of results
            min_score: Minimum relevance score (0-1)

        Returns:
            List of SearchResults, sorted by relevance
        """
        from ..components import SemanticComponent

        results: list[SearchResult] = []
        query_lower = query.lower()
        query_words = set(query_lower.split())

        for comp_id, semantic in SemanticComponent._instances.items():
            meta = semantic.semantic_meta
            component = semantic.component

            best_score = 0.0
            match_type = ""
            matched_text = ""

            # Search intents
            if search_intents and meta.intent:
                score = self._fuzzy_match(query_lower, meta.intent.lower())
                if score > best_score:
                    best_score = score
                    match_type = "intent"
                    matched_text = meta.intent

            # Search tags
            if search_tags and meta.tags:
                for tag in meta.tags:
                    score = self._fuzzy_match(query_lower, tag.lower())
                    if score > best_score:
                        best_score = score
                        match_type = "tag"
                        matched_text = tag

            # Search types
            if search_types:
                comp_type = type(component).__name__
                score = self._fuzzy_match(query_lower, comp_type.lower())
                if score > best_score:
                    best_score = score
                    match_type = "type"
                    matched_text = comp_type

            # Search labels
            if search_labels:
                label = getattr(component, "label", None)
                if label:
                    score = self._fuzzy_match(query_lower, label.lower())
                    if score > best_score:
                        best_score = score
                        match_type = "label"
                        matched_text = label

            if best_score >= min_score:
                results.append(SearchResult(
                    component_id=str(comp_id),
                    component_type=type(component).__name__,
                    intent=meta.intent,
                    score=best_score,
                    match_type=match_type,
                    matched_text=matched_text,
                    tags=meta.tags,
                    file_path=meta.file_path,
                    line_number=meta.line_number,
                ))

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        return results[:max_results]

    def search_semantic(
        self,
        query: str,
        max_results: int = 10,
    ) -> list[SearchResult]:
        """
        Semantic search using embeddings.

        Requires an embedder to be available.

        Args:
            query: Natural language query
            max_results: Maximum results

        Returns:
            List of semantically similar components
        """
        from ..components import SemanticComponent

        # Check if registry has an embedder
        embedder = SemanticComponent._embedder
        registry = SemanticComponent._registry

        if not embedder or not registry:
            # Fall back to text search
            return self.search(query, max_results=max_results)

        # Generate query embedding
        query_vector = embedder.embed(query)

        # Search registry
        matches = registry.search(query_vector, top_k=max_results)

        results = []
        for match in matches:
            meta = match.metadata
            results.append(SearchResult(
                component_id=str(meta.component_id),
                component_type=meta.component_type,
                intent=meta.intent,
                score=match.score,
                match_type="semantic",
                matched_text=meta.intent,
                tags=meta.tags,
                file_path=meta.file_path,
                line_number=meta.line_number,
            ))

        return results

    def _fuzzy_match(self, query: str, text: str) -> float:
        """
        Calculate fuzzy match score between query and text.

        Returns:
            Score from 0.0 (no match) to 1.0 (exact match)
        """
        if not query or not text:
            return 0.0

        # Exact match
        if query == text:
            return 1.0

        # Contains match
        if query in text:
            return 0.9

        # Word overlap
        query_words = set(query.split())
        text_words = set(text.split())
        overlap = len(query_words & text_words)
        if overlap > 0:
            return 0.5 + (0.4 * overlap / max(len(query_words), len(text_words)))

        # Substring match
        if any(q in text for q in query.split()):
            return 0.4

        # Prefix match
        if text.startswith(query) or query.startswith(text):
            return 0.3

        # Levenshtein-ish (simplified)
        common_chars = sum(1 for c in query if c in text)
        return 0.2 * common_chars / max(len(query), len(text))


# =============================================================================
# Convenience Functions
# =============================================================================

def search_by_intent(query: str, max_results: int = 20) -> list[SearchResult]:
    """
    Search components by intent.

    Args:
        query: Intent search query
        max_results: Maximum results

    Returns:
        Matching components
    """
    engine = SearchEngine()
    return engine.search(
        query,
        search_intents=True,
        search_tags=False,
        search_types=False,
        search_labels=False,
        max_results=max_results,
    )


def search_by_tag(tag: str, max_results: int = 20) -> list[SearchResult]:
    """
    Search components by tag.

    Args:
        tag: Tag to search for
        max_results: Maximum results

    Returns:
        Components with matching tag
    """
    engine = SearchEngine()
    return engine.search(
        tag,
        search_intents=False,
        search_tags=True,
        search_types=False,
        search_labels=False,
        max_results=max_results,
    )


def search_by_type(component_type: str, max_results: int = 20) -> list[SearchResult]:
    """
    Search components by type.

    Args:
        component_type: Component type (e.g., "Button", "Textbox")
        max_results: Maximum results

    Returns:
        Components of matching type
    """
    engine = SearchEngine()
    return engine.search(
        component_type,
        search_intents=False,
        search_tags=False,
        search_types=True,
        search_labels=False,
        max_results=max_results,
    )


def find_component(query: str) -> SearchResult | None:
    """
    Find a single component by query.

    Args:
        query: Search query

    Returns:
        Best matching component or None
    """
    engine = SearchEngine()
    results = engine.search(query, max_results=1)
    return results[0] if results else None


def list_all_intents() -> list[str]:
    """
    List all unique intents in the app.

    Returns:
        List of intent strings
    """
    from ..components import SemanticComponent

    intents = set()
    for semantic in SemanticComponent._instances.values():
        intents.add(semantic.semantic_meta.intent)
    return sorted(intents)


def list_all_tags() -> list[str]:
    """
    List all unique tags in the app.

    Returns:
        List of tag strings
    """
    from ..components import SemanticComponent

    tags = set()
    for semantic in SemanticComponent._instances.values():
        tags.update(semantic.semantic_meta.tags)
    return sorted(tags)


def list_all_types() -> list[str]:
    """
    List all component types in the app.

    Returns:
        List of type names
    """
    from ..components import SemanticComponent

    types = set()
    for semantic in SemanticComponent._instances.values():
        types.add(type(semantic.component).__name__)
    return sorted(types)
