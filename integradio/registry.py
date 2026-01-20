"""
Component Registry - HNSW vector index + SQLite metadata storage.

Provides fast semantic search across all registered components.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any
import numpy as np

logger = logging.getLogger(__name__)

try:
    import hnswlib
    HAS_HNSWLIB = True
except ImportError:
    HAS_HNSWLIB = False


@dataclass
class ComponentMetadata:
    """Metadata for a registered component."""
    component_id: int
    component_type: str  # e.g., "Textbox", "Button"
    intent: str  # Semantic description of purpose
    label: Optional[str] = None
    elem_id: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    inputs_from: list[int] = field(default_factory=list)  # Component IDs
    outputs_to: list[int] = field(default_factory=list)  # Component IDs
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ComponentMetadata":
        return cls(**data)


@dataclass
class SearchResult:
    """Result from a semantic search."""
    component_id: int
    metadata: ComponentMetadata
    score: float  # Similarity score (higher = more similar)
    distance: float  # Raw distance from HNSW


class ComponentRegistry:
    """
    Registry for vector-embedded components.

    Combines HNSW for fast vector search with SQLite for metadata.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        dimension: int = 768,
        max_elements: int = 10000,
        ef_construction: int = 200,
        M: int = 16,
    ):
        """
        Initialize the registry.

        Args:
            db_path: Path to SQLite database (None = in-memory)
            dimension: Embedding vector dimension
            max_elements: Maximum number of components
            ef_construction: HNSW construction parameter
            M: HNSW connections per layer
        """
        self.dimension = dimension
        self.max_elements = max_elements
        self._vectors: dict[int, np.ndarray] = {}
        self._metadata: dict[int, ComponentMetadata] = {}

        # Initialize SQLite
        db_str = str(db_path) if db_path else ":memory:"
        self._conn = sqlite3.connect(db_str, check_same_thread=False)
        self._init_db()

        # Initialize HNSW index
        if HAS_HNSWLIB:
            self._index = hnswlib.Index(space="cosine", dim=dimension)
            self._index.init_index(
                max_elements=max_elements,
                ef_construction=ef_construction,
                M=M,
            )
            self._index.set_ef(50)  # Query time accuracy
        else:
            self._index = None

    def _init_db(self) -> None:
        """Initialize SQLite schema."""
        try:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS components (
                    component_id INTEGER PRIMARY KEY,
                    component_type TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    label TEXT,
                    elem_id TEXT,
                    tags TEXT,  -- JSON array
                    file_path TEXT,
                    line_number INTEGER,
                    inputs_from TEXT,  -- JSON array of IDs
                    outputs_to TEXT,  -- JSON array of IDs
                    extra TEXT,  -- JSON object
                    vector BLOB  -- Raw numpy bytes
                );

                CREATE INDEX IF NOT EXISTS idx_component_type ON components(component_type);
                CREATE INDEX IF NOT EXISTS idx_elem_id ON components(elem_id);

                CREATE TABLE IF NOT EXISTS relationships (
                    source_id INTEGER,
                    target_id INTEGER,
                    relationship_type TEXT,  -- 'input', 'output', 'trigger'
                    PRIMARY KEY (source_id, target_id, relationship_type)
                );
            """)
            self._conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize database schema: {e}")
            raise

    def register(
        self,
        component_id: int,
        vector: np.ndarray,
        metadata: ComponentMetadata,
    ) -> bool:
        """
        Register a component with its embedding.

        Args:
            component_id: Unique component ID
            vector: Embedding vector
            metadata: Component metadata

        Returns:
            True if registration successful, False otherwise
        """
        try:
            # Store in memory
            self._vectors[component_id] = vector
            self._metadata[component_id] = metadata

            # Add to HNSW index
            if self._index is not None:
                self._index.add_items(
                    vector.reshape(1, -1),
                    np.array([component_id]),
                )

            # Store in SQLite
            self._conn.execute(
                """
                INSERT OR REPLACE INTO components
                (component_id, component_type, intent, label, elem_id, tags,
                 file_path, line_number, inputs_from, outputs_to, extra, vector)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    component_id,
                    metadata.component_type,
                    metadata.intent,
                    metadata.label,
                    metadata.elem_id,
                    json.dumps(metadata.tags),
                    metadata.file_path,
                    metadata.line_number,
                    json.dumps(metadata.inputs_from),
                    json.dumps(metadata.outputs_to),
                    json.dumps(metadata.extra),
                    vector.tobytes(),
                ),
            )
            self._conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Failed to register component {component_id}: {e}")
            # Rollback memory state on DB failure
            self._vectors.pop(component_id, None)
            self._metadata.pop(component_id, None)
            return False

    def add_relationship(
        self,
        source_id: int,
        target_id: int,
        relationship_type: str,
    ) -> bool:
        """
        Add a relationship between components.

        Args:
            source_id: Source component ID
            target_id: Target component ID
            relationship_type: Type of relationship ('input', 'output', 'trigger')

        Returns:
            True if relationship added successfully, False otherwise
        """
        try:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO relationships
                (source_id, target_id, relationship_type)
                VALUES (?, ?, ?)
                """,
                (source_id, target_id, relationship_type),
            )
            self._conn.commit()

            # Update metadata
            if source_id in self._metadata:
                if target_id not in self._metadata[source_id].outputs_to:
                    self._metadata[source_id].outputs_to.append(target_id)
            if target_id in self._metadata:
                if source_id not in self._metadata[target_id].inputs_from:
                    self._metadata[target_id].inputs_from.append(source_id)
            return True
        except sqlite3.Error as e:
            logger.error(f"Failed to add relationship {source_id}->{target_id}: {e}")
            return False

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        component_type: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        """
        Search for similar components.

        Args:
            query_vector: Query embedding
            k: Number of results
            component_type: Filter by component type
            tags: Filter by tags (any match)

        Returns:
            List of SearchResult ordered by similarity
        """
        if self._index is None or len(self._vectors) == 0:
            return self._fallback_search(query_vector, k, component_type, tags)

        # HNSW search
        labels, distances = self._index.knn_query(
            query_vector.reshape(1, -1),
            k=min(k * 2, len(self._vectors)),  # Over-fetch for filtering
        )

        results = []
        for label, distance in zip(labels[0], distances[0]):
            if label not in self._metadata:
                continue

            metadata = self._metadata[label]

            # Apply filters
            if component_type and metadata.component_type != component_type:
                continue
            if tags and not any(t in metadata.tags for t in tags):
                continue

            # Convert cosine distance to similarity
            score = 1.0 - distance

            results.append(SearchResult(
                component_id=int(label),
                metadata=metadata,
                score=score,
                distance=distance,
            ))

            if len(results) >= k:
                break

        return results

    def _fallback_search(
        self,
        query_vector: np.ndarray,
        k: int,
        component_type: Optional[str],
        tags: Optional[list[str]],
    ) -> list[SearchResult]:
        """Brute-force search when HNSW unavailable."""
        results = []
        # Use epsilon to prevent division by zero (2026 best practice)
        eps = np.finfo(np.float32).eps

        for comp_id, vector in self._vectors.items():
            metadata = self._metadata[comp_id]

            if component_type and metadata.component_type != component_type:
                continue
            if tags and not any(t in metadata.tags for t in tags):
                continue

            # Cosine similarity with epsilon to prevent division by zero
            query_norm = np.linalg.norm(query_vector)
            vector_norm = np.linalg.norm(vector)
            denominator = (query_norm * vector_norm) + eps
            similarity = np.dot(query_vector, vector) / denominator
            distance = 1.0 - similarity

            results.append(SearchResult(
                component_id=comp_id,
                metadata=metadata,
                score=float(similarity),
                distance=float(distance),
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    def get(self, component_id: int) -> Optional[ComponentMetadata]:
        """Get metadata for a component."""
        return self._metadata.get(component_id)

    def get_relationships(self, component_id: int) -> dict[str, list[int]]:
        """Get all relationships for a component."""
        try:
            cursor = self._conn.execute(
                """
                SELECT target_id, relationship_type FROM relationships WHERE source_id = ?
                UNION
                SELECT source_id, relationship_type FROM relationships WHERE target_id = ?
                """,
                (component_id, component_id),
            )

            relationships: dict[str, list[int]] = {}
            for target_id, rel_type in cursor:
                if rel_type not in relationships:
                    relationships[rel_type] = []
                relationships[rel_type].append(target_id)

            return relationships
        except sqlite3.Error as e:
            logger.error(f"Failed to get relationships for component {component_id}: {e}")
            return {}

    def get_dataflow(self, component_id: int) -> dict[str, Any]:
        """
        Trace data flow through a component.

        Returns dict with 'upstream' and 'downstream' component chains.
        """
        def trace_direction(start_id: int, direction: str) -> list[int]:
            visited = set()
            queue = [start_id]
            chain = []

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)

                if current != start_id:
                    chain.append(current)

                meta = self._metadata.get(current)
                if meta:
                    next_ids = (
                        meta.inputs_from if direction == "upstream"
                        else meta.outputs_to
                    )
                    queue.extend(next_ids)

            return chain

        return {
            "upstream": trace_direction(component_id, "upstream"),
            "downstream": trace_direction(component_id, "downstream"),
        }

    def all_components(self) -> list[ComponentMetadata]:
        """Get all registered components."""
        return list(self._metadata.values())

    def export_graph(self) -> dict[str, Any]:
        """
        Export component graph for visualization.

        Returns D3.js compatible format with nodes and links.
        """
        nodes = []
        links = []

        for comp_id, meta in self._metadata.items():
            nodes.append({
                "id": comp_id,
                "type": meta.component_type,
                "intent": meta.intent or "",
                "label": meta.label or meta.elem_id or f"{meta.component_type}_{comp_id}",
            })

        try:
            cursor = self._conn.execute(
                "SELECT source_id, target_id, relationship_type FROM relationships"
            )
            for source, target, rel_type in cursor:
                links.append({
                    "source": source,
                    "target": target,
                    "type": rel_type,
                })
        except sqlite3.Error as e:
            logger.error(f"Failed to export graph relationships: {e}")

        return {"nodes": nodes, "links": links}

    def clear(self) -> None:
        """Clear all registered components."""
        self._vectors.clear()
        self._metadata.clear()
        try:
            self._conn.executescript("""
                DELETE FROM components;
                DELETE FROM relationships;
            """)
            self._conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to clear database tables: {e}")

        if self._index is not None:
            self._index = hnswlib.Index(space="cosine", dim=self.dimension)
            self._index.init_index(max_elements=self.max_elements)
            self._index.set_ef(50)

    def __len__(self) -> int:
        return len(self._metadata)

    def __contains__(self, component_id: int) -> bool:
        return component_id in self._metadata
