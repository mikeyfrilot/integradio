"""
Flow Index - HNSW vector index for fast flow similarity search.

Provides semantic search over encoded sessions using the same HNSW
approach as Integradio's ComponentRegistry.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterator
import numpy as np

from .config import BehaviorModelerConfig
from .models import Session
from .store import FlowStore
from .encoder import FlowEncoder

logger = logging.getLogger(__name__)

try:
    import hnswlib
    HAS_HNSWLIB = True
except ImportError:
    HAS_HNSWLIB = False
    logger.warning("hnswlib not available, using brute-force search")


@dataclass
class SearchResult:
    """Result from a flow similarity search."""
    session_id: str
    similarity: float  # 0.0 to 1.0 (higher = more similar)
    distance: float    # Raw HNSW distance
    session: Optional[Session] = None  # Populated if include_session=True


class FlowIndex:
    """
    HNSW-based vector index for session embeddings.

    Provides fast approximate nearest neighbor search over
    encoded user flows.
    """

    def __init__(
        self,
        store: FlowStore,
        encoder: FlowEncoder,
        config: Optional[BehaviorModelerConfig] = None,
    ):
        """
        Initialize the flow index.

        Args:
            store: FlowStore for session retrieval
            encoder: FlowEncoder for query encoding
            config: Configuration options
        """
        self.store = store
        self.encoder = encoder
        self.config = config or BehaviorModelerConfig()

        self._index: Optional["hnswlib.Index"] = None
        self._session_ids: list[str] = []  # Maps HNSW label -> session_id
        self._id_to_label: dict[str, int] = {}  # session_id -> HNSW label

        self._initialized = False

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return self.config.embed_dimension

    @property
    def size(self) -> int:
        """Number of indexed sessions."""
        return len(self._session_ids)

    def _init_index(self) -> None:
        """Initialize HNSW index."""
        if not HAS_HNSWLIB:
            logger.warning("hnswlib not available, index will use brute-force")
            self._initialized = True
            return

        self._index = hnswlib.Index(space="cosine", dim=self.dimension)
        self._index.init_index(
            max_elements=self.config.hnsw_max_elements,
            ef_construction=self.config.hnsw_ef_construction,
            M=self.config.hnsw_M,
        )
        self._index.set_ef(self.config.hnsw_ef_search)
        self._initialized = True
        logger.info(f"FlowIndex initialized: max_elements={self.config.hnsw_max_elements}")

    def build(self, batch_size: int = 100, show_progress: bool = True) -> int:
        """
        Build index from all encoded sessions in store.

        Args:
            batch_size: Batch size for processing
            show_progress: Log progress

        Returns:
            Number of sessions indexed
        """
        if not self._initialized:
            self._init_index()

        # Clear existing index data
        self._session_ids.clear()
        self._id_to_label.clear()

        # Re-initialize HNSW if it exists
        if self._index is not None:
            self._index = hnswlib.Index(space="cosine", dim=self.dimension)
            self._index.init_index(
                max_elements=self.config.hnsw_max_elements,
                ef_construction=self.config.hnsw_ef_construction,
                M=self.config.hnsw_M,
            )
            self._index.set_ef(self.config.hnsw_ef_search)

        indexed = 0
        vectors_batch = []
        ids_batch = []

        for session in self.store.iter_sessions(include_events=False):
            if session.vector is None:
                continue

            label = len(self._session_ids)
            self._session_ids.append(session.session_id)
            self._id_to_label[session.session_id] = label

            vectors_batch.append(session.vector)
            ids_batch.append(label)
            indexed += 1

            # Add batch to HNSW
            if len(vectors_batch) >= batch_size:
                self._add_batch(vectors_batch, ids_batch)
                vectors_batch = []
                ids_batch = []

                if show_progress and indexed % 1000 == 0:
                    logger.info(f"Indexed {indexed} sessions...")

        # Add remaining
        if vectors_batch:
            self._add_batch(vectors_batch, ids_batch)

        logger.info(f"FlowIndex built with {indexed} sessions")
        return indexed

    def _add_batch(self, vectors: list[np.ndarray], labels: list[int]) -> None:
        """Add batch of vectors to HNSW index."""
        if self._index is None:
            return  # Brute-force mode, vectors stored via session_ids

        data = np.stack(vectors)
        ids = np.array(labels)
        self._index.add_items(data, ids)

    def add_session(self, session: Session) -> bool:
        """
        Add a single session to the index.

        Args:
            session: Session with vector embedding

        Returns:
            True if added successfully
        """
        if session.vector is None:
            logger.warning(f"Cannot index session {session.session_id}: no vector")
            return False

        if not self._initialized:
            self._init_index()

        # Check if already indexed
        if session.session_id in self._id_to_label:
            # Update: remove old and re-add
            # Note: HNSW doesn't support true updates, but we can add new label
            old_label = self._id_to_label[session.session_id]
            # Mark old as invalid (we'll filter in search)
            pass

        label = len(self._session_ids)
        self._session_ids.append(session.session_id)
        self._id_to_label[session.session_id] = label

        if self._index is not None:
            self._index.add_items(
                session.vector.reshape(1, -1),
                np.array([label]),
            )

        return True

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        min_similarity: float = 0.0,
        include_session: bool = False,
    ) -> list[SearchResult]:
        """
        Search for similar sessions.

        Args:
            query_vector: Query embedding
            k: Number of results
            min_similarity: Minimum similarity threshold
            include_session: Load full session data

        Returns:
            List of SearchResult sorted by similarity (descending)
        """
        if not self._initialized or self.size == 0:
            return []

        if self._index is not None:
            return self._search_hnsw(query_vector, k, min_similarity, include_session)
        else:
            return self._search_brute_force(query_vector, k, min_similarity, include_session)

    def _search_hnsw(
        self,
        query_vector: np.ndarray,
        k: int,
        min_similarity: float,
        include_session: bool,
    ) -> list[SearchResult]:
        """Search using HNSW index."""
        # Over-fetch to account for filtering
        fetch_k = min(k * 2, self.size)

        labels, distances = self._index.knn_query(
            query_vector.reshape(1, -1),
            k=fetch_k,
        )

        results = []
        seen_ids = set()

        for label, distance in zip(labels[0], distances[0]):
            if label >= len(self._session_ids):
                continue

            session_id = self._session_ids[label]

            # Skip duplicates (from updates)
            if session_id in seen_ids:
                continue
            seen_ids.add(session_id)

            # Convert cosine distance to similarity
            similarity = 1.0 - distance

            if similarity < min_similarity:
                continue

            result = SearchResult(
                session_id=session_id,
                similarity=float(similarity),
                distance=float(distance),
            )

            if include_session:
                result.session = self.store.get_session(session_id)

            results.append(result)

            if len(results) >= k:
                break

        return results

    def _search_brute_force(
        self,
        query_vector: np.ndarray,
        k: int,
        min_similarity: float,
        include_session: bool,
    ) -> list[SearchResult]:
        """Brute-force search when HNSW unavailable."""
        results = []
        eps = np.finfo(np.float32).eps

        for session in self.store.iter_sessions(include_events=False):
            if session.vector is None:
                continue

            # Cosine similarity
            query_norm = np.linalg.norm(query_vector)
            vec_norm = np.linalg.norm(session.vector)
            similarity = float(np.dot(query_vector, session.vector) / ((query_norm * vec_norm) + eps))

            if similarity < min_similarity:
                continue

            result = SearchResult(
                session_id=session.session_id,
                similarity=similarity,
                distance=1.0 - similarity,
            )

            if include_session:
                result.session = session

            results.append(result)

        # Sort by similarity descending
        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:k]

    def search_by_query(
        self,
        query: str,
        k: int = 10,
        min_similarity: float = 0.0,
        include_session: bool = False,
    ) -> list[SearchResult]:
        """
        Search using natural language query.

        Args:
            query: Natural language description
            k: Number of results
            min_similarity: Minimum similarity threshold
            include_session: Load full session data

        Returns:
            List of SearchResult
        """
        query_vector = self.encoder.encode_query(query)
        return self.search(query_vector, k, min_similarity, include_session)

    def search_similar(
        self,
        session_id: str,
        k: int = 10,
        min_similarity: float = 0.0,
        include_session: bool = False,
    ) -> list[SearchResult]:
        """
        Find sessions similar to a given session.

        Args:
            session_id: Session to find similar sessions for
            k: Number of results
            min_similarity: Minimum similarity threshold
            include_session: Load full session data

        Returns:
            List of SearchResult (excludes the query session)
        """
        session = self.store.get_session(session_id, include_events=False)
        if session is None or session.vector is None:
            return []

        results = self.search(
            session.vector,
            k=k + 1,  # +1 to account for self-match
            min_similarity=min_similarity,
            include_session=include_session,
        )

        # Remove self from results
        return [r for r in results if r.session_id != session_id][:k]

    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            "initialized": self._initialized,
            "size": self.size,
            "hnsw_available": HAS_HNSWLIB,
            "dimension": self.dimension,
        }


def encode_and_index_sessions(
    store: FlowStore,
    encoder: FlowEncoder,
    config: Optional[BehaviorModelerConfig] = None,
    batch_size: int = 50,
) -> tuple[int, int]:
    """
    Convenience function to encode all sessions and build index.

    Args:
        store: FlowStore containing sessions
        encoder: FlowEncoder for embeddings
        config: Configuration
        batch_size: Batch size for encoding

    Returns:
        Tuple of (encoded_count, indexed_count)
    """
    config = config or BehaviorModelerConfig()

    encoded = 0
    for session in store.iter_sessions(include_events=True):
        if session.vector is not None:
            continue  # Already encoded

        vector = encoder.encode_session(session)
        if store.update_session_vector(session.session_id, vector):
            encoded += 1

    logger.info(f"Encoded {encoded} new sessions")

    # Build index
    index = FlowIndex(store, encoder, config)
    indexed = index.build()

    return encoded, indexed
