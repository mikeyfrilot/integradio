"""
Flow Encoder - Transforms event sequences into vector representations.

Uses Ollama's nomic-embed-text model (via Integradio's Embedder) to create
semantic vectors for individual events and entire sessions.
"""

import logging
from typing import Optional, TYPE_CHECKING
import numpy as np

from .config import BehaviorModelerConfig
from .models import Session, FlowEvent

if TYPE_CHECKING:
    from integradio.embedder import Embedder

logger = logging.getLogger(__name__)


class FlowEncoder:
    """
    Encode UI flows into vector representations.

    Supports multiple encoding strategies:
    - pooled_weighted: Weighted average of event embeddings (recency bias)
    - pooled_avg: Simple average of event embeddings
    """

    def __init__(
        self,
        embedder: Optional["Embedder"] = None,
        config: Optional[BehaviorModelerConfig] = None,
    ):
        """
        Initialize the flow encoder.

        Args:
            embedder: Integradio Embedder instance (creates new if None)
            config: Configuration options
        """
        self.config = config or BehaviorModelerConfig()
        self._embedder = embedder
        self._own_embedder = False

        # Lazy initialization of embedder
        if self._embedder is None:
            self._init_embedder()

    def _init_embedder(self) -> None:
        """Initialize embedder on first use."""
        try:
            from integradio.embedder import Embedder

            self._embedder = Embedder(
                model=self.config.embed_model,
                base_url=self.config.ollama_url,
            )
            self._own_embedder = True
            logger.info(f"FlowEncoder initialized with {self.config.embed_model}")
        except ImportError:
            logger.warning("integradio.embedder not available, using fallback")
            self._embedder = None

    @property
    def dimension(self) -> int:
        """Embedding dimension."""
        return self.config.embed_dimension

    @property
    def available(self) -> bool:
        """Check if embedder is available."""
        if self._embedder is None:
            return False
        return self._embedder.available

    def _zero_vector(self) -> np.ndarray:
        """Return zero vector when embedder unavailable."""
        return np.zeros(self.dimension, dtype=np.float32)

    def _event_to_text(self, event: FlowEvent) -> str:
        """
        Convert a flow event to text for embedding.

        Combines component info, event type, and semantic context.
        """
        parts = []

        # Component type and intent
        if event.component_type:
            parts.append(event.component_type)
        if event.component_intent:
            parts.append(event.component_intent)

        # Event type
        if event.event_type:
            parts.append(event.event_type)

        # Semantic intent
        if event.intent:
            parts.append(f"intent:{event.intent}")

        # Tags (if meaningful)
        meaningful_tags = [t for t in event.tags if not t.startswith("step_")]
        if meaningful_tags:
            parts.append(" ".join(meaningful_tags))

        return " ".join(parts) if parts else "unknown_event"

    def encode_event(self, event: FlowEvent) -> np.ndarray:
        """
        Encode a single event into a vector.

        Args:
            event: FlowEvent to encode

        Returns:
            768-dimensional embedding vector
        """
        if self._embedder is None or not self.available:
            return self._zero_vector()

        text = self._event_to_text(event)
        return self._embedder.embed(text)

    def encode_events_batch(self, events: list[FlowEvent]) -> list[np.ndarray]:
        """
        Encode multiple events in batch.

        Args:
            events: List of FlowEvents

        Returns:
            List of embedding vectors
        """
        if self._embedder is None or not self.available:
            return [self._zero_vector() for _ in events]

        texts = [self._event_to_text(e) for e in events]
        return self._embedder.embed_batch(texts)

    def encode_session(
        self,
        session: Session,
        method: Optional[str] = None,
    ) -> np.ndarray:
        """
        Encode an entire session into a single vector.

        Args:
            session: Session to encode
            method: Encoding method (pooled_weighted, pooled_avg)
                   Uses config default if None

        Returns:
            768-dimensional session embedding
        """
        method = method or self.config.encoding_method

        if not session.events:
            return self._zero_vector()

        # Limit to max sequence length
        events = session.events[: self.config.max_sequence_length]

        # Encode all events
        event_vectors = self.encode_events_batch(events)

        # Check for all-zero vectors (embedder unavailable)
        if all(np.allclose(v, 0) for v in event_vectors):
            return self._zero_vector()

        # Apply pooling strategy
        if method == "pooled_weighted":
            return self._pool_weighted(event_vectors)
        elif method == "pooled_avg":
            return self._pool_average(event_vectors)
        else:
            logger.warning(f"Unknown encoding method: {method}, using pooled_avg")
            return self._pool_average(event_vectors)

    def _pool_weighted(self, vectors: list[np.ndarray]) -> np.ndarray:
        """
        Pool vectors with recency weighting.

        Later events (more recent) get higher weight.
        """
        n = len(vectors)
        if n == 0:
            return self._zero_vector()

        # Linear weights from 0.5 to 1.0 (recent events weighted higher)
        weights = np.linspace(0.5, 1.0, n)
        weights /= weights.sum()

        # Weighted average
        stacked = np.stack(vectors)
        pooled = np.average(stacked, axis=0, weights=weights)

        # L2 normalize
        norm = np.linalg.norm(pooled)
        if norm > 1e-8:
            pooled = pooled / norm

        return pooled.astype(np.float32)

    def _pool_average(self, vectors: list[np.ndarray]) -> np.ndarray:
        """Simple average pooling."""
        if not vectors:
            return self._zero_vector()

        stacked = np.stack(vectors)
        pooled = np.mean(stacked, axis=0)

        # L2 normalize
        norm = np.linalg.norm(pooled)
        if norm > 1e-8:
            pooled = pooled / norm

        return pooled.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a natural language query for searching.

        Uses the query prefix for better search performance.

        Args:
            query: Natural language search query

        Returns:
            Query embedding vector
        """
        if self._embedder is None or not self.available:
            return self._zero_vector()

        return self._embedder.embed_query(query)

    def encode_partial_flow(self, events: list[FlowEvent]) -> np.ndarray:
        """
        Encode a partial flow (for prediction).

        Args:
            events: Partial sequence of events

        Returns:
            Embedding of the partial flow
        """
        # Create a temporary session
        from datetime import datetime, timezone

        temp_session = Session(
            session_id="partial",
            started_at=events[0].timestamp if events else datetime.now(timezone.utc),
            events=events,
        )
        return self.encode_session(temp_session)

    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))


class FallbackEncoder(FlowEncoder):
    """
    Fallback encoder when Ollama is unavailable.

    Uses simple hashing-based embeddings for testing/development.
    """

    def __init__(self, config: Optional[BehaviorModelerConfig] = None):
        self.config = config or BehaviorModelerConfig()
        self._embedder = None

    @property
    def available(self) -> bool:
        return True  # Always available

    def encode_event(self, event: FlowEvent) -> np.ndarray:
        """Create deterministic embedding from event hash."""
        text = self._event_to_text(event)
        return self._hash_to_vector(text)

    def encode_events_batch(self, events: list[FlowEvent]) -> list[np.ndarray]:
        return [self.encode_event(e) for e in events]

    def encode_query(self, query: str) -> np.ndarray:
        return self._hash_to_vector(query)

    def _hash_to_vector(self, text: str) -> np.ndarray:
        """Convert text to a deterministic pseudo-random vector."""
        import hashlib

        # Use SHA256 hash as seed
        hash_bytes = hashlib.sha256(text.encode()).digest()
        seed = int.from_bytes(hash_bytes[:4], "big")

        # Generate deterministic vector
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dimension).astype(np.float32)

        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm

        return vec
