"""
Embedder module - Generate semantic vectors via Ollama/nomic-embed-text.

Handles embedding generation with caching and batching for efficiency.
Gracefully degrades when Ollama is unavailable.
"""

import hashlib
import json
import logging
import httpx
import warnings
from pathlib import Path
from typing import Optional
import numpy as np

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    get_circuit_breaker,
)
from .exceptions import (
    EmbedderUnavailableError,
    EmbedderTimeoutError,
    EmbedderResponseError,
    CircuitOpenError,
)

logger = logging.getLogger(__name__)

# Cache version - increment when changing model, prefix format, or embedding dimension
# This ensures stale embeddings are not served after changes (2026 best practice)
# See: https://sparkco.ai/blog/mastering-embedding-versioning-best-practices-future-trends
CACHE_VERSION = "v1"


class EmbedderUnavailableWarning(UserWarning):
    """Warning issued when embedder cannot connect to Ollama."""
    pass


class Embedder:
    """Generate embeddings using Ollama's nomic-embed-text model."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        cache_dir: Optional[Path] = None,
        prefix: str = "search_document: ",
        use_circuit_breaker: bool = True,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    ):
        """
        Initialize the embedder.

        Args:
            model: Ollama model name for embeddings
            base_url: Ollama API base URL
            cache_dir: Directory for embedding cache (None = no caching)
            prefix: Task prefix for nomic-embed-text
            use_circuit_breaker: Enable circuit breaker for Ollama calls
            circuit_breaker_config: Custom circuit breaker configuration
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.prefix = prefix
        self.cache_dir = cache_dir
        self._cache: dict[str, np.ndarray] = {}
        self._available: Optional[bool] = None  # Lazy check
        self._warned: bool = False

        # Circuit breaker setup
        self._use_circuit_breaker = use_circuit_breaker
        if use_circuit_breaker:
            cb_config = circuit_breaker_config or CircuitBreakerConfig(
                failure_threshold=3,
                success_threshold=1,
                timeout_seconds=30.0,
                exception_types=(httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError),
            )
            self._circuit_breaker: Optional[CircuitBreaker] = get_circuit_breaker(
                f"ollama-{base_url}",
                config=cb_config,
                fallback=self._zero_vector,
            )
        else:
            self._circuit_breaker = None

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_cache()

        logger.debug(f"Embedder initialized: model={model}, base_url={base_url}, circuit_breaker={use_circuit_breaker}")

    def _check_availability(self) -> bool:
        """Check if Ollama is available (cached result)."""
        if self._available is not None:
            return self._available

        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=2.0)
            self._available = response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            self._available = False

        if not self._available and not self._warned:
            warnings.warn(
                f"Ollama not available at {self.base_url}. "
                "Embeddings will use zero vectors. Start Ollama with: ollama serve",
                EmbedderUnavailableWarning,
                stacklevel=3,
            )
            self._warned = True

        return self._available

    @property
    def available(self) -> bool:
        """Check if the embedder service is available."""
        return self._check_availability()

    def _cache_key(self, text: str) -> str:
        """Generate versioned cache key from text content.

        Includes model name and cache version to ensure stale embeddings
        are invalidated when model or configuration changes.
        """
        # Include model and version in cache key (2026 best practice)
        key_content = f"{CACHE_VERSION}:{self.model}:{self.prefix}{text}"
        return hashlib.sha256(key_content.encode()).hexdigest()[:16]

    def _load_cache(self) -> None:
        """Load cached embeddings from disk."""
        if not self.cache_dir:
            return
        cache_file = self.cache_dir / "embeddings.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    self._cache = {k: np.array(v) for k, v in data.items()}
            except (json.JSONDecodeError, OSError, ValueError) as e:
                # Cache file corrupted or unreadable - start fresh
                warnings.warn(
                    f"Could not load embedding cache: {e}. Starting with empty cache.",
                    UserWarning,
                    stacklevel=2,
                )
                self._cache = {}

    def _save_cache(self) -> None:
        """Save embedding cache to disk."""
        if not self.cache_dir:
            return
        cache_file = self.cache_dir / "embeddings.json"
        try:
            data = {k: v.tolist() for k, v in self._cache.items()}
            with open(cache_file, "w") as f:
                json.dump(data, f)
        except (OSError, TypeError) as e:
            # Disk full, permissions, or serialization error - skip save
            warnings.warn(
                f"Could not save embedding cache: {e}. Cache not persisted.",
                UserWarning,
                stacklevel=2,
            )

    def _zero_vector(self) -> np.ndarray:
        """Return a zero vector when Ollama is unavailable."""
        return np.zeros(self.dimension, dtype=np.float32)

    def _make_embed_request(self, prompt: str) -> np.ndarray:
        """Make the actual embedding API request."""
        response = httpx.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": prompt},
            timeout=30.0,
        )
        response.raise_for_status()
        response_data = response.json()
        if "embedding" not in response_data:
            raise KeyError("API response missing 'embedding' field")
        return np.array(response_data["embedding"], dtype=np.float32)

    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array (zeros if Ollama unavailable)
        """
        key = self._cache_key(text)
        if key in self._cache:
            logger.debug(f"Cache hit for embedding: key={key[:8]}...")
            return self._cache[key]

        # Check availability first
        if not self.available:
            logger.debug("Ollama not available, returning zero vector")
            return self._zero_vector()

        # Use circuit breaker if enabled
        if self._circuit_breaker:
            try:
                embedding = self._circuit_breaker.call(
                    lambda: self._make_embed_request(f"{self.prefix}{text}")
                )
                # Cache result on success
                self._cache[key] = embedding
                if self.cache_dir:
                    self._save_cache()
                logger.debug(f"Embedding generated via circuit breaker: key={key[:8]}...")
                return embedding
            except CircuitOpenError as e:
                logger.warning(f"Circuit breaker open: {e}")
                return self._zero_vector()

        # Fallback to direct call without circuit breaker
        try:
            embedding = self._make_embed_request(f"{self.prefix}{text}")

            # Cache result
            self._cache[key] = embedding
            if self.cache_dir:
                self._save_cache()

            logger.debug(f"Embedding generated: key={key[:8]}...")
            return embedding
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError, KeyError) as e:
            # Mark as unavailable and return zero vector
            self._available = False
            if not self._warned:
                warnings.warn(
                    f"Ollama connection failed: {e}. Using zero vectors.",
                    EmbedderUnavailableWarning,
                    stacklevel=2,
                )
                self._warned = True
            logger.warning(f"Embedding failed, returning zero vector: {e}")
            return self._zero_vector()

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        results = []
        uncached_indices = []
        uncached_texts = []

        # Check cache first
        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                results.append((i, self._cache[key]))
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Embed uncached texts
        for idx, text in zip(uncached_indices, uncached_texts):
            embedding = self.embed(text)
            results.append((idx, embedding))

        # Sort by original index
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query (uses different prefix).

        Args:
            query: Search query text

        Returns:
            Query embedding vector (zeros if Ollama unavailable)
        """
        # Check availability first
        if not self.available:
            logger.debug("Ollama not available for query, returning zero vector")
            return self._zero_vector()

        # Use circuit breaker if enabled
        if self._circuit_breaker:
            try:
                embedding = self._circuit_breaker.call(
                    lambda: self._make_embed_request(f"search_query: {query}")
                )
                logger.debug(f"Query embedding generated via circuit breaker")
                return embedding
            except CircuitOpenError as e:
                logger.warning(f"Circuit breaker open for query: {e}")
                return self._zero_vector()

        # Fallback to direct call without circuit breaker
        try:
            embedding = self._make_embed_request(f"search_query: {query}")
            logger.debug("Query embedding generated")
            return embedding
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError, KeyError) as e:
            self._available = False
            if not self._warned:
                warnings.warn(
                    f"Ollama connection failed: {e}. Using zero vectors.",
                    EmbedderUnavailableWarning,
                    stacklevel=2,
                )
                self._warned = True
            logger.warning(f"Query embedding failed, returning zero vector: {e}")
            return self._zero_vector()

    @property
    def circuit_breaker_stats(self) -> dict | None:
        """Get circuit breaker statistics if enabled."""
        if self._circuit_breaker:
            return self._circuit_breaker.stats.to_dict()
        return None

    @property
    def dimension(self) -> int:
        """Get embedding dimension (768 for nomic-embed-text)."""
        return 768
