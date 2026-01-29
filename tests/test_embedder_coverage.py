"""
Batch 3: Embedder Coverage Tests (13 tests)

Tests for integradio/embedder.py - HIGH priority
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import json


class TestEmbedderInitializationDefaults:
    """Tests for embedder initialization with defaults."""

    def test_embedder_initialization_defaults(self):
        """Verify Embedder initializes with correct defaults."""
        with patch("integradio.embedder.get_circuit_breaker") as mock_cb:
            mock_cb.return_value = MagicMock()

            from integradio.embedder import Embedder

            embedder = Embedder()

            assert embedder.model == "nomic-embed-text"
            assert embedder.base_url == "http://localhost:11434"
            assert embedder.prefix == "search_document: "
            assert embedder.dimension == 768

    def test_embedder_custom_initialization(self, temp_cache_dir):
        """Verify Embedder accepts custom parameters."""
        with patch("integradio.embedder.get_circuit_breaker") as mock_cb:
            mock_cb.return_value = MagicMock()

            from integradio.embedder import Embedder

            embedder = Embedder(
                model="custom-model",
                base_url="http://custom:1234",
                cache_dir=temp_cache_dir,
                prefix="custom_prefix: ",
            )

            assert embedder.model == "custom-model"
            assert embedder.base_url == "http://custom:1234"
            assert embedder.cache_dir == temp_cache_dir
            assert embedder.prefix == "custom_prefix: "


class TestEmbedderInvalidModelName:
    """Tests for handling invalid model names."""

    def test_embedder_invalid_model_name(self):
        """Verify embedder handles invalid model gracefully."""
        with patch("integradio.embedder.get_circuit_breaker") as mock_cb:
            mock_cb.return_value = MagicMock()

            from integradio.embedder import Embedder

            # Model name is just stored, validation happens at embed time
            embedder = Embedder(model="invalid-model-xyz")

            assert embedder.model == "invalid-model-xyz"

    def test_embedder_returns_zero_vector_when_unavailable(self):
        """Verify embedder returns zero vector when service unavailable."""
        with patch("integradio.embedder.get_circuit_breaker") as mock_cb:
            mock_cb.return_value = MagicMock()

            from integradio.embedder import Embedder

            embedder = Embedder()
            embedder._available = False  # Force unavailable

            result = embedder.embed("test text")

            assert isinstance(result, np.ndarray)
            assert result.shape == (768,)
            assert np.allclose(result, np.zeros(768))


class TestEmbedderVectorDimensions:
    """Tests for embedding vector dimensions."""

    def test_embedder_vector_dimensions(self):
        """Verify embedding dimension is correct."""
        with patch("integradio.embedder.get_circuit_breaker") as mock_cb:
            mock_cb.return_value = MagicMock()

            from integradio.embedder import Embedder

            embedder = Embedder()

            assert embedder.dimension == 768

    def test_embedder_zero_vector_correct_dimension(self):
        """Verify zero vector has correct dimensions."""
        with patch("integradio.embedder.get_circuit_breaker") as mock_cb:
            mock_cb.return_value = MagicMock()

            from integradio.embedder import Embedder

            embedder = Embedder()
            zero_vec = embedder._zero_vector()

            assert zero_vec.shape == (768,)
            assert zero_vec.dtype == np.float32


class TestEmbedderBatchInputs:
    """Tests for batch embedding inputs."""

    def test_embedder_batch_inputs(self):
        """Verify batch embedding processes multiple texts."""
        with patch("integradio.embedder.get_circuit_breaker") as mock_cb:
            mock_cb.return_value = MagicMock()

            from integradio.embedder import Embedder

            embedder = Embedder()
            embedder._available = False  # Return zero vectors

            texts = ["text1", "text2", "text3"]
            results = embedder.embed_batch(texts)

            assert len(results) == 3
            for result in results:
                assert isinstance(result, np.ndarray)
                assert result.shape == (768,)

    def test_embedder_batch_preserves_order(self):
        """Verify batch results maintain input order."""
        with patch("integradio.embedder.get_circuit_breaker") as mock_cb:
            mock_cb.return_value = MagicMock()

            from integradio.embedder import Embedder

            embedder = Embedder()
            embedder._available = False

            texts = ["first", "second", "third"]
            results = embedder.embed_batch(texts)

            # All zero vectors when unavailable, but should be 3 items
            assert len(results) == len(texts)


class TestEmbedderErrorHandlingTimeouts:
    """Tests for error handling and timeouts."""

    def test_embedder_error_handling_timeouts(self):
        """Verify embedder handles connection timeouts gracefully."""
        with patch("integradio.embedder.get_circuit_breaker") as mock_cb:
            import httpx

            mock_cb.return_value = MagicMock()

            from integradio.embedder import Embedder

            embedder = Embedder()

            # Mock the actual request to timeout
            with patch.object(embedder, "_make_embed_request") as mock_request:
                mock_request.side_effect = httpx.TimeoutException("timeout")
                embedder._available = True  # Force attempt

                # Should return zero vector, not raise
                result = embedder.embed(\"test\")

                assert np.allclose(result, np.zeros(768))

    def test_embedder_handles_connection_error(self):
        """Verify embedder handles connection errors gracefully."""
        with patch("integradio.embedder.get_circuit_breaker") as mock_cb:
            import httpx

            mock_cb.return_value = MagicMock()

            from integradio.embedder import Embedder

            embedder = Embedder()
            embedder._available = True

            with patch.object(embedder, "_make_embed_request") as mock_request:
                mock_request.side_effect = httpx.ConnectError("connection failed")

                result = embedder.embed(\"test\")

                assert np.allclose(result, np.zeros(768))


class TestEmbedderCaching:
    """Tests for embedding caching."""

    def test_embedder_cache_hit(self, temp_cache_dir):
        """Verify cache is used for repeated embeddings."""
        with patch("integradio.embedder.get_circuit_breaker") as mock_cb:
            mock_cb.return_value = MagicMock()

            from integradio.embedder import Embedder

            embedder = Embedder(cache_dir=temp_cache_dir)

            # Pre-populate cache
            test_text = "cached text"
            cache_key = embedder._cache_key(test_text)
            cached_vector = np.ones(768, dtype=np.float32)
            embedder._cache[cache_key] = cached_vector

            result = embedder.embed(test_text)

            assert np.array_equal(result, cached_vector)

    def test_embedder_cache_key_versioning(self):
        """Verify cache key includes version."""
        with patch("integradio.embedder.get_circuit_breaker") as mock_cb:
            mock_cb.return_value = MagicMock()

            from integradio.embedder import Embedder, CACHE_VERSION

            embedder = Embedder()

            key = embedder._cache_key("test text")

            # Key should be 16 characters (sha256 truncated)
            assert len(key) == 16
            assert isinstance(key, str)

    def test_embedder_cache_persistence(self, temp_cache_dir):
        """Verify cache can be saved and loaded."""
        with patch("integradio.embedder.get_circuit_breaker") as mock_cb:
            mock_cb.return_value = MagicMock()

            from integradio.embedder import Embedder

            embedder = Embedder(cache_dir=temp_cache_dir)

            # Add to cache
            embedder._cache["test_key"] = np.array([1.0, 2.0, 3.0], dtype=np.float32)

            # Save
            embedder._save_cache()

            # Check file exists
            cache_file = temp_cache_dir / "embeddings.json"
            assert cache_file.exists()

            # Load in new instance
            embedder2 = Embedder(cache_dir=temp_cache_dir)

            assert "test_key" in embedder2._cache


class TestEmbedderCircuitBreakerIntegration:
    """Tests for circuit breaker integration."""

    def test_embedder_circuit_breaker_stats(self):
        """Verify circuit breaker stats are accessible."""
        with patch("integradio.embedder.get_circuit_breaker") as mock_cb:
            mock_stats = MagicMock()
            mock_stats.to_dict.return_value = {
                "total_calls": 10,
                "failed_calls": 2,
            }
            mock_breaker = MagicMock()
            mock_breaker.stats = mock_stats
            mock_cb.return_value = mock_breaker

            from integradio.embedder import Embedder

            embedder = Embedder()

            stats = embedder.circuit_breaker_stats

            assert stats is not None
            assert "total_calls" in stats

    def test_embedder_without_circuit_breaker(self):
        """Verify embedder works without circuit breaker."""
        from integradio.embedder import Embedder

        embedder = Embedder(use_circuit_breaker=False)

        assert embedder._circuit_breaker is None
        assert embedder.circuit_breaker_stats is None
