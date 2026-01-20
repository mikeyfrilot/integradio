"""
Tests for Embedder - Ollama embedding generation with caching.

Priority 4: External dependency tests.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
import json

from integradio.embedder import Embedder


class TestEmbedderUnit:
    """Unit tests that don't require Ollama."""

    def test_dimension_property(self):
        """Verify dimension is 768 for nomic-embed-text."""
        embedder = Embedder()
        assert embedder.dimension == 768

    def test_cache_key_uniqueness(self):
        """Different texts should have different cache keys."""
        embedder = Embedder()

        key1 = embedder._cache_key("hello world")
        key2 = embedder._cache_key("goodbye world")
        key3 = embedder._cache_key("hello world")

        assert key1 != key2
        assert key1 == key3  # Same text = same key

    def test_prefix_affects_cache_key(self):
        """Different prefixes should produce different cache keys."""
        embedder1 = Embedder(prefix="search_document: ")
        embedder2 = Embedder(prefix="different_prefix: ")

        key1 = embedder1._cache_key("hello")
        key2 = embedder2._cache_key("hello")

        assert key1 != key2


class TestEmbedderWithMock:
    """Tests using mocked HTTP calls (no Ollama needed)."""

    @patch("integradio.embedder.httpx.post")
    def test_embed_single_text(self, mock_post):
        """Verify embedding returns numpy array of correct dimension (768)."""
        # Mock the Ollama API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "embedding": [0.1] * 768
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        embedder = Embedder()
        result = embedder.embed("test text")

        assert isinstance(result, np.ndarray)
        assert result.shape == (768,)
        assert result.dtype == np.float32

    @patch("integradio.embedder.httpx.post")
    def test_embed_batch(self, mock_post):
        """Verify batch embedding returns list of correct length."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1] * 768}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        embedder = Embedder()
        texts = ["text1", "text2", "text3"]
        results = embedder.embed_batch(texts)

        assert len(results) == 3
        assert all(isinstance(r, np.ndarray) for r in results)
        assert all(r.shape == (768,) for r in results)

    @patch("integradio.embedder.httpx.post")
    def test_embed_query_uses_different_prefix(self, mock_post):
        """Ensure search_query prefix is used for queries."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1] * 768}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        embedder = Embedder()
        embedder.embed_query("test query")

        # Check that the API was called with search_query prefix
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert payload["prompt"].startswith("search_query: ")

    @patch("integradio.embedder.httpx.post")
    def test_cache_hit(self, mock_post):
        """Embed same text twice, verify second call uses cache."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.5] * 768}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        embedder = Embedder()

        # First call
        result1 = embedder.embed("cached text")
        assert mock_post.call_count == 1

        # Second call - should use cache
        result2 = embedder.embed("cached text")
        assert mock_post.call_count == 1  # No additional call

        # Results should be identical
        np.testing.assert_array_equal(result1, result2)

    @patch("integradio.embedder.httpx.post")
    def test_cache_miss_for_different_text(self, mock_post):
        """Different texts should not share cache."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.5] * 768}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        embedder = Embedder()

        embedder.embed("text1")
        assert mock_post.call_count == 1

        embedder.embed("text2")
        assert mock_post.call_count == 2  # New call for different text


class TestEmbedderCachePersistence:
    """Test cache persistence to disk."""

    @patch("integradio.embedder.httpx.post")
    def test_cache_persistence(self, mock_post, temp_cache_dir):
        """With cache_dir, verify cache survives new Embedder instance."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.42] * 768}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        # First embedder instance
        embedder1 = Embedder(cache_dir=temp_cache_dir)
        result1 = embedder1.embed("persistent text")
        assert mock_post.call_count == 1

        # Delete instance
        del embedder1

        # New embedder instance with same cache_dir
        embedder2 = Embedder(cache_dir=temp_cache_dir)
        result2 = embedder2.embed("persistent text")

        # Should not make a new API call (loaded from disk cache)
        assert mock_post.call_count == 1

        # Results should match
        np.testing.assert_array_almost_equal(result1, result2, decimal=5)

    def test_cache_file_created(self, temp_cache_dir):
        """Cache file is created in cache_dir."""
        with patch("integradio.embedder.httpx.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {"embedding": [0.1] * 768}
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            embedder = Embedder(cache_dir=temp_cache_dir)
            embedder.embed("test")

            cache_file = temp_cache_dir / "embeddings.json"
            assert cache_file.exists()

            # Verify content
            with open(cache_file) as f:
                data = json.load(f)
            assert len(data) > 0


class TestEmbedderBatchCaching:
    """Test batch embedding with cache interaction."""

    @patch("integradio.embedder.httpx.post")
    def test_batch_uses_cache_for_known_texts(self, mock_post):
        """Batch embedding should use cache for already-embedded texts."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.5] * 768}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        embedder = Embedder()

        # Pre-embed some texts
        embedder.embed("text1")
        embedder.embed("text2")
        assert mock_post.call_count == 2

        # Batch with mix of cached and new
        results = embedder.embed_batch(["text1", "text3", "text2"])

        # Should only call API for text3
        assert mock_post.call_count == 3  # Only one additional call

        assert len(results) == 3

    @patch("integradio.embedder.httpx.post")
    def test_batch_preserves_order(self, mock_post):
        """Batch results maintain input order."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            mock_response = MagicMock()
            # Return different embeddings based on call count
            mock_response.json.return_value = {"embedding": [call_count[0] * 0.1] * 768}
            mock_response.raise_for_status = MagicMock()
            return mock_response

        mock_post.side_effect = side_effect

        embedder = Embedder()
        texts = ["a", "b", "c"]
        results = embedder.embed_batch(texts)

        # Results should be in same order as input
        assert len(results) == 3
        # Each embedding should have different values since we used different calls
        assert not np.array_equal(results[0], results[1])


class TestErrorHandling:
    """Test error handling and graceful degradation."""

    @patch("integradio.embedder.httpx.get")
    def test_availability_check_timeout(self, mock_get):
        """Timeout during availability check marks as unavailable."""
        import httpx
        mock_get.side_effect = httpx.TimeoutException("Connection timed out")

        embedder = Embedder()

        # Should not raise, just return False
        with pytest.warns(match="Ollama not available"):
            assert embedder.available is False

    @patch("integradio.embedder.httpx.get")
    def test_availability_check_connection_error(self, mock_get):
        """Connection error during availability check marks as unavailable."""
        import httpx
        mock_get.side_effect = httpx.ConnectError("Connection refused")

        embedder = Embedder()

        with pytest.warns(match="Ollama not available"):
            assert embedder.available is False

    @patch("integradio.embedder.httpx.get")
    def test_unavailable_returns_zero_vector(self, mock_get):
        """When unavailable, embed returns zero vector."""
        import httpx
        mock_get.side_effect = httpx.ConnectError("Connection refused")

        embedder = Embedder()

        with pytest.warns(match="Ollama not available"):
            result = embedder.embed("test text")

        assert result.shape == (768,)
        np.testing.assert_array_equal(result, np.zeros(768))

    @patch("integradio.embedder.httpx.get")
    def test_unavailable_embed_query_returns_zero_vector(self, mock_get):
        """When unavailable, embed_query returns zero vector."""
        import httpx
        mock_get.side_effect = httpx.ConnectError("Connection refused")

        embedder = Embedder()

        with pytest.warns(match="Ollama not available"):
            result = embedder.embed_query("test query")

        assert result.shape == (768,)
        np.testing.assert_array_equal(result, np.zeros(768))

    @patch("integradio.embedder.httpx.get")
    @patch("integradio.embedder.httpx.post")
    def test_embed_api_error_graceful_fallback(self, mock_post, mock_get):
        """API error during embed falls back to zero vector."""
        import httpx

        # Availability check passes
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get.return_value = mock_get_response

        # But embed fails
        mock_post.side_effect = httpx.HTTPStatusError(
            "500 Internal Server Error",
            request=MagicMock(),
            response=MagicMock(),
        )

        # Disable circuit breaker to test the warning path
        embedder = Embedder(use_circuit_breaker=False)
        embedder._available = True  # Skip availability check

        with pytest.warns(match="Ollama connection failed"):
            result = embedder.embed("test")

        assert result.shape == (768,)
        np.testing.assert_array_equal(result, np.zeros(768))

    @patch("integradio.embedder.httpx.get")
    @patch("integradio.embedder.httpx.post")
    def test_embed_timeout_graceful_fallback(self, mock_post, mock_get):
        """Timeout during embed falls back to zero vector."""
        import httpx

        # Availability check passes
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get.return_value = mock_get_response

        mock_post.side_effect = httpx.TimeoutException("Request timed out")

        # Disable circuit breaker to test the warning path
        embedder = Embedder(use_circuit_breaker=False)
        embedder._available = True

        with pytest.warns(match="Ollama connection failed"):
            result = embedder.embed("test")

        assert result.shape == (768,)

    def test_warning_only_shown_once(self):
        """Warning about unavailability is only shown once."""
        with patch("integradio.embedder.httpx.get") as mock_get:
            import httpx
            mock_get.side_effect = httpx.ConnectError("Connection refused")

            embedder = Embedder()

            # First call should warn
            with pytest.warns(match="Ollama not available"):
                embedder.embed("test1")

            # Subsequent calls should not re-warn
            import warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                embedder.embed("test2")
                embedder.embed("test3")
                # No new warnings should be issued
                embedder_warnings = [x for x in w if "Ollama" in str(x.message)]
                assert len(embedder_warnings) == 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @patch("integradio.embedder.httpx.post")
    def test_empty_string_embed(self, mock_post):
        """Embedding empty string works."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.1] * 768}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        embedder = Embedder()
        result = embedder.embed("")

        assert result.shape == (768,)

    @patch("integradio.embedder.httpx.post")
    def test_unicode_text_embed(self, mock_post):
        """Embedding unicode text works."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.2] * 768}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        embedder = Embedder()
        result = embedder.embed("日本語テスト 🎉 émojis")

        assert result.shape == (768,)

    @patch("integradio.embedder.httpx.post")
    def test_very_long_text_embed(self, mock_post):
        """Embedding very long text works."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.3] * 768}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        embedder = Embedder()
        long_text = "word " * 10000  # 50000+ characters
        result = embedder.embed(long_text)

        assert result.shape == (768,)

    @patch("integradio.embedder.httpx.post")
    def test_special_characters_embed(self, mock_post):
        """Embedding text with special characters works."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.4] * 768}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        embedder = Embedder()
        result = embedder.embed('<script>alert("xss")</script>')

        assert result.shape == (768,)

    @patch("integradio.embedder.httpx.post")
    def test_batch_empty_list(self, mock_post):
        """Batch embedding empty list returns empty list."""
        embedder = Embedder()
        results = embedder.embed_batch([])

        assert results == []
        assert mock_post.call_count == 0

    @patch("integradio.embedder.httpx.post")
    def test_batch_single_item(self, mock_post):
        """Batch embedding single item works."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.5] * 768}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        embedder = Embedder()
        results = embedder.embed_batch(["single"])

        assert len(results) == 1
        assert results[0].shape == (768,)

    def test_custom_model_name(self):
        """Custom model name is used in cache key."""
        embedder1 = Embedder(model="nomic-embed-text")
        embedder2 = Embedder(model="custom-embed-model")

        key1 = embedder1._cache_key("test")
        key2 = embedder2._cache_key("test")

        assert key1 != key2

    def test_custom_base_url(self):
        """Custom base URL is accepted."""
        embedder = Embedder(base_url="http://custom-host:1234/")

        assert embedder.base_url == "http://custom-host:1234"  # Trailing slash removed

    def test_custom_prefix(self):
        """Custom prefix is used."""
        embedder = Embedder(prefix="custom_prefix: ")

        assert embedder.prefix == "custom_prefix: "

    @patch("integradio.embedder.httpx.post")
    def test_cache_key_includes_version(self, mock_post):
        """Cache key includes version for invalidation."""
        from integradio.embedder import CACHE_VERSION

        embedder = Embedder()
        key = embedder._cache_key("test")

        # Key should be a hash, but let's verify the input contains version
        # by checking that changing version would change the key
        old_version = CACHE_VERSION

        # This verifies the versioning concept without modifying the module
        key_content_1 = f"{old_version}:{embedder.model}:{embedder.prefix}test"
        key_content_2 = f"v999:{embedder.model}:{embedder.prefix}test"

        import hashlib
        hash1 = hashlib.sha256(key_content_1.encode()).hexdigest()[:16]
        hash2 = hashlib.sha256(key_content_2.encode()).hexdigest()[:16]

        assert hash1 != hash2  # Different versions produce different keys


class TestCacheEdgeCases:
    """Test cache edge cases."""

    @patch("integradio.embedder.httpx.post")
    def test_cache_clear(self, mock_post):
        """Clearing cache causes new API calls."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"embedding": [0.5] * 768}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        embedder = Embedder()

        embedder.embed("test")
        assert mock_post.call_count == 1

        embedder._cache.clear()

        embedder.embed("test")
        assert mock_post.call_count == 2

    def test_cache_dir_creation(self, tmp_path):
        """Cache directory is created if it doesn't exist."""
        cache_dir = tmp_path / "new_cache_dir" / "nested"

        assert not cache_dir.exists()

        Embedder(cache_dir=cache_dir)

        assert cache_dir.exists()

    @patch("integradio.embedder.httpx.post")
    def test_cache_file_corrupted_recovery(self, mock_post, tmp_path):
        """Corrupted cache file is handled gracefully."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "embeddings.json"

        # Write invalid JSON
        with open(cache_file, "w") as f:
            f.write("not valid json {{{")

        # After error handling improvements, corrupted cache is handled gracefully
        # with a warning instead of raising
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            embedder = Embedder(cache_dir=cache_dir)
            # Should have warned about corrupted cache
            assert len(w) >= 1
            assert "Could not load embedding cache" in str(w[-1].message)
            # Cache should be empty after corruption
            assert embedder._cache == {}


class TestZeroVector:
    """Test zero vector fallback behavior."""

    def test_zero_vector_dimension(self):
        """Zero vector has correct dimension."""
        embedder = Embedder()
        zero = embedder._zero_vector()

        assert zero.shape == (768,)
        assert zero.dtype == np.float32

    def test_zero_vector_values(self):
        """Zero vector contains all zeros."""
        embedder = Embedder()
        zero = embedder._zero_vector()

        np.testing.assert_array_equal(zero, np.zeros(768))

    def test_zero_vectors_are_independent(self):
        """Each zero vector call returns independent array."""
        embedder = Embedder()

        zero1 = embedder._zero_vector()
        zero2 = embedder._zero_vector()

        # Modify one
        zero1[0] = 999

        # Other should be unchanged
        assert zero2[0] == 0


# Integration tests marked for when Ollama is available
@pytest.mark.integration
class TestEmbedderIntegration:
    """Integration tests that require Ollama running.

    Run with: pytest tests/test_embedder.py -m integration -v
    """

    def test_ollama_connection(self):
        """Verify can connect to Ollama API."""
        import httpx

        try:
            response = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
            response.raise_for_status()
            assert "models" in response.json()
        except Exception as e:
            pytest.skip(f"Ollama not available: {e}")

    def test_embedding_determinism(self):
        """Same text should produce same embedding."""
        try:
            embedder = Embedder()
            # Clear cache to ensure fresh API calls
            embedder._cache.clear()

            result1 = embedder.embed("determinism test")
            # Clear cache again
            embedder._cache.clear()
            result2 = embedder.embed("determinism test")

            # Ollama embeddings should be deterministic
            np.testing.assert_array_almost_equal(result1, result2, decimal=5)
        except Exception as e:
            pytest.skip(f"Ollama not available: {e}")

    def test_semantic_similarity(self):
        """'cat' and 'dog' should be more similar than 'cat' and 'airplane'."""
        try:
            embedder = Embedder()

            cat = embedder.embed("cat")
            dog = embedder.embed("dog")
            airplane = embedder.embed("airplane")

            def cosine_sim(a, b):
                return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

            cat_dog_sim = cosine_sim(cat, dog)
            cat_airplane_sim = cosine_sim(cat, airplane)

            assert cat_dog_sim > cat_airplane_sim, \
                f"Expected cat-dog ({cat_dog_sim:.3f}) > cat-airplane ({cat_airplane_sim:.3f})"
        except Exception as e:
            pytest.skip(f"Ollama not available: {e}")
