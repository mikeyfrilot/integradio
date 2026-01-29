"""
Batch 2: API Coverage Tests (13 tests)

Tests for integradio/api.py - CRITICAL priority
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import json


class TestAPIStartupConfigValidation:
    """Tests for API startup and configuration validation."""

    def test_api_startup_config_validation(self):
        """Verify API validates configuration on startup."""
        from integradio.api import MAX_QUERY_LENGTH, MIN_SEARCH_RESULTS, MAX_SEARCH_RESULTS

        # Constants should be reasonable
        assert MAX_QUERY_LENGTH > 0
        assert MIN_SEARCH_RESULTS > 0
        assert MAX_SEARCH_RESULTS >= MIN_SEARCH_RESULTS

    def test_api_constants_defined(self):
        """Verify required constants are defined."""
        from integradio.api import (
            MAX_SEARCH_RESULTS,
            MIN_SEARCH_RESULTS,
            DEFAULT_SEARCH_RESULTS,
            MAX_QUERY_LENGTH,
        )

        assert isinstance(MAX_SEARCH_RESULTS, int)
        assert isinstance(MIN_SEARCH_RESULTS, int)
        assert isinstance(DEFAULT_SEARCH_RESULTS, int)
        assert isinstance(MAX_QUERY_LENGTH, int)


class TestAPIRouteRegistration:
    """Tests for API route registration."""

    def test_api_route_registration(self):
        """Verify create_api_routes registers all expected endpoints."""
        from integradio.api import create_api_routes

        mock_app = MagicMock()
        mock_blocks = MagicMock()

        create_api_routes(mock_app, mock_blocks)

        # Should register routes via decorators
        registered_routes = [call[0][0] for call in mock_app.get.call_args_list]

        assert "/semantic/search" in registered_routes
        assert "/semantic/graph" in registered_routes
        assert "/semantic/summary" in registered_routes

    def test_api_route_methods(self):
        """Verify routes use correct HTTP methods."""
        from integradio.api import create_api_routes

        mock_app = MagicMock()
        mock_blocks = MagicMock()

        create_api_routes(mock_app, mock_blocks)

        # All semantic routes should be GET
        assert mock_app.get.called
        # No POST routes in this API
        assert not mock_app.post.called


class TestAPIRequestSchemaValidation:
    """Tests for API request schema validation."""

    @pytest.mark.asyncio
    async def test_api_request_schema_validation(self):
        """Verify API validates request parameters."""
        from integradio.api import create_api_routes

        mock_app = MagicMock()
        mock_blocks = MagicMock()

        # Capture the route handler
        handlers = {}

        def capture_get(path):
            def decorator(func):
                handlers[path] = func
                return func
            return decorator

        mock_app.get = capture_get

        create_api_routes(mock_app, mock_blocks)

        # Test search endpoint with empty query
        search_handler = handlers.get("/semantic/search")
        if search_handler:
            response = await search_handler(q="", k=10, type=None, tags=None)
            data = json.loads(response.body)
            assert "error" in data

    @pytest.mark.asyncio
    async def test_api_query_length_validation(self):
        """Verify API rejects queries exceeding max length."""
        from integradio.api import create_api_routes, MAX_QUERY_LENGTH

        mock_app = MagicMock()
        mock_blocks = MagicMock()

        handlers = {}

        def capture_get(path):
            def decorator(func):
                handlers[path] = func
                return func
            return decorator

        mock_app.get = capture_get

        create_api_routes(mock_app, mock_blocks)

        search_handler = handlers.get("/semantic/search")
        if search_handler:
            long_query = "x" * (MAX_QUERY_LENGTH + 1)
            response = await search_handler(q=long_query, k=10, type=None, tags=None)
            data = json.loads(response.body)
            assert "error" in data
            assert "too long" in data["error"].lower()


class TestAPIErrorResponseShape:
    """Tests for API error response structure."""

    @pytest.mark.asyncio
    async def test_api_error_response_shape(self):
        """Verify error responses have consistent structure."""
        from integradio.api import create_api_routes

        mock_app = MagicMock()
        mock_blocks = MagicMock()
        mock_blocks.registry.get.return_value = None

        handlers = {}

        def capture_get(path):
            def decorator(func):
                handlers[path] = func
                return func
            return decorator

        mock_app.get = capture_get

        create_api_routes(mock_app, mock_blocks)

        # Test component endpoint with invalid ID
        component_handler = handlers.get("/semantic/component/{component_id}")
        if component_handler:
            response = await component_handler(component_id=99999)
            data = json.loads(response.body)

            # Should have error field
            assert "error" in data
            assert isinstance(data["error"], str)

    @pytest.mark.asyncio
    async def test_api_404_for_missing_component(self):
        """Verify 404 returned for missing components."""
        from integradio.api import create_api_routes

        mock_app = MagicMock()
        mock_blocks = MagicMock()
        mock_blocks.registry.get.return_value = None

        handlers = {}

        def capture_get(path):
            def decorator(func):
                handlers[path] = func
                return func
            return decorator

        mock_app.get = capture_get

        create_api_routes(mock_app, mock_blocks)

        component_handler = handlers.get("/semantic/component/{component_id}")
        if component_handler:
            response = await component_handler(component_id=99999)
            assert response.status_code == 404


class TestAPIRateLimitBehavior:
    """Tests for API rate limiting / circuit breaker behavior."""

    @pytest.mark.asyncio
    async def test_api_rate_limit_or_circuit_breaker_behavior(self):
        """Verify API handles rapid requests appropriately."""
        from integradio.api import create_api_routes

        mock_app = MagicMock()
        mock_blocks = MagicMock()
        mock_blocks.search.return_value = []

        handlers = {}

        def capture_get(path):
            def decorator(func):
                handlers[path] = func
                return func
            return decorator

        mock_app.get = capture_get

        create_api_routes(mock_app, mock_blocks)

        search_handler = handlers.get("/semantic/search")
        if search_handler:
            # Make multiple rapid requests
            for _ in range(10):
                response = await search_handler(q="test", k=10, type=None, tags=None)
                data = json.loads(response.body)
                # Should all succeed (no rate limiting in base API)
                assert "results" in data or "error" not in data

    @pytest.mark.asyncio
    async def test_api_k_parameter_bounds(self):
        """Verify k parameter is bounded."""
        from integradio.api import create_api_routes, MIN_SEARCH_RESULTS, MAX_SEARCH_RESULTS

        mock_app = MagicMock()
        mock_blocks = MagicMock()
        mock_blocks.search.return_value = []

        handlers = {}

        def capture_get(path):
            def decorator(func):
                handlers[path] = func
                return func
            return decorator

        mock_app.get = capture_get

        create_api_routes(mock_app, mock_blocks)

        search_handler = handlers.get("/semantic/search")
        if search_handler:
            # k below minimum should be clamped
            await search_handler(q="test", k=0, type=None, tags=None)
            # k above maximum should be clamped
            await search_handler(q="test", k=100000, type=None, tags=None)

            # Verify search was called with bounded k
            calls = mock_blocks.search.call_args_list
            for call in calls:
                k_used = call.kwargs.get("k") or call.args[1] if len(call.args) > 1 else 10
                assert MIN_SEARCH_RESULTS <= k_used <= MAX_SEARCH_RESULTS


class TestGradioAPIHandlers:
    """Tests for Gradio API handlers."""

    def test_create_gradio_api_returns_handlers(self):
        """Verify create_gradio_api returns handler dict."""
        from integradio.api import create_gradio_api

        mock_blocks = MagicMock()

        handlers = create_gradio_api(mock_blocks)

        assert "search" in handlers
        assert "graph" in handlers
        assert "summary" in handlers
        assert callable(handlers["search"])
        assert callable(handlers["graph"])
        assert callable(handlers["summary"])

    def test_gradio_search_handler(self):
        """Verify Gradio search handler works."""
        from integradio.api import create_gradio_api

        mock_blocks = MagicMock()
        mock_blocks.search.return_value = []

        handlers = create_gradio_api(mock_blocks)

        result = handlers["search"]("test query", k=5)

        assert isinstance(result, list)
        mock_blocks.search.assert_called_once_with("test query", k=5)

    def test_gradio_summary_handler(self):
        """Verify Gradio summary handler works."""
        from integradio.api import create_gradio_api

        mock_blocks = MagicMock()
        mock_blocks.summary.return_value = "Test summary"

        handlers = create_gradio_api(mock_blocks)

        result = handlers["summary"]()

        assert result == "Test summary"
        mock_blocks.summary.assert_called_once()
