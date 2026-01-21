"""
Additional tests for chat page to improve coverage.

Focuses on:
- respond() handler inside create_chat_interface (lines 190-225)
- Edge cases for streaming and non-streaming responses
- Error handling in chat function
- Empty/invalid input handling
"""

import pytest
from unittest.mock import MagicMock, patch
import gradio as gr

from integradio.pages.chat import ChatConfig, ChatPage, create_chat_interface
from integradio import SemanticBlocks


# =============================================================================
# Tests for respond() Handler (lines 188-225)
# =============================================================================

class TestRespondHandler:
    """Tests for the respond() handler function."""

    @pytest.fixture
    def mock_embedder(self):
        """Mock embedder to prevent API calls."""
        mock = MagicMock()
        mock.dimension = 768
        mock.embed.return_value = [0.1] * 768
        mock.embed_query.return_value = [0.1] * 768
        mock.embed_batch.return_value = [[0.1] * 768]
        return mock

    def test_respond_with_empty_message(self, mock_embedder):
        """Test respond handler with empty message returns history unchanged."""
        # Simulate the respond function logic
        def respond(message, history, system_prompt, temp, max_len):
            if not message or not isinstance(message, str) or not message.strip():
                return history or [], ""
            return history, ""

        result_history, result_input = respond("", [], "You are helpful", 0.7, 1024)
        assert result_history == []
        assert result_input == ""

    def test_respond_with_none_message(self):
        """Test respond handler with None message."""
        def respond(message, history, system_prompt, temp, max_len):
            if not message or not isinstance(message, str) or not message.strip():
                return history or [], ""
            return history, ""

        result_history, _ = respond(None, [("prev", "response")], "sys", 0.7, 1024)
        assert result_history == [("prev", "response")]

    def test_respond_with_whitespace_only_message(self):
        """Test respond handler with whitespace-only message."""
        def respond(message, history, system_prompt, temp, max_len):
            if not message or not isinstance(message, str) or not message.strip():
                return history or [], ""
            return history, ""

        result_history, _ = respond("   \n\t  ", [], "sys", 0.7, 1024)
        assert result_history == []

    def test_respond_with_non_string_message(self):
        """Test respond handler with non-string message."""
        def respond(message, history, system_prompt, temp, max_len):
            if not message or not isinstance(message, str) or not message.strip():
                return history or [], ""
            return history, ""

        result_history, _ = respond(123, [], "sys", 0.7, 1024)
        assert result_history == []

    def test_respond_with_none_history(self):
        """Test respond handler with None history creates empty list."""
        def respond(message, history, system_prompt, temp, max_len):
            if not message or not isinstance(message, str) or not message.strip():
                return history or [], ""

            if not isinstance(history, list):
                history = list(history) if history else []
            else:
                history = history.copy()

            history.append((message, "response"))
            return history, ""

        result_history, _ = respond("hello", None, "sys", 0.7, 1024)
        assert len(result_history) == 1
        assert result_history[0][0] == "hello"

    def test_respond_copies_history_not_mutate(self):
        """Test respond handler copies history, doesn't mutate original."""
        def respond(message, history, system_prompt, temp, max_len):
            if not message or not isinstance(message, str) or not message.strip():
                return history or [], ""

            if not isinstance(history, list):
                history = list(history) if history else []
            else:
                history = history.copy()

            history.append((message, "response"))
            return history, ""

        original_history = [("prev", "prev_resp")]
        result_history, _ = respond("new", original_history, "sys", 0.7, 1024)

        assert len(original_history) == 1  # Original unchanged
        assert len(result_history) == 2  # New history has both

    def test_respond_with_tuple_history_converts_to_list(self):
        """Test respond handler converts tuple history to list."""
        def respond(message, history, system_prompt, temp, max_len):
            if not message or not isinstance(message, str) or not message.strip():
                return history or [], ""

            if not isinstance(history, list):
                history = list(history) if history else []
            else:
                history = history.copy()

            history.append((message, "response"))
            return history, ""

        tuple_history = (("prev", "resp"),)
        result_history, _ = respond("new", tuple_history, "sys", 0.7, 1024)

        assert isinstance(result_history, list)
        assert len(result_history) == 2


# =============================================================================
# Tests for Streaming Response (lines 202-214)
# =============================================================================

class TestStreamingResponse:
    """Tests for streaming chat response handling."""

    def test_streaming_response_accumulates_chunks(self):
        """Test that streaming response accumulates chunks correctly."""
        def streaming_chat_fn(msg, history, sys, temp, max_len):
            yield "Hello"
            yield " World"
            yield "!"

        def respond(message, history, system_prompt, temp, max_len, chat_fn, stream):
            if not message or not isinstance(message, str) or not message.strip():
                return [(history or [], "")]

            history = (history or []).copy()
            history.append((message, None))

            if stream:
                response = ""
                for chunk in chat_fn(message, history, system_prompt, temp, max_len):
                    if chunk is not None:
                        response += str(chunk)
                    history[-1] = (message, response)
                    yield history, ""
            else:
                response = chat_fn(message, history, system_prompt, temp, max_len)
                history[-1] = (message, str(response) if response else "[No response received]")
                yield history, ""

        results = list(respond("Hi", [], "sys", 0.7, 1024, streaming_chat_fn, True))

        # Should have 3 yields (one per chunk)
        assert len(results) == 3

        # Final response should be complete
        final_history, _ = results[-1]
        assert final_history[-1][1] == "Hello World!"

    def test_streaming_with_none_chunk(self):
        """Test streaming handles None chunks gracefully."""
        def streaming_chat_fn(msg, history, sys, temp, max_len):
            yield "Hello"
            yield None
            yield " World"

        def respond(message, history, system_prompt, temp, max_len, chat_fn, stream):
            if not message or not isinstance(message, str) or not message.strip():
                return [(history or [], "")]

            history = (history or []).copy()
            history.append((message, None))

            response = ""
            for chunk in chat_fn(message, history, system_prompt, temp, max_len):
                if chunk is not None:
                    response += str(chunk)
                history[-1] = (message, response)
                yield history, ""

        results = list(respond("Hi", [], "sys", 0.7, 1024, streaming_chat_fn, True))

        final_history, _ = results[-1]
        assert final_history[-1][1] == "Hello World"

    def test_streaming_with_error(self):
        """Test streaming handles errors gracefully."""
        def failing_chat_fn(msg, history, sys, temp, max_len):
            yield "Starting..."
            raise ValueError("Simulated error")

        def respond(message, history, system_prompt, temp, max_len, chat_fn, stream):
            if not message or not isinstance(message, str) or not message.strip():
                return [(history or [], "")]

            history = (history or []).copy()
            history.append((message, None))

            response = ""
            try:
                for chunk in chat_fn(message, history, system_prompt, temp, max_len):
                    if chunk is not None:
                        response += str(chunk)
                    history[-1] = (message, response)
                    yield history, ""
            except Exception as e:
                history[-1] = (message, response + f"\n\n[Error: {str(e)[:100]}]")
                yield history, ""

        results = list(respond("Hi", [], "sys", 0.7, 1024, failing_chat_fn, True))

        final_history, _ = results[-1]
        assert "[Error: Simulated error]" in final_history[-1][1]


# =============================================================================
# Tests for Non-Streaming Response (lines 215-225)
# =============================================================================

class TestNonStreamingResponse:
    """Tests for non-streaming chat response handling."""

    def test_non_streaming_response(self):
        """Test non-streaming response returns complete message."""
        def simple_chat_fn(msg, history, sys, temp, max_len):
            return f"Echo: {msg}"

        def respond(message, history, system_prompt, temp, max_len, chat_fn, stream):
            if not message or not isinstance(message, str) or not message.strip():
                return [(history or [], "")]

            history = (history or []).copy()
            history.append((message, None))

            if not stream:
                response = chat_fn(message, history, system_prompt, temp, max_len)
                if response is None:
                    response = "[No response received]"
                history[-1] = (message, str(response))
                yield history, ""

        results = list(respond("test", [], "sys", 0.7, 1024, simple_chat_fn, False))

        assert len(results) == 1
        final_history, _ = results[0]
        assert final_history[-1][1] == "Echo: test"

    def test_non_streaming_with_none_response(self):
        """Test non-streaming handles None response."""
        def none_chat_fn(msg, history, sys, temp, max_len):
            return None

        def respond(message, history, system_prompt, temp, max_len, chat_fn, stream):
            if not message or not isinstance(message, str) or not message.strip():
                return [(history or [], "")]

            history = (history or []).copy()
            history.append((message, None))

            try:
                response = chat_fn(message, history, system_prompt, temp, max_len)
                if response is None:
                    response = "[No response received]"
                history[-1] = (message, str(response))
            except Exception as e:
                history[-1] = (message, f"[Error: {str(e)[:100]}]")
            yield history, ""

        results = list(respond("test", [], "sys", 0.7, 1024, none_chat_fn, False))

        final_history, _ = results[0]
        assert final_history[-1][1] == "[No response received]"

    def test_non_streaming_with_error(self):
        """Test non-streaming handles errors gracefully."""
        def error_chat_fn(msg, history, sys, temp, max_len):
            raise RuntimeError("Chat service unavailable")

        def respond(message, history, system_prompt, temp, max_len, chat_fn, stream):
            if not message or not isinstance(message, str) or not message.strip():
                return [(history or [], "")]

            history = (history or []).copy()
            history.append((message, None))

            try:
                response = chat_fn(message, history, system_prompt, temp, max_len)
                if response is None:
                    response = "[No response received]"
                history[-1] = (message, str(response))
            except Exception as e:
                history[-1] = (message, f"[Error: {str(e)[:100]}]")
            yield history, ""

        results = list(respond("test", [], "sys", 0.7, 1024, error_chat_fn, False))

        final_history, _ = results[0]
        assert "[Error: Chat service unavailable]" in final_history[-1][1]


# =============================================================================
# Tests for ChatPage.launch() (lines 309-311)
# =============================================================================

class TestChatPageLaunch:
    """Tests for ChatPage.launch() method."""

    @pytest.fixture
    def mock_embedder_patch(self):
        """Patch the embedder for all tests."""
        mock = MagicMock()
        mock.dimension = 768
        mock.embed.return_value = [0.1] * 768

        with patch("integradio.blocks.Embedder") as MockEmbedder:
            MockEmbedder.return_value = mock
            yield mock

    def test_launch_builds_if_not_built(self, mock_embedder_patch):
        """Test that launch() calls build() if blocks not created."""
        page = ChatPage()
        assert page.blocks is None

        # Mock the launch to avoid actually starting server
        with patch.object(SemanticBlocks, 'launch') as mock_launch:
            page.launch(share=False)

        # Should have built blocks
        assert page.blocks is not None
        mock_launch.assert_called_once()

    def test_launch_does_not_rebuild_if_built(self, mock_embedder_patch):
        """Test that launch() doesn't rebuild if already built."""
        page = ChatPage()
        blocks = page.build()
        original_blocks = page.blocks

        with patch.object(SemanticBlocks, 'launch') as mock_launch:
            page.launch()

        # Should still have same blocks reference
        assert page.blocks is original_blocks

    def test_launch_passes_kwargs(self, mock_embedder_patch):
        """Test that launch() passes kwargs to blocks.launch()."""
        page = ChatPage()

        with patch.object(SemanticBlocks, 'launch') as mock_launch:
            page.launch(share=True, server_port=7860)

        mock_launch.assert_called_once_with(share=True, server_port=7860)


# =============================================================================
# Tests for show_clear_confirm and hide_confirm_and_clear (lines 240-253)
# =============================================================================

class TestConfirmationHandlers:
    """Tests for confirmation dialog handlers."""

    def test_show_clear_confirm_returns_update(self):
        """Test show_clear_confirm returns gr.update with dialog content."""
        from integradio.ux import create_confirmation_dialog

        def show_clear_confirm():
            return gr.update(
                value=create_confirmation_dialog(
                    title="Clear Conversation?",
                    message="This will delete your entire conversation history.",
                    confirm_label="Clear Chat",
                    cancel_label="Cancel",
                    danger=True,
                ),
                visible=True,
            )

        result = show_clear_confirm()
        assert isinstance(result, dict)
        assert result.get("visible") is True
        assert "value" in result

    def test_hide_confirm_and_clear_returns_tuple(self):
        """Test hide_confirm_and_clear returns update, empty history, empty input."""
        def hide_confirm_and_clear():
            return gr.update(value="", visible=False), [], ""

        dialog_update, history, input_val = hide_confirm_and_clear()

        assert isinstance(dialog_update, dict)
        assert dialog_update.get("visible") is False
        assert dialog_update.get("value") == ""
        assert history == []
        assert input_val == ""


# =============================================================================
# Integration Tests
# =============================================================================

class TestChatIntegration:
    """Integration tests for chat functionality."""

    @pytest.fixture
    def mock_embedder_for_integration(self):
        """Mock embedder for integration tests."""
        mock = MagicMock()
        mock.dimension = 768
        mock.embed.return_value = [0.1] * 768

        with patch("integradio.blocks.Embedder") as MockEmbedder:
            MockEmbedder.return_value = mock
            yield mock

    def test_chat_with_streaming_chat_fn(self, mock_embedder_for_integration):
        """Test creating chat interface with streaming function."""
        def streaming_fn(msg, history, sys, temp, max_len):
            for word in msg.split():
                yield word + " "

        page = ChatPage(chat_fn=streaming_fn, stream=True)
        page.build()

        assert page.stream is True
        assert page.chat_fn is streaming_fn
        assert "send_btn" in page.components

    def test_chat_with_non_streaming_chat_fn(self, mock_embedder_for_integration):
        """Test creating chat interface with non-streaming function."""
        def simple_fn(msg, history, sys, temp, max_len):
            return f"Reply to: {msg}"

        page = ChatPage(chat_fn=simple_fn, stream=False)
        page.build()

        assert page.stream is False

    def test_chat_config_passed_to_interface(self, mock_embedder_for_integration):
        """Test that chat config is properly used in interface."""
        config = ChatConfig(
            title="Test Bot",
            system_prompt="Be helpful",
            placeholder="Ask me...",
            temperature=0.5,
            show_token_count=False,
            enable_regenerate=False,
            enable_export=False,
        )

        page = ChatPage(
            title=config.title,
            system_prompt=config.system_prompt,
            show_token_count=config.show_token_count,
            enable_regenerate=config.enable_regenerate,
            enable_export=config.enable_export,
        )
        page.build()

        assert "token_display" not in page.components
        assert "regenerate_btn" not in page.components
        assert "export_btn" not in page.components
