"""
Chat Page - Conversational AI interface with full message tracking.

Features:
- Message history with semantic tagging
- System prompt configuration
- Streaming support
- Export/clear functionality
- Token counting display
"""

from typing import Optional, Callable, Any, Generator
from dataclasses import dataclass, field

import gradio as gr

from ..components import semantic
from ..blocks import SemanticBlocks
from ..ux import create_confirmation_dialog


@dataclass
class ChatConfig:
    """Configuration for chat interface."""
    title: str = "AI Chat"
    system_prompt: str = "You are a helpful assistant."
    placeholder: str = "Type your message..."
    max_tokens: int = 4096
    temperature: float = 0.7
    show_token_count: bool = True
    show_system_prompt: bool = True
    enable_export: bool = True
    enable_regenerate: bool = True
    theme: str = "soft"


def create_chat_interface(
    config: Optional[ChatConfig] = None,
    chat_fn: Optional[Callable] = None,
    stream: bool = True,
) -> dict[str, Any]:
    """
    Create a chat interface with semantic-tracked components.

    Args:
        config: Chat configuration
        chat_fn: Function to handle chat (receives message, history)
        stream: Whether to stream responses

    Returns:
        Dict of component references for further customization
    """
    config = config or ChatConfig()

    components = {}

    # Header
    components["title"] = semantic(
        gr.Markdown(f"# {config.title}"),
        intent="displays chat interface title",
        tags=["header", "branding"],
    )

    # System prompt (collapsible)
    if config.show_system_prompt:
        with gr.Accordion("System Prompt", open=False):
            components["system_prompt"] = semantic(
                gr.Textbox(
                    value=config.system_prompt,
                    lines=3,
                    label="System Prompt",
                    elem_id="chat-system-prompt",
                ),
                intent="configures AI personality and behavior",
                tags=["config", "system"],
            )

    # Main chat area
    components["chatbot"] = semantic(
        gr.Chatbot(
            label="Conversation",
            height=500,
            elem_id="chat-history",
        ),
        intent="displays conversation history between user and AI",
        tags=["conversation", "history", "primary-display"],
    )

    # Input area
    with gr.Row():
        with gr.Column(scale=6):
            components["user_input"] = semantic(
                gr.Textbox(
                    placeholder=config.placeholder,
                    label="Message",
                    lines=2,
                    max_lines=10,
                    elem_id="chat-input",
                    show_label=False,
                ),
                intent="user types message to send to AI",
                tags=["input", "primary-input", "message"],
            )

        with gr.Column(scale=1, min_width=100):
            components["send_btn"] = semantic(
                gr.Button("Send", variant="primary", size="lg"),
                intent="sends user message to AI for response",
                tags=["action", "primary", "submit"],
            )

    # Action buttons row - default size for WCAG 2.2 touch target compliance (44x44px min)
    with gr.Row():
        components["clear_btn"] = semantic(
            gr.Button("Clear Chat", variant="secondary"),
            intent="clears entire conversation history",
            tags=["action", "destructive", "reset"],
        )

        if config.enable_regenerate:
            components["regenerate_btn"] = semantic(
                gr.Button("Regenerate", variant="secondary"),
                intent="regenerates last AI response",
                tags=["action", "retry"],
            )

        if config.enable_export:
            components["export_btn"] = semantic(
                gr.Button("Export", variant="secondary"),
                intent="exports conversation as downloadable file",
                tags=["action", "export"],
            )

    # Confirmation dialog for clear chat (2026 UX best practice)
    components["clear_confirm"] = gr.HTML(
        "",
        visible=False,
        elem_id="clear-confirm-dialog",
    )

    # Token counter
    if config.show_token_count:
        components["token_display"] = semantic(
            gr.Markdown("Tokens: 0 / " + str(config.max_tokens)),
            intent="shows current token usage and limit",
            tags=["status", "metrics"],
        )

    # Settings row with help text for accessibility (2026 UX best practice)
    with gr.Accordion("Generation Settings", open=False):
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    '<small style="color: #6b7280;">Lower = more focused and consistent, '
                    'Higher = more creative and varied</small>'
                )
                components["temperature"] = semantic(
                    gr.Slider(
                        minimum=0,
                        maximum=2,
                        value=config.temperature,
                        step=0.1,
                        label="Temperature",
                    ),
                    intent="controls randomness of AI responses",
                    tags=["config", "generation"],
                )

            with gr.Column():
                gr.Markdown(
                    '<small style="color: #6b7280;">Maximum number of tokens in the AI response. '
                    'Longer responses use more tokens.</small>'
                )
                components["max_length"] = semantic(
                    gr.Slider(
                        minimum=100,
                        maximum=config.max_tokens,
                        value=1024,
                        step=100,
                        label="Max Response Length",
                    ),
                    intent="limits maximum length of AI response",
                    tags=["config", "generation"],
                )

    # Wire up default handlers if chat_fn provided
    if chat_fn:
        def respond(message, history, system_prompt, temp, max_len):
            # Edge case: None or invalid message input
            if not message or not isinstance(message, str) or not message.strip():
                return history or [], ""

            # Edge case: Ensure history is a mutable list
            if not isinstance(history, list):
                history = list(history) if history else []
            else:
                history = history.copy()  # Don't mutate original

            # Gradio 6.x uses tuples: (user_msg, assistant_msg)
            history.append((message, None))

            if stream:
                response = ""
                try:
                    for chunk in chat_fn(message, history, system_prompt, temp, max_len):
                        # Edge case: chunk could be None
                        if chunk is not None:
                            response += str(chunk)
                        history[-1] = (message, response)
                        yield history, ""
                except Exception as e:
                    # Edge case: Error during streaming
                    history[-1] = (message, response + f"\n\n[Error: {str(e)[:100]}]")
                    yield history, ""
            else:
                try:
                    response = chat_fn(message, history, system_prompt, temp, max_len)
                    # Edge case: None response
                    if response is None:
                        response = "[No response received]"
                    history[-1] = (message, str(response))
                except Exception as e:
                    # Edge case: Error during response generation
                    history[-1] = (message, f"[Error: {str(e)[:100]}]")
                yield history, ""

        inputs = [
            components["user_input"],
            components["chatbot"],
            components.get("system_prompt", gr.State(config.system_prompt)),
            components["temperature"],
            components["max_length"],
        ]
        outputs = [components["chatbot"], components["user_input"]]

        components["send_btn"].click(fn=respond, inputs=inputs, outputs=outputs)
        components["user_input"].submit(fn=respond, inputs=inputs, outputs=outputs)

    # Confirmation flow for clear chat (separate from chat_fn to always work)
    def show_clear_confirm():
        return gr.update(
            value=create_confirmation_dialog(
                title="Clear Conversation?",
                message="This will delete your entire conversation history. This action cannot be undone.",
                confirm_label="Clear Chat",
                cancel_label="Cancel",
                danger=True,
            ),
            visible=True,
        )

    def hide_confirm_and_clear():
        return gr.update(value="", visible=False), [], ""

    # Show confirmation on clear click
    components["clear_btn"].click(
        fn=show_clear_confirm,
        outputs=[components["clear_confirm"]],
    )

    return components


class ChatPage:
    """
    Complete chat page with SemanticBlocks integration.

    Usage:
        page = ChatPage(title="My Bot", chat_fn=my_chat_function)
        page.launch()
    """

    def __init__(
        self,
        title: str = "AI Chat",
        system_prompt: str = "You are a helpful assistant.",
        chat_fn: Optional[Callable] = None,
        stream: bool = True,
        **config_kwargs,
    ):
        self.config = ChatConfig(
            title=title,
            system_prompt=system_prompt,
            **config_kwargs,
        )
        self.chat_fn = chat_fn
        self.stream = stream
        self.components: dict[str, Any] = {}
        self.blocks: Optional[SemanticBlocks] = None

    def build(self) -> SemanticBlocks:
        """Build the chat interface."""
        self.blocks = SemanticBlocks(
            title=self.config.title,
            theme=getattr(gr.themes, self.config.theme.title(), gr.themes.Soft)(),
        )

        with self.blocks:
            self.components = create_chat_interface(
                config=self.config,
                chat_fn=self.chat_fn,
                stream=self.stream,
            )

        return self.blocks

    def launch(self, **kwargs) -> None:
        """Build and launch the chat interface."""
        if self.blocks is None:
            self.build()
        self.blocks.launch(**kwargs)

    @staticmethod
    def render(
        config: Optional[ChatConfig] = None,
        chat_fn: Optional[Callable] = None,
    ) -> dict[str, Any]:
        """
        Render chat interface into existing Blocks context.

        Use inside a `with SemanticBlocks()` or `with gr.Blocks()`.
        """
        return create_chat_interface(config=config, chat_fn=chat_fn)
