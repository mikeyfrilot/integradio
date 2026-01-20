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

    # Action buttons row
    with gr.Row():
        components["clear_btn"] = semantic(
            gr.Button("Clear Chat", variant="secondary", size="sm"),
            intent="clears entire conversation history",
            tags=["action", "destructive", "reset"],
        )

        if config.enable_regenerate:
            components["regenerate_btn"] = semantic(
                gr.Button("Regenerate", variant="secondary", size="sm"),
                intent="regenerates last AI response",
                tags=["action", "retry"],
            )

        if config.enable_export:
            components["export_btn"] = semantic(
                gr.Button("Export", variant="secondary", size="sm"),
                intent="exports conversation as downloadable file",
                tags=["action", "export"],
            )

    # Token counter
    if config.show_token_count:
        components["token_display"] = semantic(
            gr.Markdown("Tokens: 0 / " + str(config.max_tokens)),
            intent="shows current token usage and limit",
            tags=["status", "metrics"],
        )

    # Settings row
    with gr.Accordion("Generation Settings", open=False):
        with gr.Row():
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
            if not message.strip():
                return history, ""

            history = history or []
            # Gradio 6.x uses tuples: (user_msg, assistant_msg)
            history.append((message, None))

            if stream:
                response = ""
                for chunk in chat_fn(message, history, system_prompt, temp, max_len):
                    response += chunk
                    history[-1] = (message, response)
                    yield history, ""
            else:
                response = chat_fn(message, history, system_prompt, temp, max_len)
                history[-1] = (message, response)
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

        # Clear handler
        components["clear_btn"].click(
            fn=lambda: ([], ""),
            outputs=[components["chatbot"], components["user_input"]],
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
