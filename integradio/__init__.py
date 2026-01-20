"""
Integradio - Vector-embedded UI components for semantic codebase navigation.

Components carry semantic embeddings that make them discoverable and mappable.
Search your UI by intent, trace data flows, and visualize component relationships.

Basic Usage:
    from integradio import SemanticBlocks, semantic

    with SemanticBlocks() as demo:
        inp = semantic(gr.Textbox(label="Query"), intent="user search input")
        btn = semantic(gr.Button("Search"), intent="trigger search")
        out = semantic(gr.Markdown(), intent="display results")

        btn.click(fn=search, inputs=inp, outputs=out)

    # Search components by intent
    results = demo.search("where does user input go")

    # Visualize component graph
    demo.map()

Specialized Wrappers (for complex components):
    from integradio import semantic_multimodal, semantic_chatbot, semantic_plot

    # Rich metadata for AI chat interfaces
    chat = semantic_chatbot(
        gr.Chatbot(),
        persona="assistant",
        supports_streaming=True,
    )

    # Enhanced image editor for inpainting
    editor = semantic_image_editor(
        gr.ImageEditor(),
        use_case="inpainting",
        supports_masks=True,
    )
"""

from .blocks import SemanticBlocks
from .components import semantic, SemanticComponent
from .registry import ComponentRegistry
from .embedder import Embedder, EmbedderUnavailableWarning
from .introspect import get_source_location, extract_dataflow

# Exception hierarchy
from .exceptions import (
    IntegradioError,
    EmbedderError,
    EmbedderUnavailableError,
    EmbedderTimeoutError,
    EmbedderResponseError,
    CacheError,
    RegistryError,
    RegistryDatabaseError,
    ComponentNotFoundError,
    ComponentRegistrationError,
    ComponentError,
    InvalidComponentError,
    ComponentIdError,
    VisualizationError,
    GraphSerializationError,
    APIError,
    ValidationError,
    CircuitBreakerError,
    CircuitOpenError,
)

# Circuit breaker pattern
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerStats,
    CircuitState,
    get_circuit_breaker,
    circuit_registry,
)

# Structured logging
from .logging_config import (
    configure_logging,
    get_logger,
    StructuredFormatter,
    HumanReadableFormatter,
    LogContext,
    OperationContext,
)

# Specialized wrappers for complex components
from .specialized import (
    # Classes
    SemanticMultimodal,
    SemanticImageEditor,
    SemanticAnnotatedImage,
    SemanticHighlightedText,
    SemanticChatbot,
    SemanticPlot,
    SemanticModel3D,
    SemanticDataFrame,
    SemanticFileExplorer,
    # Factory functions
    semantic_multimodal,
    semantic_image_editor,
    semantic_annotated_image,
    semantic_highlighted_text,
    semantic_chatbot,
    semantic_plot,
    semantic_model3d,
    semantic_dataframe,
    semantic_file_explorer,
    # Metadata classes
    MultimodalMetadata,
    ImageEditorMetadata,
    AnnotationMetadata,
    ChatMetadata,
    VisualizationMetadata,
    Model3DMetadata,
)

# Lazy imports for optional modules (events, visual)
def __getattr__(name: str):
    """Lazy import optional modules."""
    import importlib
    if name == "events":
        module = importlib.import_module(".events", __name__)
        globals()["events"] = module
        return module
    if name == "visual":
        module = importlib.import_module(".visual", __name__)
        globals()["visual"] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.3.0"
__all__ = [
    # Core
    "SemanticBlocks",
    "semantic",
    "SemanticComponent",
    "ComponentRegistry",
    "Embedder",
    "EmbedderUnavailableWarning",
    "get_source_location",
    "extract_dataflow",
    # Exceptions
    "IntegradioError",
    "EmbedderError",
    "EmbedderUnavailableError",
    "EmbedderTimeoutError",
    "EmbedderResponseError",
    "CacheError",
    "RegistryError",
    "RegistryDatabaseError",
    "ComponentNotFoundError",
    "ComponentRegistrationError",
    "ComponentError",
    "InvalidComponentError",
    "ComponentIdError",
    "VisualizationError",
    "GraphSerializationError",
    "APIError",
    "ValidationError",
    "CircuitBreakerError",
    "CircuitOpenError",
    # Circuit breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerStats",
    "CircuitState",
    "get_circuit_breaker",
    "circuit_registry",
    # Logging
    "configure_logging",
    "get_logger",
    "StructuredFormatter",
    "HumanReadableFormatter",
    "LogContext",
    "OperationContext",
    # Specialized wrapper classes
    "SemanticMultimodal",
    "SemanticImageEditor",
    "SemanticAnnotatedImage",
    "SemanticHighlightedText",
    "SemanticChatbot",
    "SemanticPlot",
    "SemanticModel3D",
    "SemanticDataFrame",
    "SemanticFileExplorer",
    # Factory functions
    "semantic_multimodal",
    "semantic_image_editor",
    "semantic_annotated_image",
    "semantic_highlighted_text",
    "semantic_chatbot",
    "semantic_plot",
    "semantic_model3d",
    "semantic_dataframe",
    "semantic_file_explorer",
    # Metadata classes
    "MultimodalMetadata",
    "ImageEditorMetadata",
    "AnnotationMetadata",
    "ChatMetadata",
    "VisualizationMetadata",
    "Model3DMetadata",
    # Events (access via integradio.events)
    "events",
    # Visual specifications (access via integradio.visual)
    "visual",
]
