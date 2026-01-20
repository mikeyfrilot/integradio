"""
Semantic Components - Wrapped Gradio components with embedded vectors.

Now with optional visual specification support for complete UI contracts.
"""

import logging
from typing import Optional, Any, TypeVar, TYPE_CHECKING
from dataclasses import dataclass, field
import weakref

if TYPE_CHECKING:
    import gradio as gr
    from .registry import ComponentRegistry
    from .embedder import Embedder
    from .visual import VisualSpec

from .introspect import (
    get_source_location,
    extract_component_info,
    build_intent_text,
    infer_tags,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class SemanticMetadata:
    """Metadata attached to a semantic component."""
    intent: str
    tags: list[str] = field(default_factory=list)
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    embedded: bool = False
    extra: dict[str, Any] = field(default_factory=dict)
    # Visual specification (optional)
    visual_spec: Optional["VisualSpec"] = None


class SemanticComponent:
    """
    Wrapper that adds vector embedding capabilities to a Gradio component.

    This doesn't subclass the component - it wraps it and delegates attribute
    access, allowing any Gradio component to become semantic.
    """

    # Registry reference (set by SemanticBlocks)
    _registry: Optional["ComponentRegistry"] = None
    _embedder: Optional["Embedder"] = None

    # Track all semantic components
    _instances: weakref.WeakValueDictionary = weakref.WeakValueDictionary()

    def __init__(
        self,
        component: Any,
        intent: Optional[str] = None,
        tags: Optional[list[str]] = None,
        auto_embed: bool = True,
        visual: Optional["VisualSpec"] = None,
        **extra: Any,
    ):
        """
        Wrap a Gradio component with semantic capabilities.

        Args:
            component: The Gradio component to wrap
            intent: Semantic description of the component's purpose
            tags: Additional semantic tags
            auto_embed: Whether to automatically generate embedding
            visual: Optional VisualSpec defining the component's appearance
            **extra: Additional metadata to store
        """
        self._component = component
        self._auto_embed = auto_embed

        # Get source location
        source = get_source_location(depth=3)

        # Build intent from component info if not provided
        if intent is None:
            info = extract_component_info(component)
            intent = info.get("label") or info.get("elem_id") or info["type"]

        # Infer tags if not provided
        if tags is None:
            tags = infer_tags(component)
        else:
            tags = list(tags) + infer_tags(component)

        # Auto-populate visual spec component_id if provided
        if visual is not None and not visual.component_id:
            visual.component_id = str(getattr(component, "_id", id(component)))
            if not visual.component_type:
                visual.component_type = type(component).__name__

        self._semantic_meta = SemanticMetadata(
            intent=intent,
            tags=list(set(tags)),
            file_path=source.file_path if source else None,
            line_number=source.line_number if source else None,
            extra=extra,
            visual_spec=visual,
        )

        # Track instance
        if hasattr(component, "_id"):
            SemanticComponent._instances[component._id] = self

    @property
    def component(self) -> Any:
        """Get the wrapped Gradio component."""
        return self._component

    @property
    def semantic_meta(self) -> SemanticMetadata:
        """Get semantic metadata."""
        return self._semantic_meta

    @property
    def intent(self) -> str:
        """Get the component intent."""
        return self._semantic_meta.intent

    @intent.setter
    def intent(self, value: str) -> None:
        """Update the intent (triggers re-embedding if registered)."""
        self._semantic_meta.intent = value
        self._semantic_meta.embedded = False
        if self._registry and self._embedder:
            self._register_to_registry()

    def add_tags(self, *tags: str) -> "SemanticComponent":
        """Add tags to the component."""
        self._semantic_meta.tags.extend(tags)
        self._semantic_meta.tags = list(set(self._semantic_meta.tags))
        return self

    @property
    def visual(self) -> Optional["VisualSpec"]:
        """Get the visual specification."""
        return self._semantic_meta.visual_spec

    @visual.setter
    def visual(self, spec: "VisualSpec") -> None:
        """Set the visual specification."""
        # Auto-populate IDs if not set
        if not spec.component_id:
            spec.component_id = str(getattr(self._component, "_id", id(self._component)))
        if not spec.component_type:
            spec.component_type = type(self._component).__name__
        self._semantic_meta.visual_spec = spec

    def set_visual(
        self,
        background: Optional[str] = None,
        text_color: Optional[str] = None,
        border_color: Optional[str] = None,
        padding: Optional[int] = None,
        **kwargs: Any,
    ) -> "SemanticComponent":
        """
        Quick way to set visual properties.

        Creates a VisualSpec if one doesn't exist.

        Args:
            background: Background color (hex)
            text_color: Text color (hex)
            border_color: Border color (hex)
            padding: Padding in pixels
            **kwargs: Additional properties

        Returns:
            Self for chaining
        """
        from .visual import VisualSpec, DimensionValue

        if self._semantic_meta.visual_spec is None:
            self._semantic_meta.visual_spec = VisualSpec(
                component_id=str(getattr(self._component, "_id", id(self._component))),
                component_type=type(self._component).__name__,
            )

        spec = self._semantic_meta.visual_spec

        if background or text_color or border_color:
            spec.set_colors(
                background=background,
                text=text_color,
                border=border_color,
            )

        if padding is not None:
            spec.set_spacing(padding=DimensionValue(padding, "px"))

        return self

    def to_css(self, selector: Optional[str] = None) -> str:
        """
        Generate CSS for this component.

        Args:
            selector: CSS selector (defaults to #component_id)

        Returns:
            CSS string
        """
        if self._semantic_meta.visual_spec:
            return self._semantic_meta.visual_spec.to_css(selector)
        return ""

    def _register_to_registry(self) -> bool:
        """
        Register this component to the global registry.

        Returns:
            True if registration successful, False otherwise
        """
        if self._registry is None or self._embedder is None:
            return False
        if not hasattr(self._component, "_id"):
            return False

        try:
            from .registry import ComponentMetadata

            # Build embedding text
            embed_text = build_intent_text(self._component, self._semantic_meta.intent)

            # Generate embedding
            vector = self._embedder.embed(embed_text)

            # Create metadata
            info = extract_component_info(self._component)
            metadata = ComponentMetadata(
                component_id=self._component._id,
                component_type=info["type"],
                intent=self._semantic_meta.intent,
                label=info.get("label"),
                elem_id=info.get("elem_id"),
                tags=self._semantic_meta.tags,
                file_path=self._semantic_meta.file_path,
                line_number=self._semantic_meta.line_number,
                extra=self._semantic_meta.extra,
            )

            # Register
            success = self._registry.register(self._component._id, vector, metadata)
            if success:
                self._semantic_meta.embedded = True
            return success
        except Exception as e:
            logger.error(f"Failed to register component {getattr(self._component, '_id', '?')}: {e}")
            return False

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to wrapped component."""
        # Allow Gradio's internal attributes to pass through
        if name in ("_id", "_parent", "_elem_id", "_elem_classes"):
            return getattr(self._component, name)
        # Block our own private attributes
        if name.startswith("_") or name in ("component", "semantic_meta", "intent", "visual"):
            raise AttributeError(name)
        return getattr(self._component, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Handle attribute setting."""
        if name.startswith("_") or name in ("component", "semantic_meta", "intent", "visual"):
            object.__setattr__(self, name, value)
        else:
            setattr(self._component, name, value)

    def __repr__(self) -> str:
        comp_type = type(self._component).__name__
        comp_id = getattr(self._component, "_id", "?")
        return f"SemanticComponent({comp_type}, id={comp_id}, intent='{self.intent}')"


def semantic(
    component: T,
    intent: Optional[str] = None,
    tags: Optional[list[str]] = None,
    visual: Optional["VisualSpec"] = None,
    **extra: Any,
) -> T:
    """
    Wrap a Gradio component with semantic capabilities.

    This is the main API for creating semantic components.

    Args:
        component: Gradio component to wrap
        intent: Semantic description of purpose
        tags: Additional semantic tags
        visual: Optional VisualSpec for appearance definition
        **extra: Additional metadata

    Returns:
        SemanticComponent wrapping the input (typed as original for IDE support)

    Example:
        inp = semantic(gr.Textbox(label="Query"), intent="user search input")

        # With visual specification
        from integradio.visual import VisualSpec
        btn = semantic(
            gr.Button("Search"),
            intent="trigger search",
            visual=VisualSpec(component_id="search-btn").set_colors(background="#3b82f6"),
        )
    """
    return SemanticComponent(component, intent=intent, tags=tags, visual=visual, **extra)  # type: ignore


def get_semantic(component_id: int) -> Optional[SemanticComponent]:
    """
    Get a SemanticComponent by its component ID.

    Args:
        component_id: Gradio component ID

    Returns:
        SemanticComponent if found, None otherwise
    """
    return SemanticComponent._instances.get(component_id)
