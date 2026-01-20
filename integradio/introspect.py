"""
Introspection utilities - Extract source locations and dataflow from components.
"""

import inspect
from typing import Optional, Any, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    import gradio as gr


@dataclass
class SourceLocation:
    """Source code location for a component."""
    file_path: str
    line_number: int
    function_name: Optional[str] = None


def get_source_location(depth: int = 2) -> Optional[SourceLocation]:
    """
    Get source location of the calling code.

    Args:
        depth: Stack depth to inspect (2 = caller's caller)

    Returns:
        SourceLocation or None if unavailable
    """
    try:
        frame = inspect.currentframe()
        for _ in range(depth):
            if frame is None:
                return None
            frame = frame.f_back

        if frame is None:
            return None

        info = inspect.getframeinfo(frame)
        return SourceLocation(
            file_path=info.filename,
            line_number=info.lineno,
            function_name=info.function if info.function != "<module>" else None,
        )
    except Exception:
        return None


def extract_component_info(component: Any) -> dict[str, Any]:
    """
    Extract useful information from a Gradio component.

    Args:
        component: Gradio component instance

    Returns:
        Dict with extracted info (label, elem_id, type, etc.)
    """
    info: dict[str, Any] = {
        "type": type(component).__name__,
    }

    # Common attributes
    for attr in ["label", "elem_id", "elem_classes", "visible", "interactive"]:
        if hasattr(component, attr):
            value = getattr(component, attr)
            if value is not None:
                info[attr] = value

    # Get component ID if available
    if hasattr(component, "_id"):
        info["component_id"] = component._id

    # Get placeholder/value for inputs
    for attr in ["placeholder", "value", "choices"]:
        if hasattr(component, attr):
            value = getattr(component, attr)
            if value is not None:
                info[attr] = value

    return info


def extract_dataflow(blocks: "gr.Blocks") -> list[dict[str, Any]]:
    """
    Extract dataflow relationships from a Blocks instance.

    Parses event listeners to find input/output connections.

    Args:
        blocks: Gradio Blocks instance

    Returns:
        List of dataflow relationships
    """
    relationships = []

    # Access internal function registry
    if not hasattr(blocks, "fns"):
        return relationships

    for fn_id, block_fn in blocks.fns.items():
        if not hasattr(block_fn, "inputs") or not hasattr(block_fn, "outputs"):
            continue

        inputs = block_fn.inputs or []
        outputs = block_fn.outputs or []
        triggers = getattr(block_fn, "triggers", [])

        # Get trigger component IDs
        trigger_ids = []
        for trigger in triggers:
            if hasattr(trigger, "block") and hasattr(trigger.block, "_id"):
                trigger_ids.append(trigger.block._id)

        # Map input -> output relationships
        input_ids = [
            inp._id for inp in inputs
            if hasattr(inp, "_id")
        ]
        output_ids = [
            out._id for out in outputs
            if hasattr(out, "_id")
        ]

        if input_ids or output_ids:
            relationships.append({
                "fn_id": fn_id,
                "triggers": trigger_ids,
                "inputs": input_ids,
                "outputs": output_ids,
            })

    return relationships


def build_intent_text(component: Any, explicit_intent: Optional[str] = None) -> str:
    """
    Build a text description for embedding.

    Combines explicit intent with component attributes.

    Args:
        component: Gradio component
        explicit_intent: User-provided intent description

    Returns:
        Text suitable for embedding
    """
    parts = []

    # Explicit intent first (most important)
    if explicit_intent:
        parts.append(explicit_intent)

    # Component type
    comp_type = type(component).__name__
    parts.append(f"Gradio {comp_type} component")

    # Label
    if hasattr(component, "label") and component.label:
        parts.append(f"labeled '{component.label}'")

    # elem_id
    if hasattr(component, "elem_id") and component.elem_id:
        parts.append(f"with id '{component.elem_id}'")

    # Placeholder
    if hasattr(component, "placeholder") and component.placeholder:
        parts.append(f"placeholder: {component.placeholder}")

    # Choices (for dropdowns, radios)
    if hasattr(component, "choices") and component.choices:
        choices_str = ", ".join(str(c) for c in component.choices[:5])
        if len(component.choices) > 5:
            choices_str += f"... ({len(component.choices)} total)"
        parts.append(f"choices: {choices_str}")

    return " | ".join(parts)


def infer_tags(component: Any) -> list[str]:
    """
    Infer semantic tags from component type and attributes.

    Supports all Gradio 6 components including new navigation and layout elements.

    Args:
        component: Gradio component

    Returns:
        List of inferred tags
    """
    tags = []
    comp_type = type(component).__name__.lower()

    # Type-based tags - comprehensive mapping for all Gradio 6 components
    type_tags = {
        # === Text Input Components ===
        "textbox": ["input", "text"],
        "textarea": ["input", "text", "multiline"],
        "multimodaltextbox": ["input", "text", "multimodal", "file"],
        "code": ["input", "code", "text"],

        # === Numeric Input Components ===
        "number": ["input", "numeric"],
        "slider": ["input", "numeric", "range"],

        # === Selection Components ===
        "checkbox": ["input", "boolean"],
        "checkboxgroup": ["input", "multi-select"],
        "radio": ["input", "single-select"],
        "dropdown": ["input", "single-select"],

        # === Date/Time Components ===
        "datetime": ["input", "temporal", "date"],

        # === Color Components ===
        "colorpicker": ["input", "color"],

        # === Button/Trigger Components ===
        "button": ["trigger", "action"],
        "uploadbutton": ["trigger", "upload"],
        "downloadbutton": ["trigger", "download"],
        "clearbutton": ["trigger", "clear", "action"],
        "duplicatebutton": ["trigger", "duplicate", "action"],
        "loginbutton": ["trigger", "auth", "action"],
        "timer": ["trigger", "temporal", "automation"],

        # === File Components ===
        "file": ["input", "upload", "file"],
        "fileexplorer": ["input", "navigation", "filesystem"],

        # === Media Input Components ===
        "image": ["media", "visual", "input"],
        "imageeditor": ["input", "media", "editor", "visual"],
        "audio": ["media", "audio"],
        "video": ["media", "video"],
        "model3d": ["media", "3d", "visualization"],

        # === Media Output Components ===
        "gallery": ["output", "media", "collection", "visual"],
        "annotatedimage": ["output", "media", "annotation", "visual"],
        "simpleimage": ["output", "media", "visual"],
        "imageslider": ["input", "media", "comparison", "visual"],

        # === Text Output Components ===
        "markdown": ["output", "text", "display"],
        "html": ["output", "display", "web"],
        "highlightedtext": ["output", "text", "annotation", "nlp"],
        "label": ["output", "classification"],

        # === Data Components ===
        "dataframe": ["data", "tabular"],
        "dataset": ["data", "collection", "examples"],
        "json": ["data", "structured"],
        "state": ["data", "session", "hidden"],
        "paramviewer": ["data", "parameters", "display"],

        # === Visualization Components ===
        "plot": ["output", "visualization"],
        "scatterplot": ["output", "visualization", "data", "scatter"],
        "lineplot": ["output", "visualization", "timeseries"],
        "barplot": ["output", "visualization", "categorical"],

        # === Conversation Components ===
        "chatbot": ["io", "conversation", "ai"],
        "chatinterface": ["io", "conversation", "ai", "interface"],

        # === Layout Components ===
        "accordion": ["layout", "collapsible"],
        "tab": ["layout", "navigation", "tabbed"],
        "tabs": ["layout", "navigation", "tabbed"],
        "row": ["layout", "horizontal"],
        "column": ["layout", "vertical"],
        "group": ["layout", "container"],
        "blocks": ["layout", "container", "root"],

        # === Gradio 6 Navigation Components ===
        "sidebar": ["layout", "navigation", "panel"],
        "navbar": ["layout", "navigation", "header"],
        "dialogue": ["io", "modal", "interaction"],
        "walkthrough": ["layout", "tutorial", "guided"],

        # === Interface Components ===
        "interface": ["layout", "interface", "root"],
        "tabbedinterface": ["layout", "interface", "tabbed"],
    }

    if comp_type in type_tags:
        tags.extend(type_tags[comp_type])
    else:
        # Generic fallback
        tags.append("component")

    # Attribute-based tags
    if hasattr(component, "interactive"):
        if component.interactive:
            tags.append("interactive")
        else:
            tags.append("static")

    # Additional attribute-based inference
    if hasattr(component, "streaming") and component.streaming:
        tags.append("streaming")

    if hasattr(component, "type"):
        # File type hints
        file_type = getattr(component, "type", None)
        if file_type == "filepath":
            tags.append("filepath")
        elif file_type == "binary":
            tags.append("binary")

    if hasattr(component, "sources"):
        # Media source hints (webcam, upload, etc.)
        sources = getattr(component, "sources", [])
        if "webcam" in sources:
            tags.append("webcam")
        if "microphone" in sources:
            tags.append("microphone")
        if "upload" in sources:
            tags.append("upload")

    if hasattr(component, "show_copy_button") and component.show_copy_button:
        tags.append("copyable")

    if hasattr(component, "rtl") and component.rtl:
        tags.append("rtl")

    return list(set(tags))  # Dedupe
