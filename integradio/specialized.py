"""
Specialized Semantic Wrappers - Enhanced semantic metadata for complex Gradio components.

These wrappers provide richer intent inference, domain-specific tags, and
component-specific metadata extraction for complex Gradio 6 components.
"""

from typing import Optional, Any, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    import gradio as gr

from .components import SemanticComponent, semantic


# =============================================================================
# Metadata Classes for Rich Component Information
# =============================================================================

@dataclass
class MultimodalMetadata:
    """Extended metadata for multimodal inputs."""
    accepts_images: bool = True
    accepts_files: bool = True
    accepts_audio: bool = False
    max_files: Optional[int] = None
    file_types: list[str] = field(default_factory=list)
    use_case: Optional[str] = None  # "chat", "document_qa", "image_analysis"


@dataclass
class ImageEditorMetadata:
    """Extended metadata for image editing components."""
    tools: list[str] = field(default_factory=list)  # ["brush", "eraser", "crop", "layers"]
    supports_layers: bool = False
    supports_masks: bool = False
    output_format: str = "png"
    use_case: Optional[str] = None  # "inpainting", "annotation", "segmentation"


@dataclass
class AnnotationMetadata:
    """Extended metadata for annotation components."""
    annotation_type: str = "generic"  # "ner", "bbox", "segmentation", "highlight"
    entity_types: list[str] = field(default_factory=list)
    color_map: dict[str, str] = field(default_factory=dict)
    supports_overlapping: bool = False


@dataclass
class ChatMetadata:
    """Extended metadata for chat/conversation components."""
    supports_streaming: bool = True
    supports_retry: bool = True
    supports_undo: bool = True
    supports_like: bool = False
    avatar_style: Optional[str] = None  # "circle", "square", "none"
    message_format: str = "markdown"  # "markdown", "html", "plain"


@dataclass
class VisualizationMetadata:
    """Extended metadata for visualization components."""
    chart_type: str = "generic"  # "line", "bar", "scatter", "pie", "heatmap"
    interactive: bool = True
    supports_zoom: bool = False
    supports_pan: bool = False
    axes: list[str] = field(default_factory=list)  # ["x", "y", "z"]
    data_format: str = "pandas"  # "pandas", "dict", "numpy"


@dataclass
class Model3DMetadata:
    """Extended metadata for 3D model components."""
    supported_formats: list[str] = field(default_factory=lambda: ["obj", "glb", "gltf"])
    supports_animation: bool = False
    supports_textures: bool = True
    camera_controls: bool = True
    lighting: str = "default"  # "default", "studio", "outdoor"


# =============================================================================
# Specialized Semantic Wrapper Classes
# =============================================================================

class SemanticMultimodal(SemanticComponent):
    """
    Specialized wrapper for MultimodalTextbox with enhanced metadata.

    Provides rich semantic understanding for multimodal AI inputs.

    Example:
        inp = SemanticMultimodal(
            gr.MultimodalTextbox(label="Ask about images"),
            intent="user uploads images and asks questions",
            use_case="image_analysis",
            accepts_audio=False,
        )
    """

    def __init__(
        self,
        component: Any,
        intent: Optional[str] = None,
        tags: Optional[list[str]] = None,
        use_case: Optional[str] = None,
        accepts_images: bool = True,
        accepts_files: bool = True,
        accepts_audio: bool = False,
        max_files: Optional[int] = None,
        file_types: Optional[list[str]] = None,
        **extra: Any,
    ):
        # Build enhanced tags based on capabilities
        enhanced_tags = list(tags) if tags else []

        if use_case:
            enhanced_tags.append(use_case)
        if accepts_images:
            enhanced_tags.extend(["vision", "image-input"])
        if accepts_audio:
            enhanced_tags.extend(["audio-input", "speech"])
        if accepts_files:
            enhanced_tags.append("document-input")

        # Infer use case specific tags
        use_case_tags = {
            "chat": ["conversational", "llm"],
            "document_qa": ["rag", "document", "qa"],
            "image_analysis": ["vision", "analysis", "vlm"],
            "code_review": ["code", "analysis"],
        }
        if use_case and use_case in use_case_tags:
            enhanced_tags.extend(use_case_tags[use_case])

        # Build enhanced intent if not provided
        if intent is None:
            parts = ["multimodal input"]
            if accepts_images:
                parts.append("with image support")
            if accepts_files:
                parts.append("with file upload")
            if use_case:
                parts.append(f"for {use_case.replace('_', ' ')}")
            intent = " ".join(parts)

        super().__init__(component, intent=intent, tags=enhanced_tags, **extra)

        # Store extended metadata
        self.multimodal_meta = MultimodalMetadata(
            accepts_images=accepts_images,
            accepts_files=accepts_files,
            accepts_audio=accepts_audio,
            max_files=max_files,
            file_types=file_types or [],
            use_case=use_case,
        )


class SemanticImageEditor(SemanticComponent):
    """
    Specialized wrapper for ImageEditor with enhanced metadata.

    Provides rich semantic understanding for image editing operations.

    Example:
        editor = SemanticImageEditor(
            gr.ImageEditor(label="Edit Image"),
            intent="user edits images for inpainting",
            use_case="inpainting",
            tools=["brush", "eraser"],
            supports_masks=True,
        )
    """

    def __init__(
        self,
        component: Any,
        intent: Optional[str] = None,
        tags: Optional[list[str]] = None,
        use_case: Optional[str] = None,
        tools: Optional[list[str]] = None,
        supports_layers: bool = False,
        supports_masks: bool = False,
        output_format: str = "png",
        **extra: Any,
    ):
        enhanced_tags = list(tags) if tags else []
        tools = tools or ["brush", "eraser"]

        # Add tool-specific tags
        for tool in tools:
            enhanced_tags.append(f"tool-{tool}")

        if supports_masks:
            enhanced_tags.extend(["masking", "segmentation-input"])
        if supports_layers:
            enhanced_tags.append("layered")

        # Use case specific tags
        use_case_tags = {
            "inpainting": ["generative", "inpaint", "fill"],
            "annotation": ["labeling", "bbox", "markup"],
            "segmentation": ["mask", "segment", "region"],
            "photo_editing": ["filter", "adjust", "enhance"],
        }
        if use_case:
            enhanced_tags.append(use_case)
            if use_case in use_case_tags:
                enhanced_tags.extend(use_case_tags[use_case])

        if intent is None:
            intent = f"image editor for {use_case or 'editing'}"
            if supports_masks:
                intent += " with mask support"

        super().__init__(component, intent=intent, tags=enhanced_tags, **extra)

        self.editor_meta = ImageEditorMetadata(
            tools=tools,
            supports_layers=supports_layers,
            supports_masks=supports_masks,
            output_format=output_format,
            use_case=use_case,
        )


class SemanticAnnotatedImage(SemanticComponent):
    """
    Specialized wrapper for AnnotatedImage with enhanced metadata.

    Provides rich semantic understanding for object detection and segmentation outputs.

    Example:
        output = SemanticAnnotatedImage(
            gr.AnnotatedImage(label="Detections"),
            intent="displays object detection results",
            annotation_type="bbox",
            entity_types=["person", "car", "dog"],
        )
    """

    def __init__(
        self,
        component: Any,
        intent: Optional[str] = None,
        tags: Optional[list[str]] = None,
        annotation_type: str = "bbox",
        entity_types: Optional[list[str]] = None,
        color_map: Optional[dict[str, str]] = None,
        supports_overlapping: bool = False,
        **extra: Any,
    ):
        enhanced_tags = list(tags) if tags else []
        entity_types = entity_types or []

        # Add annotation type tags
        type_tags = {
            "bbox": ["bounding-box", "detection", "localization"],
            "segmentation": ["mask", "pixel-wise", "semantic"],
            "polygon": ["polygon", "region", "outline"],
            "keypoint": ["pose", "landmarks", "points"],
        }
        enhanced_tags.append(annotation_type)
        if annotation_type in type_tags:
            enhanced_tags.extend(type_tags[annotation_type])

        # Add entity type tags
        for entity in entity_types[:5]:  # Limit to avoid tag explosion
            enhanced_tags.append(f"detects-{entity.lower()}")

        if intent is None:
            intent = f"displays {annotation_type} annotations"
            if entity_types:
                intent += f" for {', '.join(entity_types[:3])}"

        super().__init__(component, intent=intent, tags=enhanced_tags, **extra)

        self.annotation_meta = AnnotationMetadata(
            annotation_type=annotation_type,
            entity_types=entity_types,
            color_map=color_map or {},
            supports_overlapping=supports_overlapping,
        )


class SemanticHighlightedText(SemanticComponent):
    """
    Specialized wrapper for HighlightedText with enhanced NLP metadata.

    Provides rich semantic understanding for NER and text annotation outputs.

    Example:
        output = SemanticHighlightedText(
            gr.HighlightedText(label="Entities"),
            intent="displays named entity recognition results",
            annotation_type="ner",
            entity_types=["PERSON", "ORG", "LOC", "DATE"],
        )
    """

    def __init__(
        self,
        component: Any,
        intent: Optional[str] = None,
        tags: Optional[list[str]] = None,
        annotation_type: str = "ner",
        entity_types: Optional[list[str]] = None,
        color_map: Optional[dict[str, str]] = None,
        **extra: Any,
    ):
        enhanced_tags = list(tags) if tags else []
        entity_types = entity_types or []

        # Add NLP-specific tags
        nlp_tags = {
            "ner": ["named-entity", "entity-recognition", "extraction"],
            "pos": ["part-of-speech", "grammar", "syntax"],
            "sentiment": ["sentiment", "opinion", "emotion"],
            "classification": ["label", "category", "class"],
            "highlight": ["emphasis", "important", "key-phrases"],
        }
        enhanced_tags.append(annotation_type)
        if annotation_type in nlp_tags:
            enhanced_tags.extend(nlp_tags[annotation_type])

        # Add common NER entity tags
        ner_entity_mapping = {
            "PERSON": "person-entity",
            "PER": "person-entity",
            "ORG": "organization-entity",
            "LOC": "location-entity",
            "GPE": "geo-entity",
            "DATE": "date-entity",
            "TIME": "time-entity",
            "MONEY": "money-entity",
            "PRODUCT": "product-entity",
        }
        for entity in entity_types:
            if entity.upper() in ner_entity_mapping:
                enhanced_tags.append(ner_entity_mapping[entity.upper()])

        if intent is None:
            intent = f"displays {annotation_type} text annotations"
            if entity_types:
                intent += f" ({', '.join(entity_types[:4])})"

        super().__init__(component, intent=intent, tags=enhanced_tags, **extra)

        self.annotation_meta = AnnotationMetadata(
            annotation_type=annotation_type,
            entity_types=entity_types,
            color_map=color_map or {},
        )


class SemanticChatbot(SemanticComponent):
    """
    Specialized wrapper for Chatbot with enhanced conversation metadata.

    Provides rich semantic understanding for AI chat interfaces.

    Example:
        chat = SemanticChatbot(
            gr.Chatbot(label="Assistant"),
            intent="AI assistant conversation display",
            supports_streaming=True,
            supports_like=True,
            message_format="markdown",
        )
    """

    def __init__(
        self,
        component: Any,
        intent: Optional[str] = None,
        tags: Optional[list[str]] = None,
        supports_streaming: bool = True,
        supports_retry: bool = True,
        supports_undo: bool = True,
        supports_like: bool = False,
        message_format: str = "markdown",
        persona: Optional[str] = None,
        **extra: Any,
    ):
        enhanced_tags = list(tags) if tags else []

        if supports_streaming:
            enhanced_tags.append("streaming")
        if supports_retry:
            enhanced_tags.append("retry-capable")
        if supports_like:
            enhanced_tags.extend(["feedback", "likeable"])
        if message_format == "markdown":
            enhanced_tags.append("markdown-output")

        # Persona-based tags
        persona_tags = {
            "assistant": ["helpful", "general-purpose"],
            "coder": ["code-assistant", "programming"],
            "tutor": ["educational", "explanatory"],
            "creative": ["creative-writing", "storytelling"],
            "analyst": ["data-analysis", "reasoning"],
        }
        if persona:
            enhanced_tags.append(f"persona-{persona}")
            if persona in persona_tags:
                enhanced_tags.extend(persona_tags[persona])

        if intent is None:
            intent = "AI conversation display"
            if persona:
                intent = f"{persona} AI conversation display"
            if supports_streaming:
                intent += " with streaming"

        super().__init__(component, intent=intent, tags=enhanced_tags, **extra)

        self.chat_meta = ChatMetadata(
            supports_streaming=supports_streaming,
            supports_retry=supports_retry,
            supports_undo=supports_undo,
            supports_like=supports_like,
            message_format=message_format,
        )


class SemanticPlot(SemanticComponent):
    """
    Specialized wrapper for Plot/Chart components with enhanced visualization metadata.

    Provides rich semantic understanding for data visualizations.

    Example:
        chart = SemanticPlot(
            gr.LinePlot(x="date", y="value"),
            intent="displays time series metrics",
            chart_type="line",
            axes=["date", "value"],
            data_domain="metrics",
        )
    """

    def __init__(
        self,
        component: Any,
        intent: Optional[str] = None,
        tags: Optional[list[str]] = None,
        chart_type: Optional[str] = None,
        axes: Optional[list[str]] = None,
        interactive: bool = True,
        supports_zoom: bool = False,
        supports_pan: bool = False,
        data_domain: Optional[str] = None,
        **extra: Any,
    ):
        enhanced_tags = list(tags) if tags else []
        axes = axes or []

        # Infer chart type from component if not provided
        if chart_type is None:
            comp_name = type(component).__name__.lower()
            if "line" in comp_name:
                chart_type = "line"
            elif "bar" in comp_name:
                chart_type = "bar"
            elif "scatter" in comp_name:
                chart_type = "scatter"
            else:
                chart_type = "generic"

        # Chart type tags
        chart_tags = {
            "line": ["timeseries", "trend", "continuous"],
            "bar": ["categorical", "comparison", "discrete"],
            "scatter": ["correlation", "distribution", "points"],
            "pie": ["proportion", "percentage", "parts"],
            "heatmap": ["matrix", "intensity", "grid"],
            "histogram": ["distribution", "frequency", "binned"],
        }
        enhanced_tags.append(f"chart-{chart_type}")
        if chart_type in chart_tags:
            enhanced_tags.extend(chart_tags[chart_type])

        if interactive:
            enhanced_tags.append("interactive-viz")
        if supports_zoom:
            enhanced_tags.append("zoomable")

        # Data domain tags
        if data_domain:
            enhanced_tags.append(f"domain-{data_domain}")

        if intent is None:
            intent = f"{chart_type} chart visualization"
            if data_domain:
                intent = f"{data_domain} {chart_type} chart"
            if axes:
                intent += f" ({' vs '.join(axes[:2])})"

        super().__init__(component, intent=intent, tags=enhanced_tags, **extra)

        self.viz_meta = VisualizationMetadata(
            chart_type=chart_type,
            interactive=interactive,
            supports_zoom=supports_zoom,
            supports_pan=supports_pan,
            axes=axes,
        )


class SemanticModel3D(SemanticComponent):
    """
    Specialized wrapper for Model3D with enhanced 3D metadata.

    Provides rich semantic understanding for 3D model viewers.

    Example:
        viewer = SemanticModel3D(
            gr.Model3D(label="3D Model"),
            intent="displays generated 3D mesh",
            use_case="mesh_generation",
            supports_animation=True,
        )
    """

    def __init__(
        self,
        component: Any,
        intent: Optional[str] = None,
        tags: Optional[list[str]] = None,
        use_case: Optional[str] = None,
        supported_formats: Optional[list[str]] = None,
        supports_animation: bool = False,
        supports_textures: bool = True,
        camera_controls: bool = True,
        **extra: Any,
    ):
        enhanced_tags = list(tags) if tags else []
        supported_formats = supported_formats or ["obj", "glb", "gltf"]

        # Format tags
        for fmt in supported_formats:
            enhanced_tags.append(f"format-{fmt}")

        if supports_animation:
            enhanced_tags.extend(["animated", "rigged"])
        if supports_textures:
            enhanced_tags.append("textured")
        if camera_controls:
            enhanced_tags.append("orbitable")

        # Use case tags
        use_case_tags = {
            "mesh_generation": ["generative", "ai-generated"],
            "cad_viewer": ["engineering", "technical"],
            "game_asset": ["game-dev", "asset"],
            "medical": ["medical-imaging", "anatomical"],
            "architectural": ["architecture", "building"],
        }
        if use_case:
            enhanced_tags.append(use_case)
            if use_case in use_case_tags:
                enhanced_tags.extend(use_case_tags[use_case])

        if intent is None:
            intent = "3D model viewer"
            if use_case:
                intent = f"3D {use_case.replace('_', ' ')} viewer"

        super().__init__(component, intent=intent, tags=enhanced_tags, **extra)

        self.model3d_meta = Model3DMetadata(
            supported_formats=supported_formats,
            supports_animation=supports_animation,
            supports_textures=supports_textures,
            camera_controls=camera_controls,
        )


class SemanticDataFrame(SemanticComponent):
    """
    Specialized wrapper for DataFrame with enhanced data metadata.

    Provides rich semantic understanding for tabular data components.

    Example:
        table = SemanticDataFrame(
            gr.DataFrame(label="Results"),
            intent="displays query results",
            data_domain="database",
            editable=True,
            columns=["id", "name", "value"],
        )
    """

    def __init__(
        self,
        component: Any,
        intent: Optional[str] = None,
        tags: Optional[list[str]] = None,
        data_domain: Optional[str] = None,
        editable: bool = False,
        columns: Optional[list[str]] = None,
        row_count: Optional[int] = None,
        **extra: Any,
    ):
        enhanced_tags = list(tags) if tags else []
        columns = columns or []

        if editable:
            enhanced_tags.extend(["editable", "interactive-data"])

        # Data domain tags
        domain_tags = {
            "database": ["sql", "query-results"],
            "spreadsheet": ["excel", "csv"],
            "metrics": ["kpi", "measurements"],
            "logs": ["logging", "events"],
            "inventory": ["stock", "items"],
            "users": ["accounts", "profiles"],
        }
        if data_domain:
            enhanced_tags.append(f"domain-{data_domain}")
            if data_domain in domain_tags:
                enhanced_tags.extend(domain_tags[data_domain])

        # Column-based inference
        column_hints = {
            "date": "temporal",
            "time": "temporal",
            "timestamp": "temporal",
            "price": "financial",
            "amount": "financial",
            "cost": "financial",
            "name": "entity",
            "email": "contact",
            "id": "identifier",
            "status": "categorical",
        }
        for col in columns:
            col_lower = col.lower()
            for hint, tag in column_hints.items():
                if hint in col_lower:
                    enhanced_tags.append(f"has-{tag}")
                    break

        if intent is None:
            intent = "tabular data display"
            if data_domain:
                intent = f"{data_domain} data table"
            if editable:
                intent += " (editable)"

        super().__init__(component, intent=intent, tags=enhanced_tags, **extra)

        self._data_domain = data_domain
        self._columns = columns
        self._editable = editable


class SemanticFileExplorer(SemanticComponent):
    """
    Specialized wrapper for FileExplorer with enhanced filesystem metadata.

    Provides rich semantic understanding for file navigation components.

    Example:
        explorer = SemanticFileExplorer(
            gr.FileExplorer(label="Project Files"),
            intent="navigate project source files",
            root_type="code_project",
            file_types=[".py", ".js", ".ts"],
        )
    """

    def __init__(
        self,
        component: Any,
        intent: Optional[str] = None,
        tags: Optional[list[str]] = None,
        root_type: Optional[str] = None,
        file_types: Optional[list[str]] = None,
        **extra: Any,
    ):
        enhanced_tags = list(tags) if tags else []
        file_types = file_types or []

        # Root type tags
        root_tags = {
            "code_project": ["source-code", "repository"],
            "documents": ["docs", "files"],
            "media": ["images", "videos", "assets"],
            "data": ["datasets", "csv", "json"],
            "config": ["settings", "configuration"],
        }
        if root_type:
            enhanced_tags.append(root_type)
            if root_type in root_tags:
                enhanced_tags.extend(root_tags[root_type])

        # File type inference
        code_extensions = {".py", ".js", ".ts", ".java", ".cpp", ".go", ".rs"}
        doc_extensions = {".md", ".txt", ".pdf", ".docx"}
        data_extensions = {".csv", ".json", ".xml", ".yaml"}

        file_types_set = set(file_types)
        if file_types_set & code_extensions:
            enhanced_tags.append("code-files")
        if file_types_set & doc_extensions:
            enhanced_tags.append("document-files")
        if file_types_set & data_extensions:
            enhanced_tags.append("data-files")

        if intent is None:
            intent = "file explorer"
            if root_type:
                intent = f"{root_type.replace('_', ' ')} file explorer"

        super().__init__(component, intent=intent, tags=enhanced_tags, **extra)

        self._root_type = root_type
        self._file_types = file_types


# =============================================================================
# Factory Functions for Convenient Usage
# =============================================================================

def semantic_multimodal(
    component: Any,
    intent: Optional[str] = None,
    use_case: Optional[str] = None,
    **kwargs: Any,
) -> SemanticMultimodal:
    """Create a SemanticMultimodal wrapper. See SemanticMultimodal for full options."""
    return SemanticMultimodal(component, intent=intent, use_case=use_case, **kwargs)


def semantic_image_editor(
    component: Any,
    intent: Optional[str] = None,
    use_case: Optional[str] = None,
    **kwargs: Any,
) -> SemanticImageEditor:
    """Create a SemanticImageEditor wrapper. See SemanticImageEditor for full options."""
    return SemanticImageEditor(component, intent=intent, use_case=use_case, **kwargs)


def semantic_annotated_image(
    component: Any,
    intent: Optional[str] = None,
    annotation_type: str = "bbox",
    **kwargs: Any,
) -> SemanticAnnotatedImage:
    """Create a SemanticAnnotatedImage wrapper. See SemanticAnnotatedImage for full options."""
    return SemanticAnnotatedImage(component, intent=intent, annotation_type=annotation_type, **kwargs)


def semantic_highlighted_text(
    component: Any,
    intent: Optional[str] = None,
    annotation_type: str = "ner",
    **kwargs: Any,
) -> SemanticHighlightedText:
    """Create a SemanticHighlightedText wrapper. See SemanticHighlightedText for full options."""
    return SemanticHighlightedText(component, intent=intent, annotation_type=annotation_type, **kwargs)


def semantic_chatbot(
    component: Any,
    intent: Optional[str] = None,
    persona: Optional[str] = None,
    **kwargs: Any,
) -> SemanticChatbot:
    """Create a SemanticChatbot wrapper. See SemanticChatbot for full options."""
    return SemanticChatbot(component, intent=intent, persona=persona, **kwargs)


def semantic_plot(
    component: Any,
    intent: Optional[str] = None,
    chart_type: Optional[str] = None,
    **kwargs: Any,
) -> SemanticPlot:
    """Create a SemanticPlot wrapper. See SemanticPlot for full options."""
    return SemanticPlot(component, intent=intent, chart_type=chart_type, **kwargs)


def semantic_model3d(
    component: Any,
    intent: Optional[str] = None,
    use_case: Optional[str] = None,
    **kwargs: Any,
) -> SemanticModel3D:
    """Create a SemanticModel3D wrapper. See SemanticModel3D for full options."""
    return SemanticModel3D(component, intent=intent, use_case=use_case, **kwargs)


def semantic_dataframe(
    component: Any,
    intent: Optional[str] = None,
    data_domain: Optional[str] = None,
    **kwargs: Any,
) -> SemanticDataFrame:
    """Create a SemanticDataFrame wrapper. See SemanticDataFrame for full options."""
    return SemanticDataFrame(component, intent=intent, data_domain=data_domain, **kwargs)


def semantic_file_explorer(
    component: Any,
    intent: Optional[str] = None,
    root_type: Optional[str] = None,
    **kwargs: Any,
) -> SemanticFileExplorer:
    """Create a SemanticFileExplorer wrapper. See SemanticFileExplorer for full options."""
    return SemanticFileExplorer(component, intent=intent, root_type=root_type, **kwargs)
