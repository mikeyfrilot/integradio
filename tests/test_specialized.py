"""
Tests for specialized semantic wrappers - Enhanced semantic metadata for complex Gradio components.

Tests cover:
- 6 metadata dataclasses
- 9 specialized wrapper classes
- 9 factory functions
- Tag inference, intent building, use case mapping
"""

import pytest
from unittest.mock import MagicMock, patch


# =============================================================================
# Metadata Dataclass Tests
# =============================================================================


class TestMultimodalMetadata:
    """Tests for MultimodalMetadata dataclass."""

    def test_default_values(self):
        """Default values are set correctly."""
        from integradio.specialized import MultimodalMetadata

        meta = MultimodalMetadata()

        assert meta.accepts_images is True
        assert meta.accepts_files is True
        assert meta.accepts_audio is False
        assert meta.max_files is None
        assert meta.file_types == []
        assert meta.use_case is None

    def test_custom_values(self):
        """Custom values override defaults."""
        from integradio.specialized import MultimodalMetadata

        meta = MultimodalMetadata(
            accepts_images=False,
            accepts_files=False,
            accepts_audio=True,
            max_files=5,
            file_types=[".mp3", ".wav"],
            use_case="audio_qa",
        )

        assert meta.accepts_images is False
        assert meta.accepts_audio is True
        assert meta.max_files == 5
        assert meta.file_types == [".mp3", ".wav"]
        assert meta.use_case == "audio_qa"


class TestImageEditorMetadata:
    """Tests for ImageEditorMetadata dataclass."""

    def test_default_values(self):
        """Default values are set correctly."""
        from integradio.specialized import ImageEditorMetadata

        meta = ImageEditorMetadata()

        assert meta.tools == []
        assert meta.supports_layers is False
        assert meta.supports_masks is False
        assert meta.output_format == "png"
        assert meta.use_case is None

    def test_custom_values(self):
        """Custom values override defaults."""
        from integradio.specialized import ImageEditorMetadata

        meta = ImageEditorMetadata(
            tools=["brush", "eraser", "crop"],
            supports_layers=True,
            supports_masks=True,
            output_format="webp",
            use_case="inpainting",
        )

        assert meta.tools == ["brush", "eraser", "crop"]
        assert meta.supports_layers is True
        assert meta.supports_masks is True
        assert meta.output_format == "webp"
        assert meta.use_case == "inpainting"


class TestAnnotationMetadata:
    """Tests for AnnotationMetadata dataclass."""

    def test_default_values(self):
        """Default values are set correctly."""
        from integradio.specialized import AnnotationMetadata

        meta = AnnotationMetadata()

        assert meta.annotation_type == "generic"
        assert meta.entity_types == []
        assert meta.color_map == {}
        assert meta.supports_overlapping is False

    def test_custom_values(self):
        """Custom values override defaults."""
        from integradio.specialized import AnnotationMetadata

        meta = AnnotationMetadata(
            annotation_type="ner",
            entity_types=["PERSON", "ORG"],
            color_map={"PERSON": "blue", "ORG": "green"},
            supports_overlapping=True,
        )

        assert meta.annotation_type == "ner"
        assert meta.entity_types == ["PERSON", "ORG"]
        assert meta.color_map == {"PERSON": "blue", "ORG": "green"}
        assert meta.supports_overlapping is True


class TestChatMetadata:
    """Tests for ChatMetadata dataclass."""

    def test_default_values(self):
        """Default values are set correctly."""
        from integradio.specialized import ChatMetadata

        meta = ChatMetadata()

        assert meta.supports_streaming is True
        assert meta.supports_retry is True
        assert meta.supports_undo is True
        assert meta.supports_like is False
        assert meta.avatar_style is None
        assert meta.message_format == "markdown"

    def test_custom_values(self):
        """Custom values override defaults."""
        from integradio.specialized import ChatMetadata

        meta = ChatMetadata(
            supports_streaming=False,
            supports_retry=False,
            supports_like=True,
            avatar_style="circle",
            message_format="html",
        )

        assert meta.supports_streaming is False
        assert meta.supports_like is True
        assert meta.avatar_style == "circle"
        assert meta.message_format == "html"


class TestVisualizationMetadata:
    """Tests for VisualizationMetadata dataclass."""

    def test_default_values(self):
        """Default values are set correctly."""
        from integradio.specialized import VisualizationMetadata

        meta = VisualizationMetadata()

        assert meta.chart_type == "generic"
        assert meta.interactive is True
        assert meta.supports_zoom is False
        assert meta.supports_pan is False
        assert meta.axes == []
        assert meta.data_format == "pandas"

    def test_custom_values(self):
        """Custom values override defaults."""
        from integradio.specialized import VisualizationMetadata

        meta = VisualizationMetadata(
            chart_type="scatter",
            interactive=True,
            supports_zoom=True,
            supports_pan=True,
            axes=["x", "y", "color"],
            data_format="dict",
        )

        assert meta.chart_type == "scatter"
        assert meta.supports_zoom is True
        assert meta.axes == ["x", "y", "color"]
        assert meta.data_format == "dict"


class TestModel3DMetadata:
    """Tests for Model3DMetadata dataclass."""

    def test_default_values(self):
        """Default values are set correctly."""
        from integradio.specialized import Model3DMetadata

        meta = Model3DMetadata()

        assert meta.supported_formats == ["obj", "glb", "gltf"]
        assert meta.supports_animation is False
        assert meta.supports_textures is True
        assert meta.camera_controls is True
        assert meta.lighting == "default"

    def test_custom_values(self):
        """Custom values override defaults."""
        from integradio.specialized import Model3DMetadata

        meta = Model3DMetadata(
            supported_formats=["fbx", "obj"],
            supports_animation=True,
            supports_textures=False,
            camera_controls=False,
            lighting="studio",
        )

        assert meta.supported_formats == ["fbx", "obj"]
        assert meta.supports_animation is True
        assert meta.supports_textures is False
        assert meta.lighting == "studio"


# =============================================================================
# SemanticMultimodal Tests
# =============================================================================


class TestSemanticMultimodal:
    """Tests for SemanticMultimodal specialized wrapper."""

    def test_basic_creation(self):
        """Creates wrapper with default settings."""
        from integradio.specialized import SemanticMultimodal

        mock_component = MagicMock()
        mock_component._id = 1

        with patch("integradio.components.infer_tags", return_value=["input", "text"]):
            wrapper = SemanticMultimodal(mock_component, intent="test input")

        assert wrapper.component is mock_component
        assert wrapper.intent == "test input"
        assert wrapper.multimodal_meta.accepts_images is True
        assert wrapper.multimodal_meta.accepts_files is True

    def test_image_tags_added(self):
        """Vision tags added when accepts_images=True."""
        from integradio.specialized import SemanticMultimodal

        mock_component = MagicMock()
        mock_component._id = 2

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticMultimodal(mock_component, accepts_images=True)

        assert "vision" in wrapper.semantic_meta.tags
        assert "image-input" in wrapper.semantic_meta.tags

    def test_audio_tags_added(self):
        """Audio tags added when accepts_audio=True."""
        from integradio.specialized import SemanticMultimodal

        mock_component = MagicMock()
        mock_component._id = 3

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticMultimodal(mock_component, accepts_audio=True)

        assert "audio-input" in wrapper.semantic_meta.tags
        assert "speech" in wrapper.semantic_meta.tags

    def test_document_tags_added(self):
        """Document tags added when accepts_files=True."""
        from integradio.specialized import SemanticMultimodal

        mock_component = MagicMock()
        mock_component._id = 4

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticMultimodal(mock_component, accepts_files=True)

        assert "document-input" in wrapper.semantic_meta.tags

    def test_use_case_tags_chat(self):
        """Chat use case adds conversational and llm tags."""
        from integradio.specialized import SemanticMultimodal

        mock_component = MagicMock()
        mock_component._id = 5

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticMultimodal(mock_component, use_case="chat")

        assert "chat" in wrapper.semantic_meta.tags
        assert "conversational" in wrapper.semantic_meta.tags
        assert "llm" in wrapper.semantic_meta.tags

    def test_use_case_tags_document_qa(self):
        """Document QA use case adds rag and qa tags."""
        from integradio.specialized import SemanticMultimodal

        mock_component = MagicMock()
        mock_component._id = 6

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticMultimodal(mock_component, use_case="document_qa")

        assert "document_qa" in wrapper.semantic_meta.tags
        assert "rag" in wrapper.semantic_meta.tags
        assert "document" in wrapper.semantic_meta.tags
        assert "qa" in wrapper.semantic_meta.tags

    def test_use_case_tags_image_analysis(self):
        """Image analysis use case adds vision and vlm tags."""
        from integradio.specialized import SemanticMultimodal

        mock_component = MagicMock()
        mock_component._id = 7

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticMultimodal(mock_component, use_case="image_analysis")

        assert "image_analysis" in wrapper.semantic_meta.tags
        assert "vlm" in wrapper.semantic_meta.tags

    def test_auto_intent_generation(self):
        """Intent auto-generated from capabilities when not provided."""
        from integradio.specialized import SemanticMultimodal

        mock_component = MagicMock()
        mock_component._id = 8

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticMultimodal(
                mock_component,
                accepts_images=True,
                accepts_files=True,
                use_case="image_analysis",
            )

        assert "multimodal input" in wrapper.intent
        assert "image support" in wrapper.intent
        assert "file upload" in wrapper.intent
        assert "image analysis" in wrapper.intent

    def test_metadata_stored(self):
        """MultimodalMetadata is properly stored."""
        from integradio.specialized import SemanticMultimodal

        mock_component = MagicMock()
        mock_component._id = 9

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticMultimodal(
                mock_component,
                use_case="chat",
                accepts_audio=True,
                max_files=10,
                file_types=[".pdf", ".txt"],
            )

        assert wrapper.multimodal_meta.use_case == "chat"
        assert wrapper.multimodal_meta.accepts_audio is True
        assert wrapper.multimodal_meta.max_files == 10
        assert wrapper.multimodal_meta.file_types == [".pdf", ".txt"]


# =============================================================================
# SemanticImageEditor Tests
# =============================================================================


class TestSemanticImageEditor:
    """Tests for SemanticImageEditor specialized wrapper."""

    def test_basic_creation(self):
        """Creates wrapper with default settings."""
        from integradio.specialized import SemanticImageEditor

        mock_component = MagicMock()
        mock_component._id = 10

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticImageEditor(mock_component, intent="edit image")

        assert wrapper.component is mock_component
        assert wrapper.intent == "edit image"

    def test_default_tools(self):
        """Default tools are brush and eraser."""
        from integradio.specialized import SemanticImageEditor

        mock_component = MagicMock()
        mock_component._id = 11

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticImageEditor(mock_component)

        assert wrapper.editor_meta.tools == ["brush", "eraser"]
        assert "tool-brush" in wrapper.semantic_meta.tags
        assert "tool-eraser" in wrapper.semantic_meta.tags

    def test_custom_tools(self):
        """Custom tools are added as tags."""
        from integradio.specialized import SemanticImageEditor

        mock_component = MagicMock()
        mock_component._id = 12

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticImageEditor(
                mock_component,
                tools=["brush", "crop", "layers"],
            )

        assert "tool-brush" in wrapper.semantic_meta.tags
        assert "tool-crop" in wrapper.semantic_meta.tags
        assert "tool-layers" in wrapper.semantic_meta.tags

    def test_mask_support_tags(self):
        """Mask support adds masking and segmentation-input tags."""
        from integradio.specialized import SemanticImageEditor

        mock_component = MagicMock()
        mock_component._id = 13

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticImageEditor(mock_component, supports_masks=True)

        assert "masking" in wrapper.semantic_meta.tags
        assert "segmentation-input" in wrapper.semantic_meta.tags

    def test_layer_support_tags(self):
        """Layer support adds layered tag."""
        from integradio.specialized import SemanticImageEditor

        mock_component = MagicMock()
        mock_component._id = 14

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticImageEditor(mock_component, supports_layers=True)

        assert "layered" in wrapper.semantic_meta.tags

    def test_inpainting_use_case(self):
        """Inpainting use case adds generative and fill tags."""
        from integradio.specialized import SemanticImageEditor

        mock_component = MagicMock()
        mock_component._id = 15

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticImageEditor(mock_component, use_case="inpainting")

        assert "inpainting" in wrapper.semantic_meta.tags
        assert "generative" in wrapper.semantic_meta.tags
        assert "inpaint" in wrapper.semantic_meta.tags
        assert "fill" in wrapper.semantic_meta.tags

    def test_annotation_use_case(self):
        """Annotation use case adds labeling tags."""
        from integradio.specialized import SemanticImageEditor

        mock_component = MagicMock()
        mock_component._id = 16

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticImageEditor(mock_component, use_case="annotation")

        assert "annotation" in wrapper.semantic_meta.tags
        assert "labeling" in wrapper.semantic_meta.tags
        assert "bbox" in wrapper.semantic_meta.tags

    def test_auto_intent_generation(self):
        """Intent auto-generated with use case and mask info."""
        from integradio.specialized import SemanticImageEditor

        mock_component = MagicMock()
        mock_component._id = 17

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticImageEditor(
                mock_component,
                use_case="inpainting",
                supports_masks=True,
            )

        assert "inpainting" in wrapper.intent
        assert "mask support" in wrapper.intent

    def test_metadata_stored(self):
        """ImageEditorMetadata is properly stored."""
        from integradio.specialized import SemanticImageEditor

        mock_component = MagicMock()
        mock_component._id = 18

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticImageEditor(
                mock_component,
                tools=["brush"],
                supports_layers=True,
                output_format="webp",
                use_case="photo_editing",
            )

        assert wrapper.editor_meta.tools == ["brush"]
        assert wrapper.editor_meta.supports_layers is True
        assert wrapper.editor_meta.output_format == "webp"
        assert wrapper.editor_meta.use_case == "photo_editing"


# =============================================================================
# SemanticAnnotatedImage Tests
# =============================================================================


class TestSemanticAnnotatedImage:
    """Tests for SemanticAnnotatedImage specialized wrapper."""

    def test_basic_creation(self):
        """Creates wrapper with default settings."""
        from integradio.specialized import SemanticAnnotatedImage

        mock_component = MagicMock()
        mock_component._id = 20

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticAnnotatedImage(mock_component, intent="show detections")

        assert wrapper.component is mock_component
        assert "bbox" in wrapper.semantic_meta.tags

    def test_bbox_annotation_type(self):
        """Bbox type adds detection and localization tags."""
        from integradio.specialized import SemanticAnnotatedImage

        mock_component = MagicMock()
        mock_component._id = 21

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticAnnotatedImage(mock_component, annotation_type="bbox")

        assert "bbox" in wrapper.semantic_meta.tags
        assert "bounding-box" in wrapper.semantic_meta.tags
        assert "detection" in wrapper.semantic_meta.tags
        assert "localization" in wrapper.semantic_meta.tags

    def test_segmentation_annotation_type(self):
        """Segmentation type adds mask and semantic tags."""
        from integradio.specialized import SemanticAnnotatedImage

        mock_component = MagicMock()
        mock_component._id = 22

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticAnnotatedImage(mock_component, annotation_type="segmentation")

        assert "segmentation" in wrapper.semantic_meta.tags
        assert "mask" in wrapper.semantic_meta.tags
        assert "pixel-wise" in wrapper.semantic_meta.tags

    def test_keypoint_annotation_type(self):
        """Keypoint type adds pose and landmarks tags."""
        from integradio.specialized import SemanticAnnotatedImage

        mock_component = MagicMock()
        mock_component._id = 23

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticAnnotatedImage(mock_component, annotation_type="keypoint")

        assert "keypoint" in wrapper.semantic_meta.tags
        assert "pose" in wrapper.semantic_meta.tags
        assert "landmarks" in wrapper.semantic_meta.tags

    def test_entity_type_tags(self):
        """Entity types add detects-* tags."""
        from integradio.specialized import SemanticAnnotatedImage

        mock_component = MagicMock()
        mock_component._id = 24

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticAnnotatedImage(
                mock_component,
                entity_types=["person", "car", "dog"],
            )

        assert "detects-person" in wrapper.semantic_meta.tags
        assert "detects-car" in wrapper.semantic_meta.tags
        assert "detects-dog" in wrapper.semantic_meta.tags

    def test_entity_type_limit(self):
        """Only first 5 entity types are added as tags."""
        from integradio.specialized import SemanticAnnotatedImage

        mock_component = MagicMock()
        mock_component._id = 25

        entities = ["a", "b", "c", "d", "e", "f", "g"]
        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticAnnotatedImage(mock_component, entity_types=entities)

        # First 5 should be tags
        assert "detects-a" in wrapper.semantic_meta.tags
        assert "detects-e" in wrapper.semantic_meta.tags
        # 6th and beyond should not
        assert "detects-f" not in wrapper.semantic_meta.tags

    def test_auto_intent_generation(self):
        """Intent auto-generated with annotation type and entities."""
        from integradio.specialized import SemanticAnnotatedImage

        mock_component = MagicMock()
        mock_component._id = 26

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticAnnotatedImage(
                mock_component,
                annotation_type="bbox",
                entity_types=["person", "vehicle", "animal"],
            )

        assert "bbox" in wrapper.intent
        assert "person" in wrapper.intent

    def test_metadata_stored(self):
        """AnnotationMetadata is properly stored."""
        from integradio.specialized import SemanticAnnotatedImage

        mock_component = MagicMock()
        mock_component._id = 27

        color_map = {"person": "red", "car": "blue"}
        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticAnnotatedImage(
                mock_component,
                annotation_type="polygon",
                entity_types=["person", "car"],
                color_map=color_map,
                supports_overlapping=True,
            )

        assert wrapper.annotation_meta.annotation_type == "polygon"
        assert wrapper.annotation_meta.entity_types == ["person", "car"]
        assert wrapper.annotation_meta.color_map == color_map
        assert wrapper.annotation_meta.supports_overlapping is True


# =============================================================================
# SemanticHighlightedText Tests
# =============================================================================


class TestSemanticHighlightedText:
    """Tests for SemanticHighlightedText specialized wrapper."""

    def test_basic_creation(self):
        """Creates wrapper with default NER type."""
        from integradio.specialized import SemanticHighlightedText

        mock_component = MagicMock()
        mock_component._id = 30

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticHighlightedText(mock_component, intent="show entities")

        assert wrapper.component is mock_component
        assert "ner" in wrapper.semantic_meta.tags

    def test_ner_annotation_type(self):
        """NER type adds entity recognition tags."""
        from integradio.specialized import SemanticHighlightedText

        mock_component = MagicMock()
        mock_component._id = 31

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticHighlightedText(mock_component, annotation_type="ner")

        assert "ner" in wrapper.semantic_meta.tags
        assert "named-entity" in wrapper.semantic_meta.tags
        assert "entity-recognition" in wrapper.semantic_meta.tags
        assert "extraction" in wrapper.semantic_meta.tags

    def test_pos_annotation_type(self):
        """POS type adds grammar and syntax tags."""
        from integradio.specialized import SemanticHighlightedText

        mock_component = MagicMock()
        mock_component._id = 32

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticHighlightedText(mock_component, annotation_type="pos")

        assert "pos" in wrapper.semantic_meta.tags
        assert "part-of-speech" in wrapper.semantic_meta.tags
        assert "grammar" in wrapper.semantic_meta.tags

    def test_sentiment_annotation_type(self):
        """Sentiment type adds opinion and emotion tags."""
        from integradio.specialized import SemanticHighlightedText

        mock_component = MagicMock()
        mock_component._id = 33

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticHighlightedText(mock_component, annotation_type="sentiment")

        assert "sentiment" in wrapper.semantic_meta.tags
        assert "opinion" in wrapper.semantic_meta.tags
        assert "emotion" in wrapper.semantic_meta.tags

    def test_ner_entity_mapping(self):
        """NER entities map to standardized tags."""
        from integradio.specialized import SemanticHighlightedText

        mock_component = MagicMock()
        mock_component._id = 34

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticHighlightedText(
                mock_component,
                entity_types=["PERSON", "ORG", "LOC", "DATE"],
            )

        assert "person-entity" in wrapper.semantic_meta.tags
        assert "organization-entity" in wrapper.semantic_meta.tags
        assert "location-entity" in wrapper.semantic_meta.tags
        assert "date-entity" in wrapper.semantic_meta.tags

    def test_ner_entity_case_insensitive(self):
        """NER entity mapping works with different cases."""
        from integradio.specialized import SemanticHighlightedText

        mock_component = MagicMock()
        mock_component._id = 35

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticHighlightedText(
                mock_component,
                entity_types=["person", "per", "org"],  # lowercase
            )

        assert "person-entity" in wrapper.semantic_meta.tags
        assert "organization-entity" in wrapper.semantic_meta.tags

    def test_auto_intent_generation(self):
        """Intent auto-generated with type and entities."""
        from integradio.specialized import SemanticHighlightedText

        mock_component = MagicMock()
        mock_component._id = 36

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticHighlightedText(
                mock_component,
                annotation_type="ner",
                entity_types=["PERSON", "ORG"],
            )

        assert "ner" in wrapper.intent
        assert "PERSON" in wrapper.intent

    def test_metadata_stored(self):
        """AnnotationMetadata is properly stored."""
        from integradio.specialized import SemanticHighlightedText

        mock_component = MagicMock()
        mock_component._id = 37

        color_map = {"PERSON": "#ff0000", "ORG": "#00ff00"}
        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticHighlightedText(
                mock_component,
                annotation_type="ner",
                entity_types=["PERSON", "ORG"],
                color_map=color_map,
            )

        assert wrapper.annotation_meta.annotation_type == "ner"
        assert wrapper.annotation_meta.entity_types == ["PERSON", "ORG"]
        assert wrapper.annotation_meta.color_map == color_map


# =============================================================================
# SemanticChatbot Tests
# =============================================================================


class TestSemanticChatbot:
    """Tests for SemanticChatbot specialized wrapper."""

    def test_basic_creation(self):
        """Creates wrapper with default settings."""
        from integradio.specialized import SemanticChatbot

        mock_component = MagicMock()
        mock_component._id = 40

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticChatbot(mock_component, intent="AI chat")

        assert wrapper.component is mock_component
        assert wrapper.chat_meta.supports_streaming is True

    def test_streaming_tag(self):
        """Streaming support adds streaming tag."""
        from integradio.specialized import SemanticChatbot

        mock_component = MagicMock()
        mock_component._id = 41

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticChatbot(mock_component, supports_streaming=True)

        assert "streaming" in wrapper.semantic_meta.tags

    def test_no_streaming_tag_when_disabled(self):
        """No streaming tag when disabled."""
        from integradio.specialized import SemanticChatbot

        mock_component = MagicMock()
        mock_component._id = 42

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticChatbot(mock_component, supports_streaming=False)

        assert "streaming" not in wrapper.semantic_meta.tags

    def test_retry_tag(self):
        """Retry support adds retry-capable tag."""
        from integradio.specialized import SemanticChatbot

        mock_component = MagicMock()
        mock_component._id = 43

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticChatbot(mock_component, supports_retry=True)

        assert "retry-capable" in wrapper.semantic_meta.tags

    def test_like_tags(self):
        """Like support adds feedback and likeable tags."""
        from integradio.specialized import SemanticChatbot

        mock_component = MagicMock()
        mock_component._id = 44

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticChatbot(mock_component, supports_like=True)

        assert "feedback" in wrapper.semantic_meta.tags
        assert "likeable" in wrapper.semantic_meta.tags

    def test_markdown_format_tag(self):
        """Markdown format adds markdown-output tag."""
        from integradio.specialized import SemanticChatbot

        mock_component = MagicMock()
        mock_component._id = 45

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticChatbot(mock_component, message_format="markdown")

        assert "markdown-output" in wrapper.semantic_meta.tags

    def test_persona_assistant(self):
        """Assistant persona adds helpful tags."""
        from integradio.specialized import SemanticChatbot

        mock_component = MagicMock()
        mock_component._id = 46

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticChatbot(mock_component, persona="assistant")

        assert "persona-assistant" in wrapper.semantic_meta.tags
        assert "helpful" in wrapper.semantic_meta.tags
        assert "general-purpose" in wrapper.semantic_meta.tags

    def test_persona_coder(self):
        """Coder persona adds programming tags."""
        from integradio.specialized import SemanticChatbot

        mock_component = MagicMock()
        mock_component._id = 47

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticChatbot(mock_component, persona="coder")

        assert "persona-coder" in wrapper.semantic_meta.tags
        assert "code-assistant" in wrapper.semantic_meta.tags
        assert "programming" in wrapper.semantic_meta.tags

    def test_persona_tutor(self):
        """Tutor persona adds educational tags."""
        from integradio.specialized import SemanticChatbot

        mock_component = MagicMock()
        mock_component._id = 48

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticChatbot(mock_component, persona="tutor")

        assert "persona-tutor" in wrapper.semantic_meta.tags
        assert "educational" in wrapper.semantic_meta.tags
        assert "explanatory" in wrapper.semantic_meta.tags

    def test_auto_intent_generation(self):
        """Intent auto-generated with persona and streaming."""
        from integradio.specialized import SemanticChatbot

        mock_component = MagicMock()
        mock_component._id = 49

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticChatbot(
                mock_component,
                persona="coder",
                supports_streaming=True,
            )

        assert "coder" in wrapper.intent
        assert "streaming" in wrapper.intent

    def test_metadata_stored(self):
        """ChatMetadata is properly stored."""
        from integradio.specialized import SemanticChatbot

        mock_component = MagicMock()
        mock_component._id = 50

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticChatbot(
                mock_component,
                supports_streaming=False,
                supports_retry=False,
                supports_undo=False,
                supports_like=True,
                message_format="html",
            )

        assert wrapper.chat_meta.supports_streaming is False
        assert wrapper.chat_meta.supports_retry is False
        assert wrapper.chat_meta.supports_like is True
        assert wrapper.chat_meta.message_format == "html"


# =============================================================================
# SemanticPlot Tests
# =============================================================================


class TestSemanticPlot:
    """Tests for SemanticPlot specialized wrapper."""

    def test_basic_creation(self):
        """Creates wrapper with default settings."""
        from integradio.specialized import SemanticPlot

        mock_component = MagicMock()
        mock_component._id = 60
        mock_component.__class__.__name__ = "Plot"

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticPlot(mock_component, intent="show data")

        assert wrapper.component is mock_component
        assert wrapper.viz_meta.chart_type == "generic"

    def test_line_chart_inference(self):
        """LinePlot infers chart_type=line."""
        from integradio.specialized import SemanticPlot

        mock_component = MagicMock()
        mock_component._id = 61
        mock_component.__class__.__name__ = "LinePlot"

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticPlot(mock_component)

        assert wrapper.viz_meta.chart_type == "line"
        assert "chart-line" in wrapper.semantic_meta.tags
        assert "timeseries" in wrapper.semantic_meta.tags
        assert "trend" in wrapper.semantic_meta.tags

    def test_bar_chart_inference(self):
        """BarPlot infers chart_type=bar."""
        from integradio.specialized import SemanticPlot

        mock_component = MagicMock()
        mock_component._id = 62
        mock_component.__class__.__name__ = "BarPlot"

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticPlot(mock_component)

        assert wrapper.viz_meta.chart_type == "bar"
        assert "chart-bar" in wrapper.semantic_meta.tags
        assert "categorical" in wrapper.semantic_meta.tags

    def test_scatter_chart_inference(self):
        """ScatterPlot infers chart_type=scatter."""
        from integradio.specialized import SemanticPlot

        mock_component = MagicMock()
        mock_component._id = 63
        mock_component.__class__.__name__ = "ScatterPlot"

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticPlot(mock_component)

        assert wrapper.viz_meta.chart_type == "scatter"
        assert "chart-scatter" in wrapper.semantic_meta.tags
        assert "correlation" in wrapper.semantic_meta.tags

    def test_explicit_chart_type(self):
        """Explicit chart_type overrides inference."""
        from integradio.specialized import SemanticPlot

        mock_component = MagicMock()
        mock_component._id = 64
        mock_component.__class__.__name__ = "LinePlot"

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticPlot(mock_component, chart_type="heatmap")

        assert wrapper.viz_meta.chart_type == "heatmap"
        assert "chart-heatmap" in wrapper.semantic_meta.tags
        assert "matrix" in wrapper.semantic_meta.tags

    def test_interactive_tag(self):
        """Interactive adds interactive-viz tag."""
        from integradio.specialized import SemanticPlot

        mock_component = MagicMock()
        mock_component._id = 65
        mock_component.__class__.__name__ = "Plot"

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticPlot(mock_component, interactive=True)

        assert "interactive-viz" in wrapper.semantic_meta.tags

    def test_zoom_tag(self):
        """Zoom support adds zoomable tag."""
        from integradio.specialized import SemanticPlot

        mock_component = MagicMock()
        mock_component._id = 66
        mock_component.__class__.__name__ = "Plot"

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticPlot(mock_component, supports_zoom=True)

        assert "zoomable" in wrapper.semantic_meta.tags

    def test_data_domain_tag(self):
        """Data domain adds domain-* tag."""
        from integradio.specialized import SemanticPlot

        mock_component = MagicMock()
        mock_component._id = 67
        mock_component.__class__.__name__ = "Plot"

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticPlot(mock_component, data_domain="metrics")

        assert "domain-metrics" in wrapper.semantic_meta.tags

    def test_auto_intent_with_axes(self):
        """Intent includes axes when provided."""
        from integradio.specialized import SemanticPlot

        mock_component = MagicMock()
        mock_component._id = 68
        mock_component.__class__.__name__ = "LinePlot"

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticPlot(
                mock_component,
                data_domain="sales",
                axes=["date", "revenue"],
            )

        assert "sales" in wrapper.intent
        assert "date vs revenue" in wrapper.intent

    def test_metadata_stored(self):
        """VisualizationMetadata is properly stored."""
        from integradio.specialized import SemanticPlot

        mock_component = MagicMock()
        mock_component._id = 69
        mock_component.__class__.__name__ = "Plot"

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticPlot(
                mock_component,
                chart_type="pie",
                interactive=False,
                supports_zoom=True,
                supports_pan=True,
                axes=["category", "value"],
            )

        assert wrapper.viz_meta.chart_type == "pie"
        assert wrapper.viz_meta.interactive is False
        assert wrapper.viz_meta.supports_zoom is True
        assert wrapper.viz_meta.supports_pan is True
        assert wrapper.viz_meta.axes == ["category", "value"]


# =============================================================================
# SemanticModel3D Tests
# =============================================================================


class TestSemanticModel3D:
    """Tests for SemanticModel3D specialized wrapper."""

    def test_basic_creation(self):
        """Creates wrapper with default settings."""
        from integradio.specialized import SemanticModel3D

        mock_component = MagicMock()
        mock_component._id = 70

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticModel3D(mock_component, intent="view 3D model")

        assert wrapper.component is mock_component
        assert wrapper.model3d_meta.supported_formats == ["obj", "glb", "gltf"]

    def test_format_tags(self):
        """Supported formats add format-* tags."""
        from integradio.specialized import SemanticModel3D

        mock_component = MagicMock()
        mock_component._id = 71

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticModel3D(
                mock_component,
                supported_formats=["obj", "fbx", "stl"],
            )

        assert "format-obj" in wrapper.semantic_meta.tags
        assert "format-fbx" in wrapper.semantic_meta.tags
        assert "format-stl" in wrapper.semantic_meta.tags

    def test_animation_tags(self):
        """Animation support adds animated and rigged tags."""
        from integradio.specialized import SemanticModel3D

        mock_component = MagicMock()
        mock_component._id = 72

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticModel3D(mock_component, supports_animation=True)

        assert "animated" in wrapper.semantic_meta.tags
        assert "rigged" in wrapper.semantic_meta.tags

    def test_texture_tag(self):
        """Texture support adds textured tag."""
        from integradio.specialized import SemanticModel3D

        mock_component = MagicMock()
        mock_component._id = 73

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticModel3D(mock_component, supports_textures=True)

        assert "textured" in wrapper.semantic_meta.tags

    def test_camera_tag(self):
        """Camera controls add orbitable tag."""
        from integradio.specialized import SemanticModel3D

        mock_component = MagicMock()
        mock_component._id = 74

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticModel3D(mock_component, camera_controls=True)

        assert "orbitable" in wrapper.semantic_meta.tags

    def test_mesh_generation_use_case(self):
        """Mesh generation use case adds generative tags."""
        from integradio.specialized import SemanticModel3D

        mock_component = MagicMock()
        mock_component._id = 75

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticModel3D(mock_component, use_case="mesh_generation")

        assert "mesh_generation" in wrapper.semantic_meta.tags
        assert "generative" in wrapper.semantic_meta.tags
        assert "ai-generated" in wrapper.semantic_meta.tags

    def test_cad_viewer_use_case(self):
        """CAD viewer use case adds engineering tags."""
        from integradio.specialized import SemanticModel3D

        mock_component = MagicMock()
        mock_component._id = 76

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticModel3D(mock_component, use_case="cad_viewer")

        assert "cad_viewer" in wrapper.semantic_meta.tags
        assert "engineering" in wrapper.semantic_meta.tags
        assert "technical" in wrapper.semantic_meta.tags

    def test_medical_use_case(self):
        """Medical use case adds anatomical tags."""
        from integradio.specialized import SemanticModel3D

        mock_component = MagicMock()
        mock_component._id = 77

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticModel3D(mock_component, use_case="medical")

        assert "medical" in wrapper.semantic_meta.tags
        assert "medical-imaging" in wrapper.semantic_meta.tags
        assert "anatomical" in wrapper.semantic_meta.tags

    def test_auto_intent_generation(self):
        """Intent auto-generated with use case."""
        from integradio.specialized import SemanticModel3D

        mock_component = MagicMock()
        mock_component._id = 78

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticModel3D(mock_component, use_case="game_asset")

        assert "game asset" in wrapper.intent

    def test_metadata_stored(self):
        """Model3DMetadata is properly stored."""
        from integradio.specialized import SemanticModel3D

        mock_component = MagicMock()
        mock_component._id = 79

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticModel3D(
                mock_component,
                supported_formats=["obj", "glb"],
                supports_animation=True,
                supports_textures=False,
                camera_controls=False,
            )

        assert wrapper.model3d_meta.supported_formats == ["obj", "glb"]
        assert wrapper.model3d_meta.supports_animation is True
        assert wrapper.model3d_meta.supports_textures is False
        assert wrapper.model3d_meta.camera_controls is False


# =============================================================================
# SemanticDataFrame Tests
# =============================================================================


class TestSemanticDataFrame:
    """Tests for SemanticDataFrame specialized wrapper."""

    def test_basic_creation(self):
        """Creates wrapper with default settings."""
        from integradio.specialized import SemanticDataFrame

        mock_component = MagicMock()
        mock_component._id = 80

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticDataFrame(mock_component, intent="show data")

        assert wrapper.component is mock_component

    def test_editable_tags(self):
        """Editable adds editable and interactive-data tags."""
        from integradio.specialized import SemanticDataFrame

        mock_component = MagicMock()
        mock_component._id = 81

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticDataFrame(mock_component, editable=True)

        assert "editable" in wrapper.semantic_meta.tags
        assert "interactive-data" in wrapper.semantic_meta.tags

    def test_database_domain(self):
        """Database domain adds sql and query-results tags."""
        from integradio.specialized import SemanticDataFrame

        mock_component = MagicMock()
        mock_component._id = 82

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticDataFrame(mock_component, data_domain="database")

        assert "domain-database" in wrapper.semantic_meta.tags
        assert "sql" in wrapper.semantic_meta.tags
        assert "query-results" in wrapper.semantic_meta.tags

    def test_metrics_domain(self):
        """Metrics domain adds kpi and measurements tags."""
        from integradio.specialized import SemanticDataFrame

        mock_component = MagicMock()
        mock_component._id = 83

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticDataFrame(mock_component, data_domain="metrics")

        assert "domain-metrics" in wrapper.semantic_meta.tags
        assert "kpi" in wrapper.semantic_meta.tags
        assert "measurements" in wrapper.semantic_meta.tags

    def test_logs_domain(self):
        """Logs domain adds logging and events tags."""
        from integradio.specialized import SemanticDataFrame

        mock_component = MagicMock()
        mock_component._id = 84

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticDataFrame(mock_component, data_domain="logs")

        assert "domain-logs" in wrapper.semantic_meta.tags
        assert "logging" in wrapper.semantic_meta.tags
        assert "events" in wrapper.semantic_meta.tags

    def test_column_inference_temporal(self):
        """Date/time columns add temporal tag."""
        from integradio.specialized import SemanticDataFrame

        mock_component = MagicMock()
        mock_component._id = 85

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticDataFrame(
                mock_component,
                columns=["created_date", "timestamp"],
            )

        assert "has-temporal" in wrapper.semantic_meta.tags

    def test_column_inference_financial(self):
        """Price/amount columns add financial tag."""
        from integradio.specialized import SemanticDataFrame

        mock_component = MagicMock()
        mock_component._id = 86

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticDataFrame(
                mock_component,
                columns=["price", "total_amount"],
            )

        assert "has-financial" in wrapper.semantic_meta.tags

    def test_column_inference_identifier(self):
        """ID columns add identifier tag."""
        from integradio.specialized import SemanticDataFrame

        mock_component = MagicMock()
        mock_component._id = 87

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticDataFrame(
                mock_component,
                columns=["user_id", "product_id"],
            )

        assert "has-identifier" in wrapper.semantic_meta.tags

    def test_auto_intent_generation(self):
        """Intent auto-generated with domain and editable."""
        from integradio.specialized import SemanticDataFrame

        mock_component = MagicMock()
        mock_component._id = 88

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticDataFrame(
                mock_component,
                data_domain="inventory",
                editable=True,
            )

        assert "inventory" in wrapper.intent
        assert "editable" in wrapper.intent

    def test_internal_attributes_stored(self):
        """Internal attributes are stored."""
        from integradio.specialized import SemanticDataFrame

        mock_component = MagicMock()
        mock_component._id = 89

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticDataFrame(
                mock_component,
                data_domain="users",
                columns=["name", "email"],
                editable=True,
            )

        assert wrapper._data_domain == "users"
        assert wrapper._columns == ["name", "email"]
        assert wrapper._editable is True


# =============================================================================
# SemanticFileExplorer Tests
# =============================================================================


class TestSemanticFileExplorer:
    """Tests for SemanticFileExplorer specialized wrapper."""

    def test_basic_creation(self):
        """Creates wrapper with default settings."""
        from integradio.specialized import SemanticFileExplorer

        mock_component = MagicMock()
        mock_component._id = 90

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticFileExplorer(mock_component, intent="browse files")

        assert wrapper.component is mock_component

    def test_code_project_root(self):
        """Code project root adds source-code and repository tags."""
        from integradio.specialized import SemanticFileExplorer

        mock_component = MagicMock()
        mock_component._id = 91

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticFileExplorer(mock_component, root_type="code_project")

        assert "code_project" in wrapper.semantic_meta.tags
        assert "source-code" in wrapper.semantic_meta.tags
        assert "repository" in wrapper.semantic_meta.tags

    def test_documents_root(self):
        """Documents root adds docs and files tags."""
        from integradio.specialized import SemanticFileExplorer

        mock_component = MagicMock()
        mock_component._id = 92

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticFileExplorer(mock_component, root_type="documents")

        assert "documents" in wrapper.semantic_meta.tags
        assert "docs" in wrapper.semantic_meta.tags
        assert "files" in wrapper.semantic_meta.tags

    def test_media_root(self):
        """Media root adds images, videos, assets tags."""
        from integradio.specialized import SemanticFileExplorer

        mock_component = MagicMock()
        mock_component._id = 93

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticFileExplorer(mock_component, root_type="media")

        assert "media" in wrapper.semantic_meta.tags
        assert "images" in wrapper.semantic_meta.tags
        assert "videos" in wrapper.semantic_meta.tags
        assert "assets" in wrapper.semantic_meta.tags

    def test_config_root(self):
        """Config root adds settings and configuration tags."""
        from integradio.specialized import SemanticFileExplorer

        mock_component = MagicMock()
        mock_component._id = 94

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticFileExplorer(mock_component, root_type="config")

        assert "config" in wrapper.semantic_meta.tags
        assert "settings" in wrapper.semantic_meta.tags
        assert "configuration" in wrapper.semantic_meta.tags

    def test_code_file_types(self):
        """Code file types add code-files tag."""
        from integradio.specialized import SemanticFileExplorer

        mock_component = MagicMock()
        mock_component._id = 95

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticFileExplorer(
                mock_component,
                file_types=[".py", ".js", ".ts"],
            )

        assert "code-files" in wrapper.semantic_meta.tags

    def test_document_file_types(self):
        """Document file types add document-files tag."""
        from integradio.specialized import SemanticFileExplorer

        mock_component = MagicMock()
        mock_component._id = 96

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticFileExplorer(
                mock_component,
                file_types=[".md", ".txt", ".pdf"],
            )

        assert "document-files" in wrapper.semantic_meta.tags

    def test_data_file_types(self):
        """Data file types add data-files tag."""
        from integradio.specialized import SemanticFileExplorer

        mock_component = MagicMock()
        mock_component._id = 97

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticFileExplorer(
                mock_component,
                file_types=[".csv", ".json", ".yaml"],
            )

        assert "data-files" in wrapper.semantic_meta.tags

    def test_mixed_file_types(self):
        """Mixed file types add multiple category tags."""
        from integradio.specialized import SemanticFileExplorer

        mock_component = MagicMock()
        mock_component._id = 98

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticFileExplorer(
                mock_component,
                file_types=[".py", ".md", ".json"],
            )

        assert "code-files" in wrapper.semantic_meta.tags
        assert "document-files" in wrapper.semantic_meta.tags
        assert "data-files" in wrapper.semantic_meta.tags

    def test_auto_intent_generation(self):
        """Intent auto-generated with root type."""
        from integradio.specialized import SemanticFileExplorer

        mock_component = MagicMock()
        mock_component._id = 99

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticFileExplorer(mock_component, root_type="code_project")

        assert "code project" in wrapper.intent

    def test_internal_attributes_stored(self):
        """Internal attributes are stored."""
        from integradio.specialized import SemanticFileExplorer

        mock_component = MagicMock()
        mock_component._id = 100

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticFileExplorer(
                mock_component,
                root_type="data",
                file_types=[".csv", ".json"],
            )

        assert wrapper._root_type == "data"
        assert wrapper._file_types == [".csv", ".json"]


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions that create specialized wrappers."""

    def test_semantic_multimodal(self):
        """semantic_multimodal() creates SemanticMultimodal."""
        from integradio.specialized import semantic_multimodal, SemanticMultimodal

        mock_component = MagicMock()
        mock_component._id = 110

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = semantic_multimodal(
                mock_component,
                intent="test",
                use_case="chat",
            )

        assert isinstance(wrapper, SemanticMultimodal)
        assert wrapper.multimodal_meta.use_case == "chat"

    def test_semantic_image_editor(self):
        """semantic_image_editor() creates SemanticImageEditor."""
        from integradio.specialized import semantic_image_editor, SemanticImageEditor

        mock_component = MagicMock()
        mock_component._id = 111

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = semantic_image_editor(
                mock_component,
                intent="test",
                use_case="inpainting",
            )

        assert isinstance(wrapper, SemanticImageEditor)
        assert wrapper.editor_meta.use_case == "inpainting"

    def test_semantic_annotated_image(self):
        """semantic_annotated_image() creates SemanticAnnotatedImage."""
        from integradio.specialized import semantic_annotated_image, SemanticAnnotatedImage

        mock_component = MagicMock()
        mock_component._id = 112

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = semantic_annotated_image(
                mock_component,
                intent="test",
                annotation_type="segmentation",
            )

        assert isinstance(wrapper, SemanticAnnotatedImage)
        assert wrapper.annotation_meta.annotation_type == "segmentation"

    def test_semantic_highlighted_text(self):
        """semantic_highlighted_text() creates SemanticHighlightedText."""
        from integradio.specialized import semantic_highlighted_text, SemanticHighlightedText

        mock_component = MagicMock()
        mock_component._id = 113

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = semantic_highlighted_text(
                mock_component,
                intent="test",
                annotation_type="pos",
            )

        assert isinstance(wrapper, SemanticHighlightedText)
        assert wrapper.annotation_meta.annotation_type == "pos"

    def test_semantic_chatbot(self):
        """semantic_chatbot() creates SemanticChatbot."""
        from integradio.specialized import semantic_chatbot, SemanticChatbot

        mock_component = MagicMock()
        mock_component._id = 114

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = semantic_chatbot(
                mock_component,
                intent="test",
                persona="coder",
            )

        assert isinstance(wrapper, SemanticChatbot)
        assert "persona-coder" in wrapper.semantic_meta.tags

    def test_semantic_plot(self):
        """semantic_plot() creates SemanticPlot."""
        from integradio.specialized import semantic_plot, SemanticPlot

        mock_component = MagicMock()
        mock_component._id = 115
        mock_component.__class__.__name__ = "Plot"

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = semantic_plot(
                mock_component,
                intent="test",
                chart_type="heatmap",
            )

        assert isinstance(wrapper, SemanticPlot)
        assert wrapper.viz_meta.chart_type == "heatmap"

    def test_semantic_model3d(self):
        """semantic_model3d() creates SemanticModel3D."""
        from integradio.specialized import semantic_model3d, SemanticModel3D

        mock_component = MagicMock()
        mock_component._id = 116

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = semantic_model3d(
                mock_component,
                intent="test",
                use_case="medical",
            )

        assert isinstance(wrapper, SemanticModel3D)
        assert "medical" in wrapper.semantic_meta.tags

    def test_semantic_dataframe(self):
        """semantic_dataframe() creates SemanticDataFrame."""
        from integradio.specialized import semantic_dataframe, SemanticDataFrame

        mock_component = MagicMock()
        mock_component._id = 117

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = semantic_dataframe(
                mock_component,
                intent="test",
                data_domain="logs",
            )

        assert isinstance(wrapper, SemanticDataFrame)
        assert wrapper._data_domain == "logs"

    def test_semantic_file_explorer(self):
        """semantic_file_explorer() creates SemanticFileExplorer."""
        from integradio.specialized import semantic_file_explorer, SemanticFileExplorer

        mock_component = MagicMock()
        mock_component._id = 118

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = semantic_file_explorer(
                mock_component,
                intent="test",
                root_type="media",
            )

        assert isinstance(wrapper, SemanticFileExplorer)
        assert wrapper._root_type == "media"

    def test_factory_passes_kwargs(self):
        """Factory functions pass extra kwargs to wrapper."""
        from integradio.specialized import semantic_multimodal

        mock_component = MagicMock()
        mock_component._id = 119

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = semantic_multimodal(
                mock_component,
                accepts_audio=True,
                max_files=5,
                tags=["custom"],
            )

        assert wrapper.multimodal_meta.accepts_audio is True
        assert wrapper.multimodal_meta.max_files == 5
        assert "custom" in wrapper.semantic_meta.tags


# =============================================================================
# Integration Tests
# =============================================================================


class TestSpecializedIntegration:
    """Integration tests for specialized wrappers."""

    def test_wrapper_inherits_from_semantic_component(self):
        """All specialized wrappers inherit from SemanticComponent."""
        from integradio.specialized import (
            SemanticMultimodal,
            SemanticImageEditor,
            SemanticAnnotatedImage,
            SemanticHighlightedText,
            SemanticChatbot,
            SemanticPlot,
            SemanticModel3D,
            SemanticDataFrame,
            SemanticFileExplorer,
        )
        from integradio.components import SemanticComponent

        wrappers = [
            SemanticMultimodal,
            SemanticImageEditor,
            SemanticAnnotatedImage,
            SemanticHighlightedText,
            SemanticChatbot,
            SemanticPlot,
            SemanticModel3D,
            SemanticDataFrame,
            SemanticFileExplorer,
        ]

        for wrapper_class in wrappers:
            assert issubclass(wrapper_class, SemanticComponent)

    def test_custom_tags_preserved(self):
        """Custom tags from user are preserved alongside auto-tags."""
        from integradio.specialized import SemanticChatbot

        mock_component = MagicMock()
        mock_component._id = 120

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticChatbot(
                mock_component,
                tags=["my-custom-tag", "another-tag"],
                persona="coder",
            )

        # Custom tags preserved
        assert "my-custom-tag" in wrapper.semantic_meta.tags
        assert "another-tag" in wrapper.semantic_meta.tags
        # Auto tags added
        assert "persona-coder" in wrapper.semantic_meta.tags

    def test_explicit_intent_not_overwritten(self):
        """Explicit intent is not overwritten by auto-generation."""
        from integradio.specialized import SemanticPlot

        mock_component = MagicMock()
        mock_component._id = 121
        mock_component.__class__.__name__ = "LinePlot"

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapper = SemanticPlot(
                mock_component,
                intent="my explicit intent",
                chart_type="line",
                data_domain="sales",
            )

        assert wrapper.intent == "my explicit intent"
        # Not auto-generated
        assert "line chart" not in wrapper.intent
        assert "sales" not in wrapper.intent
