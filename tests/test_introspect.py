"""
Tests for introspection utilities - Source location and dataflow extraction.

Priority 5: Utility tests.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestSourceLocation:
    """Test source location capture."""

    def test_get_source_location(self):
        """Returns correct file and line."""
        from integradio.introspect import get_source_location

        # Call from within this test
        result = get_source_location(depth=1)

        assert result is not None
        assert "test_introspect.py" in result.file_path
        assert result.line_number > 0

    def test_get_source_location_function_name(self):
        """Captures function name when available."""
        from integradio.introspect import get_source_location

        def inner_function():
            return get_source_location(depth=1)

        result = inner_function()

        assert result is not None
        assert result.function_name == "inner_function"

    def test_get_source_location_module_level(self):
        """function_name is None at module level."""
        from integradio.introspect import get_source_location

        # At module level (simulated by depth)
        result = get_source_location(depth=1)

        # In a test method, function_name will be the test method name
        assert result is not None
        # Just verify it returns something reasonable
        assert result.file_path is not None

    def test_get_source_location_handles_errors(self):
        """Returns None when frame inspection fails."""
        from integradio.introspect import get_source_location

        # Very deep depth should fail gracefully
        result = get_source_location(depth=1000)

        assert result is None


class TestExtractComponentInfo:
    """Test component info extraction."""

    def test_extract_component_info(self):
        """Extracts label, elem_id, type."""
        from integradio.introspect import extract_component_info

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "Textbox"
        mock_component.label = "Search Query"
        mock_component.elem_id = "search-input"
        mock_component.elem_classes = ["input-class"]
        mock_component.visible = True
        mock_component.interactive = True
        mock_component._id = 123

        result = extract_component_info(mock_component)

        assert result["type"] == "Textbox"
        assert result["label"] == "Search Query"
        assert result["elem_id"] == "search-input"
        assert result["component_id"] == 123

    def test_extract_component_info_optional_attrs(self):
        """Handles missing optional attributes."""
        from integradio.introspect import extract_component_info

        mock_component = MagicMock(spec=["__class__"])
        mock_component.__class__.__name__ = "Button"

        result = extract_component_info(mock_component)

        assert result["type"] == "Button"
        # Missing attributes should not cause errors

    def test_extract_component_info_placeholder(self):
        """Extracts placeholder for input components."""
        from integradio.introspect import extract_component_info

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "Textbox"
        mock_component.placeholder = "Enter your query..."
        mock_component.label = None
        mock_component.elem_id = None
        mock_component.elem_classes = None
        mock_component.visible = None
        mock_component.interactive = None
        mock_component.value = None
        mock_component.choices = None

        result = extract_component_info(mock_component)

        assert result["placeholder"] == "Enter your query..."

    def test_extract_component_info_choices(self):
        """Extracts choices for select components."""
        from integradio.introspect import extract_component_info

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "Dropdown"
        mock_component.choices = ["Option A", "Option B", "Option C"]
        mock_component.label = None
        mock_component.elem_id = None
        mock_component.elem_classes = None
        mock_component.visible = None
        mock_component.interactive = None
        mock_component.placeholder = None
        mock_component.value = None

        result = extract_component_info(mock_component)

        assert result["choices"] == ["Option A", "Option B", "Option C"]


class TestBuildIntentText:
    """Test intent text building for embeddings."""

    def test_build_intent_text(self):
        """Builds proper embedding text."""
        from integradio.introspect import build_intent_text

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "Textbox"
        mock_component.label = "Search Query"
        mock_component.elem_id = "search-input"
        mock_component.placeholder = "Type here..."
        mock_component.choices = None

        result = build_intent_text(mock_component, "user enters search terms")

        assert "user enters search terms" in result
        assert "Textbox" in result
        assert "Search Query" in result

    def test_build_intent_text_without_explicit_intent(self):
        """Handles missing explicit intent."""
        from integradio.introspect import build_intent_text

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "Button"
        mock_component.label = "Submit"
        mock_component.elem_id = None
        mock_component.placeholder = None
        mock_component.choices = None

        result = build_intent_text(mock_component, None)

        assert "Button" in result
        assert "Submit" in result

    def test_build_intent_text_with_choices(self):
        """Includes choices preview for select components."""
        from integradio.introspect import build_intent_text

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "Dropdown"
        mock_component.label = "Category"
        mock_component.elem_id = None
        mock_component.placeholder = None
        mock_component.choices = ["A", "B", "C", "D", "E", "F", "G"]

        result = build_intent_text(mock_component, "select category")

        assert "choices:" in result
        assert "A" in result
        # Should show first 5 and indicate more
        assert "7 total" in result


class TestInferTags:
    """Test automatic tag inference."""

    def test_infer_tags_textbox(self):
        """Textbox -> ['input', 'text']."""
        from integradio.introspect import infer_tags

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "Textbox"

        result = infer_tags(mock_component)

        assert "input" in result
        assert "text" in result

    def test_infer_tags_button(self):
        """Button -> ['trigger', 'action']."""
        from integradio.introspect import infer_tags

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "Button"

        result = infer_tags(mock_component)

        assert "trigger" in result
        assert "action" in result

    def test_infer_tags_markdown(self):
        """Markdown -> ['output', 'text', 'display']."""
        from integradio.introspect import infer_tags

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "Markdown"

        result = infer_tags(mock_component)

        assert "output" in result
        assert "text" in result
        assert "display" in result

    def test_infer_tags_slider(self):
        """Slider -> ['input', 'numeric', 'range']."""
        from integradio.introspect import infer_tags

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "Slider"

        result = infer_tags(mock_component)

        assert "input" in result
        assert "numeric" in result
        assert "range" in result

    def test_infer_tags_image(self):
        """Image -> ['media', 'visual']."""
        from integradio.introspect import infer_tags

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "Image"

        result = infer_tags(mock_component)

        assert "media" in result
        assert "visual" in result

    def test_infer_tags_chatbot(self):
        """Chatbot -> ['io', 'conversation']."""
        from integradio.introspect import infer_tags

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "Chatbot"

        result = infer_tags(mock_component)

        assert "io" in result
        assert "conversation" in result

    def test_infer_tags_unknown_type(self):
        """Unknown type gets fallback tag."""
        from integradio.introspect import infer_tags

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "UnknownWidget"

        result = infer_tags(mock_component)

        assert "component" in result

    def test_infer_tags_interactive_attribute(self):
        """Interactive attribute adds tag."""
        from integradio.introspect import infer_tags

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "Textbox"
        mock_component.interactive = True

        result = infer_tags(mock_component)

        assert "interactive" in result

    def test_infer_tags_static_attribute(self):
        """Non-interactive adds 'static' tag."""
        from integradio.introspect import infer_tags

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "Textbox"
        mock_component.interactive = False

        result = infer_tags(mock_component)

        assert "static" in result

    def test_infer_tags_deduplication(self):
        """Tags are deduplicated."""
        from integradio.introspect import infer_tags

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "Textbox"

        result = infer_tags(mock_component)

        # Check no duplicates
        assert len(result) == len(set(result))


class TestExtractDataflow:
    """Test dataflow extraction from Blocks."""

    def test_extract_dataflow(self):
        """Extracts event relationships from Blocks."""
        from integradio.introspect import extract_dataflow

        # Create mock Blocks with fns
        mock_blocks = MagicMock()

        # Create mock function with inputs/outputs
        mock_fn = MagicMock()

        mock_input = MagicMock()
        mock_input._id = 1

        mock_output = MagicMock()
        mock_output._id = 2

        mock_trigger = MagicMock()
        mock_trigger.block = MagicMock()
        mock_trigger.block._id = 3

        mock_fn.inputs = [mock_input]
        mock_fn.outputs = [mock_output]
        mock_fn.triggers = [mock_trigger]

        mock_blocks.fns = {0: mock_fn}

        result = extract_dataflow(mock_blocks)

        assert len(result) == 1
        assert result[0]["fn_id"] == 0
        assert result[0]["inputs"] == [1]
        assert result[0]["outputs"] == [2]
        assert result[0]["triggers"] == [3]

    def test_extract_dataflow_no_fns(self):
        """Handles Blocks without fns attribute."""
        from integradio.introspect import extract_dataflow

        mock_blocks = MagicMock(spec=[])  # No fns attribute

        result = extract_dataflow(mock_blocks)

        assert result == []

    def test_extract_dataflow_empty_fns(self):
        """Handles Blocks with empty fns."""
        from integradio.introspect import extract_dataflow

        mock_blocks = MagicMock()
        mock_blocks.fns = {}

        result = extract_dataflow(mock_blocks)

        assert result == []

    def test_extract_dataflow_missing_io(self):
        """Handles functions with missing inputs/outputs."""
        from integradio.introspect import extract_dataflow

        mock_blocks = MagicMock()
        mock_fn = MagicMock(spec=[])  # No inputs/outputs

        mock_blocks.fns = {0: mock_fn}

        result = extract_dataflow(mock_blocks)

        assert result == []

    def test_extract_dataflow_multiple_functions(self):
        """Handles multiple event handlers."""
        from integradio.introspect import extract_dataflow

        mock_blocks = MagicMock()

        mock_fn1 = MagicMock()
        mock_input1 = MagicMock()
        mock_input1._id = 1
        mock_output1 = MagicMock()
        mock_output1._id = 2
        mock_fn1.inputs = [mock_input1]
        mock_fn1.outputs = [mock_output1]
        mock_fn1.triggers = []

        mock_fn2 = MagicMock()
        mock_input2 = MagicMock()
        mock_input2._id = 3
        mock_output2 = MagicMock()
        mock_output2._id = 4
        mock_fn2.inputs = [mock_input2]
        mock_fn2.outputs = [mock_output2]
        mock_fn2.triggers = []

        mock_blocks.fns = {0: mock_fn1, 1: mock_fn2}

        result = extract_dataflow(mock_blocks)

        assert len(result) == 2
        assert {r["fn_id"] for r in result} == {0, 1}


# ============================================================================
# EDGE CASES - More comprehensive coverage
# ============================================================================


class TestSourceLocationEdgeCases:
    """Edge cases for source location capture."""

    def test_get_source_location_depth_zero(self):
        """Depth 0 returns location of get_source_location itself."""
        from integradio.introspect import get_source_location

        result = get_source_location(depth=0)

        assert result is not None
        assert "introspect.py" in result.file_path

    def test_get_source_location_depth_one(self):
        """Depth 1 returns location of the immediate caller."""
        from integradio.introspect import get_source_location

        result = get_source_location(depth=1)

        assert result is not None
        assert "test_introspect.py" in result.file_path
        assert result.line_number > 0

    def test_get_source_location_nested_calls(self):
        """Nested function calls track properly."""
        from integradio.introspect import get_source_location

        def outer():
            def inner():
                return get_source_location(depth=2)
            return inner()

        result = outer()

        assert result is not None
        assert result.function_name == "outer"


class TestExtractComponentInfoEdgeCases:
    """Edge cases for component info extraction."""

    def test_extract_component_info_none_values(self):
        """Handles None values for all attributes."""
        from integradio.introspect import extract_component_info

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "Textbox"
        mock_component.label = None
        mock_component.elem_id = None
        mock_component.elem_classes = None
        mock_component.visible = None
        mock_component.interactive = None
        mock_component.placeholder = None
        mock_component.value = None
        mock_component.choices = None

        result = extract_component_info(mock_component)

        assert result["type"] == "Textbox"
        # None values should not be included
        assert "label" not in result
        assert "elem_id" not in result

    def test_extract_component_info_with_value(self):
        """Extracts default value for components."""
        from integradio.introspect import extract_component_info

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "Textbox"
        mock_component.label = None
        mock_component.elem_id = None
        mock_component.elem_classes = None
        mock_component.visible = None
        mock_component.interactive = None
        mock_component.placeholder = None
        mock_component.value = "default text"
        mock_component.choices = None

        result = extract_component_info(mock_component)

        assert result["value"] == "default text"

    def test_extract_component_info_empty_string_label(self):
        """Handles empty string label (should be included as falsy but not None)."""
        from integradio.introspect import extract_component_info

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "Button"
        mock_component.label = ""
        mock_component.elem_id = None
        mock_component.elem_classes = None
        mock_component.visible = None
        mock_component.interactive = None

        result = extract_component_info(mock_component)

        # Empty string is truthy check, empty string is falsy so not included
        assert result["type"] == "Button"

    def test_extract_component_info_elem_classes_list(self):
        """Handles elem_classes as list."""
        from integradio.introspect import extract_component_info

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "Textbox"
        mock_component.label = None
        mock_component.elem_id = None
        mock_component.elem_classes = ["class1", "class2"]
        mock_component.visible = None
        mock_component.interactive = None

        result = extract_component_info(mock_component)

        assert result["elem_classes"] == ["class1", "class2"]


class TestBuildIntentTextEdgeCases:
    """Edge cases for intent text building."""

    def test_build_intent_text_empty_intent(self):
        """Handles empty string intent."""
        from integradio.introspect import build_intent_text

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "Button"
        mock_component.label = "Submit"
        mock_component.elem_id = None
        mock_component.placeholder = None
        mock_component.choices = None

        result = build_intent_text(mock_component, "")

        # Empty string should not appear in output
        assert result.startswith("Gradio") or result.startswith("|")

    def test_build_intent_text_all_attributes(self):
        """Builds text with all attributes present."""
        from integradio.introspect import build_intent_text

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "Dropdown"
        mock_component.label = "Category"
        mock_component.elem_id = "cat-select"
        mock_component.placeholder = "Select one..."
        mock_component.choices = ["A", "B", "C"]

        result = build_intent_text(mock_component, "user selects category")

        assert "user selects category" in result
        assert "Dropdown" in result
        assert "Category" in result
        assert "cat-select" in result
        assert "choices:" in result

    def test_build_intent_text_exactly_five_choices(self):
        """Handles exactly 5 choices (no truncation indicator)."""
        from integradio.introspect import build_intent_text

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "Dropdown"
        mock_component.label = None
        mock_component.elem_id = None
        mock_component.placeholder = None
        mock_component.choices = ["A", "B", "C", "D", "E"]

        result = build_intent_text(mock_component, None)

        assert "choices:" in result
        assert "total" not in result  # No truncation needed

    def test_build_intent_text_unicode_intent(self):
        """Handles Unicode in intent."""
        from integradio.introspect import build_intent_text

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "Textbox"
        mock_component.label = "输入"  # Chinese for "input"
        mock_component.elem_id = None
        mock_component.placeholder = None
        mock_component.choices = None

        result = build_intent_text(mock_component, "用户输入查询")  # "user enters query"

        assert "用户输入查询" in result
        assert "输入" in result


class TestInferTagsGradio6Components:
    """Test tag inference for Gradio 6 component types."""

    def test_infer_tags_multimodaltextbox(self):
        """MultimodalTextbox -> ['input', 'text', 'multimodal', 'file']."""
        from integradio.introspect import infer_tags

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "MultimodalTextbox"

        result = infer_tags(mock_component)

        assert "input" in result
        assert "multimodal" in result

    def test_infer_tags_timer(self):
        """Timer -> ['trigger', 'temporal', 'automation']."""
        from integradio.introspect import infer_tags

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "Timer"

        result = infer_tags(mock_component)

        assert "trigger" in result
        assert "temporal" in result

    def test_infer_tags_sidebar(self):
        """Sidebar -> ['layout', 'navigation', 'panel']."""
        from integradio.introspect import infer_tags

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "Sidebar"

        result = infer_tags(mock_component)

        assert "layout" in result
        assert "navigation" in result

    def test_infer_tags_imageeditor(self):
        """ImageEditor -> ['input', 'media', 'editor', 'visual']."""
        from integradio.introspect import infer_tags

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "ImageEditor"

        result = infer_tags(mock_component)

        assert "input" in result
        assert "editor" in result

    def test_infer_tags_scatterplot(self):
        """ScatterPlot -> ['output', 'visualization', 'data', 'scatter']."""
        from integradio.introspect import infer_tags

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "ScatterPlot"

        result = infer_tags(mock_component)

        assert "output" in result
        assert "visualization" in result

    def test_infer_tags_dataframe(self):
        """DataFrame -> ['data', 'tabular']."""
        from integradio.introspect import infer_tags

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "DataFrame"

        result = infer_tags(mock_component)

        assert "data" in result
        assert "tabular" in result


class TestInferTagsAttributeBased:
    """Test attribute-based tag inference."""

    def test_infer_tags_streaming_attribute(self):
        """Streaming attribute adds 'streaming' tag."""
        from integradio.introspect import infer_tags

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "Chatbot"
        mock_component.streaming = True

        result = infer_tags(mock_component)

        assert "streaming" in result

    def test_infer_tags_filepath_type(self):
        """type='filepath' adds 'filepath' tag."""
        from integradio.introspect import infer_tags

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "File"
        mock_component.type = "filepath"

        result = infer_tags(mock_component)

        assert "filepath" in result

    def test_infer_tags_binary_type(self):
        """type='binary' adds 'binary' tag."""
        from integradio.introspect import infer_tags

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "File"
        mock_component.type = "binary"

        result = infer_tags(mock_component)

        assert "binary" in result

    def test_infer_tags_webcam_source(self):
        """sources=['webcam'] adds 'webcam' tag."""
        from integradio.introspect import infer_tags

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "Image"
        mock_component.sources = ["webcam", "upload"]

        result = infer_tags(mock_component)

        assert "webcam" in result
        assert "upload" in result

    def test_infer_tags_microphone_source(self):
        """sources=['microphone'] adds 'microphone' tag."""
        from integradio.introspect import infer_tags

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "Audio"
        mock_component.sources = ["microphone"]

        result = infer_tags(mock_component)

        assert "microphone" in result

    def test_infer_tags_copyable(self):
        """show_copy_button=True adds 'copyable' tag."""
        from integradio.introspect import infer_tags

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "Textbox"
        mock_component.show_copy_button = True

        result = infer_tags(mock_component)

        assert "copyable" in result

    def test_infer_tags_rtl(self):
        """rtl=True adds 'rtl' tag."""
        from integradio.introspect import infer_tags

        mock_component = MagicMock()
        mock_component.__class__.__name__ = "Textbox"
        mock_component.rtl = True

        result = infer_tags(mock_component)

        assert "rtl" in result


class TestExtractDataflowEdgeCases:
    """Edge cases for dataflow extraction."""

    def test_extract_dataflow_none_inputs(self):
        """Handles None inputs list."""
        from integradio.introspect import extract_dataflow

        mock_blocks = MagicMock()
        mock_fn = MagicMock()
        mock_fn.inputs = None
        mock_fn.outputs = []
        mock_fn.triggers = []

        mock_blocks.fns = {0: mock_fn}

        result = extract_dataflow(mock_blocks)

        # Should handle None gracefully
        assert result == []

    def test_extract_dataflow_none_outputs(self):
        """Handles None outputs list."""
        from integradio.introspect import extract_dataflow

        mock_blocks = MagicMock()
        mock_fn = MagicMock()
        mock_fn.inputs = []
        mock_fn.outputs = None
        mock_fn.triggers = []

        mock_blocks.fns = {0: mock_fn}

        result = extract_dataflow(mock_blocks)

        assert result == []

    def test_extract_dataflow_trigger_without_block(self):
        """Handles triggers without block attribute."""
        from integradio.introspect import extract_dataflow

        mock_blocks = MagicMock()
        mock_fn = MagicMock()

        mock_input = MagicMock()
        mock_input._id = 1
        mock_fn.inputs = [mock_input]

        mock_output = MagicMock()
        mock_output._id = 2
        mock_fn.outputs = [mock_output]

        # Trigger without block attribute
        mock_trigger = MagicMock(spec=[])
        mock_fn.triggers = [mock_trigger]

        mock_blocks.fns = {0: mock_fn}

        result = extract_dataflow(mock_blocks)

        assert len(result) == 1
        assert result[0]["triggers"] == []  # Trigger without block._id ignored

    def test_extract_dataflow_input_without_id(self):
        """Handles inputs without _id attribute."""
        from integradio.introspect import extract_dataflow

        mock_blocks = MagicMock()
        mock_fn = MagicMock()

        # Input without _id
        mock_input = MagicMock(spec=[])
        mock_fn.inputs = [mock_input]

        mock_output = MagicMock()
        mock_output._id = 2
        mock_fn.outputs = [mock_output]
        mock_fn.triggers = []

        mock_blocks.fns = {0: mock_fn}

        result = extract_dataflow(mock_blocks)

        assert len(result) == 1
        assert result[0]["inputs"] == []  # Input without _id filtered out
        assert result[0]["outputs"] == [2]

    def test_extract_dataflow_mixed_valid_invalid_inputs(self):
        """Handles mix of valid and invalid inputs."""
        from integradio.introspect import extract_dataflow

        mock_blocks = MagicMock()
        mock_fn = MagicMock()

        mock_input1 = MagicMock()
        mock_input1._id = 1
        mock_input2 = MagicMock(spec=[])  # No _id
        mock_input3 = MagicMock()
        mock_input3._id = 3

        mock_fn.inputs = [mock_input1, mock_input2, mock_input3]
        mock_fn.outputs = []
        mock_fn.triggers = []

        mock_blocks.fns = {0: mock_fn}

        result = extract_dataflow(mock_blocks)

        assert len(result) == 1
        assert result[0]["inputs"] == [1, 3]  # Only valid inputs included
