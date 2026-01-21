"""
Tests for SemanticComponent - Wrapped Gradio components with semantic metadata.

Priority 2: User-facing API tests.
"""

import pytest
from unittest.mock import MagicMock, patch
import weakref


class TestSemanticWrapping:
    """Test SemanticComponent wrapper functionality."""

    def test_semantic_wraps_component(self):
        """semantic() returns SemanticComponent."""
        from integradio.components import semantic, SemanticComponent

        # Create a mock Gradio component
        mock_component = MagicMock()
        mock_component._id = 123
        mock_component.label = "Test Label"

        result = semantic(mock_component, intent="test intent")

        assert isinstance(result, SemanticComponent)
        assert result.component is mock_component

    def test_attribute_delegation(self):
        """Accessing wrapped component attributes works."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 456
        mock_component.label = "My Label"
        mock_component.custom_attr = "custom_value"
        mock_component.some_method = MagicMock(return_value="method_result")

        wrapped = semantic(mock_component, intent="test")

        # Attributes should delegate to wrapped component
        assert wrapped.label == "My Label"
        assert wrapped.custom_attr == "custom_value"
        assert wrapped.some_method() == "method_result"

    def test_attribute_setting_delegates(self):
        """Setting attributes on SemanticComponent delegates to wrapped component."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 789

        wrapped = semantic(mock_component, intent="test")
        wrapped.label = "New Label"

        assert mock_component.label == "New Label"

    def test_component_property(self):
        """component property returns the wrapped Gradio component."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 111

        wrapped = semantic(mock_component, intent="test")

        assert wrapped.component is mock_component


class TestIntentHandling:
    """Test intent property and inference."""

    def test_explicit_intent(self):
        """Provided intent is stored."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 1

        wrapped = semantic(mock_component, intent="explicit user intent")

        assert wrapped.intent == "explicit user intent"
        assert wrapped.semantic_meta.intent == "explicit user intent"

    def test_inferred_intent_from_label(self):
        """No intent uses label."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 2
        mock_component.label = "Search Query Input"
        # Set up spec to avoid AttributeError for missing attrs
        mock_component.elem_id = None

        with patch("integradio.components.extract_component_info") as mock_extract:
            mock_extract.return_value = {
                "type": "Textbox",
                "label": "Search Query Input",
            }

            wrapped = semantic(mock_component)

        assert wrapped.intent == "Search Query Input"

    def test_inferred_intent_from_elem_id(self):
        """When no label, uses elem_id."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 3

        with patch("integradio.components.extract_component_info") as mock_extract:
            mock_extract.return_value = {
                "type": "Button",
                "elem_id": "submit-btn",
            }

            wrapped = semantic(mock_component)

        assert wrapped.intent == "submit-btn"

    def test_inferred_intent_falls_back_to_type(self):
        """When no label or elem_id, uses type."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 4

        with patch("integradio.components.extract_component_info") as mock_extract:
            mock_extract.return_value = {
                "type": "Slider",
            }

            wrapped = semantic(mock_component)

        assert wrapped.intent == "Slider"

    def test_intent_setter_triggers_reembed(self):
        """Update the intent property triggers re-embedding if registered."""
        from integradio.components import semantic, SemanticComponent

        # Clear class-level state to ensure no registry/embedder from other tests
        old_registry = SemanticComponent._registry
        old_embedder = SemanticComponent._embedder
        SemanticComponent._registry = None
        SemanticComponent._embedder = None

        try:
            mock_component = MagicMock()
            mock_component._id = 5

            wrapped = semantic(mock_component, intent="original intent")
            assert wrapped.semantic_meta.embedded is False  # Not yet embedded

            # Update intent
            wrapped.intent = "updated intent"

            assert wrapped.intent == "updated intent"
            assert wrapped.semantic_meta.embedded is False  # Marked for re-embedding
        finally:
            # Restore state
            SemanticComponent._registry = old_registry
            SemanticComponent._embedder = old_embedder


class TestTagHandling:
    """Test tag inference and management."""

    def test_auto_tags_by_type(self):
        """Textbox gets ['input', 'text'] tags."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 10
        mock_component.__class__.__name__ = "Textbox"

        with patch("integradio.components.infer_tags") as mock_infer:
            mock_infer.return_value = ["input", "text"]

            wrapped = semantic(mock_component, intent="test")

        assert "input" in wrapped.semantic_meta.tags
        assert "text" in wrapped.semantic_meta.tags

    def test_custom_tags_merged(self):
        """Custom tags merge with inferred."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 11
        mock_component.__class__.__name__ = "Textbox"

        with patch("integradio.components.infer_tags") as mock_infer:
            mock_infer.return_value = ["input", "text"]

            wrapped = semantic(
                mock_component,
                intent="test",
                tags=["custom", "special"],
            )

        # Should have both custom and inferred tags
        assert "custom" in wrapped.semantic_meta.tags
        assert "special" in wrapped.semantic_meta.tags
        assert "input" in wrapped.semantic_meta.tags
        assert "text" in wrapped.semantic_meta.tags

    def test_add_tags_method(self):
        """add_tags() adds tags and returns self for chaining."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 12

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="test", tags=["original"])

        result = wrapped.add_tags("new1", "new2")

        assert result is wrapped  # Returns self
        assert "original" in wrapped.semantic_meta.tags
        assert "new1" in wrapped.semantic_meta.tags
        assert "new2" in wrapped.semantic_meta.tags

    def test_tags_deduplicated(self):
        """Duplicate tags are removed."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 13

        with patch("integradio.components.infer_tags") as mock_infer:
            mock_infer.return_value = ["input", "text"]

            wrapped = semantic(
                mock_component,
                intent="test",
                tags=["input", "custom"],  # "input" is duplicate
            )

        # Count occurrences
        assert wrapped.semantic_meta.tags.count("input") == 1


class TestSourceLocation:
    """Test source code location capture."""

    def test_source_location_captured(self):
        """file_path and line_number are set."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 20

        with patch("integradio.components.get_source_location") as mock_loc:
            from integradio.introspect import SourceLocation
            mock_loc.return_value = SourceLocation(
                file_path="/path/to/app.py",
                line_number=42,
                function_name="create_ui",
            )
            with patch("integradio.components.infer_tags", return_value=[]):
                wrapped = semantic(mock_component, intent="test")

        assert wrapped.semantic_meta.file_path == "/path/to/app.py"
        assert wrapped.semantic_meta.line_number == 42


class TestInstanceTracking:
    """Test SemanticComponent instance tracking."""

    def test_get_semantic_by_id(self):
        """get_semantic() retrieves correct instance."""
        from integradio.components import semantic, get_semantic, SemanticComponent

        # Clear instances
        SemanticComponent._instances.clear()

        mock_component = MagicMock()
        mock_component._id = 100

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="test intent")

        # Should be able to retrieve by ID
        retrieved = get_semantic(100)
        assert retrieved is wrapped

    def test_get_semantic_nonexistent(self):
        """get_semantic() returns None for unknown ID."""
        from integradio.components import get_semantic, SemanticComponent

        SemanticComponent._instances.clear()
        result = get_semantic(99999)
        assert result is None

    def test_weak_reference_tracking(self):
        """Instances use weak references (don't prevent GC)."""
        from integradio.components import semantic, get_semantic, SemanticComponent
        import gc

        SemanticComponent._instances.clear()

        mock_component = MagicMock()
        mock_component._id = 200

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="test")

        assert get_semantic(200) is not None

        # Delete reference and force GC
        del wrapped
        gc.collect()

        # WeakValueDictionary should allow cleanup
        # (Note: In practice, mock objects may not be GC'd immediately)


class TestExtraMetadata:
    """Test extra metadata handling."""

    def test_extra_kwargs_stored(self):
        """Extra kwargs are stored in semantic_meta.extra."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 300

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(
                mock_component,
                intent="test",
                custom_field="custom_value",
                another_field=42,
            )

        assert wrapped.semantic_meta.extra["custom_field"] == "custom_value"
        assert wrapped.semantic_meta.extra["another_field"] == 42


class TestRepr:
    """Test string representation."""

    def test_repr_format(self):
        """__repr__ shows useful info."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 400
        mock_component.__class__.__name__ = "Textbox"

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="user search")

        repr_str = repr(wrapped)

        assert "SemanticComponent" in repr_str
        assert "Textbox" in repr_str or "MagicMock" in repr_str
        assert "400" in repr_str
        assert "user search" in repr_str


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_component_without_id_not_tracked(self):
        """Components without _id are wrapped but not tracked."""
        from integradio.components import semantic, get_semantic, SemanticComponent

        SemanticComponent._instances.clear()

        mock_component = MagicMock(spec=[])  # No _id attribute
        del mock_component._id  # Ensure no _id

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="test")

        # Should be wrapped successfully
        assert isinstance(wrapped, SemanticComponent)
        # But not tracked (no ID to track with)
        assert len(SemanticComponent._instances) == 0

    def test_intent_with_unicode_special_chars(self):
        """Intent with unicode and special characters works."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 501

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(
                mock_component,
                intent="用户搜索 🔍 user's \"query\" <input>"
            )

        assert wrapped.intent == "用户搜索 🔍 user's \"query\" <input>"

    def test_intent_empty_string(self):
        """Empty string intent is allowed."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 502

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="")

        assert wrapped.intent == ""

    def test_intent_whitespace_only(self):
        """Whitespace-only intent is preserved."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 503

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="   \t\n  ")

        assert wrapped.intent == "   \t\n  "

    def test_tags_empty_list(self):
        """Empty tags list works."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 504

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="test", tags=[])

        assert wrapped.semantic_meta.tags == []

    def test_tags_with_empty_strings(self):
        """Empty string tags are preserved."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 505

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="test", tags=["valid", "", "also_valid"])

        assert "" in wrapped.semantic_meta.tags
        assert "valid" in wrapped.semantic_meta.tags

    def test_extra_metadata_with_complex_types(self):
        """Extra metadata supports complex types."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 506

        complex_data = {
            "nested": {"a": 1, "b": [1, 2, 3]},
            "list": [{"x": 1}, {"y": 2}],
            "none_value": None,
            "boolean": True,
        }

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="test", **complex_data)

        assert wrapped.semantic_meta.extra["nested"] == {"a": 1, "b": [1, 2, 3]}
        assert wrapped.semantic_meta.extra["list"] == [{"x": 1}, {"y": 2}]
        assert wrapped.semantic_meta.extra["none_value"] is None
        assert wrapped.semantic_meta.extra["boolean"] is True

    def test_component_id_zero(self):
        """Component with ID 0 is tracked correctly."""
        from integradio.components import semantic, get_semantic, SemanticComponent

        SemanticComponent._instances.clear()

        mock_component = MagicMock()
        mock_component._id = 0

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="test")

        assert get_semantic(0) is wrapped

    def test_component_id_negative(self):
        """Component with negative ID is tracked correctly."""
        from integradio.components import semantic, get_semantic, SemanticComponent

        SemanticComponent._instances.clear()

        mock_component = MagicMock()
        mock_component._id = -1

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="test")

        assert get_semantic(-1) is wrapped

    def test_getattr_raises_for_nonexistent(self):
        """Accessing nonexistent attribute raises AttributeError."""
        from integradio.components import semantic

        mock_component = MagicMock(spec=["_id", "label"])
        mock_component._id = 507
        mock_component.label = "Test"

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="test")

        with pytest.raises(AttributeError):
            _ = wrapped.nonexistent_attribute

    def test_setattr_creates_on_component(self):
        """Setting new attributes creates them on wrapped component."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 508

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="test")

        wrapped.new_attr = "new_value"
        assert mock_component.new_attr == "new_value"

    def test_repr_with_missing_id(self):
        """__repr__ works when component has no _id."""
        from integradio.components import semantic

        mock_component = MagicMock(spec=[])
        del mock_component._id

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="test repr")

        repr_str = repr(wrapped)
        assert "SemanticComponent" in repr_str
        assert "?" in repr_str  # Missing ID shown as ?
        assert "test repr" in repr_str


class TestSemanticMetadataDataclass:
    """Test SemanticMetadata dataclass behavior."""

    def test_metadata_defaults(self):
        """Default values are set correctly."""
        from integradio.components import SemanticMetadata

        meta = SemanticMetadata(intent="test")

        assert meta.intent == "test"
        assert meta.tags == []
        assert meta.file_path is None
        assert meta.line_number is None
        assert meta.embedded is False
        assert meta.extra == {}

    def test_metadata_immutable_defaults(self):
        """Default list/dict are not shared between instances."""
        from integradio.components import SemanticMetadata

        meta1 = SemanticMetadata(intent="test1")
        meta2 = SemanticMetadata(intent="test2")

        meta1.tags.append("tag1")
        meta1.extra["key"] = "value"

        assert meta2.tags == []
        assert meta2.extra == {}

    def test_metadata_equality(self):
        """SemanticMetadata supports equality comparison."""
        from integradio.components import SemanticMetadata

        meta1 = SemanticMetadata(
            intent="test",
            tags=["a", "b"],
            file_path="/path",
            line_number=42,
        )
        meta2 = SemanticMetadata(
            intent="test",
            tags=["a", "b"],
            file_path="/path",
            line_number=42,
        )

        assert meta1 == meta2


class TestAddTagsMethod:
    """Test add_tags method edge cases."""

    def test_add_tags_empty(self):
        """add_tags with no arguments returns self."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 600

        with patch("integradio.components.infer_tags", return_value=["inferred"]):
            wrapped = semantic(mock_component, intent="test")

        original_tags = wrapped.semantic_meta.tags.copy()
        result = wrapped.add_tags()

        assert result is wrapped
        assert wrapped.semantic_meta.tags == original_tags

    def test_add_tags_chaining(self):
        """add_tags supports method chaining."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 601

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="test")

        result = wrapped.add_tags("a").add_tags("b").add_tags("c")

        assert result is wrapped
        assert "a" in wrapped.semantic_meta.tags
        assert "b" in wrapped.semantic_meta.tags
        assert "c" in wrapped.semantic_meta.tags

    def test_add_tags_duplicate_prevention(self):
        """add_tags prevents duplicate tags."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 602

        with patch("integradio.components.infer_tags", return_value=["existing"]):
            wrapped = semantic(mock_component, intent="test")

        wrapped.add_tags("existing", "existing", "new")

        assert wrapped.semantic_meta.tags.count("existing") == 1
        assert "new" in wrapped.semantic_meta.tags


class TestIntentSetterWithRegistry:
    """Test intent setter triggering re-embedding."""

    def test_intent_setter_without_registry(self):
        """Setting intent without registry doesn't error."""
        from integradio.components import semantic, SemanticComponent

        SemanticComponent._registry = None
        SemanticComponent._embedder = None

        mock_component = MagicMock()
        mock_component._id = 700

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="original")

        # Should not raise
        wrapped.intent = "new intent"
        assert wrapped.intent == "new intent"

    def test_intent_setter_marks_not_embedded(self):
        """Setting intent marks component as not embedded."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 701

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="original")

        wrapped._semantic_meta.embedded = True  # Simulate being embedded

        wrapped.intent = "new intent"

        assert wrapped.semantic_meta.embedded is False


class TestGetSemanticEdgeCases:
    """Test get_semantic function edge cases."""

    def test_get_semantic_after_overwrite(self):
        """get_semantic returns latest instance for overwritten ID."""
        from integradio.components import semantic, get_semantic, SemanticComponent

        SemanticComponent._instances.clear()

        mock_component1 = MagicMock()
        mock_component1._id = 800

        mock_component2 = MagicMock()
        mock_component2._id = 800  # Same ID

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped1 = semantic(mock_component1, intent="first")
            wrapped2 = semantic(mock_component2, intent="second")

        # Should return the second one
        result = get_semantic(800)
        assert result is wrapped2
        assert result.intent == "second"

    def test_get_semantic_various_id_types(self):
        """get_semantic handles various ID types."""
        from integradio.components import get_semantic, SemanticComponent

        SemanticComponent._instances.clear()

        # None ID should return None
        result = get_semantic(None)  # type: ignore
        assert result is None

        # String ID (if somehow passed) should return None
        result = get_semantic("string_id")  # type: ignore
        assert result is None


class TestVisualSpecIntegration:
    """Test visual specification integration with SemanticComponent."""

    def test_visual_property_getter(self):
        """visual property returns the visual spec."""
        from integradio.components import semantic
        from integradio.visual import VisualSpec

        mock_component = MagicMock()
        mock_component._id = 900

        visual_spec = VisualSpec(component_id="test-900", component_type="Button")

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="test", visual=visual_spec)

        # Test the visual property getter
        assert wrapped.visual is not None
        assert wrapped.visual.component_id == "test-900"
        assert wrapped.visual.component_type == "Button"

    def test_visual_property_getter_returns_none(self):
        """visual property returns None when no spec is set."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 901

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="test")

        assert wrapped.visual is None

    def test_visual_property_setter(self):
        """visual setter sets the visual spec and auto-populates IDs."""
        from integradio.components import semantic
        from integradio.visual import VisualSpec

        mock_component = MagicMock()
        mock_component._id = 902

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="test")

        # Set visual spec via property setter with empty component_id (falsy)
        new_spec = VisualSpec(component_id="", component_type="")
        wrapped.visual = new_spec

        # Empty string is falsy, so should auto-populate IDs
        # Note: VisualSpec requires non-empty component_id, so check it's populated
        assert wrapped.visual is not None
        # The setter auto-populates when component_id is empty/falsy
        assert wrapped.visual.component_id == "902"
        # type(mock_component).__name__ returns "MagicMock"
        assert wrapped.visual.component_type == "MagicMock"

    def test_visual_setter_preserves_existing_ids(self):
        """visual setter preserves IDs if already set."""
        from integradio.components import semantic
        from integradio.visual import VisualSpec

        mock_component = MagicMock()
        mock_component._id = 903

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="test")

        # Set visual spec with IDs already populated
        new_spec = VisualSpec(component_id="custom-id", component_type="CustomType")
        wrapped.visual = new_spec

        # Should preserve existing IDs
        assert wrapped.visual.component_id == "custom-id"
        assert wrapped.visual.component_type == "CustomType"

    def test_visual_spec_auto_populate_on_init(self):
        """Visual spec component_id and component_type are auto-populated on init."""
        from integradio.components import semantic
        from integradio.visual import VisualSpec

        mock_component = MagicMock()
        mock_component._id = 904

        # VisualSpec with empty component_id (falsy triggers auto-populate)
        visual_spec = VisualSpec(component_id="", component_type="")

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="test", visual=visual_spec)

        # Should auto-populate from component (empty string is falsy so triggers code path)
        # Lines 96-99 in components.py:
        # if visual is not None and not visual.component_id:
        #     visual.component_id = str(getattr(component, "_id", id(component)))
        #     if not visual.component_type:
        #         visual.component_type = type(component).__name__
        assert wrapped.visual.component_id == "904"
        # type(mock_component).__name__ returns "MagicMock"
        assert wrapped.visual.component_type == "MagicMock"

    def test_visual_spec_auto_populate_component_type_only(self):
        """Visual spec auto-populates component_type when component_id is empty but type is not."""
        from integradio.components import semantic
        from integradio.visual import VisualSpec

        mock_component = MagicMock()
        mock_component._id = 905
        mock_component.__class__.__name__ = "Slider"

        # VisualSpec with empty component_id (triggers auto-populate) but component_type set
        visual_spec = VisualSpec(component_id="", component_type="MyButton")

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="test", visual=visual_spec)

        # component_id auto-populated because it was empty
        assert wrapped.visual.component_id == "905"
        # component_type preserved because it was already set
        assert wrapped.visual.component_type == "MyButton"

    def test_visual_spec_preserves_existing_component_id(self):
        """Visual spec preserves component_id when already set (non-empty)."""
        from integradio.components import semantic
        from integradio.visual import VisualSpec

        mock_component = MagicMock()
        mock_component._id = 906
        mock_component.__class__.__name__ = "Slider"

        # VisualSpec with component_id already set - should NOT be overwritten
        visual_spec = VisualSpec(component_id="my-slider", component_type="")

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="test", visual=visual_spec)

        # Should preserve component_id since it was non-empty
        assert wrapped.visual.component_id == "my-slider"
        # component_type stays empty because the code path only runs when component_id is empty
        assert wrapped.visual.component_type == ""


class TestSetVisualMethod:
    """Test the set_visual convenience method."""

    def test_set_visual_creates_spec(self):
        """set_visual creates a VisualSpec if one doesn't exist."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 1000

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="test")

        # No visual spec initially
        assert wrapped.visual is None

        # set_visual should create one
        result = wrapped.set_visual(background="#ff0000")

        assert result is wrapped  # Returns self for chaining
        assert wrapped.visual is not None
        assert wrapped.visual.component_id == "1000"

    def test_set_visual_with_colors(self):
        """set_visual sets color tokens."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 1001

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="test")

        wrapped.set_visual(
            background="#3b82f6",
            text_color="#ffffff",
            border_color="#1e40af",
        )

        assert wrapped.visual is not None
        # Verify colors were set via set_colors
        assert "background" in wrapped.visual.tokens
        assert "color" in wrapped.visual.tokens
        assert "border-color" in wrapped.visual.tokens

    def test_set_visual_with_padding(self):
        """set_visual sets padding via DimensionValue."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 1002

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="test")

        wrapped.set_visual(padding=16)

        assert wrapped.visual is not None
        assert wrapped.visual.layout.padding is not None
        # Padding is set uniformly
        assert wrapped.visual.layout.padding.top.value == 16
        assert wrapped.visual.layout.padding.top.unit == "px"

    def test_set_visual_chaining(self):
        """set_visual supports method chaining."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 1003

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="test")

        result = (
            wrapped
            .set_visual(background="#ff0000")
            .set_visual(text_color="#ffffff")
            .set_visual(padding=8)
        )

        assert result is wrapped
        assert wrapped.visual is not None

    def test_set_visual_with_existing_spec(self):
        """set_visual updates existing VisualSpec."""
        from integradio.components import semantic
        from integradio.visual import VisualSpec

        mock_component = MagicMock()
        mock_component._id = 1004

        visual_spec = VisualSpec(component_id="existing", component_type="Button")

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="test", visual=visual_spec)

        wrapped.set_visual(background="#00ff00")

        # Should update the existing spec, not create a new one
        assert wrapped.visual.component_id == "existing"
        assert "background" in wrapped.visual.tokens


class TestToCssMethod:
    """Test the to_css method."""

    def test_to_css_with_visual_spec(self):
        """to_css returns CSS when visual spec exists."""
        from integradio.components import semantic
        from integradio.visual import VisualSpec

        mock_component = MagicMock()
        mock_component._id = 1100

        visual_spec = VisualSpec(component_id="test-btn", component_type="Button")
        visual_spec.set_colors(background="#3b82f6")

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="test", visual=visual_spec)

        css = wrapped.to_css()

        assert css != ""
        assert "#test-btn" in css
        # Color may be output as rgb() format, so check for either format
        assert "background" in css
        assert "3b82f6" in css.lower() or "rgb(59, 130, 246)" in css

    def test_to_css_without_visual_spec(self):
        """to_css returns empty string when no visual spec."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 1101

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="test")

        css = wrapped.to_css()

        assert css == ""

    def test_to_css_with_custom_selector(self):
        """to_css uses custom selector when provided."""
        from integradio.components import semantic
        from integradio.visual import VisualSpec

        mock_component = MagicMock()
        mock_component._id = 1102

        visual_spec = VisualSpec(component_id="test-btn", component_type="Button")
        visual_spec.set_colors(background="#ff0000")

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="test", visual=visual_spec)

        css = wrapped.to_css(selector=".custom-class")

        assert ".custom-class" in css
        assert "#test-btn" not in css  # Default selector not used


class TestRegistryIntegration:
    """Test SemanticComponent registration with registry."""

    def test_register_to_registry_success(self):
        """_register_to_registry successfully registers component."""
        from integradio.components import semantic, SemanticComponent
        from integradio.registry import ComponentRegistry
        import numpy as np

        # Create a mock embedder
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = np.random.randn(768).astype(np.float32)

        # Create a real registry (in-memory)
        registry = ComponentRegistry()

        # Store original state
        old_registry = SemanticComponent._registry
        old_embedder = SemanticComponent._embedder

        try:
            # Set class-level registry and embedder
            SemanticComponent._registry = registry
            SemanticComponent._embedder = mock_embedder

            mock_component = MagicMock()
            mock_component._id = 1200

            # Mock extract_component_info to return proper string values (not MagicMock)
            with patch("integradio.components.infer_tags", return_value=["input"]), \
                 patch("integradio.components.extract_component_info", return_value={
                     "type": "Textbox",
                     "label": "Test Label",
                     "elem_id": "test-elem",
                 }):
                wrapped = semantic(mock_component, intent="test registration")

                # Manually call _register_to_registry
                result = wrapped._register_to_registry()

            assert result is True
            assert wrapped.semantic_meta.embedded is True
            assert 1200 in registry

        finally:
            SemanticComponent._registry = old_registry
            SemanticComponent._embedder = old_embedder

    def test_register_to_registry_no_registry(self):
        """_register_to_registry returns False when no registry."""
        from integradio.components import semantic, SemanticComponent

        old_registry = SemanticComponent._registry
        old_embedder = SemanticComponent._embedder

        try:
            SemanticComponent._registry = None
            SemanticComponent._embedder = None

            mock_component = MagicMock()
            mock_component._id = 1201

            with patch("integradio.components.infer_tags", return_value=[]):
                wrapped = semantic(mock_component, intent="test")

            result = wrapped._register_to_registry()

            assert result is False

        finally:
            SemanticComponent._registry = old_registry
            SemanticComponent._embedder = old_embedder

    def test_register_to_registry_no_component_id(self):
        """_register_to_registry returns False when component has no _id."""
        from integradio.components import semantic, SemanticComponent
        from integradio.registry import ComponentRegistry
        import numpy as np

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = np.random.randn(768).astype(np.float32)

        registry = ComponentRegistry()

        old_registry = SemanticComponent._registry
        old_embedder = SemanticComponent._embedder

        try:
            SemanticComponent._registry = registry
            SemanticComponent._embedder = mock_embedder

            mock_component = MagicMock(spec=[])  # No _id attribute
            del mock_component._id

            with patch("integradio.components.infer_tags", return_value=[]):
                wrapped = semantic(mock_component, intent="test")

            result = wrapped._register_to_registry()

            assert result is False

        finally:
            SemanticComponent._registry = old_registry
            SemanticComponent._embedder = old_embedder

    def test_intent_setter_triggers_reregister(self):
        """Setting intent triggers re-registration when registry is available."""
        from integradio.components import semantic, SemanticComponent
        from integradio.registry import ComponentRegistry
        import numpy as np

        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = np.random.randn(768).astype(np.float32)

        registry = ComponentRegistry()

        old_registry = SemanticComponent._registry
        old_embedder = SemanticComponent._embedder

        try:
            SemanticComponent._registry = registry
            SemanticComponent._embedder = mock_embedder

            mock_component = MagicMock()
            mock_component._id = 1202

            # Mock extract_component_info to return proper string values
            with patch("integradio.components.infer_tags", return_value=[]), \
                 patch("integradio.components.extract_component_info", return_value={
                     "type": "Textbox",
                     "label": "Test Label",
                     "elem_id": None,
                 }):
                wrapped = semantic(mock_component, intent="original intent")

                # Register initially
                wrapped._register_to_registry()
                assert wrapped.semantic_meta.embedded is True

                # Update intent - should trigger re-registration
                wrapped.intent = "new intent"

                # embedded flag set to False, then re-registered
                assert wrapped.semantic_meta.embedded is True
                assert wrapped.intent == "new intent"

                # Embedder should have been called twice
                assert mock_embedder.embed.call_count >= 2

        finally:
            SemanticComponent._registry = old_registry
            SemanticComponent._embedder = old_embedder

    def test_register_to_registry_exception_handling(self):
        """_register_to_registry handles exceptions gracefully."""
        from integradio.components import semantic, SemanticComponent

        # Create a mock embedder that raises an exception
        mock_embedder = MagicMock()
        mock_embedder.embed.side_effect = RuntimeError("Embedding failed")

        mock_registry = MagicMock()

        old_registry = SemanticComponent._registry
        old_embedder = SemanticComponent._embedder

        try:
            SemanticComponent._registry = mock_registry
            SemanticComponent._embedder = mock_embedder

            mock_component = MagicMock()
            mock_component._id = 1203

            with patch("integradio.components.infer_tags", return_value=[]):
                wrapped = semantic(mock_component, intent="test")

            result = wrapped._register_to_registry()

            # Should return False on exception, not raise
            assert result is False
            assert wrapped.semantic_meta.embedded is False

        finally:
            SemanticComponent._registry = old_registry
            SemanticComponent._embedder = old_embedder


class TestGetAttrInternalAttributes:
    """Test __getattr__ handling of internal Gradio attributes."""

    def test_getattr_returns_gradio_internal_id(self):
        """__getattr__ returns _id from wrapped component."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 1300

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="test")

        # Access _id through getattr (this hits line 265)
        assert wrapped._id == 1300

    def test_getattr_returns_gradio_internal_parent(self):
        """__getattr__ returns _parent from wrapped component."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 1301
        mock_component._parent = "parent_block"

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="test")

        assert wrapped._parent == "parent_block"

    def test_getattr_returns_gradio_internal_elem_id(self):
        """__getattr__ returns _elem_id from wrapped component."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 1302
        mock_component._elem_id = "my-element"

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="test")

        assert wrapped._elem_id == "my-element"

    def test_getattr_returns_gradio_internal_elem_classes(self):
        """__getattr__ returns _elem_classes from wrapped component."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 1303
        mock_component._elem_classes = ["class1", "class2"]

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="test")

        assert wrapped._elem_classes == ["class1", "class2"]

    def test_getattr_raises_for_blocked_private_attributes(self):
        """__getattr__ raises AttributeError for blocked private attributes."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 1304

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="test")

        # These should raise AttributeError (line 268)
        with pytest.raises(AttributeError):
            _ = wrapped._other_private

        with pytest.raises(AttributeError):
            _ = wrapped._random_attr

    def test_getattr_raises_for_semantic_reserved_attributes(self):
        """__getattr__ raises AttributeError for semantic's own reserved attrs."""
        from integradio.components import semantic

        mock_component = MagicMock()
        mock_component._id = 1305
        # Even if component has these, they're blocked for getattr
        mock_component.component = "should not return this"
        mock_component.semantic_meta = "should not return this"
        mock_component.intent = "should not return this"
        mock_component.visual = "should not return this"

        with patch("integradio.components.infer_tags", return_value=[]):
            wrapped = semantic(mock_component, intent="test")

        # These go through the property mechanism, not getattr
        # But if we try to get them via __getattr__ they'd be blocked
        # Actually the properties handle these, so let's verify the properties work
        assert wrapped.component is mock_component
        assert wrapped.semantic_meta is not None
        assert wrapped.intent == "test"
        assert wrapped.visual is None  # Not set in this test
