"""
Batch 1: Components Coverage Tests (13 tests)

Tests for integradio/components.py - CRITICAL priority
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np


class TestComponentValidation:
    """Tests for component validation and required fields."""

    def test_component_validation_required_fields(self):
        """Verify SemanticComponent requires a wrapped component."""
        from integradio.components import SemanticComponent

        # Create mock component
        mock_component = MagicMock()
        mock_component._id = 123

        sc = SemanticComponent(mock_component)

        assert sc._component is mock_component
        assert sc.semantic_meta is not None
        assert sc.semantic_meta.intent is not None  # Auto-generated from component info

    def test_component_missing_component_raises(self):
        """Verify proper handling when component is None."""
        from integradio.components import SemanticComponent

        # Components module accepts None, but attribute access will fail
        with pytest.raises((TypeError, AttributeError)):
            sc = SemanticComponent(None)
            # Accessing component attributes should fail
            _ = sc._id

    def test_semantic_metadata_required_fields(self):
        """Verify SemanticMetadata has required intent field."""
        from integradio.components import SemanticMetadata

        meta = SemanticMetadata(intent="test intent")

        assert meta.intent == "test intent"
        assert meta.tags == []
        assert meta.embedded is False
        assert meta.extra == {}


class TestComponentSerialization:
    """Tests for component serialization roundtrip."""

    def test_component_serialization_roundtrip(self):
        """Verify component metadata can be serialized and restored."""
        from integradio.components import SemanticMetadata

        original = SemanticMetadata(
            intent="user input field",
            tags=["input", "text", "required"],
            file_path="app.py",
            line_number=42,
            embedded=True,
            extra={"custom_key": "custom_value"},
        )

        # Simulate serialization via dict
        serialized = {
            "intent": original.intent,
            "tags": original.tags,
            "file_path": original.file_path,
            "line_number": original.line_number,
            "embedded": original.embedded,
            "extra": original.extra,
        }

        # Reconstruct
        restored = SemanticMetadata(**serialized)

        assert restored.intent == original.intent
        assert restored.tags == original.tags
        assert restored.file_path == original.file_path
        assert restored.line_number == original.line_number
        assert restored.extra == original.extra

    def test_component_repr_contains_info(self):
        """Verify __repr__ includes useful information."""
        from integradio.components import SemanticComponent

        mock_component = MagicMock()
        mock_component._id = 456
        type(mock_component).__name__ = "Textbox"

        sc = SemanticComponent(mock_component, intent="search query")

        repr_str = repr(sc)
        assert "SemanticComponent" in repr_str
        assert "456" in repr_str
        assert "search query" in repr_str


class TestComponentDefaults:
    """Tests for component default values."""

    def test_component_default_values(self):
        """Verify SemanticComponent applies correct defaults."""
        from integradio.components import SemanticComponent

        mock_component = MagicMock()
        mock_component._id = 789
        mock_component.label = "Test Label"

        sc = SemanticComponent(mock_component)

        # Default auto_embed is True
        assert sc._auto_embed is True
        # Tags should be auto-inferred
        assert isinstance(sc.semantic_meta.tags, list)

    def test_semantic_metadata_default_values(self):
        """Verify SemanticMetadata default values."""
        from integradio.components import SemanticMetadata

        meta = SemanticMetadata(intent="test")

        assert meta.tags == []
        assert meta.file_path is None
        assert meta.line_number is None
        assert meta.embedded is False
        assert meta.extra == {}
        assert meta.visual_spec is None


class TestComponentTypeValidation:
    """Tests for type validation on components."""

    def test_component_invalid_type_rejected(self):
        """Verify invalid component types are handled properly."""
        from integradio.components import SemanticComponent

        # String is not a valid component
        with pytest.raises((TypeError, AttributeError)):
            sc = SemanticComponent("not a component")
            # Trying to access attributes should fail
            _ = sc.label

    def test_component_accepts_gradio_like_objects(self):
        """Verify component accepts objects with Gradio-like interface."""
        from integradio.components import SemanticComponent

        # Create mock with Gradio-like interface
        mock_component = MagicMock()
        mock_component._id = 100
        mock_component._elem_id = "my-element"
        mock_component.label = "My Label"

        sc = SemanticComponent(mock_component, intent="test component")

        # Should be able to access delegated attributes
        assert sc._id == 100
        assert sc.label == "My Label"


class TestComponentEventBinding:
    """Tests for component event binding."""

    def test_component_event_binding(self):
        """Verify event handlers can be bound through SemanticComponent."""
        from integradio.components import SemanticComponent

        mock_component = MagicMock()
        mock_component._id = 200
        mock_component.change = MagicMock()

        sc = SemanticComponent(mock_component, intent="test")

        # Should delegate to wrapped component
        handler = lambda x: x
        sc.change(handler)

        mock_component.change.assert_called_once_with(handler)

    def test_component_attribute_delegation(self):
        """Verify attribute access is properly delegated."""
        from integradio.components import SemanticComponent

        mock_component = MagicMock()
        mock_component._id = 300
        mock_component.value = "test value"
        mock_component.custom_attr = "custom"

        sc = SemanticComponent(mock_component, intent="test")

        assert sc.value == "test value"
        assert sc.custom_attr == "custom"

    def test_component_attribute_setting(self):
        """Verify attribute setting is properly delegated."""
        from integradio.components import SemanticComponent

        mock_component = MagicMock()
        mock_component._id = 400

        sc = SemanticComponent(mock_component, intent="test")

        # Setting non-private attribute should delegate
        sc.value = "new value"

        assert mock_component.value == "new value"


class TestSemanticFunction:
    """Tests for the semantic() helper function."""

    def test_semantic_function_wraps_component(self):
        """Verify semantic() function creates SemanticComponent."""
        from integradio.components import semantic, SemanticComponent

        mock_component = MagicMock()
        mock_component._id = 500

        wrapped = semantic(mock_component, intent="test intent")

        assert isinstance(wrapped, SemanticComponent)
        assert wrapped.intent == "test intent"

    def test_get_semantic_retrieves_by_id(self):
        """Verify get_semantic() retrieves registered components."""
        from integradio.components import SemanticComponent, get_semantic

        mock_component = MagicMock()
        mock_component._id = 600

        sc = SemanticComponent(mock_component, intent="test")

        # Should be registered in instances
        retrieved = get_semantic(600)

        assert retrieved is sc
