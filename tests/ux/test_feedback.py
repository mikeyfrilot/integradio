"""
Integration tests for feedback components (toasts, confirmations, status badges).

Tests focus on real behavior and integration between feedback components.
"""

import pytest
import time
from integradio.ux.feedback import (
    ToastType,
    StatusType,
    ToastConfig,
    Toast,
    ToastManager,
    FEEDBACK_CSS,
    create_toast,
    create_toast_container,
    create_confirmation_dialog,
    create_status_badge,
    create_inline_status,
    get_feedback_css,
)


class TestToastTypeEnum:
    """Tests for ToastType enum."""

    def test_toast_type_values(self):
        """Verify all ToastType values."""
        assert ToastType.INFO.value == "info"
        assert ToastType.SUCCESS.value == "success"
        assert ToastType.WARNING.value == "warning"
        assert ToastType.ERROR.value == "error"

    def test_toast_type_iteration(self):
        """Test ToastType can be iterated."""
        all_types = list(ToastType)
        assert len(all_types) == 4


class TestStatusTypeEnum:
    """Tests for StatusType enum."""

    def test_status_type_values(self):
        """Verify all StatusType values."""
        assert StatusType.DEFAULT.value == "default"
        assert StatusType.INFO.value == "info"
        assert StatusType.SUCCESS.value == "success"
        assert StatusType.WARNING.value == "warning"
        assert StatusType.ERROR.value == "error"
        assert StatusType.PENDING.value == "pending"

    def test_status_type_iteration(self):
        """Test StatusType can be iterated."""
        all_types = list(StatusType)
        assert len(all_types) == 6


class TestToastConfigDataclass:
    """Tests for ToastConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ToastConfig()
        assert config.auto_dismiss_ms == 5000
        assert config.dismissible is True
        assert config.position == "bottom-right"
        assert config.max_visible == 3
        assert config.stack_direction == "up"

    def test_custom_config(self):
        """Test custom configuration."""
        config = ToastConfig(
            auto_dismiss_ms=3000,
            dismissible=False,
            position="top-left",
            max_visible=5,
            stack_direction="down"
        )
        assert config.auto_dismiss_ms == 3000
        assert config.dismissible is False
        assert config.position == "top-left"
        assert config.max_visible == 5
        assert config.stack_direction == "down"

    def test_config_all_positions(self):
        """Test all valid positions."""
        positions = ["top-left", "top-right", "bottom-left", "bottom-right"]
        for pos in positions:
            config = ToastConfig(position=pos)
            assert config.position == pos


class TestToastDataclass:
    """Tests for Toast dataclass."""

    def test_toast_creation(self):
        """Test basic toast creation."""
        toast = Toast(message="Test message")
        assert toast.message == "Test message"
        assert toast.toast_type == ToastType.INFO
        assert toast.title is None
        assert toast.action_label is None
        assert toast.action_callback is None
        assert toast.id.startswith("toast-")

    def test_toast_with_all_fields(self):
        """Test toast with all fields."""
        callback = lambda: None
        toast = Toast(
            message="Message",
            toast_type=ToastType.SUCCESS,
            title="Title",
            action_label="Undo",
            action_callback=callback
        )
        assert toast.message == "Message"
        assert toast.toast_type == ToastType.SUCCESS
        assert toast.title == "Title"
        assert toast.action_label == "Undo"
        assert toast.action_callback is callback

    def test_toast_created_at(self):
        """Test toast has created_at timestamp."""
        before = time.time()
        toast = Toast(message="Test")
        after = time.time()
        assert before <= toast.created_at <= after

    def test_toast_unique_ids(self):
        """Test toasts have unique IDs."""
        toast1 = Toast(message="Test 1")
        time.sleep(0.001)  # Ensure different timestamp
        toast2 = Toast(message="Test 2")
        assert toast1.id != toast2.id


class TestCreateToast:
    """Tests for create_toast function."""

    def test_basic_toast(self):
        """Test basic toast HTML structure."""
        html = create_toast("Test message")

        assert 'class="toast info"' in html
        assert 'role="alert"' in html
        assert 'aria-live="polite"' in html
        assert "Test message" in html
        assert 'class="toast-icon"' in html
        assert 'class="toast-content"' in html

    def test_toast_types(self):
        """Test all toast types."""
        for toast_type in ToastType:
            html = create_toast("Message", toast_type=toast_type)
            assert f'class="toast {toast_type.value}"' in html

    def test_toast_with_title(self):
        """Test toast with title."""
        html = create_toast("Message", title="Important")
        assert 'class="toast-title"' in html
        assert "Important" in html

    def test_toast_without_title(self):
        """Test toast without title."""
        html = create_toast("Message")
        assert 'class="toast-title"' not in html

    def test_toast_dismissible(self):
        """Test dismissible toast."""
        html = create_toast("Message", dismissible=True)
        assert 'class="toast-dismiss"' in html
        assert 'aria-label="Dismiss"' in html

    def test_toast_not_dismissible(self):
        """Test non-dismissible toast."""
        html = create_toast("Message", dismissible=False)
        assert 'class="toast-dismiss"' not in html

    def test_toast_with_action(self):
        """Test toast with action button."""
        html = create_toast("Message", action_label="Undo")
        assert 'class="toast-action"' in html
        assert "Undo" in html

    def test_toast_without_action(self):
        """Test toast without action."""
        html = create_toast("Message")
        assert 'class="toast-action"' not in html

    def test_toast_icons(self):
        """Test toast icons for each type."""
        # Each type should have an icon
        for toast_type in ToastType:
            html = create_toast("Message", toast_type=toast_type)
            assert 'class="toast-icon"' in html


class TestCreateToastContainer:
    """Tests for create_toast_container function."""

    def test_container_structure(self):
        """Test container HTML structure."""
        html = create_toast_container()
        assert 'class="toast-container' in html
        assert 'aria-live="polite"' in html

    def test_container_positions(self):
        """Test all container positions."""
        positions = ["top-left", "top-right", "bottom-left", "bottom-right"]
        for pos in positions:
            html = create_toast_container(position=pos)
            assert pos in html

    def test_container_stack_directions(self):
        """Test stack directions."""
        html_up = create_toast_container(stack_direction="up")
        html_down = create_toast_container(stack_direction="down")

        assert "stack-up" in html_up
        assert "stack-down" in html_down


class TestCreateConfirmationDialog:
    """Tests for create_confirmation_dialog function."""

    def test_dialog_structure(self):
        """Test dialog HTML structure."""
        html = create_confirmation_dialog(
            title="Confirm Delete",
            message="Are you sure you want to delete this item?"
        )

        assert 'class="confirmation-dialog"' in html
        assert 'role="dialog"' in html
        assert 'aria-modal="true"' in html
        assert 'aria-labelledby="confirm-title"' in html
        assert 'id="confirm-title"' in html
        assert "Confirm Delete" in html
        assert "Are you sure you want to delete this item?" in html

    def test_dialog_buttons(self):
        """Test dialog button labels."""
        html = create_confirmation_dialog(
            title="Title",
            message="Message",
            confirm_label="Delete",
            cancel_label="Keep"
        )

        assert "Delete" in html
        assert "Keep" in html
        assert 'class="confirmation-btn cancel"' in html
        assert 'class="confirmation-btn confirm"' in html

    def test_dialog_default_buttons(self):
        """Test dialog default button labels."""
        html = create_confirmation_dialog(
            title="Title",
            message="Message"
        )

        assert "Confirm" in html
        assert "Cancel" in html

    def test_dialog_icon_types(self):
        """Test dialog icon types."""
        for icon_type in ["warning", "danger", "info"]:
            html = create_confirmation_dialog(
                title="Title",
                message="Message",
                icon_type=icon_type
            )
            assert f'class="confirmation-icon {icon_type}"' in html

    def test_dialog_danger_mode(self):
        """Test dialog danger mode."""
        html = create_confirmation_dialog(
            title="Delete",
            message="Are you sure?",
            danger=True
        )
        # Should still have confirm class
        assert 'class="confirmation-btn confirm"' in html


class TestCreateStatusBadge:
    """Tests for create_status_badge function."""

    def test_badge_structure(self):
        """Test badge HTML structure."""
        html = create_status_badge("Active")
        assert 'class="status-badge' in html
        assert "Active" in html

    def test_badge_types(self):
        """Test all badge types."""
        for status_type in StatusType:
            html = create_status_badge("Label", status_type=status_type)
            assert f'class="status-badge {status_type.value}"' in html

    def test_badge_with_dot(self):
        """Test badge with status dot."""
        html = create_status_badge("Active", show_dot=True)
        assert 'class="status-badge-dot"' in html

    def test_badge_without_dot(self):
        """Test badge without status dot."""
        html = create_status_badge("Active", show_dot=False)
        assert 'class="status-badge-dot"' not in html

    def test_badge_default_type(self):
        """Test badge default type."""
        html = create_status_badge("Label")
        assert "default" in html


class TestCreateInlineStatus:
    """Tests for create_inline_status function."""

    def test_inline_status_structure(self):
        """Test inline status HTML structure."""
        html = create_inline_status("Operation successful", StatusType.SUCCESS)
        assert 'class="inline-status' in html
        assert 'role="status"' in html
        assert "Operation successful" in html

    def test_inline_status_types(self):
        """Test all inline status types."""
        for status_type in StatusType:
            html = create_inline_status("Message", status_type=status_type)
            assert status_type.value in html

    def test_inline_status_icons(self):
        """Test inline status has icons."""
        for status_type in StatusType:
            html = create_inline_status("Message", status_type=status_type)
            assert 'class="inline-status-icon"' in html


class TestGetFeedbackCSS:
    """Tests for get_feedback_css function."""

    def test_css_returns_string(self):
        """Test CSS function returns string."""
        css = get_feedback_css()
        assert isinstance(css, str)
        assert len(css) > 0

    def test_css_matches_constant(self):
        """Test CSS function returns the constant."""
        assert get_feedback_css() == FEEDBACK_CSS

    def test_css_contains_toast_classes(self):
        """Test CSS contains toast-related classes."""
        css = get_feedback_css()
        required_classes = [
            ".toast-container",
            ".toast",
            ".toast-icon",
            ".toast-content",
            ".toast-title",
            ".toast-message",
            ".toast-dismiss",
            ".toast-action"
        ]
        for cls in required_classes:
            assert cls in css, f"Missing class: {cls}"

    def test_css_contains_confirmation_classes(self):
        """Test CSS contains confirmation dialog classes."""
        css = get_feedback_css()
        required_classes = [
            ".confirmation-dialog",
            ".confirmation-content",
            ".confirmation-icon",
            ".confirmation-title",
            ".confirmation-message",
            ".confirmation-actions",
            ".confirmation-btn"
        ]
        for cls in required_classes:
            assert cls in css, f"Missing class: {cls}"

    def test_css_contains_status_badge_classes(self):
        """Test CSS contains status badge classes."""
        css = get_feedback_css()
        required_classes = [
            ".status-badge",
            ".status-badge.default",
            ".status-badge.info",
            ".status-badge.success",
            ".status-badge.warning",
            ".status-badge.error",
            ".status-badge.pending",
            ".status-badge-dot"
        ]
        for cls in required_classes:
            assert cls in css, f"Missing class: {cls}"

    def test_css_contains_animations(self):
        """Test CSS contains animations."""
        css = get_feedback_css()
        required_animations = [
            "@keyframes toast-slide-in",
            "@keyframes toast-slide-out",
            "@keyframes fade-in",
            "@keyframes scale-in",
            "@keyframes pulse-dot"
        ]
        for anim in required_animations:
            assert anim in css, f"Missing animation: {anim}"

    def test_css_contains_dark_mode(self):
        """Test CSS contains dark mode styles."""
        css = get_feedback_css()
        assert '[data-theme="dark"]' in css or ".dark" in css


class TestToastManager:
    """Tests for ToastManager class."""

    def test_manager_creation(self):
        """Test manager creation with default config."""
        manager = ToastManager()
        assert manager.config is not None
        assert manager.toasts == []

    def test_manager_custom_config(self):
        """Test manager with custom config."""
        config = ToastConfig(max_visible=5)
        manager = ToastManager(config=config)
        assert manager.config.max_visible == 5

    def test_add_toast(self):
        """Test adding a toast."""
        manager = ToastManager()
        toast = manager.add("Test message")

        assert len(manager.toasts) == 1
        assert toast.message == "Test message"
        assert toast.toast_type == ToastType.INFO

    def test_add_toast_with_type(self):
        """Test adding toast with specific type."""
        manager = ToastManager()
        toast = manager.add("Error!", ToastType.ERROR)

        assert toast.toast_type == ToastType.ERROR

    def test_add_toast_with_title(self):
        """Test adding toast with title."""
        manager = ToastManager()
        toast = manager.add("Message", title="Title")

        assert toast.title == "Title"

    def test_max_visible_limit(self):
        """Test max visible toast limit."""
        config = ToastConfig(max_visible=2)
        manager = ToastManager(config=config)

        manager.add("Toast 1")
        manager.add("Toast 2")
        manager.add("Toast 3")

        assert len(manager.toasts) == 2
        # Should keep the most recent
        assert manager.toasts[0].message == "Toast 2"
        assert manager.toasts[1].message == "Toast 3"

    def test_remove_toast(self):
        """Test removing a toast by ID."""
        manager = ToastManager()
        toast = manager.add("Test")
        manager.remove(toast.id)

        assert len(manager.toasts) == 0

    def test_remove_nonexistent_toast(self):
        """Test removing nonexistent toast."""
        manager = ToastManager()
        manager.add("Test")
        manager.remove("nonexistent-id")

        assert len(manager.toasts) == 1

    def test_clear_toasts(self):
        """Test clearing all toasts."""
        manager = ToastManager()
        manager.add("Toast 1")
        manager.add("Toast 2")
        manager.clear()

        assert len(manager.toasts) == 0

    def test_render_empty(self):
        """Test rendering empty toast manager."""
        manager = ToastManager()
        html = manager.render()

        assert 'class="toast-container' in html
        assert 'class="toast ' not in html

    def test_render_with_toasts(self):
        """Test rendering with toasts."""
        manager = ToastManager()
        manager.add("Toast 1")
        manager.add("Toast 2", ToastType.SUCCESS)

        html = manager.render()

        assert 'class="toast-container' in html
        assert "Toast 1" in html
        assert "Toast 2" in html
        assert 'class="toast info"' in html
        assert 'class="toast success"' in html

    def test_render_position(self):
        """Test rendering uses correct position."""
        config = ToastConfig(position="top-left")
        manager = ToastManager(config=config)
        html = manager.render()

        assert "top-left" in html

    def test_render_stack_direction(self):
        """Test rendering uses correct stack direction."""
        config = ToastConfig(stack_direction="down")
        manager = ToastManager(config=config)
        html = manager.render()

        assert "stack-down" in html


class TestFeedbackIntegration:
    """Integration tests for feedback workflows."""

    def test_toast_manager_workflow(self):
        """Test complete toast manager workflow."""
        config = ToastConfig(
            position="bottom-right",
            max_visible=3,
            dismissible=True
        )
        manager = ToastManager(config=config)

        # Add various toasts
        manager.add("Info message", ToastType.INFO)
        manager.add("Success!", ToastType.SUCCESS, title="Complete")
        manager.add("Warning", ToastType.WARNING)
        manager.add("Error occurred", ToastType.ERROR)

        # Should only have last 3
        assert len(manager.toasts) == 3

        # Render and verify
        html = manager.render()
        assert "Success!" in html
        assert "Warning" in html
        assert "Error occurred" in html
        assert "bottom-right" in html

    def test_all_components_valid_html(self):
        """Test all feedback components produce valid HTML."""
        components = [
            create_toast("Message"),
            create_toast("Message", ToastType.SUCCESS, "Title", True, "Action"),
            create_toast_container(),
            create_confirmation_dialog("Title", "Message"),
            create_status_badge("Label"),
            create_inline_status("Message"),
        ]

        for html in components:
            # Check for balanced div tags
            open_divs = html.count('<div')
            close_divs = html.count('</div>')
            assert open_divs == close_divs, f"Unbalanced divs in: {html[:100]}"

    def test_toast_accessibility_complete(self):
        """Test toasts have proper accessibility."""
        html = create_toast("Message", ToastType.ERROR, "Error Title")

        assert 'role="alert"' in html
        assert 'aria-live="polite"' in html

    def test_dialog_accessibility_complete(self):
        """Test dialogs have proper accessibility."""
        html = create_confirmation_dialog("Delete Item", "Are you sure?")

        assert 'role="dialog"' in html
        assert 'aria-modal="true"' in html
        assert 'aria-labelledby="confirm-title"' in html

    def test_status_badge_all_states(self):
        """Test status badges for all states."""
        states = [
            (StatusType.DEFAULT, "Inactive"),
            (StatusType.INFO, "Info"),
            (StatusType.SUCCESS, "Active"),
            (StatusType.WARNING, "Warning"),
            (StatusType.ERROR, "Error"),
            (StatusType.PENDING, "Loading"),
        ]

        for status_type, label in states:
            html = create_status_badge(label, status_type)
            assert label in html
            assert status_type.value in html
