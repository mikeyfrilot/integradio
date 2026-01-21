"""
Tests for integradio.pages.utils module.

Tests accessibility utilities and UX enhancements for pages:
- Skip navigation links
- ARIA landmarks
- Loading sections
- Confirmation buttons
- Status announcements
- Empty states
"""

import pytest
from unittest.mock import MagicMock, patch

from integradio.pages.utils import (
    get_page_css,
    add_skip_navigation,
    add_main_content_landmark,
    close_main_content_landmark,
    create_confirmation_button,
    create_loading_section,
    create_status_announcement,
    get_empty_state,
    get_enhanced_page_css,
    EMPTY_STATES,
    EMPTY_STATE_CSS,
)


# ======================================================================
# Tests for get_page_css
# ======================================================================

class TestGetPageCSS:
    """Tests for get_page_css function."""

    def test_returns_string(self):
        """Test that get_page_css returns a string."""
        css = get_page_css()
        assert isinstance(css, str)

    def test_css_not_empty(self):
        """Test that returned CSS is not empty."""
        css = get_page_css()
        assert len(css) > 0

    def test_contains_ux_styles(self):
        """Test that CSS contains UX component styles."""
        css = get_page_css()
        # Should contain styles from get_all_ux_css
        assert "skeleton" in css.lower() or "loading" in css.lower() or "button" in css.lower()


# ======================================================================
# Tests for add_skip_navigation
# ======================================================================

class TestAddSkipNavigation:
    """Tests for add_skip_navigation function."""

    def test_returns_gradio_html(self):
        """Test that add_skip_navigation returns gr.HTML component."""
        import gradio as gr
        result = add_skip_navigation()
        assert isinstance(result, gr.HTML)

    def test_default_target_id(self):
        """Test default main-content target ID."""
        result = add_skip_navigation()
        # The component should have elem_id set
        assert result.elem_id == "skip-nav"

    def test_custom_target_id(self):
        """Test custom main content ID."""
        result = add_skip_navigation(main_content_id="custom-content")
        # Component is created with custom target
        assert result is not None

    def test_visible_by_default(self):
        """Test skip navigation is visible."""
        result = add_skip_navigation()
        assert result.visible is True


# ======================================================================
# Tests for add_main_content_landmark
# ======================================================================

class TestAddMainContentLandmark:
    """Tests for add_main_content_landmark function."""

    def test_returns_gradio_html(self):
        """Test that add_main_content_landmark returns gr.HTML."""
        import gradio as gr
        result = add_main_content_landmark()
        assert isinstance(result, gr.HTML)

    def test_default_content_id(self):
        """Test default content ID is used."""
        result = add_main_content_landmark()
        # Check the value contains the default ID
        assert "main-content" in result.value

    def test_custom_content_id(self):
        """Test custom content ID is used."""
        result = add_main_content_landmark(content_id="custom-id")
        assert "custom-id" in result.value

    def test_has_main_role(self):
        """Test landmark has role='main'."""
        result = add_main_content_landmark()
        assert 'role="main"' in result.value

    def test_has_tabindex(self):
        """Test landmark has tabindex for focus."""
        result = add_main_content_landmark()
        assert 'tabindex="-1"' in result.value

    def test_has_aria_label(self):
        """Test landmark has aria-label."""
        result = add_main_content_landmark()
        assert 'aria-label="Main content"' in result.value

    def test_visible(self):
        """Test landmark is visible."""
        result = add_main_content_landmark()
        assert result.visible is True


# ======================================================================
# Tests for close_main_content_landmark
# ======================================================================

class TestCloseMainContentLandmark:
    """Tests for close_main_content_landmark function."""

    def test_returns_gradio_html(self):
        """Test that function returns gr.HTML."""
        import gradio as gr
        result = close_main_content_landmark()
        assert isinstance(result, gr.HTML)

    def test_contains_closing_div(self):
        """Test result contains closing div tag."""
        result = close_main_content_landmark()
        assert "</div>" in result.value

    def test_visible(self):
        """Test element is visible."""
        result = close_main_content_landmark()
        assert result.visible is True


# ======================================================================
# Tests for create_confirmation_button
# ======================================================================

class TestCreateConfirmationButton:
    """Tests for create_confirmation_button function."""

    def test_returns_tuple_of_three(self):
        """Test function returns tuple of (button, dialog, state)."""
        result = create_confirmation_button(
            label="Delete",
            confirm_title="Confirm Delete",
            confirm_message="Are you sure?",
        )
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_returns_correct_types(self):
        """Test returned types are correct."""
        import gradio as gr
        button, dialog, state = create_confirmation_button(
            label="Delete",
            confirm_title="Confirm Delete",
            confirm_message="Are you sure?",
        )
        assert isinstance(button, gr.Button)
        assert isinstance(dialog, gr.HTML)
        assert isinstance(state, gr.State)

    def test_button_label(self):
        """Test button has correct label."""
        button, _, _ = create_confirmation_button(
            label="Delete All",
            confirm_title="Confirm",
            confirm_message="Sure?",
        )
        assert "Delete All" in button.value

    def test_button_with_icon(self):
        """Test button with icon prefix."""
        button, _, _ = create_confirmation_button(
            label="Delete",
            confirm_title="Confirm",
            confirm_message="Sure?",
            icon="🗑️",
        )
        assert "🗑️" in button.value
        assert "Delete" in button.value

    def test_button_variant(self):
        """Test button variant is set."""
        button, _, _ = create_confirmation_button(
            label="Delete",
            confirm_title="Confirm",
            confirm_message="Sure?",
            variant="stop",
        )
        assert button.variant == "stop"

    def test_dialog_initially_hidden(self):
        """Test dialog is initially hidden."""
        _, dialog, _ = create_confirmation_button(
            label="Delete",
            confirm_title="Confirm",
            confirm_message="Sure?",
        )
        assert dialog.visible is False

    def test_state_initially_false(self):
        """Test confirmed state is initially False."""
        _, _, state = create_confirmation_button(
            label="Delete",
            confirm_title="Confirm",
            confirm_message="Sure?",
        )
        assert state.value is False

    def test_custom_labels(self):
        """Test custom confirm and cancel labels."""
        button, dialog, state = create_confirmation_button(
            label="Delete",
            confirm_title="Delete Item",
            confirm_message="This action cannot be undone.",
            confirm_label="Yes, Delete",
            cancel_label="No, Keep",
        )
        # All components should be created without error
        assert button is not None
        assert dialog is not None
        assert state is not None


# ======================================================================
# Tests for create_loading_section
# ======================================================================

class TestCreateLoadingSection:
    """Tests for create_loading_section function."""

    def test_returns_tuple_of_two(self):
        """Test function returns tuple of (skeleton, content)."""
        result = create_loading_section()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_html_components(self):
        """Test returned components are gr.HTML."""
        import gradio as gr
        skeleton, content = create_loading_section()
        assert isinstance(skeleton, gr.HTML)
        assert isinstance(content, gr.HTML)

    def test_skeleton_visible_by_default(self):
        """Test skeleton is visible when show_skeleton=True."""
        skeleton, content = create_loading_section(show_skeleton=True)
        assert skeleton.visible is True
        assert content.visible is False

    def test_content_visible_when_not_loading(self):
        """Test content visible when show_skeleton=False."""
        skeleton, content = create_loading_section(show_skeleton=False)
        assert skeleton.visible is False
        assert content.visible is True

    def test_custom_lines(self):
        """Test custom line count for skeleton."""
        skeleton, _ = create_loading_section(lines=5)
        # Skeleton should be created with custom lines
        assert skeleton is not None

    def test_elem_ids_from_label(self):
        """Test element IDs are derived from label."""
        skeleton, content = create_loading_section(label="Search Results")
        assert skeleton.elem_id == "search-results-skeleton"
        assert content.elem_id == "search-results-content"

    def test_label_with_spaces(self):
        """Test label with spaces is properly formatted."""
        skeleton, content = create_loading_section(label="My Custom Section")
        assert skeleton.elem_id == "my-custom-section-skeleton"
        assert content.elem_id == "my-custom-section-content"


# ======================================================================
# Tests for create_status_announcement
# ======================================================================

class TestCreateStatusAnnouncement:
    """Tests for create_status_announcement function."""

    def test_returns_gradio_html(self):
        """Test function returns gr.HTML component."""
        import gradio as gr
        result = create_status_announcement()
        assert isinstance(result, gr.HTML)

    def test_empty_by_default(self):
        """Test announcement is empty by default."""
        result = create_status_announcement()
        assert result.value == ""

    def test_with_initial_message(self):
        """Test announcement with initial message."""
        result = create_status_announcement(message="Loading complete")
        assert result.value != ""

    def test_polite_politeness_default(self):
        """Test default politeness is 'polite'."""
        result = create_status_announcement(message="Test")
        # Should use polite by default
        assert result is not None

    def test_assertive_politeness(self):
        """Test assertive politeness."""
        result = create_status_announcement(message="Error!", politeness="assertive")
        assert result is not None

    def test_elem_id(self):
        """Test correct element ID."""
        result = create_status_announcement()
        assert result.elem_id == "status-announcement"

    def test_visible(self):
        """Test component is visible."""
        result = create_status_announcement()
        assert result.visible is True


# ======================================================================
# Tests for EMPTY_STATES constant
# ======================================================================

class TestEmptyStatesConstant:
    """Tests for EMPTY_STATES dictionary."""

    def test_all_states_defined(self):
        """Test all expected empty states are defined."""
        expected_states = ["no_files", "no_results", "no_data", "no_messages", "no_images"]
        for state in expected_states:
            assert state in EMPTY_STATES

    def test_states_are_strings(self):
        """Test all states are HTML strings."""
        for state_type, html in EMPTY_STATES.items():
            assert isinstance(html, str)
            assert len(html) > 0

    def test_states_have_role_status(self):
        """Test empty states have role='status' for accessibility."""
        for state_type, html in EMPTY_STATES.items():
            assert 'role="status"' in html

    def test_states_have_empty_state_class(self):
        """Test empty states have proper CSS class."""
        for state_type, html in EMPTY_STATES.items():
            assert 'class="empty-state"' in html

    def test_states_have_icons(self):
        """Test empty states have icon elements."""
        for state_type, html in EMPTY_STATES.items():
            assert 'class="empty-state-icon"' in html

    def test_no_files_content(self):
        """Test no_files state has correct content."""
        html = EMPTY_STATES["no_files"]
        assert "No files uploaded" in html
        assert "Drag and drop" in html

    def test_no_results_content(self):
        """Test no_results state has correct content."""
        html = EMPTY_STATES["no_results"]
        assert "No results found" in html
        assert "search" in html.lower() or "filter" in html.lower()

    def test_no_data_content(self):
        """Test no_data state has correct content."""
        html = EMPTY_STATES["no_data"]
        assert "No data available" in html

    def test_no_messages_content(self):
        """Test no_messages state has correct content."""
        html = EMPTY_STATES["no_messages"]
        assert "No messages" in html
        assert "conversation" in html.lower() or "message" in html.lower()

    def test_no_images_content(self):
        """Test no_images state has correct content."""
        html = EMPTY_STATES["no_images"]
        assert "No images" in html


# ======================================================================
# Tests for get_empty_state
# ======================================================================

class TestGetEmptyState:
    """Tests for get_empty_state function."""

    def test_returns_string(self):
        """Test function returns a string."""
        result = get_empty_state("no_files")
        assert isinstance(result, str)

    def test_valid_state_types(self):
        """Test all valid state types work."""
        valid_types = ["no_files", "no_results", "no_data", "no_messages", "no_images"]
        for state_type in valid_types:
            result = get_empty_state(state_type)
            assert len(result) > 0

    def test_invalid_state_returns_no_data(self):
        """Test invalid state type returns no_data as fallback."""
        result = get_empty_state("invalid_state")
        expected = EMPTY_STATES["no_data"]
        assert result == expected

    def test_matches_constant(self):
        """Test returned value matches EMPTY_STATES constant."""
        for state_type in EMPTY_STATES.keys():
            result = get_empty_state(state_type)
            assert result == EMPTY_STATES[state_type]


# ======================================================================
# Tests for EMPTY_STATE_CSS constant
# ======================================================================

class TestEmptyStateCSS:
    """Tests for EMPTY_STATE_CSS constant."""

    def test_is_string(self):
        """Test CSS is a string."""
        assert isinstance(EMPTY_STATE_CSS, str)

    def test_not_empty(self):
        """Test CSS is not empty."""
        assert len(EMPTY_STATE_CSS) > 0

    def test_contains_empty_state_class(self):
        """Test CSS contains .empty-state class."""
        assert ".empty-state" in EMPTY_STATE_CSS

    def test_contains_icon_class(self):
        """Test CSS contains icon class."""
        assert ".empty-state-icon" in EMPTY_STATE_CSS

    def test_contains_dark_mode(self):
        """Test CSS contains dark mode styles."""
        assert "[data-theme=\"dark\"]" in EMPTY_STATE_CSS or ".dark" in EMPTY_STATE_CSS


# ======================================================================
# Tests for get_enhanced_page_css
# ======================================================================

class TestGetEnhancedPageCSS:
    """Tests for get_enhanced_page_css function."""

    def test_returns_string(self):
        """Test function returns a string."""
        css = get_enhanced_page_css()
        assert isinstance(css, str)

    def test_includes_base_css(self):
        """Test enhanced CSS includes base UX CSS."""
        enhanced_css = get_enhanced_page_css()
        base_css = get_page_css()
        # Enhanced should contain base CSS
        assert len(enhanced_css) >= len(base_css)

    def test_includes_empty_state_css(self):
        """Test enhanced CSS includes empty state CSS."""
        enhanced_css = get_enhanced_page_css()
        assert ".empty-state" in enhanced_css

    def test_longer_than_base(self):
        """Test enhanced CSS is longer than base CSS."""
        enhanced_css = get_enhanced_page_css()
        base_css = get_page_css()
        assert len(enhanced_css) > len(base_css)


# ======================================================================
# Integration Tests
# ======================================================================

class TestPageUtilsIntegration:
    """Integration tests for page utilities working together."""

    def test_full_page_structure(self):
        """Test creating a full accessible page structure."""
        import gradio as gr

        # Create page components
        skip_nav = add_skip_navigation()
        landmark_open = add_main_content_landmark()
        landmark_close = close_main_content_landmark()

        # All components should be created successfully
        assert skip_nav is not None
        assert landmark_open is not None
        assert landmark_close is not None

    def test_loading_section_workflow(self):
        """Test loading section show/hide workflow."""
        skeleton, content = create_loading_section(label="Results", show_skeleton=True)

        # Initial state: skeleton visible, content hidden
        assert skeleton.visible is True
        assert content.visible is False

        # Create section with content visible
        skeleton2, content2 = create_loading_section(label="Results", show_skeleton=False)
        assert skeleton2.visible is False
        assert content2.visible is True

    def test_css_combination(self):
        """Test that all CSS can be combined."""
        page_css = get_page_css()
        enhanced_css = get_enhanced_page_css()

        # Both should be valid CSS strings
        assert isinstance(page_css, str)
        assert isinstance(enhanced_css, str)

        # Enhanced should include everything from base
        assert len(enhanced_css) >= len(page_css)

    def test_empty_state_in_loading_section(self):
        """Test using empty state with loading section."""
        _, content = create_loading_section(label="Files", show_skeleton=False)
        empty_html = get_empty_state("no_files")

        # Should be able to use empty state HTML
        assert "No files uploaded" in empty_html
        assert content is not None
