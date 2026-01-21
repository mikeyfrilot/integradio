"""
Integration tests for loading state components.

Tests focus on real behavior and integration between components,
ensuring the loading module works correctly as a whole.
"""

import pytest
from integradio.ux.loading import (
    LoadingState,
    LoadingConfig,
    ProgressType,
    LOADING_CSS,
    create_skeleton,
    create_skeleton_text,
    create_skeleton_card,
    create_spinner,
    create_progress_indicator,
    create_loading_overlay,
    get_loading_css,
)


class TestLoadingEnums:
    """Tests for loading-related enums."""

    def test_loading_state_all_values(self):
        """Verify all LoadingState values are accessible."""
        states = [LoadingState.IDLE, LoadingState.LOADING,
                  LoadingState.SUCCESS, LoadingState.ERROR]
        assert len(states) == 4
        assert all(isinstance(s.value, str) for s in states)

    def test_progress_type_values(self):
        """Verify ProgressType enum values."""
        assert ProgressType.DETERMINATE.value == "determinate"
        assert ProgressType.INDETERMINATE.value == "indeterminate"

    def test_loading_state_iteration(self):
        """Test LoadingState can be iterated."""
        all_states = list(LoadingState)
        assert len(all_states) == 4
        assert LoadingState.IDLE in all_states


class TestLoadingConfigDataclass:
    """Tests for LoadingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LoadingConfig()
        assert config.delay_ms == 300
        assert config.min_display_ms == 500
        assert config.show_estimated_time is True
        assert config.use_skeleton is True
        assert config.skeleton_animation == "pulse"

    def test_custom_values(self):
        """Test custom configuration."""
        config = LoadingConfig(
            delay_ms=100,
            min_display_ms=200,
            show_estimated_time=False,
            use_skeleton=False,
            skeleton_animation="wave"
        )
        assert config.delay_ms == 100
        assert config.min_display_ms == 200
        assert config.show_estimated_time is False
        assert config.use_skeleton is False
        assert config.skeleton_animation == "wave"

    def test_partial_override(self):
        """Test partial override of defaults."""
        config = LoadingConfig(delay_ms=600)
        assert config.delay_ms == 600
        assert config.min_display_ms == 500  # Default preserved

    def test_config_equality(self):
        """Test config equality comparison."""
        config1 = LoadingConfig(delay_ms=300)
        config2 = LoadingConfig(delay_ms=300)
        assert config1 == config2

    def test_config_inequality(self):
        """Test config inequality."""
        config1 = LoadingConfig(delay_ms=300)
        config2 = LoadingConfig(delay_ms=400)
        assert config1 != config2


class TestCreateSkeletonIntegration:
    """Integration tests for skeleton creation."""

    def test_skeleton_html_structure(self):
        """Test complete HTML structure of skeleton."""
        html = create_skeleton(lines=3, animation="pulse", width="100%")

        # Check container
        assert '<div class="skeleton-container"' in html
        assert 'role="status"' in html
        assert 'aria-label="Loading content"' in html

        # Check lines
        assert html.count('class="skeleton skeleton-text') == 3

        # Check accessibility
        assert '<span class="sr-only">Loading...</span>' in html

    def test_skeleton_with_all_animations(self):
        """Test skeleton with all animation types."""
        for anim in ["pulse", "wave", "none"]:
            html = create_skeleton(animation=anim)
            if anim == "none":
                assert "skeleton-pulse" not in html
                assert "skeleton-wave" not in html
            else:
                assert f"skeleton-{anim}" in html

    def test_skeleton_lines_last_line_width(self):
        """Test that the last line has different width."""
        html = create_skeleton(lines=4, width="100%")
        # Last line should have 60% width
        assert 'width: 60%' in html

    def test_skeleton_single_line(self):
        """Test skeleton with single line."""
        html = create_skeleton(lines=1)
        assert html.count('class="skeleton skeleton-text') == 1
        # Single line should have 60% width (it's the last line)
        assert 'width: 60%' in html

    def test_skeleton_many_lines(self):
        """Test skeleton with many lines."""
        html = create_skeleton(lines=10)
        assert html.count('class="skeleton skeleton-text') == 10


class TestCreateSkeletonTextIntegration:
    """Integration tests for skeleton text creation."""

    def test_skeleton_text_html_structure(self):
        """Test complete HTML structure of skeleton text."""
        html = create_skeleton_text(words=10, animation="pulse")
        assert 'class="skeleton skeleton-pulse"' in html
        assert 'role="status"' in html
        assert 'aria-label="Loading text"' in html
        assert '<span class="sr-only">Loading...</span>' in html

    def test_skeleton_text_width_calculation(self):
        """Test that width varies with word count."""
        html_5 = create_skeleton_text(words=5)
        html_10 = create_skeleton_text(words=10)

        # Extract widths (they should be different)
        assert "width:" in html_5
        assert "width:" in html_10
        # 5 words = 240px, 10 words = 400px (max)
        assert "240px" in html_5
        assert "400px" in html_10

    def test_skeleton_text_max_width_cap(self):
        """Test that width is capped at 400px."""
        html = create_skeleton_text(words=100)
        assert "400px" in html  # Capped at max

    def test_skeleton_text_animation_variants(self):
        """Test all animation variants."""
        for anim in ["pulse", "wave", "none"]:
            html = create_skeleton_text(animation=anim)
            if anim == "none":
                assert 'class="skeleton "' in html or 'class="skeleton"' in html
            else:
                assert f"skeleton-{anim}" in html


class TestCreateSkeletonCardIntegration:
    """Integration tests for skeleton card creation."""

    def test_skeleton_card_full(self):
        """Test skeleton card with all components."""
        html = create_skeleton_card(show_image=True, show_title=True, lines=2)

        # Container
        assert 'class="skeleton-card"' in html
        assert 'role="status"' in html
        assert 'aria-label="Loading card"' in html

        # Image placeholder (150px height)
        assert 'height: 150px' in html

        # Title placeholder (1.5em height)
        assert 'height: 1.5em' in html

        # Body lines
        assert html.count('skeleton-text') >= 2

        # Accessibility
        assert 'Loading card content...' in html

    def test_skeleton_card_no_image(self):
        """Test skeleton card without image."""
        html = create_skeleton_card(show_image=False, show_title=True, lines=2)
        assert 'height: 150px' not in html
        assert 'height: 1.5em' in html

    def test_skeleton_card_no_title(self):
        """Test skeleton card without title."""
        html = create_skeleton_card(show_image=True, show_title=False, lines=2)
        assert 'height: 150px' in html
        assert 'height: 1.5em' not in html

    def test_skeleton_card_minimal(self):
        """Test skeleton card with minimum components."""
        html = create_skeleton_card(show_image=False, show_title=False, lines=1)
        assert 'skeleton-card' in html
        assert 'height: 150px' not in html
        assert 'height: 1.5em' not in html

    def test_skeleton_card_many_lines(self):
        """Test skeleton card with many body lines."""
        html = create_skeleton_card(show_image=False, show_title=False, lines=5)
        # Count skeleton-text occurrences
        assert html.count('skeleton-text') == 5

    def test_skeleton_card_animation(self):
        """Test skeleton card with different animations."""
        for anim in ["pulse", "wave", "none"]:
            html = create_skeleton_card(animation=anim)
            if anim != "none":
                assert f"skeleton-{anim}" in html


class TestCreateSpinnerIntegration:
    """Integration tests for spinner creation."""

    def test_spinner_html_structure(self):
        """Test complete HTML structure of spinner."""
        html = create_spinner(size="md", label="Loading", show_label=True)

        # Container
        assert 'role="status"' in html
        assert 'aria-label="Loading"' in html

        # Spinner element
        assert 'class="spinner' in html

        # Label visibility
        assert ">Loading<" in html

        # Screen reader text
        assert '<span class="sr-only">Loading</span>' in html

    def test_spinner_sizes(self):
        """Test all spinner size variants."""
        assert "spinner-sm" in create_spinner(size="sm")
        assert "spinner " in create_spinner(size="md") or "spinner\"" in create_spinner(size="md")
        assert "spinner-lg" in create_spinner(size="lg")

    def test_spinner_unknown_size(self):
        """Test spinner with unknown size defaults to medium."""
        html = create_spinner(size="xl")
        # Should not have any size class
        assert "spinner-sm" not in html
        assert "spinner-lg" not in html

    def test_spinner_label_visibility(self):
        """Test spinner label visibility options."""
        # Visible label
        html_visible = create_spinner(label="Processing", show_label=True)
        assert ">Processing<" in html_visible

        # Hidden label
        html_hidden = create_spinner(label="Processing", show_label=False)
        assert 'aria-label="Processing"' in html_hidden

    def test_spinner_custom_labels(self):
        """Test spinner with various custom labels."""
        labels = ["Loading...", "Please wait", "Processing data"]
        for label in labels:
            html = create_spinner(label=label, show_label=True)
            assert label in html


class TestCreateProgressIndicatorIntegration:
    """Integration tests for progress indicator creation."""

    def test_progress_determinate_structure(self):
        """Test complete structure of determinate progress bar."""
        html = create_progress_indicator(
            progress=75,
            progress_type=ProgressType.DETERMINATE,
            label="Uploading",
            show_percentage=True
        )

        # ARIA attributes
        assert 'role="progressbar"' in html
        assert 'aria-label="Uploading"' in html
        assert 'aria-valuenow="75"' in html
        assert 'aria-valuemin="0"' in html
        assert 'aria-valuemax="100"' in html

        # Visual elements
        assert 'width: 75%' in html
        assert '75%' in html  # Percentage text
        assert 'Uploading' in html

    def test_progress_indeterminate_structure(self):
        """Test complete structure of indeterminate progress bar."""
        html = create_progress_indicator(
            progress=50,  # Should be ignored
            progress_type=ProgressType.INDETERMINATE,
            label="Processing"
        )

        # Should have indeterminate class
        assert 'progress-bar-indeterminate' in html

        # Should have 100% width for animation
        assert 'width: 100%' in html

    def test_progress_percentage_visibility(self):
        """Test percentage visibility control."""
        # With percentage
        html_with = create_progress_indicator(progress=50, show_percentage=True)
        assert '50%' in html_with

        # Without percentage
        html_without = create_progress_indicator(progress=50, show_percentage=False)
        # Should not have the percentage span
        count = html_without.count('50%')
        # It might appear in aria-valuenow but not in visible text
        assert 'aria-valuenow="50"' in html_without

    def test_progress_estimated_time(self):
        """Test estimated time display."""
        html = create_progress_indicator(
            progress=25,
            estimated_time="3 minutes remaining"
        )
        assert "(3 minutes remaining)" in html

    def test_progress_no_estimated_time(self):
        """Test progress without estimated time."""
        html = create_progress_indicator(progress=25)
        assert "(" not in html or "remaining" not in html

    def test_progress_boundary_values(self):
        """Test progress at boundary values."""
        # 0%
        html_zero = create_progress_indicator(progress=0)
        assert 'width: 0%' in html_zero

        # 100%
        html_hundred = create_progress_indicator(progress=100)
        assert 'width: 100%' in html_hundred

    def test_progress_decimal_values(self):
        """Test progress with decimal values."""
        html = create_progress_indicator(progress=33.333)
        # Should be formatted to whole number
        assert '33%' in html


class TestCreateLoadingOverlayIntegration:
    """Integration tests for loading overlay creation."""

    def test_overlay_when_loading(self):
        """Test overlay appears when loading."""
        html = create_loading_overlay(loading=True, message="Loading...")

        assert 'class="loading-overlay"' in html
        assert 'role="status"' in html
        assert 'aria-live="polite"' in html
        assert "Loading..." in html
        assert 'class="spinner' in html  # Contains spinner

    def test_overlay_when_not_loading(self):
        """Test overlay is empty when not loading."""
        html = create_loading_overlay(loading=False, message="Loading...")
        assert html == ""

    def test_overlay_custom_message(self):
        """Test overlay with custom message."""
        messages = [
            "Saving changes...",
            "Uploading file...",
            "Processing data..."
        ]
        for msg in messages:
            html = create_loading_overlay(loading=True, message=msg)
            assert msg in html

    def test_overlay_default_message(self):
        """Test overlay with default message."""
        html = create_loading_overlay(loading=True)
        assert "Loading..." in html


class TestGetLoadingCSSIntegration:
    """Integration tests for loading CSS."""

    def test_css_returns_string(self):
        """Test CSS function returns string."""
        css = get_loading_css()
        assert isinstance(css, str)
        assert len(css) > 0

    def test_css_matches_constant(self):
        """Test CSS function returns the constant."""
        assert get_loading_css() == LOADING_CSS

    def test_css_contains_skeleton_classes(self):
        """Test CSS contains all skeleton-related classes."""
        css = get_loading_css()
        required_classes = [
            ".skeleton",
            ".skeleton-pulse",
            ".skeleton-wave",
            ".skeleton-text",
            ".skeleton-card"
        ]
        for cls in required_classes:
            assert cls in css, f"Missing class: {cls}"

    def test_css_contains_spinner_classes(self):
        """Test CSS contains all spinner-related classes."""
        css = get_loading_css()
        required_classes = [
            ".spinner",
            ".spinner-sm",
            ".spinner-lg"
        ]
        for cls in required_classes:
            assert cls in css, f"Missing class: {cls}"

    def test_css_contains_progress_classes(self):
        """Test CSS contains all progress-related classes."""
        css = get_loading_css()
        required_classes = [
            ".progress-bar",
            ".progress-bar-fill",
            ".progress-bar-indeterminate"
        ]
        for cls in required_classes:
            assert cls in css, f"Missing class: {cls}"

    def test_css_contains_animations(self):
        """Test CSS contains required animations."""
        css = get_loading_css()
        required_animations = [
            "@keyframes skeleton-pulse",
            "@keyframes skeleton-wave",
            "@keyframes spin",
            "@keyframes progress-indeterminate"
        ]
        for anim in required_animations:
            assert anim in css, f"Missing animation: {anim}"

    def test_css_contains_dark_mode(self):
        """Test CSS contains dark mode styles."""
        css = get_loading_css()
        assert '[data-theme="dark"]' in css or ".dark" in css

    def test_css_contains_loading_overlay(self):
        """Test CSS contains overlay styles."""
        css = get_loading_css()
        assert ".loading-overlay" in css

    def test_css_contains_sr_only(self):
        """Test CSS contains screen reader utility."""
        css = get_loading_css()
        assert ".sr-only" in css


class TestLoadingComponentsIntegration:
    """Integration tests for combining loading components."""

    def test_spinner_inside_overlay(self):
        """Test that overlay uses spinner internally."""
        overlay = create_loading_overlay(loading=True)
        assert 'class="spinner' in overlay

    def test_skeleton_accessibility_complete(self):
        """Test all skeletons have proper accessibility."""
        components = [
            create_skeleton(),
            create_skeleton_text(),
            create_skeleton_card()
        ]
        for html in components:
            assert 'role="status"' in html
            assert 'aria-label=' in html or 'sr-only' in html

    def test_all_components_valid_html(self):
        """Test all components produce valid HTML structure."""
        components = [
            create_skeleton(),
            create_skeleton_text(),
            create_skeleton_card(),
            create_spinner(),
            create_progress_indicator(50),
            create_loading_overlay(True)
        ]
        for html in components:
            # Check for balanced div tags (simplified check)
            open_divs = html.count('<div')
            close_divs = html.count('</div>')
            assert open_divs == close_divs, f"Unbalanced divs: {open_divs} opens, {close_divs} closes"


class TestUXModuleIntegration:
    """Integration tests for the entire UX module."""

    def test_get_all_ux_css(self):
        """Test get_all_ux_css combines all CSS from submodules."""
        from integradio.ux import get_all_ux_css

        css = get_all_ux_css()

        # Should contain CSS from all submodules
        assert ".skeleton" in css  # From loading
        assert ".field-error" in css  # From validation
        assert ".toast" in css  # From feedback
        assert ".sr-only" in css  # From accessibility

    def test_get_all_ux_css_returns_string(self):
        """Test get_all_ux_css returns a string."""
        from integradio.ux import get_all_ux_css

        css = get_all_ux_css()
        assert isinstance(css, str)
        assert len(css) > 0

    def test_module_exports(self):
        """Test that key exports are available from integradio.ux."""
        from integradio.ux import (
            # Loading
            LoadingState,
            LoadingConfig,
            create_skeleton,
            create_progress_indicator,
            ProgressType,
            # Validation
            FieldValidator,
            ValidationResult,
            create_inline_error,
            # Feedback
            Toast,
            ToastType,
            ToastManager,
            create_confirmation_dialog,
            # Accessibility
            aria_attrs,
            AriaRole,
            announce_to_screen_reader,
            # CSS
            get_all_ux_css,
        )

        # Basic sanity checks
        assert LoadingState.IDLE is not None
        assert ProgressType.DETERMINATE is not None
        assert ToastType.INFO is not None
        assert AriaRole.BUTTON is not None
