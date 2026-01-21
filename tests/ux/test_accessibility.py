"""
Integration tests for accessibility utilities.

Tests focus on ARIA attributes, keyboard navigation support,
and screen reader announcements.
"""

import pytest
from integradio.ux.accessibility import (
    AriaRole,
    AriaLive,
    AriaAttrs,
    ACCESSIBILITY_CSS,
    aria_attrs,
    announce_to_screen_reader,
    create_skip_link,
    create_focus_trap,
    keyboard_shortcut,
    visible_focus_styles,
    get_accessibility_css,
    create_landmark,
    create_loading_region,
    format_field_description,
)


class TestAriaRoleEnum:
    """Tests for AriaRole enum."""

    def test_common_roles(self):
        """Verify common ARIA roles exist."""
        common_roles = [
            ("ALERT", "alert"),
            ("BUTTON", "button"),
            ("DIALOG", "dialog"),
            ("FORM", "form"),
            ("NAVIGATION", "navigation"),
            ("SEARCH", "search"),
            ("STATUS", "status"),
            ("TAB", "tab"),
            ("TABLIST", "tablist"),
            ("TABPANEL", "tabpanel"),
        ]
        for attr, value in common_roles:
            assert hasattr(AriaRole, attr)
            assert getattr(AriaRole, attr).value == value

    def test_all_roles_have_values(self):
        """Test all roles have string values."""
        for role in AriaRole:
            assert isinstance(role.value, str)
            assert len(role.value) > 0

    def test_role_iteration(self):
        """Test AriaRole can be iterated."""
        all_roles = list(AriaRole)
        assert len(all_roles) > 0


class TestAriaLiveEnum:
    """Tests for AriaLive enum."""

    def test_live_values(self):
        """Verify AriaLive values."""
        assert AriaLive.OFF.value == "off"
        assert AriaLive.POLITE.value == "polite"
        assert AriaLive.ASSERTIVE.value == "assertive"

    def test_live_iteration(self):
        """Test AriaLive can be iterated."""
        all_levels = list(AriaLive)
        assert len(all_levels) == 3


class TestAriaAttrsDataclass:
    """Tests for AriaAttrs dataclass."""

    def test_default_attrs(self):
        """Test default attribute values."""
        attrs = AriaAttrs()
        assert attrs.role is None
        assert attrs.label is None
        assert attrs.live is None
        assert attrs.disabled is None

    def test_custom_attrs(self):
        """Test custom attribute values."""
        attrs = AriaAttrs(
            role=AriaRole.BUTTON,
            label="Submit form",
            disabled=False,
            pressed=True
        )
        assert attrs.role == AriaRole.BUTTON
        assert attrs.label == "Submit form"
        assert attrs.disabled is False
        assert attrs.pressed is True

    def test_to_html_attrs_role(self):
        """Test HTML output for role."""
        attrs = AriaAttrs(role=AriaRole.BUTTON)
        html = attrs.to_html_attrs()
        assert 'role="button"' in html

    def test_to_html_attrs_label(self):
        """Test HTML output for label."""
        attrs = AriaAttrs(label="Search")
        html = attrs.to_html_attrs()
        assert 'aria-label="Search"' in html

    def test_to_html_attrs_labelledby(self):
        """Test HTML output for labelledby."""
        attrs = AriaAttrs(labelledby="title-id")
        html = attrs.to_html_attrs()
        assert 'aria-labelledby="title-id"' in html

    def test_to_html_attrs_describedby(self):
        """Test HTML output for describedby."""
        attrs = AriaAttrs(describedby="desc-id")
        html = attrs.to_html_attrs()
        assert 'aria-describedby="desc-id"' in html

    def test_to_html_attrs_live(self):
        """Test HTML output for live region."""
        attrs = AriaAttrs(live=AriaLive.POLITE)
        html = attrs.to_html_attrs()
        assert 'aria-live="polite"' in html

    def test_to_html_attrs_atomic(self):
        """Test HTML output for atomic."""
        attrs = AriaAttrs(atomic=True)
        html = attrs.to_html_attrs()
        assert 'aria-atomic="true"' in html

    def test_to_html_attrs_busy(self):
        """Test HTML output for busy."""
        attrs = AriaAttrs(busy=True)
        html = attrs.to_html_attrs()
        assert 'aria-busy="true"' in html

    def test_to_html_attrs_controls(self):
        """Test HTML output for controls."""
        attrs = AriaAttrs(controls="menu-panel")
        html = attrs.to_html_attrs()
        assert 'aria-controls="menu-panel"' in html

    def test_to_html_attrs_current(self):
        """Test HTML output for current."""
        attrs = AriaAttrs(current="page")
        html = attrs.to_html_attrs()
        assert 'aria-current="page"' in html

    def test_to_html_attrs_disabled(self):
        """Test HTML output for disabled."""
        attrs = AriaAttrs(disabled=True)
        html = attrs.to_html_attrs()
        assert 'aria-disabled="true"' in html

    def test_to_html_attrs_expanded(self):
        """Test HTML output for expanded."""
        attrs = AriaAttrs(expanded=False)
        html = attrs.to_html_attrs()
        assert 'aria-expanded="false"' in html

    def test_to_html_attrs_haspopup(self):
        """Test HTML output for haspopup."""
        attrs = AriaAttrs(haspopup="menu")
        html = attrs.to_html_attrs()
        assert 'aria-haspopup="menu"' in html

    def test_to_html_attrs_hidden(self):
        """Test HTML output for hidden."""
        attrs = AriaAttrs(hidden=True)
        html = attrs.to_html_attrs()
        assert 'aria-hidden="true"' in html

    def test_to_html_attrs_invalid(self):
        """Test HTML output for invalid."""
        attrs = AriaAttrs(invalid=True)
        html = attrs.to_html_attrs()
        assert 'aria-invalid="true"' in html

    def test_to_html_attrs_pressed(self):
        """Test HTML output for pressed."""
        attrs = AriaAttrs(pressed=True)
        html = attrs.to_html_attrs()
        assert 'aria-pressed="true"' in html

    def test_to_html_attrs_required(self):
        """Test HTML output for required."""
        attrs = AriaAttrs(required=True)
        html = attrs.to_html_attrs()
        assert 'aria-required="true"' in html

    def test_to_html_attrs_selected(self):
        """Test HTML output for selected."""
        attrs = AriaAttrs(selected=True)
        html = attrs.to_html_attrs()
        assert 'aria-selected="true"' in html

    def test_to_html_attrs_valuenow(self):
        """Test HTML output for valuenow."""
        attrs = AriaAttrs(valuenow=50)
        html = attrs.to_html_attrs()
        assert 'aria-valuenow="50"' in html

    def test_to_html_attrs_valuemin_valuemax(self):
        """Test HTML output for value range."""
        attrs = AriaAttrs(valuemin=0, valuemax=100)
        html = attrs.to_html_attrs()
        assert 'aria-valuemin="0"' in html
        assert 'aria-valuemax="100"' in html

    def test_to_html_attrs_valuetext(self):
        """Test HTML output for valuetext."""
        attrs = AriaAttrs(valuetext="50 percent")
        html = attrs.to_html_attrs()
        assert 'aria-valuetext="50 percent"' in html

    def test_to_html_attrs_multiple(self):
        """Test HTML output with multiple attributes."""
        attrs = AriaAttrs(
            role=AriaRole.BUTTON,
            label="Submit",
            disabled=False,
            expanded=True
        )
        html = attrs.to_html_attrs()
        assert 'role="button"' in html
        assert 'aria-label="Submit"' in html
        assert 'aria-disabled="false"' in html
        assert 'aria-expanded="true"' in html

    def test_to_html_attrs_empty(self):
        """Test HTML output with no attributes."""
        attrs = AriaAttrs()
        html = attrs.to_html_attrs()
        assert html == ""


class TestAriaAttrsFunction:
    """Tests for aria_attrs helper function."""

    def test_basic_attrs(self):
        """Test basic attribute generation."""
        html = aria_attrs(label="Search", expanded=False)
        assert 'aria-label="Search"' in html
        assert 'aria-expanded="false"' in html

    def test_role_attr(self):
        """Test role attribute."""
        html = aria_attrs(role="button")
        assert 'role="button"' in html

    def test_role_enum(self):
        """Test role with enum."""
        html = aria_attrs(role=AriaRole.DIALOG)
        assert 'role="dialog"' in html

    def test_boolean_values(self):
        """Test boolean value conversion."""
        html = aria_attrs(disabled=True, hidden=False)
        assert 'aria-disabled="true"' in html
        assert 'aria-hidden="false"' in html

    def test_live_enum(self):
        """Test AriaLive enum conversion."""
        html = aria_attrs(live=AriaLive.ASSERTIVE)
        assert 'aria-live="assertive"' in html

    def test_none_values_skipped(self):
        """Test None values are skipped."""
        html = aria_attrs(label="Test", expanded=None)
        assert 'aria-label="Test"' in html
        assert 'aria-expanded' not in html

    def test_snake_case_conversion(self):
        """Test snake_case to aria-kebab-case conversion."""
        # Note: This depends on implementation - some may not convert
        html = aria_attrs(controls="panel")
        assert 'aria-controls="panel"' in html


class TestAnnounceToScreenReader:
    """Tests for announce_to_screen_reader function."""

    def test_basic_announcement(self):
        """Test basic screen reader announcement."""
        html = announce_to_screen_reader("Form submitted successfully")
        assert 'class="sr-only"' in html
        assert 'aria-live="polite"' in html
        assert 'aria-atomic="true"' in html
        assert "Form submitted successfully" in html

    def test_polite_announcement(self):
        """Test polite announcement."""
        html = announce_to_screen_reader("Message", AriaLive.POLITE)
        assert 'aria-live="polite"' in html

    def test_assertive_announcement(self):
        """Test assertive announcement."""
        html = announce_to_screen_reader("Error!", AriaLive.ASSERTIVE)
        assert 'aria-live="assertive"' in html

    def test_atomic_true(self):
        """Test atomic=True."""
        html = announce_to_screen_reader("Message", atomic=True)
        assert 'aria-atomic="true"' in html

    def test_atomic_false(self):
        """Test atomic=False."""
        html = announce_to_screen_reader("Message", atomic=False)
        assert 'aria-atomic="false"' in html


class TestCreateSkipLink:
    """Tests for create_skip_link function."""

    def test_basic_skip_link(self):
        """Test basic skip link."""
        html = create_skip_link()
        assert '<a href="#main-content"' in html
        assert 'class="skip-link"' in html
        assert "Skip to main content" in html

    def test_custom_target(self):
        """Test skip link with custom target."""
        html = create_skip_link(target_id="content-area")
        assert 'href="#content-area"' in html

    def test_custom_label(self):
        """Test skip link with custom label."""
        html = create_skip_link(label="Skip navigation")
        assert "Skip navigation" in html


class TestCreateFocusTrap:
    """Tests for create_focus_trap function."""

    def test_basic_focus_trap(self):
        """Test basic focus trap."""
        content = "<div>Modal content</div>"
        html = create_focus_trap(content)

        assert 'id="focus-trap"' in html
        assert 'class="focus-trap"' in html
        assert 'tabindex="-1"' in html
        assert 'class="focus-trap-start"' in html
        assert 'class="focus-trap-end"' in html
        assert "Modal content" in html

    def test_custom_trap_id(self):
        """Test focus trap with custom ID."""
        html = create_focus_trap("<div>Content</div>", trap_id="modal-trap")
        assert 'id="modal-trap"' in html

    def test_trap_sentinels(self):
        """Test focus trap sentinel elements."""
        html = create_focus_trap("<div>Content</div>")

        # Start sentinel
        assert 'class="focus-trap-start" tabindex="0"' in html
        assert 'aria-hidden="true"' in html

        # End sentinel
        assert 'class="focus-trap-end" tabindex="0"' in html


class TestKeyboardShortcut:
    """Tests for keyboard_shortcut function."""

    def test_single_key(self):
        """Test single key shortcut."""
        html = keyboard_shortcut("Enter")
        assert '<kbd class="kbd">Enter</kbd>' in html
        assert 'class="keyboard-shortcut"' in html

    def test_key_with_modifier(self):
        """Test key with modifier."""
        html = keyboard_shortcut("S", modifier="Ctrl")
        assert '<kbd class="kbd">Ctrl</kbd>' in html
        assert '<kbd class="kbd">S</kbd>' in html
        assert "+" in html

    def test_key_with_description(self):
        """Test key with description."""
        html = keyboard_shortcut("S", modifier="Ctrl", description="Save")
        assert "Save" in html
        assert 'class="shortcut-desc"' in html

    def test_key_without_description(self):
        """Test key without description."""
        html = keyboard_shortcut("Escape")
        assert 'class="shortcut-desc"' not in html


class TestVisibleFocusStyles:
    """Tests for visible_focus_styles function."""

    def test_focus_styles_content(self):
        """Test focus styles CSS content."""
        css = visible_focus_styles()

        assert ":focus-visible" in css
        assert "outline:" in css
        assert "outline-offset:" in css

    def test_focus_styles_for_inputs(self):
        """Test focus styles for form inputs."""
        css = visible_focus_styles()

        assert "input:focus-visible" in css
        assert "textarea:focus-visible" in css
        assert "select:focus-visible" in css
        assert "button:focus-visible" in css

    def test_no_focus_outline_for_mouse(self):
        """Test focus outline removed for mouse users."""
        css = visible_focus_styles()
        assert ":focus:not(:focus-visible)" in css


class TestGetAccessibilityCSS:
    """Tests for get_accessibility_css function."""

    def test_css_returns_string(self):
        """Test CSS function returns string."""
        css = get_accessibility_css()
        assert isinstance(css, str)
        assert len(css) > 0

    def test_css_matches_constant(self):
        """Test CSS function returns the constant."""
        assert get_accessibility_css() == ACCESSIBILITY_CSS

    def test_css_contains_sr_only(self):
        """Test CSS contains screen reader utility."""
        css = get_accessibility_css()
        assert ".sr-only" in css
        assert "position: absolute" in css
        assert "clip: rect(0, 0, 0, 0)" in css

    def test_css_contains_skip_link(self):
        """Test CSS contains skip link styles."""
        css = get_accessibility_css()
        assert ".skip-link" in css
        assert ".skip-link:focus" in css

    def test_css_contains_focus_trap(self):
        """Test CSS contains focus trap styles."""
        css = get_accessibility_css()
        assert ".focus-trap-start" in css
        assert ".focus-trap-end" in css

    def test_css_contains_keyboard_styles(self):
        """Test CSS contains keyboard shortcut styles."""
        css = get_accessibility_css()
        assert ".keyboard-shortcut" in css
        assert ".kbd" in css

    def test_css_contains_reduced_motion(self):
        """Test CSS contains reduced motion media query."""
        css = get_accessibility_css()
        assert "@media (prefers-reduced-motion: reduce)" in css

    def test_css_contains_high_contrast(self):
        """Test CSS contains high contrast media query."""
        css = get_accessibility_css()
        assert "@media (prefers-contrast: high)" in css

    def test_css_contains_focus_visible(self):
        """Test CSS contains focus-visible styles."""
        css = get_accessibility_css()
        assert ":focus-visible" in css
        assert ":focus:not(:focus-visible)" in css


class TestCreateLandmark:
    """Tests for create_landmark function."""

    def test_basic_landmark(self):
        """Test basic landmark creation."""
        html = create_landmark(
            "<p>Content</p>",
            AriaRole.NAVIGATION,
            "Main navigation"
        )

        assert 'role="navigation"' in html
        assert 'aria-label="Main navigation"' in html
        assert "Content" in html

    def test_landmark_with_id(self):
        """Test landmark with ID."""
        html = create_landmark(
            "<div>Content</div>",
            AriaRole.REGION,
            "Results",
            id="results-region"
        )

        assert 'id="results-region"' in html

    def test_landmark_without_id(self):
        """Test landmark without ID."""
        html = create_landmark(
            "<div>Content</div>",
            AriaRole.SEARCH,
            "Search"
        )

        # Should not have id attribute
        assert 'id=""' not in html

    def test_landmark_roles(self):
        """Test various landmark roles."""
        roles = [
            AriaRole.NAVIGATION,
            AriaRole.SEARCH,
            AriaRole.REGION,
            AriaRole.FORM,
        ]
        for role in roles:
            html = create_landmark("<div>Content</div>", role, "Label")
            assert f'role="{role.value}"' in html


class TestCreateLoadingRegion:
    """Tests for create_loading_region function."""

    def test_loading_region_not_loading(self):
        """Test loading region when not loading."""
        html = create_loading_region(
            "<div>Content</div>",
            loading=False,
            label="Results"
        )

        assert 'role="region"' in html
        assert 'aria-label="Results"' in html
        assert 'aria-live="polite"' in html
        assert 'aria-busy="false"' in html
        assert "Content" in html

    def test_loading_region_loading(self):
        """Test loading region when loading."""
        html = create_loading_region(
            "<div>Content</div>",
            loading=True,
            label="Results"
        )

        assert 'aria-busy="true"' in html
        # Should include screen reader announcement
        assert 'class="sr-only"' in html
        assert "Loading..." in html

    def test_loading_region_default_label(self):
        """Test loading region with default label."""
        html = create_loading_region("<div>Content</div>")
        assert 'aria-label="Content area"' in html


class TestFormatFieldDescription:
    """Tests for format_field_description function."""

    def test_description_only(self):
        """Test with description only."""
        html, describedby = format_field_description("Enter your email")

        assert 'id="desc"' in html
        assert "Enter your email" in html
        assert describedby == "desc"

    def test_with_hint(self):
        """Test with description and hint."""
        html, describedby = format_field_description(
            "Enter your email",
            hint="We'll never share your email"
        )

        assert 'id="desc"' in html
        assert 'id="hint"' in html
        assert 'class="field-hint"' in html
        assert "We'll never share your email" in html
        assert "desc" in describedby
        assert "hint" in describedby

    def test_with_error(self):
        """Test with error message."""
        html, describedby = format_field_description(
            "Enter your email",
            error="Email is required"
        )

        assert 'id="error"' in html
        assert 'class="field-error"' in html
        assert 'role="alert"' in html
        assert "Email is required" in html
        assert "error" in describedby

    def test_all_parts(self):
        """Test with all parts."""
        html, describedby = format_field_description(
            "Enter your email",
            error="Invalid email",
            hint="Use your work email"
        )

        assert 'id="desc"' in html
        assert 'id="hint"' in html
        assert 'id="error"' in html
        assert "desc" in describedby
        assert "hint" in describedby
        assert "error" in describedby

    def test_empty_description(self):
        """Test with empty description."""
        html, describedby = format_field_description("")
        # Should be empty
        assert describedby == ""

    def test_describedby_space_separated(self):
        """Test describedby IDs are space-separated."""
        _, describedby = format_field_description(
            "Description",
            hint="Hint",
            error="Error"
        )
        # Should have spaces between IDs
        parts = describedby.split()
        assert len(parts) == 3


class TestAccessibilityIntegration:
    """Integration tests for accessibility features."""

    def test_complete_accessible_form_field(self):
        """Test complete accessible form field setup."""
        # Create field description
        html, describedby = format_field_description(
            "Enter your username",
            hint="3-20 characters, letters and numbers only"
        )

        # Create ARIA attributes for input
        attrs = AriaAttrs(
            describedby=describedby,
            required=True,
            invalid=False
        )
        aria = attrs.to_html_attrs()

        assert 'aria-describedby="desc hint"' in aria
        assert 'aria-required="true"' in aria
        assert 'aria-invalid="false"' in aria

    def test_accessible_dialog(self):
        """Test accessible dialog setup."""
        content = "<form>Dialog content</form>"

        # Wrap in focus trap
        trapped = create_focus_trap(content, "dialog-trap")

        # Create landmark
        dialog = create_landmark(
            trapped,
            AriaRole.DIALOG,
            "Edit Profile",
            id="profile-dialog"
        )

        assert 'role="dialog"' in dialog
        assert 'aria-label="Edit Profile"' in dialog
        assert 'class="focus-trap"' in dialog
        assert "Dialog content" in dialog

    def test_accessible_loading_state(self):
        """Test accessible loading state."""
        # Create loading region
        content = "<div class='results'>Results will appear here</div>"
        region = create_loading_region(content, loading=True, label="Search results")

        assert 'aria-busy="true"' in region
        assert 'aria-live="polite"' in region
        assert 'class="sr-only"' in region

        # Announce completion
        announcement = announce_to_screen_reader(
            "Search complete. 10 results found.",
            AriaLive.POLITE
        )
        assert "Search complete" in announcement

    def test_keyboard_navigation_setup(self):
        """Test keyboard navigation setup."""
        # Skip link
        skip = create_skip_link("main", "Skip to main content")

        # Keyboard shortcut display
        shortcut1 = keyboard_shortcut("K", "Ctrl", "Open search")
        shortcut2 = keyboard_shortcut("Escape", description="Close dialog")

        assert 'href="#main"' in skip
        assert "Ctrl" in shortcut1
        assert "K" in shortcut1
        assert "Escape" in shortcut2

    def test_css_includes_all_accessibility_features(self):
        """Test CSS includes all accessibility features."""
        css = get_accessibility_css()

        # Must-have accessibility classes
        required_features = [
            ".sr-only",  # Screen reader only
            ".skip-link",  # Skip navigation
            ".focus-trap",  # Focus management
            ".kbd",  # Keyboard shortcuts
            ":focus-visible",  # Visible focus
            "@media (prefers-reduced-motion",  # Motion preference
            "@media (prefers-contrast",  # Contrast preference
        ]

        for feature in required_features:
            assert feature in css, f"Missing accessibility feature: {feature}"

    def test_all_components_have_aria_attributes(self):
        """Test all components produce proper ARIA or semantic attributes."""
        # Components that should have ARIA attributes
        aria_components = [
            announce_to_screen_reader("Message"),
            create_focus_trap("<div>Content</div>"),
            create_landmark("<div>Content</div>", AriaRole.REGION, "Label"),
            create_loading_region("<div>Content</div>", True),
        ]

        for html in aria_components:
            # Each should have at least one ARIA attribute
            has_aria = (
                'aria-' in html or
                'role=' in html or
                'tabindex=' in html
            )
            assert has_aria, f"Missing ARIA attributes in: {html[:100]}"

        # Skip link is semantic HTML (anchor with href) - it's accessible by design
        skip_link = create_skip_link()
        assert 'href="#' in skip_link
        assert 'class="skip-link"' in skip_link
