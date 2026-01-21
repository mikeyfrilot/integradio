"""
Tests for Visual Validation module.

Tests:
- Severity enum
- ValidationIssue dataclass
- ValidationReport dataclass
- Color utility functions (hex_to_rgb, relative_luminance, contrast_ratio, is_color_accessible)
- SpecValidator class
- TemplateValidator class
- CSSValidator class
- Convenience functions (validate_spec, validate_templates, validate_css, validate_all, get_validation_score)
"""

import pytest
from pathlib import Path

from integradio.visual.validation import (
    # Types
    Severity,
    ValidationIssue,
    ValidationReport,
    # Color utilities
    hex_to_rgb,
    relative_luminance,
    contrast_ratio,
    is_color_accessible,
    # Validators
    SpecValidator,
    TemplateValidator,
    CSSValidator,
    # Convenience functions
    validate_spec,
    validate_templates,
    validate_css,
    validate_all,
    get_validation_score,
)
from integradio.visual.spec import (
    UISpec,
    PageSpec,
    VisualSpec,
    LayoutSpec,
    FlexSpec,
    GridSpec,
    Display,
)
from integradio.visual.tokens import (
    DesignToken,
    TokenGroup,
    TokenType,
    ColorValue,
    DimensionValue,
)


# =============================================================================
# Severity Enum Tests
# =============================================================================

class TestSeverity:
    """Tests for Severity enum."""

    def test_all_severity_values(self):
        """Test all severity values exist."""
        assert Severity.ERROR == "error"
        assert Severity.WARNING == "warning"
        assert Severity.INFO == "info"

    def test_severity_is_string(self):
        """Test that severity values are strings."""
        for severity in Severity:
            assert isinstance(severity.value, str)


# =============================================================================
# ValidationIssue Tests
# =============================================================================

class TestValidationIssue:
    """Tests for ValidationIssue dataclass."""

    def test_basic_creation(self):
        """Test basic issue creation."""
        issue = ValidationIssue(
            code="TEST_ERROR",
            message="A test error occurred",
            severity=Severity.ERROR,
            path="pages.home.button",
        )

        assert issue.code == "TEST_ERROR"
        assert issue.message == "A test error occurred"
        assert issue.severity == Severity.ERROR
        assert issue.path == "pages.home.button"
        assert issue.suggestion == ""
        assert issue.details == {}

    def test_creation_with_all_fields(self):
        """Test issue creation with all fields."""
        issue = ValidationIssue(
            code="LOW_CONTRAST",
            message="Contrast ratio is too low",
            severity=Severity.WARNING,
            path="pages.home.text",
            suggestion="Increase contrast between colors",
            details={"ratio": 3.5, "fg": "#666666", "bg": "#ffffff"},
        )

        assert issue.code == "LOW_CONTRAST"
        assert issue.suggestion == "Increase contrast between colors"
        assert issue.details["ratio"] == 3.5

    def test_str_representation_error(self):
        """Test string representation for error severity."""
        issue = ValidationIssue(
            code="TEST_ERROR",
            message="Error message",
            severity=Severity.ERROR,
            path="test.path",
        )

        result = str(issue)
        assert "[X]" in result
        assert "TEST_ERROR" in result
        assert "Error message" in result
        assert "test.path" in result

    def test_str_representation_warning(self):
        """Test string representation for warning severity."""
        issue = ValidationIssue(
            code="TEST_WARNING",
            message="Warning message",
            severity=Severity.WARNING,
            path="test.path",
        )

        result = str(issue)
        assert "[!]" in result

    def test_str_representation_info(self):
        """Test string representation for info severity."""
        issue = ValidationIssue(
            code="TEST_INFO",
            message="Info message",
            severity=Severity.INFO,
            path="test.path",
        )

        result = str(issue)
        assert "[i]" in result


# =============================================================================
# ValidationReport Tests
# =============================================================================

class TestValidationReport:
    """Tests for ValidationReport dataclass."""

    def test_basic_creation(self):
        """Test basic report creation."""
        report = ValidationReport(spec_name="test-spec")

        assert report.spec_name == "test-spec"
        assert report.issues == []
        assert report.passed == 0
        assert report.total_checks == 0

    def test_errors_property(self):
        """Test errors property filters correctly."""
        report = ValidationReport(spec_name="test")
        report.issues.append(ValidationIssue(
            code="ERR1", message="Error 1", severity=Severity.ERROR, path="a"
        ))
        report.issues.append(ValidationIssue(
            code="WARN1", message="Warning 1", severity=Severity.WARNING, path="b"
        ))
        report.issues.append(ValidationIssue(
            code="ERR2", message="Error 2", severity=Severity.ERROR, path="c"
        ))

        errors = report.errors
        assert len(errors) == 2
        assert all(e.severity == Severity.ERROR for e in errors)

    def test_warnings_property(self):
        """Test warnings property filters correctly."""
        report = ValidationReport(spec_name="test")
        report.issues.append(ValidationIssue(
            code="ERR1", message="Error 1", severity=Severity.ERROR, path="a"
        ))
        report.issues.append(ValidationIssue(
            code="WARN1", message="Warning 1", severity=Severity.WARNING, path="b"
        ))
        report.issues.append(ValidationIssue(
            code="INFO1", message="Info 1", severity=Severity.INFO, path="c"
        ))

        warnings = report.warnings
        assert len(warnings) == 1
        assert warnings[0].code == "WARN1"

    def test_is_valid_no_errors(self):
        """Test is_valid returns True when no errors."""
        report = ValidationReport(spec_name="test")
        report.issues.append(ValidationIssue(
            code="WARN1", message="Warning", severity=Severity.WARNING, path="a"
        ))

        assert report.is_valid is True

    def test_is_valid_with_errors(self):
        """Test is_valid returns False when errors exist."""
        report = ValidationReport(spec_name="test")
        report.issues.append(ValidationIssue(
            code="ERR1", message="Error", severity=Severity.ERROR, path="a"
        ))

        assert report.is_valid is False

    def test_score_no_checks(self):
        """Test score returns 100 when no checks performed."""
        report = ValidationReport(spec_name="test")
        assert report.score == 100.0

    def test_score_all_passed(self):
        """Test score returns 100 when all checks pass."""
        report = ValidationReport(spec_name="test")
        report.passed = 10
        report.total_checks = 10

        assert report.score == 100.0

    def test_score_partial_pass(self):
        """Test score calculation with partial pass."""
        report = ValidationReport(spec_name="test")
        report.passed = 7
        report.total_checks = 10

        assert report.score == 70.0

    def test_add_issue(self):
        """Test add_issue method."""
        report = ValidationReport(spec_name="test")
        issue = ValidationIssue(
            code="TEST", message="Test", severity=Severity.ERROR, path="path"
        )

        report.add_issue(issue)

        assert len(report.issues) == 1
        assert report.issues[0] == issue

    def test_add_pass(self):
        """Test add_pass method."""
        report = ValidationReport(spec_name="test")

        report.add_pass()
        report.add_pass()

        assert report.passed == 2
        assert report.total_checks == 2

    def test_add_fail(self):
        """Test add_fail method."""
        report = ValidationReport(spec_name="test")
        issue = ValidationIssue(
            code="FAIL", message="Failure", severity=Severity.ERROR, path="path"
        )

        report.add_fail(issue)

        assert len(report.issues) == 1
        assert report.total_checks == 1
        assert report.passed == 0

    def test_to_dict(self):
        """Test to_dict method."""
        report = ValidationReport(spec_name="test-spec")
        report.passed = 5
        report.total_checks = 6
        report.issues.append(ValidationIssue(
            code="WARN1",
            message="Warning message",
            severity=Severity.WARNING,
            path="test.path",
            suggestion="Fix it",
        ))

        result = report.to_dict()

        assert result["spec_name"] == "test-spec"
        assert result["is_valid"] is True
        assert result["score"] == 83.3  # (5/6) * 100 rounded to 1 decimal
        assert result["passed"] == 5
        assert result["total_checks"] == 6
        assert result["errors"] == 0
        assert result["warnings"] == 1
        assert len(result["issues"]) == 1
        assert result["issues"][0]["code"] == "WARN1"
        assert result["issues"][0]["severity"] == "warning"

    def test_str_representation(self):
        """Test string representation."""
        report = ValidationReport(spec_name="my-spec")
        report.passed = 8
        report.total_checks = 10
        report.issues.append(ValidationIssue(
            code="ERR1", message="Error", severity=Severity.ERROR, path="a"
        ))
        report.issues.append(ValidationIssue(
            code="WARN1", message="Warning", severity=Severity.WARNING, path="b"
        ))

        result = str(report)

        assert "Validation Report: my-spec" in result
        assert "80.0%" in result
        assert "8/10" in result
        assert "Errors: 1" in result
        assert "Warnings: 1" in result


# =============================================================================
# Color Utility Tests
# =============================================================================

class TestHexToRgb:
    """Tests for hex_to_rgb function."""

    def test_black(self):
        """Test black color conversion."""
        r, g, b = hex_to_rgb("#000000")
        assert (r, g, b) == (0, 0, 0)

    def test_white(self):
        """Test white color conversion."""
        r, g, b = hex_to_rgb("#ffffff")
        assert (r, g, b) == (255, 255, 255)

    def test_red(self):
        """Test red color conversion."""
        r, g, b = hex_to_rgb("#ff0000")
        assert (r, g, b) == (255, 0, 0)

    def test_without_hash(self):
        """Test conversion without # prefix."""
        r, g, b = hex_to_rgb("00ff00")
        assert (r, g, b) == (0, 255, 0)

    def test_mixed_case(self):
        """Test mixed case hex values."""
        r, g, b = hex_to_rgb("#AbCdEf")
        assert (r, g, b) == (171, 205, 239)


class TestRelativeLuminance:
    """Tests for relative_luminance function."""

    def test_black_luminance(self):
        """Test black has 0 luminance."""
        lum = relative_luminance(0, 0, 0)
        assert lum == 0.0

    def test_white_luminance(self):
        """Test white has 1 luminance."""
        lum = relative_luminance(255, 255, 255)
        assert abs(lum - 1.0) < 0.001

    def test_gray_luminance(self):
        """Test gray has intermediate luminance."""
        lum = relative_luminance(128, 128, 128)
        assert 0.2 < lum < 0.3

    def test_low_value_branch(self):
        """Test the low sRGB value branch (<=0.03928)."""
        # 10/255 = 0.039 which is above threshold
        # 9/255 = 0.035 which is still above, but close
        # Need value where c/255 <= 0.03928, so c <= 10
        lum = relative_luminance(10, 10, 10)
        assert lum > 0


class TestContrastRatio:
    """Tests for contrast_ratio function."""

    def test_max_contrast(self):
        """Test maximum contrast (black vs white)."""
        ratio = contrast_ratio("#000000", "#ffffff")
        assert abs(ratio - 21.0) < 0.1

    def test_no_contrast(self):
        """Test no contrast (same color)."""
        ratio = contrast_ratio("#888888", "#888888")
        assert abs(ratio - 1.0) < 0.01

    def test_order_independent(self):
        """Test that color order doesn't matter."""
        ratio1 = contrast_ratio("#000000", "#ffffff")
        ratio2 = contrast_ratio("#ffffff", "#000000")
        assert abs(ratio1 - ratio2) < 0.01

    def test_moderate_contrast(self):
        """Test moderate contrast values."""
        ratio = contrast_ratio("#333333", "#ffffff")
        assert 10 < ratio < 15


class TestIsColorAccessible:
    """Tests for is_color_accessible function."""

    def test_wcag_aa_normal_text_pass(self):
        """Test WCAG AA for normal text passes with high contrast."""
        assert is_color_accessible("#000000", "#ffffff", "AA", False) is True

    def test_wcag_aa_normal_text_fail(self):
        """Test WCAG AA for normal text fails with low contrast."""
        assert is_color_accessible("#777777", "#888888", "AA", False) is False

    def test_wcag_aa_large_text_pass(self):
        """Test WCAG AA for large text has lower threshold."""
        # This ratio might pass for large text but fail for normal
        # 3:1 is threshold for large text
        assert is_color_accessible("#000000", "#ffffff", "AA", True) is True

    def test_wcag_aaa_normal_text(self):
        """Test WCAG AAA for normal text requires 7:1."""
        # Black on white passes AAA
        assert is_color_accessible("#000000", "#ffffff", "AAA", False) is True

    def test_wcag_aaa_large_text(self):
        """Test WCAG AAA for large text requires 4.5:1."""
        assert is_color_accessible("#000000", "#ffffff", "AAA", True) is True

    def test_wcag_aaa_borderline(self):
        """Test WCAG AAA borderline case."""
        # A contrast ratio around 5:1 should fail AAA normal but pass AAA large
        # Gray on white - using #666666 which gives ~5.74:1
        ratio = contrast_ratio("#666666", "#ffffff")
        assert 4.5 < ratio < 7
        assert is_color_accessible("#666666", "#ffffff", "AAA", False) is False
        assert is_color_accessible("#666666", "#ffffff", "AAA", True) is True


# =============================================================================
# SpecValidator Tests
# =============================================================================

class TestSpecValidator:
    """Tests for SpecValidator class."""

    def test_empty_spec_validation(self):
        """Test validation of minimal spec."""
        spec = UISpec(name="empty-spec")
        validator = SpecValidator(spec)
        report = validator.validate()

        assert report.spec_name == "empty-spec"
        # Should have warnings about missing tokens and pages
        assert len(report.warnings) >= 1

    def test_spec_with_color_tokens(self):
        """Test validation passes when color tokens exist."""
        spec = UISpec(name="color-spec")
        tokens = TokenGroup()
        tokens.add("colors", TokenGroup())
        tokens.tokens["colors"].add("primary", DesignToken.color("#3b82f6"))
        spec.tokens = tokens

        validator = SpecValidator(spec)
        report = validator.validate()

        # Should not have MISSING_COLOR_TOKENS issue
        color_issues = [i for i in report.issues if i.code == "MISSING_COLOR_TOKENS"]
        assert len(color_issues) == 0

    def test_spec_with_spacing_tokens(self):
        """Test validation passes when spacing tokens exist."""
        spec = UISpec(name="spacing-spec")
        tokens = TokenGroup()
        tokens.add("spacing", TokenGroup())
        tokens.tokens["spacing"].add("sm", DesignToken.dimension(8))
        spec.tokens = tokens

        validator = SpecValidator(spec)
        report = validator.validate()

        # Should not have MISSING_SPACING_TOKENS issue
        spacing_issues = [i for i in report.issues if i.code == "MISSING_SPACING_TOKENS"]
        assert len(spacing_issues) == 0

    def test_broken_alias_detection(self):
        """Test detection of broken token aliases."""
        spec = UISpec(name="alias-spec")
        tokens = TokenGroup()
        tokens.add("colors", TokenGroup())
        # Add alias to non-existent token
        broken_alias = DesignToken.reference("colors.nonexistent", TokenType.COLOR)
        tokens.tokens["colors"].add("broken", broken_alias)
        spec.tokens = tokens

        validator = SpecValidator(spec)
        report = validator.validate()

        # Should have BROKEN_ALIAS error
        alias_issues = [i for i in report.issues if i.code == "BROKEN_ALIAS"]
        assert len(alias_issues) == 1
        assert alias_issues[0].severity == Severity.ERROR

    def test_valid_alias(self):
        """Test valid alias passes validation."""
        spec = UISpec(name="alias-spec")
        tokens = TokenGroup()
        colors = TokenGroup()
        colors.add("primary", DesignToken.color("#3b82f6"))
        colors.add("main", DesignToken.reference("colors.primary", TokenType.COLOR))
        tokens.add("colors", colors)
        spec.tokens = tokens

        validator = SpecValidator(spec)
        report = validator.validate()

        # Should not have BROKEN_ALIAS error
        alias_issues = [i for i in report.issues if i.code == "BROKEN_ALIAS"]
        assert len(alias_issues) == 0

    def test_pure_black_detection(self):
        """Test detection of pure black color."""
        spec = UISpec(name="black-spec")
        tokens = TokenGroup()
        colors = TokenGroup()
        colors.add("black", DesignToken.color("#000000"))
        tokens.add("colors", colors)
        spec.tokens = tokens

        validator = SpecValidator(spec)
        report = validator.validate()

        # Should have PURE_BLACK info
        black_issues = [i for i in report.issues if i.code == "PURE_BLACK"]
        assert len(black_issues) == 1
        assert black_issues[0].severity == Severity.INFO

    def test_pure_white_detection(self):
        """Test detection of pure white color."""
        spec = UISpec(name="white-spec")
        tokens = TokenGroup()
        colors = TokenGroup()
        colors.add("white", DesignToken.color("#ffffff"))
        tokens.add("colors", colors)
        spec.tokens = tokens

        validator = SpecValidator(spec)
        report = validator.validate()

        # Should have PURE_WHITE info
        white_issues = [i for i in report.issues if i.code == "PURE_WHITE"]
        assert len(white_issues) == 1
        assert white_issues[0].severity == Severity.INFO

    def test_non_pure_color_passes(self):
        """Test that non-pure colors pass validation."""
        spec = UISpec(name="color-spec")
        tokens = TokenGroup()
        colors = TokenGroup()
        # Not pure black or white
        colors.add("dark", DesignToken.color("#111827"))
        colors.add("light", DesignToken.color("#f9fafb"))
        tokens.add("colors", colors)
        spec.tokens = tokens

        validator = SpecValidator(spec)
        report = validator.validate()

        # Should not have PURE_BLACK or PURE_WHITE issues
        pure_issues = [i for i in report.issues if i.code in ("PURE_BLACK", "PURE_WHITE")]
        assert len(pure_issues) == 0

    def test_no_pages_warning(self):
        """Test warning when no pages defined."""
        spec = UISpec(name="no-pages")

        validator = SpecValidator(spec)
        report = validator.validate()

        no_pages_issues = [i for i in report.issues if i.code == "NO_PAGES"]
        assert len(no_pages_issues) == 1
        assert no_pages_issues[0].severity == Severity.WARNING

    def test_empty_page_warning(self):
        """Test warning when page has no components."""
        spec = UISpec(name="empty-page-spec")
        page = PageSpec(name="Home", route="/")
        spec.add_page(page)

        validator = SpecValidator(spec)
        report = validator.validate()

        empty_page_issues = [i for i in report.issues if i.code == "EMPTY_PAGE"]
        assert len(empty_page_issues) == 1

    def test_missing_component_type_warning(self):
        """Test warning when component type not specified."""
        spec = UISpec(name="component-spec")
        page = PageSpec(name="Home", route="/")
        page.add_component(VisualSpec(
            component_id="btn",
            component_type="",  # Empty type
        ))
        spec.add_page(page)

        validator = SpecValidator(spec)
        report = validator.validate()

        type_issues = [i for i in report.issues if i.code == "MISSING_COMPONENT_TYPE"]
        assert len(type_issues) == 1

    def test_no_colors_info(self):
        """Test info when component has no color tokens."""
        spec = UISpec(name="no-colors-spec")
        page = PageSpec(name="Home", route="/")
        page.add_component(VisualSpec(
            component_id="btn",
            component_type="Button",
        ))
        spec.add_page(page)

        validator = SpecValidator(spec)
        report = validator.validate()

        color_issues = [i for i in report.issues if i.code == "NO_COLORS"]
        assert len(color_issues) == 1
        assert color_issues[0].severity == Severity.INFO

    def test_component_with_colors(self):
        """Test component with colors passes."""
        spec = UISpec(name="colored-spec")
        page = PageSpec(name="Home", route="/")
        comp = VisualSpec(component_id="btn", component_type="Button")
        comp.set_colors(background="#3b82f6", text="#ffffff")
        page.add_component(comp)
        spec.add_page(page)

        validator = SpecValidator(spec)
        report = validator.validate()

        color_issues = [i for i in report.issues if i.code == "NO_COLORS"]
        assert len(color_issues) == 0

    def test_component_with_only_background(self):
        """Test component with only background color passes."""
        spec = UISpec(name="bg-only-spec")
        page = PageSpec(name="Home", route="/")
        comp = VisualSpec(component_id="box", component_type="Box")
        comp.set_colors(background="#3b82f6")  # Only background, no text color
        page.add_component(comp)
        spec.add_page(page)

        validator = SpecValidator(spec)
        report = validator.validate()

        # Should not have NO_COLORS since we have background
        color_issues = [i for i in report.issues if i.code == "NO_COLORS"]
        assert len(color_issues) == 0

    def test_component_with_only_text_color(self):
        """Test component with only text color passes."""
        spec = UISpec(name="text-only-spec")
        page = PageSpec(name="Home", route="/")
        comp = VisualSpec(component_id="label", component_type="Label")
        comp.set_colors(text="#1f2937")  # Only text color, no background
        page.add_component(comp)
        spec.add_page(page)

        validator = SpecValidator(spec)
        report = validator.validate()

        # Should not have NO_COLORS since we have color
        color_issues = [i for i in report.issues if i.code == "NO_COLORS"]
        assert len(color_issues) == 0

    def test_flex_without_config_warning(self):
        """Test warning when display=flex but no FlexSpec."""
        spec = UISpec(name="flex-spec")
        page = PageSpec(name="Home", route="/")
        comp = VisualSpec(
            component_id="container",
            component_type="Container",
            layout=LayoutSpec(display=Display.FLEX, flex=None),
        )
        comp.set_colors(background="#ffffff")
        page.add_component(comp)
        spec.add_page(page)

        validator = SpecValidator(spec)
        report = validator.validate()

        flex_issues = [i for i in report.issues if i.code == "FLEX_WITHOUT_CONFIG"]
        assert len(flex_issues) == 1

    def test_flex_with_config_passes(self):
        """Test flex with FlexSpec passes."""
        spec = UISpec(name="flex-spec")
        page = PageSpec(name="Home", route="/")
        comp = VisualSpec(
            component_id="container",
            component_type="Container",
            layout=LayoutSpec(display=Display.FLEX, flex=FlexSpec()),
        )
        comp.set_colors(background="#ffffff")
        page.add_component(comp)
        spec.add_page(page)

        validator = SpecValidator(spec)
        report = validator.validate()

        flex_issues = [i for i in report.issues if i.code == "FLEX_WITHOUT_CONFIG"]
        assert len(flex_issues) == 0

    def test_grid_without_config_warning(self):
        """Test warning when display=grid but no GridSpec."""
        spec = UISpec(name="grid-spec")
        page = PageSpec(name="Home", route="/")
        comp = VisualSpec(
            component_id="container",
            component_type="Container",
            layout=LayoutSpec(display=Display.GRID, grid=None),
        )
        comp.set_colors(background="#ffffff")
        page.add_component(comp)
        spec.add_page(page)

        validator = SpecValidator(spec)
        report = validator.validate()

        grid_issues = [i for i in report.issues if i.code == "GRID_WITHOUT_CONFIG"]
        assert len(grid_issues) == 1

    def test_grid_with_config_passes(self):
        """Test grid with GridSpec passes."""
        spec = UISpec(name="grid-spec")
        page = PageSpec(name="Home", route="/")
        comp = VisualSpec(
            component_id="container",
            component_type="Container",
            layout=LayoutSpec(display=Display.GRID, grid=GridSpec()),
        )
        comp.set_colors(background="#ffffff")
        page.add_component(comp)
        spec.add_page(page)

        validator = SpecValidator(spec)
        report = validator.validate()

        grid_issues = [i for i in report.issues if i.code == "GRID_WITHOUT_CONFIG"]
        assert len(grid_issues) == 0

    def test_block_display_no_layout_issues(self):
        """Test block display doesn't trigger flex/grid warnings."""
        spec = UISpec(name="block-spec")
        page = PageSpec(name="Home", route="/")
        comp = VisualSpec(
            component_id="container",
            component_type="Container",
            layout=LayoutSpec(display=Display.BLOCK),  # Default block display
        )
        comp.set_colors(background="#ffffff")
        page.add_component(comp)
        spec.add_page(page)

        validator = SpecValidator(spec)
        report = validator.validate()

        # Should not have flex or grid issues
        layout_issues = [i for i in report.issues if "FLEX" in i.code or "GRID" in i.code]
        assert len(layout_issues) == 0

    def test_low_contrast_error(self):
        """Test error when contrast ratio is very low."""
        spec = UISpec(name="contrast-spec")
        page = PageSpec(name="Home", route="/")
        comp = VisualSpec(component_id="text", component_type="Text")
        # Very low contrast: similar grays
        comp.set_colors(background="#808080", text="#888888")
        page.add_component(comp)
        spec.add_page(page)

        validator = SpecValidator(spec)
        report = validator.validate()

        contrast_issues = [i for i in report.issues if i.code == "CONTRAST_FAIL"]
        assert len(contrast_issues) == 1
        assert contrast_issues[0].severity == Severity.ERROR

    def test_medium_contrast_warning(self):
        """Test warning when contrast is between 3:1 and 4.5:1."""
        spec = UISpec(name="contrast-spec")
        page = PageSpec(name="Home", route="/")
        comp = VisualSpec(component_id="text", component_type="Text")
        # Contrast between 3:1 and 4.5:1 - #888888 on white gives ~3.54:1
        comp.set_colors(background="#ffffff", text="#888888")
        page.add_component(comp)
        spec.add_page(page)

        validator = SpecValidator(spec)
        report = validator.validate()

        # Should have CONTRAST_LOW warning (not fail, since ratio > 3)
        contrast_low = [i for i in report.issues if i.code == "CONTRAST_LOW"]
        contrast_fail = [i for i in report.issues if i.code == "CONTRAST_FAIL"]
        assert len(contrast_low) == 1
        assert len(contrast_fail) == 0
        assert contrast_low[0].severity == Severity.WARNING

    def test_good_contrast_passes(self):
        """Test good contrast passes validation."""
        spec = UISpec(name="contrast-spec")
        page = PageSpec(name="Home", route="/")
        comp = VisualSpec(component_id="text", component_type="Text")
        # High contrast: black on white
        comp.set_colors(background="#ffffff", text="#000000")
        page.add_component(comp)
        spec.add_page(page)

        validator = SpecValidator(spec)
        report = validator.validate()

        contrast_issues = [i for i in report.issues if "CONTRAST" in i.code]
        assert len(contrast_issues) == 0

    def test_undefined_token_ref_error(self):
        """Test error when component references undefined token."""
        spec = UISpec(name="ref-spec")
        spec.tokens = TokenGroup()  # Empty tokens

        page = PageSpec(name="Home", route="/")
        comp = VisualSpec(component_id="btn", component_type="Button")
        # Add alias token that references non-existent token
        comp.tokens["background"] = DesignToken.reference("colors.primary", TokenType.COLOR)
        page.add_component(comp)
        spec.add_page(page)

        validator = SpecValidator(spec)
        report = validator.validate()

        ref_issues = [i for i in report.issues if i.code == "UNDEFINED_TOKEN_REF"]
        assert len(ref_issues) == 1
        assert ref_issues[0].severity == Severity.ERROR

    def test_defined_token_ref_passes(self):
        """Test that defined token references pass validation."""
        spec = UISpec(name="ref-spec")
        tokens = TokenGroup()
        colors = TokenGroup()
        colors.add("primary", DesignToken.color("#3b82f6"))
        tokens.add("colors", colors)
        spec.tokens = tokens

        page = PageSpec(name="Home", route="/")
        comp = VisualSpec(component_id="btn", component_type="Button")
        # Add alias token that references existing token
        comp.tokens["background"] = DesignToken.reference("colors.primary", TokenType.COLOR)
        page.add_component(comp)
        spec.add_page(page)

        validator = SpecValidator(spec)
        report = validator.validate()

        ref_issues = [i for i in report.issues if i.code == "UNDEFINED_TOKEN_REF"]
        assert len(ref_issues) == 0

    def test_token_to_hex_with_string(self):
        """Test _token_to_hex handles string hex values."""
        spec = UISpec(name="hex-spec")
        page = PageSpec(name="Home", route="/")
        comp = VisualSpec(component_id="text", component_type="Text")
        # Use string value starting with #
        comp.tokens["background"] = DesignToken(
            value="#ffffff",
            type=TokenType.COLOR,
        )
        comp.tokens["color"] = DesignToken(
            value="#000000",
            type=TokenType.COLOR,
        )
        page.add_component(comp)
        spec.add_page(page)

        validator = SpecValidator(spec)
        report = validator.validate()

        # Should process without error
        assert report is not None

    def test_token_to_hex_returns_none_for_non_color(self):
        """Test _token_to_hex returns None for non-color values."""
        spec = UISpec(name="non-color-spec")
        page = PageSpec(name="Home", route="/")
        comp = VisualSpec(component_id="text", component_type="Text")
        # Non-color value
        comp.tokens["background"] = DesignToken(
            value=DimensionValue(10, "px"),
            type=TokenType.DIMENSION,
        )
        page.add_component(comp)
        spec.add_page(page)

        validator = SpecValidator(spec)
        report = validator.validate()

        # Should not crash, accessibility check skipped
        assert report is not None


# =============================================================================
# TemplateValidator Tests
# =============================================================================

class TestTemplateValidator:
    """Tests for TemplateValidator class."""

    def test_template_dir_not_found(self, tmp_path: Path):
        """Test error when template directory doesn't exist."""
        spec = UISpec(name="test-spec")
        non_existent = tmp_path / "nonexistent"

        validator = TemplateValidator(spec, non_existent)
        report = validator.validate()

        dir_issues = [i for i in report.issues if i.code == "TEMPLATE_DIR_NOT_FOUND"]
        assert len(dir_issues) == 1
        assert dir_issues[0].severity == Severity.ERROR

    def test_finds_html_templates(self, tmp_path: Path):
        """Test finding HTML template files."""
        spec = UISpec(name="test-spec")
        templates = tmp_path / "templates"
        templates.mkdir()
        (templates / "index.html").write_text('<div id="main-content">Hello</div>')

        page = PageSpec(name="Home", route="/")
        page.add_component(VisualSpec(component_id="main-content", component_type="Div"))
        spec.add_page(page)

        validator = TemplateValidator(spec, templates)
        report = validator.validate()

        # Component should be found
        missing_issues = [i for i in report.issues if i.code == "MISSING_VISUAL_SPEC"]
        assert len(missing_issues) == 0

    def test_finds_jinja_templates(self, tmp_path: Path):
        """Test finding Jinja template files."""
        spec = UISpec(name="test-spec")
        templates = tmp_path / "templates"
        templates.mkdir()
        (templates / "base.jinja").write_text('<button id="submit-btn">Submit</button>')
        (templates / "page.jinja2").write_text('<input id="search-input" />')

        page = PageSpec(name="Home", route="/")
        page.add_component(VisualSpec(component_id="submit-btn", component_type="Button"))
        page.add_component(VisualSpec(component_id="search-input", component_type="Input"))
        spec.add_page(page)

        validator = TemplateValidator(spec, templates)
        report = validator.validate()

        missing_issues = [i for i in report.issues if i.code == "MISSING_VISUAL_SPEC"]
        assert len(missing_issues) == 0

    def test_missing_visual_spec_warning(self, tmp_path: Path):
        """Test warning when template has component without spec."""
        spec = UISpec(name="test-spec")
        templates = tmp_path / "templates"
        templates.mkdir()
        (templates / "page.html").write_text('<div id="unspecified-component">Content</div>')

        # Empty spec with no matching component
        page = PageSpec(name="Home", route="/")
        spec.add_page(page)

        validator = TemplateValidator(spec, templates)
        report = validator.validate()

        missing_issues = [i for i in report.issues if i.code == "MISSING_VISUAL_SPEC"]
        assert len(missing_issues) == 1
        assert "unspecified-component" in missing_issues[0].message

    def test_extracts_elem_id_pattern(self, tmp_path: Path):
        """Test extraction of Gradio elem_id pattern."""
        spec = UISpec(name="test-spec")
        templates = tmp_path / "templates"
        templates.mkdir()
        (templates / "gradio.html").write_text('<div elem_id="gr-component">Gradio</div>')

        page = PageSpec(name="Home", route="/")
        page.add_component(VisualSpec(component_id="gr-component", component_type="Div"))
        spec.add_page(page)

        validator = TemplateValidator(spec, templates)
        report = validator.validate()

        missing_issues = [i for i in report.issues if i.code == "MISSING_VISUAL_SPEC"]
        assert len(missing_issues) == 0

    def test_extracts_data_component_pattern(self, tmp_path: Path):
        """Test extraction of data-component pattern."""
        spec = UISpec(name="test-spec")
        templates = tmp_path / "templates"
        templates.mkdir()
        (templates / "page.html").write_text('<div data-component="custom-widget">Widget</div>')

        page = PageSpec(name="Home", route="/")
        page.add_component(VisualSpec(component_id="custom-widget", component_type="Widget"))
        spec.add_page(page)

        validator = TemplateValidator(spec, templates)
        report = validator.validate()

        missing_issues = [i for i in report.issues if i.code == "MISSING_VISUAL_SPEC"]
        assert len(missing_issues) == 0

    def test_nested_templates(self, tmp_path: Path):
        """Test finding templates in nested directories."""
        spec = UISpec(name="test-spec")
        templates = tmp_path / "templates"
        templates.mkdir()
        nested = templates / "pages" / "admin"
        nested.mkdir(parents=True)
        (nested / "dashboard.html").write_text('<div id="dashboard-panel">Dashboard</div>')

        page = PageSpec(name="Admin", route="/admin")
        page.add_component(VisualSpec(component_id="dashboard-panel", component_type="Panel"))
        spec.add_page(page)

        validator = TemplateValidator(spec, templates)
        report = validator.validate()

        missing_issues = [i for i in report.issues if i.code == "MISSING_VISUAL_SPEC"]
        assert len(missing_issues) == 0

    def test_deduplicates_component_ids(self, tmp_path: Path):
        """Test that duplicate IDs in same template are deduplicated."""
        spec = UISpec(name="test-spec")
        templates = tmp_path / "templates"
        templates.mkdir()
        # Same ID appears multiple times
        (templates / "page.html").write_text('''
            <div id="repeated-id">First</div>
            <div id="repeated-id">Second</div>
        ''')

        validator = TemplateValidator(spec, templates)
        report = validator.validate()

        # Should only report once
        missing_issues = [i for i in report.issues if i.code == "MISSING_VISUAL_SPEC"]
        assert len(missing_issues) == 1


# =============================================================================
# CSSValidator Tests
# =============================================================================

class TestCSSValidator:
    """Tests for CSSValidator class."""

    def test_css_not_found(self, tmp_path: Path):
        """Test error when CSS file doesn't exist."""
        spec = UISpec(name="test-spec")
        non_existent = tmp_path / "styles.css"

        validator = CSSValidator(spec, non_existent)
        report = validator.validate()

        not_found_issues = [i for i in report.issues if i.code == "CSS_NOT_FOUND"]
        assert len(not_found_issues) == 1
        assert not_found_issues[0].severity == Severity.ERROR

    def test_no_hardcoded_colors_passes(self, tmp_path: Path):
        """Test CSS with no hardcoded colors passes."""
        spec = UISpec(name="test-spec")
        css_file = tmp_path / "styles.css"
        css_file.write_text("""
            .button {
                background: var(--colors-primary);
                color: var(--colors-text);
            }
        """)

        validator = CSSValidator(spec, css_file)
        report = validator.validate()

        color_issues = [i for i in report.issues if i.code == "HARDCODED_COLORS"]
        assert len(color_issues) == 0

    def test_hardcoded_colors_warning(self, tmp_path: Path):
        """Test warning for hardcoded colors."""
        spec = UISpec(name="test-spec")
        css_file = tmp_path / "styles.css"
        css_file.write_text("""
            .button {
                background: #3b82f6;
                color: #ffffff;
            }
            .link {
                color: #1d4ed8;
            }
        """)

        validator = CSSValidator(spec, css_file)
        report = validator.validate()

        color_issues = [i for i in report.issues if i.code == "HARDCODED_COLORS"]
        assert len(color_issues) == 1
        assert color_issues[0].severity == Severity.WARNING
        assert "3" in color_issues[0].message  # 3 unique colors

    def test_consistent_spacing_passes(self, tmp_path: Path):
        """Test CSS with consistent 4px spacing scale passes."""
        spec = UISpec(name="test-spec")
        css_file = tmp_path / "styles.css"
        css_file.write_text("""
            .element {
                padding: 4px;
                margin: 8px;
                gap: 16px;
                border-radius: 12px;
            }
        """)

        validator = CSSValidator(spec, css_file)
        report = validator.validate()

        spacing_issues = [i for i in report.issues if i.code == "INCONSISTENT_SPACING"]
        assert len(spacing_issues) == 0

    def test_inconsistent_spacing_info(self, tmp_path: Path):
        """Test info for inconsistent spacing values."""
        spec = UISpec(name="test-spec")
        css_file = tmp_path / "styles.css"
        css_file.write_text("""
            .element1 { padding: 5px; }
            .element2 { margin: 7px; }
            .element3 { gap: 9px; }
            .element4 { padding: 11px; }
            .element5 { margin: 13px; }
            .element6 { gap: 15px; }
        """)

        validator = CSSValidator(spec, css_file)
        report = validator.validate()

        spacing_issues = [i for i in report.issues if i.code == "INCONSISTENT_SPACING"]
        assert len(spacing_issues) == 1
        assert spacing_issues[0].severity == Severity.INFO

    def test_small_values_ignored_in_spacing_check(self, tmp_path: Path):
        """Test that small values (1px borders) don't trigger spacing warning."""
        spec = UISpec(name="test-spec")
        css_file = tmp_path / "styles.css"
        css_file.write_text("""
            .element {
                border: 1px solid black;
                padding: 8px;
            }
        """)

        validator = CSSValidator(spec, css_file)
        report = validator.validate()

        # 1px is <= 1, so ignored
        spacing_issues = [i for i in report.issues if i.code == "INCONSISTENT_SPACING"]
        assert len(spacing_issues) == 0

    def test_limited_details_in_report(self, tmp_path: Path):
        """Test that details are limited to first 10 items."""
        spec = UISpec(name="test-spec")
        css_file = tmp_path / "styles.css"
        # Many unique colors
        colors = "\n".join(f".c{i} {{ color: #{i:06x}; }}" for i in range(20))
        css_file.write_text(colors)

        validator = CSSValidator(spec, css_file)
        report = validator.validate()

        color_issues = [i for i in report.issues if i.code == "HARDCODED_COLORS"]
        assert len(color_issues) == 1
        # Details should have at most 10 colors
        assert len(color_issues[0].details.get("colors", [])) <= 10


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_validate_spec(self):
        """Test validate_spec convenience function."""
        spec = UISpec(name="test")
        report = validate_spec(spec)

        assert isinstance(report, ValidationReport)
        assert report.spec_name == "test"

    def test_validate_templates(self, tmp_path: Path):
        """Test validate_templates convenience function."""
        spec = UISpec(name="test")
        templates = tmp_path / "templates"
        templates.mkdir()

        report = validate_templates(spec, templates)

        assert isinstance(report, ValidationReport)
        assert "templates" in report.spec_name

    def test_validate_css(self, tmp_path: Path):
        """Test validate_css convenience function."""
        spec = UISpec(name="test")
        css_file = tmp_path / "styles.css"
        css_file.write_text(".test { color: red; }")

        report = validate_css(spec, css_file)

        assert isinstance(report, ValidationReport)
        assert "CSS" in report.spec_name

    def test_validate_all_spec_only(self):
        """Test validate_all with spec only."""
        spec = UISpec(name="test")
        results = validate_all(spec)

        assert "spec" in results
        assert isinstance(results["spec"], ValidationReport)
        assert "templates" not in results
        assert "css" not in results

    def test_validate_all_with_templates(self, tmp_path: Path):
        """Test validate_all with templates."""
        spec = UISpec(name="test")
        templates = tmp_path / "templates"
        templates.mkdir()

        results = validate_all(spec, template_dir=templates)

        assert "spec" in results
        assert "templates" in results
        assert "css" not in results

    def test_validate_all_with_css(self, tmp_path: Path):
        """Test validate_all with CSS."""
        spec = UISpec(name="test")
        css_file = tmp_path / "styles.css"
        css_file.write_text(".test {}")

        results = validate_all(spec, css_path=css_file)

        assert "spec" in results
        assert "css" in results
        assert "templates" not in results

    def test_validate_all_full(self, tmp_path: Path):
        """Test validate_all with all options."""
        spec = UISpec(name="test")
        templates = tmp_path / "templates"
        templates.mkdir()
        css_file = tmp_path / "styles.css"
        css_file.write_text(".test {}")

        results = validate_all(spec, template_dir=templates, css_path=css_file)

        assert "spec" in results
        assert "templates" in results
        assert "css" in results

    def test_get_validation_score_empty(self):
        """Test get_validation_score with no reports."""
        reports = {}
        score = get_validation_score(reports)
        assert score == 100.0

    def test_get_validation_score_all_pass(self):
        """Test get_validation_score with all passing."""
        report1 = ValidationReport(spec_name="test1")
        report1.passed = 10
        report1.total_checks = 10
        report2 = ValidationReport(spec_name="test2")
        report2.passed = 5
        report2.total_checks = 5

        score = get_validation_score({"spec": report1, "css": report2})
        assert score == 100.0

    def test_get_validation_score_partial(self):
        """Test get_validation_score with partial pass."""
        report1 = ValidationReport(spec_name="test1")
        report1.passed = 8
        report1.total_checks = 10
        report2 = ValidationReport(spec_name="test2")
        report2.passed = 6
        report2.total_checks = 10

        # Total: 14 passed out of 20 = 70%
        score = get_validation_score({"spec": report1, "css": report2})
        assert score == 70.0

    def test_get_validation_score_no_checks(self):
        """Test get_validation_score when no checks performed."""
        report = ValidationReport(spec_name="empty")
        score = get_validation_score({"spec": report})
        assert score == 100.0


# =============================================================================
# Integration Tests
# =============================================================================

class TestValidationIntegration:
    """Integration tests for complete validation workflows."""

    def test_full_spec_validation_flow(self, tmp_path: Path):
        """Test complete validation workflow with realistic spec."""
        # Create a realistic UISpec
        spec = UISpec(name="my-app", version="1.0.0")

        # Add global tokens
        colors = TokenGroup()
        colors.add("primary", DesignToken.color("#3b82f6"))
        colors.add("secondary", DesignToken.color("#6366f1"))
        colors.add("background", DesignToken.color("#ffffff"))
        colors.add("text", DesignToken.color("#1f2937"))

        spacing = TokenGroup()
        spacing.add("sm", DesignToken.dimension(8))
        spacing.add("md", DesignToken.dimension(16))
        spacing.add("lg", DesignToken.dimension(24))

        spec.tokens.add("colors", colors)
        spec.tokens.add("spacing", spacing)

        # Add a page with components
        home = PageSpec(name="Home", route="/")

        header = VisualSpec(component_id="header", component_type="Header")
        header.set_colors(background="#1f2937", text="#ffffff")
        header.layout = LayoutSpec(display=Display.FLEX, flex=FlexSpec())
        home.add_component(header)

        button = VisualSpec(component_id="cta-button", component_type="Button")
        button.set_colors(background="#3b82f6", text="#ffffff")
        home.add_component(button)

        spec.add_page(home)

        # Create templates
        templates = tmp_path / "templates"
        templates.mkdir()
        (templates / "index.html").write_text('''
            <header id="header">
                <nav></nav>
            </header>
            <main>
                <button id="cta-button">Get Started</button>
            </main>
        ''')

        # Create CSS
        css_file = tmp_path / "styles.css"
        css_file.write_text('''
            :root {
                --colors-primary: #3b82f6;
            }
            #header {
                background: var(--colors-text);
                padding: 16px;
            }
            #cta-button {
                background: var(--colors-primary);
                padding: 12px 24px;
            }
        ''')

        # Run full validation
        results = validate_all(spec, template_dir=templates, css_path=css_file)

        # Check results
        assert results["spec"].is_valid
        assert results["templates"].is_valid
        assert results["css"].is_valid

        overall_score = get_validation_score(results)
        assert overall_score > 80

    def test_validation_catches_issues(self, tmp_path: Path):
        """Test that validation catches multiple issue types."""
        spec = UISpec(name="problematic-app")

        # Page with component that has contrast issues
        page = PageSpec(name="Bad Page", route="/bad")
        comp = VisualSpec(component_id="bad-text", component_type="Text")
        comp.set_colors(background="#cccccc", text="#aaaaaa")  # Low contrast
        page.add_component(comp)
        spec.add_page(page)

        # Create template with unspecified component
        templates = tmp_path / "templates"
        templates.mkdir()
        (templates / "bad.html").write_text('<div id="unspecified">Content</div>')

        # Create CSS with hardcoded colors
        css_file = tmp_path / "styles.css"
        css_file.write_text('''
            .element {
                background: #ff0000;
                color: #00ff00;
                padding: 7px;
                margin: 13px;
            }
        ''')

        results = validate_all(spec, template_dir=templates, css_path=css_file)

        # Should have issues
        all_issues = []
        for report in results.values():
            all_issues.extend(report.issues)

        # Check for expected issue types
        issue_codes = {i.code for i in all_issues}
        assert "MISSING_COLOR_TOKENS" in issue_codes or "MISSING_SPACING_TOKENS" in issue_codes
        assert "MISSING_VISUAL_SPEC" in issue_codes  # unspecified component
        assert "HARDCODED_COLORS" in issue_codes

    def test_report_serialization(self):
        """Test that reports can be serialized to dict."""
        spec = UISpec(name="test")
        report = validate_spec(spec)

        # Convert to dict
        data = report.to_dict()

        # Should be JSON serializable
        import json
        json_str = json.dumps(data)
        parsed = json.loads(json_str)

        assert parsed["spec_name"] == "test"
        assert "is_valid" in parsed
        assert "score" in parsed
        assert "issues" in parsed
