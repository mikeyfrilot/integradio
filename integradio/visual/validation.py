"""
Visual Validation - Spec completeness and consistency checking.

This module provides "tests" for things that can't be traditionally tested:
- Template completeness (all components have visual specs)
- CSS consistency (color contrast, spacing scale adherence)
- Layout relationships (hierarchy, positioning)
- Asset existence (icons, images)
- Accessibility compliance (color contrast ratios)

Think of this as "type-checking for visual design."
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable
import math

from .tokens import (
    DesignToken,
    TokenType,
    TokenGroup,
    ColorValue,
    DimensionValue,
)
from .spec import (
    VisualSpec,
    PageSpec,
    UISpec,
    LayoutSpec,
    BREAKPOINTS,
)


# =============================================================================
# Validation Result Types
# =============================================================================

class Severity(str, Enum):
    """Validation issue severity."""
    ERROR = "error"      # Must fix - spec is broken
    WARNING = "warning"  # Should fix - best practice violation
    INFO = "info"        # Nice to know - suggestion


@dataclass
class ValidationIssue:
    """A single validation issue."""
    code: str           # e.g., "MISSING_COLOR", "LOW_CONTRAST"
    message: str        # Human-readable description
    severity: Severity
    path: str           # e.g., "pages.home.components.search-btn"
    suggestion: str = ""  # How to fix it
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        icon = {"error": "X", "warning": "!", "info": "i"}[self.severity.value]
        return f"[{icon}] {self.code}: {self.message} @ {self.path}"


@dataclass
class ValidationReport:
    """Complete validation report."""
    spec_name: str
    issues: list[ValidationIssue] = field(default_factory=list)
    passed: int = 0
    total_checks: int = 0

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    @property
    def is_valid(self) -> bool:
        """True if no errors (warnings are OK)."""
        return len(self.errors) == 0

    @property
    def score(self) -> float:
        """Validation score as percentage (0-100)."""
        if self.total_checks == 0:
            return 100.0
        return (self.passed / self.total_checks) * 100

    def add_issue(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)

    def add_pass(self) -> None:
        self.passed += 1
        self.total_checks += 1

    def add_fail(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)
        self.total_checks += 1

    def to_dict(self) -> dict:
        return {
            "spec_name": self.spec_name,
            "is_valid": self.is_valid,
            "score": round(self.score, 1),
            "passed": self.passed,
            "total_checks": self.total_checks,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "issues": [
                {
                    "code": i.code,
                    "message": i.message,
                    "severity": i.severity.value,
                    "path": i.path,
                    "suggestion": i.suggestion,
                }
                for i in self.issues
            ],
        }

    def __str__(self) -> str:
        lines = [
            f"Validation Report: {self.spec_name}",
            f"Score: {self.score:.1f}% ({self.passed}/{self.total_checks} checks passed)",
            f"Errors: {len(self.errors)}, Warnings: {len(self.warnings)}",
            "",
        ]
        for issue in self.issues:
            lines.append(str(issue))
        return "\n".join(lines)


# =============================================================================
# Color Utilities
# =============================================================================

def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def relative_luminance(r: int, g: int, b: int) -> float:
    """Calculate relative luminance per WCAG 2.1."""
    def adjust(c: int) -> float:
        c_srgb = c / 255
        if c_srgb <= 0.03928:
            return c_srgb / 12.92
        return ((c_srgb + 0.055) / 1.055) ** 2.4

    return 0.2126 * adjust(r) + 0.7152 * adjust(g) + 0.0722 * adjust(b)


def contrast_ratio(color1: str, color2: str) -> float:
    """
    Calculate WCAG contrast ratio between two colors.

    Returns value from 1 (no contrast) to 21 (max contrast).
    WCAG AA requires 4.5:1 for normal text, 3:1 for large text.
    WCAG AAA requires 7:1 for normal text, 4.5:1 for large text.
    """
    r1, g1, b1 = hex_to_rgb(color1)
    r2, g2, b2 = hex_to_rgb(color2)

    l1 = relative_luminance(r1, g1, b1)
    l2 = relative_luminance(r2, g2, b2)

    lighter = max(l1, l2)
    darker = min(l1, l2)

    return (lighter + 0.05) / (darker + 0.05)


def is_color_accessible(fg: str, bg: str, level: str = "AA", large_text: bool = False) -> bool:
    """Check if foreground/background colors meet WCAG contrast requirements."""
    ratio = contrast_ratio(fg, bg)

    if level == "AAA":
        threshold = 4.5 if large_text else 7.0
    else:  # AA
        threshold = 3.0 if large_text else 4.5

    return ratio >= threshold


# =============================================================================
# Validators
# =============================================================================

class SpecValidator:
    """
    Validates UISpec completeness and consistency.

    Usage:
        validator = SpecValidator(spec)
        report = validator.validate()
        print(report)
    """

    def __init__(self, spec: UISpec):
        self.spec = spec
        self.report = ValidationReport(spec_name=spec.name)

    def validate(self) -> ValidationReport:
        """Run all validation checks."""
        self._validate_tokens()
        self._validate_pages()
        self._validate_accessibility()
        self._validate_consistency()
        return self.report

    def _validate_tokens(self) -> None:
        """Validate global design tokens."""
        flat = self.spec.tokens.flatten()

        # Check for required token categories
        has_colors = any("color" in k.lower() for k in flat.keys())
        has_spacing = any("spacing" in k.lower() or "space" in k.lower() for k in flat.keys())

        if not has_colors:
            self.report.add_fail(ValidationIssue(
                code="MISSING_COLOR_TOKENS",
                message="No color tokens defined",
                severity=Severity.WARNING,
                path="tokens",
                suggestion="Add a 'colors' token group with primary, secondary, etc.",
            ))
        else:
            self.report.add_pass()

        if not has_spacing:
            self.report.add_fail(ValidationIssue(
                code="MISSING_SPACING_TOKENS",
                message="No spacing tokens defined",
                severity=Severity.INFO,
                path="tokens",
                suggestion="Add a 'spacing' token group with sm, md, lg values.",
            ))
        else:
            self.report.add_pass()

        # Validate each token
        for path, token in flat.items():
            self._validate_token(path, token)

    def _validate_token(self, path: str, token: DesignToken) -> None:
        """Validate a single token."""
        # Check aliases resolve
        if token.is_alias:
            ref_path = token._reference
            target = self.spec.tokens.get(ref_path)
            if target is None:
                self.report.add_fail(ValidationIssue(
                    code="BROKEN_ALIAS",
                    message=f"Token alias '{ref_path}' does not exist",
                    severity=Severity.ERROR,
                    path=f"tokens.{path}",
                    suggestion=f"Create token at '{ref_path}' or fix the reference.",
                ))
            else:
                self.report.add_pass()

        # Validate color values
        if token.type == TokenType.COLOR and isinstance(token.value, ColorValue):
            # Check for pure black/white (often unintentional)
            r, g, b = token.value.components
            if (r, g, b) == (0, 0, 0):
                self.report.add_fail(ValidationIssue(
                    code="PURE_BLACK",
                    message="Pure black (#000000) can be harsh; consider softer dark",
                    severity=Severity.INFO,
                    path=f"tokens.{path}",
                    suggestion="Try #111827 or #1f2937 for a softer dark.",
                ))
            elif (r, g, b) == (1, 1, 1):
                self.report.add_fail(ValidationIssue(
                    code="PURE_WHITE",
                    message="Pure white (#ffffff) can be harsh; consider off-white",
                    severity=Severity.INFO,
                    path=f"tokens.{path}",
                    suggestion="Try #f9fafb or #fafafa for a softer light.",
                ))
            else:
                self.report.add_pass()

    def _validate_pages(self) -> None:
        """Validate all pages and their components."""
        if not self.spec.pages:
            self.report.add_fail(ValidationIssue(
                code="NO_PAGES",
                message="No pages defined in spec",
                severity=Severity.WARNING,
                path="pages",
                suggestion="Add at least one PageSpec to the UISpec.",
            ))
            return

        self.report.add_pass()

        for route, page in self.spec.pages.items():
            self._validate_page(route, page)

    def _validate_page(self, route: str, page: PageSpec) -> None:
        """Validate a single page."""
        path = f"pages.{route}"

        if not page.components:
            self.report.add_fail(ValidationIssue(
                code="EMPTY_PAGE",
                message=f"Page '{page.name}' has no components",
                severity=Severity.WARNING,
                path=path,
                suggestion="Add VisualSpec components to the page.",
            ))
        else:
            self.report.add_pass()

        for comp_id, comp in page.components.items():
            self._validate_component(f"{path}.{comp_id}", comp)

    def _validate_component(self, path: str, spec: VisualSpec) -> None:
        """Validate a single component visual spec."""
        # Check required fields
        if not spec.component_type:
            self.report.add_fail(ValidationIssue(
                code="MISSING_COMPONENT_TYPE",
                message="Component type not specified",
                severity=Severity.WARNING,
                path=path,
                suggestion="Set component_type (e.g., 'Button', 'Textbox').",
            ))
        else:
            self.report.add_pass()

        # Check for minimal styling
        has_bg = "background" in spec.tokens
        has_color = "color" in spec.tokens

        if not has_bg and not has_color:
            self.report.add_fail(ValidationIssue(
                code="NO_COLORS",
                message="Component has no color tokens defined",
                severity=Severity.INFO,
                path=path,
                suggestion="Use spec.set_colors() to define appearance.",
            ))
        else:
            self.report.add_pass()

        # Validate layout
        self._validate_layout(path, spec.layout)

    def _validate_layout(self, path: str, layout: LayoutSpec) -> None:
        """Validate layout specification."""
        # Check for flex/grid without proper setup
        from .spec import Display

        if layout.display == Display.FLEX and layout.flex is None:
            self.report.add_fail(ValidationIssue(
                code="FLEX_WITHOUT_CONFIG",
                message="Display is 'flex' but no FlexSpec provided",
                severity=Severity.WARNING,
                path=f"{path}.layout",
                suggestion="Add a FlexSpec to configure flex layout.",
            ))
        elif layout.display == Display.FLEX:
            self.report.add_pass()

        if layout.display == Display.GRID and layout.grid is None:
            self.report.add_fail(ValidationIssue(
                code="GRID_WITHOUT_CONFIG",
                message="Display is 'grid' but no GridSpec provided",
                severity=Severity.WARNING,
                path=f"{path}.layout",
                suggestion="Add a GridSpec to configure grid layout.",
            ))
        elif layout.display == Display.GRID:
            self.report.add_pass()

    def _validate_accessibility(self) -> None:
        """Validate accessibility requirements."""
        for route, page in self.spec.pages.items():
            for comp_id, comp in page.components.items():
                self._validate_component_accessibility(
                    f"pages.{route}.{comp_id}",
                    comp,
                )

    def _validate_component_accessibility(self, path: str, spec: VisualSpec) -> None:
        """Check accessibility for a component."""
        bg_token = spec.tokens.get("background")
        fg_token = spec.tokens.get("color")

        if bg_token and fg_token:
            # Extract hex colors
            bg_hex = self._token_to_hex(bg_token)
            fg_hex = self._token_to_hex(fg_token)

            if bg_hex and fg_hex:
                ratio = contrast_ratio(fg_hex, bg_hex)

                if ratio < 3.0:
                    self.report.add_fail(ValidationIssue(
                        code="CONTRAST_FAIL",
                        message=f"Contrast ratio {ratio:.1f}:1 fails WCAG AA (needs 4.5:1)",
                        severity=Severity.ERROR,
                        path=path,
                        suggestion="Increase contrast between text and background colors.",
                        details={"ratio": ratio, "fg": fg_hex, "bg": bg_hex},
                    ))
                elif ratio < 4.5:
                    self.report.add_fail(ValidationIssue(
                        code="CONTRAST_LOW",
                        message=f"Contrast ratio {ratio:.1f}:1 only passes for large text",
                        severity=Severity.WARNING,
                        path=path,
                        suggestion="Consider increasing contrast for better readability.",
                        details={"ratio": ratio, "fg": fg_hex, "bg": bg_hex},
                    ))
                else:
                    self.report.add_pass()

    def _token_to_hex(self, token: DesignToken) -> str | None:
        """Extract hex color from a token."""
        if isinstance(token.value, ColorValue):
            return token.value.to_hex()
        if isinstance(token.value, str) and token.value.startswith("#"):
            return token.value
        return None

    def _validate_consistency(self) -> None:
        """Validate consistency across the spec."""
        # Check that all pages use consistent token references
        all_token_refs = set()
        for page in self.spec.pages.values():
            for comp in page.components.values():
                for name, token in comp.tokens.items():
                    if token.is_alias:
                        all_token_refs.add(token._reference)

        # Verify all references exist
        flat_tokens = self.spec.tokens.flatten()
        for ref in all_token_refs:
            if ref not in flat_tokens:
                self.report.add_fail(ValidationIssue(
                    code="UNDEFINED_TOKEN_REF",
                    message=f"Components reference undefined token '{ref}'",
                    severity=Severity.ERROR,
                    path="tokens",
                    suggestion=f"Define token at path '{ref}' or fix references.",
                ))
            else:
                self.report.add_pass()


# =============================================================================
# Template Validator
# =============================================================================

class TemplateValidator:
    """
    Validates that templates (HTML/Jinja) have corresponding visual specs.

    This bridges the gap between templates (which can't be unit tested)
    and visual specs (which can be validated).
    """

    def __init__(self, spec: UISpec, template_dir: str | Path):
        self.spec = spec
        self.template_dir = Path(template_dir)
        self.report = ValidationReport(spec_name=f"{spec.name} (templates)")

    def validate(self) -> ValidationReport:
        """Scan templates and check for missing specs."""
        if not self.template_dir.exists():
            self.report.add_fail(ValidationIssue(
                code="TEMPLATE_DIR_NOT_FOUND",
                message=f"Template directory not found: {self.template_dir}",
                severity=Severity.ERROR,
                path=str(self.template_dir),
            ))
            return self.report

        # Find all template files
        templates = list(self.template_dir.glob("**/*.html"))
        templates.extend(self.template_dir.glob("**/*.jinja"))
        templates.extend(self.template_dir.glob("**/*.jinja2"))

        for template_path in templates:
            self._validate_template(template_path)

        return self.report

    def _validate_template(self, path: Path) -> None:
        """Validate a single template file."""
        content = path.read_text(encoding="utf-8", errors="ignore")

        # Extract component IDs from template
        component_ids = self._extract_component_ids(content)

        # Check each has a visual spec
        for comp_id in component_ids:
            found = self._find_component_spec(comp_id)
            if not found:
                self.report.add_fail(ValidationIssue(
                    code="MISSING_VISUAL_SPEC",
                    message=f"Template component '{comp_id}' has no visual spec",
                    severity=Severity.WARNING,
                    path=str(path),
                    suggestion=f"Add VisualSpec(component_id='{comp_id}') to a page.",
                ))
            else:
                self.report.add_pass()

    def _extract_component_ids(self, content: str) -> list[str]:
        """Extract component IDs from HTML/template content."""
        ids = []

        # Match id="..." attributes
        id_pattern = r'id=["\']([^"\']+)["\']'
        ids.extend(re.findall(id_pattern, content))

        # Match elem_id="..." (Gradio)
        elem_id_pattern = r'elem_id=["\']([^"\']+)["\']'
        ids.extend(re.findall(elem_id_pattern, content))

        # Match data-component="..." (common pattern)
        data_pattern = r'data-component=["\']([^"\']+)["\']'
        ids.extend(re.findall(data_pattern, content))

        return list(set(ids))

    def _find_component_spec(self, comp_id: str) -> bool:
        """Check if a component ID has a visual spec."""
        for page in self.spec.pages.values():
            if comp_id in page.components:
                return True
        return False


# =============================================================================
# CSS Validator
# =============================================================================

class CSSValidator:
    """
    Validates CSS files against the visual spec.

    Ensures CSS doesn't contradict the spec and uses design tokens.
    """

    def __init__(self, spec: UISpec, css_path: str | Path):
        self.spec = spec
        self.css_path = Path(css_path)
        self.report = ValidationReport(spec_name=f"{spec.name} (CSS)")

    def validate(self) -> ValidationReport:
        """Validate CSS file."""
        if not self.css_path.exists():
            self.report.add_fail(ValidationIssue(
                code="CSS_NOT_FOUND",
                message=f"CSS file not found: {self.css_path}",
                severity=Severity.ERROR,
                path=str(self.css_path),
            ))
            return self.report

        content = self.css_path.read_text(encoding="utf-8")
        self._check_hardcoded_values(content)
        self._check_inconsistent_spacing(content)

        return self.report

    def _check_hardcoded_values(self, css: str) -> None:
        """Check for hardcoded values that should use tokens."""
        # Find hardcoded colors (not var())
        color_pattern = r'(?<!var\()#[0-9a-fA-F]{3,8}\b'
        hardcoded_colors = re.findall(color_pattern, css)

        if hardcoded_colors:
            unique_colors = set(hardcoded_colors)
            self.report.add_fail(ValidationIssue(
                code="HARDCODED_COLORS",
                message=f"Found {len(unique_colors)} hardcoded colors in CSS",
                severity=Severity.WARNING,
                path=str(self.css_path),
                suggestion="Use CSS custom properties: var(--colors-primary)",
                details={"colors": list(unique_colors)[:10]},  # First 10
            ))
        else:
            self.report.add_pass()

    def _check_inconsistent_spacing(self, css: str) -> None:
        """Check for inconsistent spacing values."""
        # Find all pixel values
        px_pattern = r'(\d+)px'
        px_values = [int(m) for m in re.findall(px_pattern, css)]

        if px_values:
            unique_values = set(px_values)
            # Check if using a consistent scale (multiples of 4 or 8)
            non_scale_4 = [v for v in unique_values if v % 4 != 0 and v > 1]
            if len(non_scale_4) > 5:
                self.report.add_fail(ValidationIssue(
                    code="INCONSISTENT_SPACING",
                    message=f"Many spacing values don't follow 4px scale",
                    severity=Severity.INFO,
                    path=str(self.css_path),
                    suggestion="Use consistent spacing scale (4, 8, 12, 16, 24, 32...)",
                    details={"non_scale_values": sorted(non_scale_4)[:10]},
                ))
            else:
                self.report.add_pass()


# =============================================================================
# Convenience Functions
# =============================================================================

def validate_spec(spec: UISpec) -> ValidationReport:
    """Quick validation of a UISpec."""
    validator = SpecValidator(spec)
    return validator.validate()


def validate_templates(
    spec: UISpec,
    template_dir: str | Path,
) -> ValidationReport:
    """Validate templates against spec."""
    validator = TemplateValidator(spec, template_dir)
    return validator.validate()


def validate_css(spec: UISpec, css_path: str | Path) -> ValidationReport:
    """Validate CSS against spec."""
    validator = CSSValidator(spec, css_path)
    return validator.validate()


def validate_all(
    spec: UISpec,
    template_dir: str | Path | None = None,
    css_path: str | Path | None = None,
) -> dict[str, ValidationReport]:
    """
    Run all validators and return combined results.

    Returns:
        Dict mapping validator name to report
    """
    results = {
        "spec": validate_spec(spec),
    }

    if template_dir:
        results["templates"] = validate_templates(spec, template_dir)

    if css_path:
        results["css"] = validate_css(spec, css_path)

    return results


def get_validation_score(reports: dict[str, ValidationReport]) -> float:
    """Calculate overall validation score from multiple reports."""
    total_passed = sum(r.passed for r in reports.values())
    total_checks = sum(r.total_checks for r in reports.values())

    if total_checks == 0:
        return 100.0
    return (total_passed / total_checks) * 100
