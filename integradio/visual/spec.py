"""
Visual Specification - Complete visual contract for UI components.

Extends W3C DTCG tokens with layout, responsive behavior, and component mapping.
This captures what tests can't: the visual appearance of the interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Callable
from enum import Enum
import json
from pathlib import Path

from .tokens import (
    DesignToken,
    TokenGroup,
    TokenType,
    ColorValue,
    DimensionValue,
    DurationValue,
    TransitionValue,
    ShadowValue,
    BorderValue,
    TypographyValue,
)


# =============================================================================
# Layout Types (Extending DTCG)
# =============================================================================

class Display(str, Enum):
    """CSS display values."""
    BLOCK = "block"
    INLINE = "inline"
    INLINE_BLOCK = "inline-block"
    FLEX = "flex"
    INLINE_FLEX = "inline-flex"
    GRID = "grid"
    INLINE_GRID = "inline-grid"
    NONE = "none"


class Position(str, Enum):
    """CSS position values."""
    STATIC = "static"
    RELATIVE = "relative"
    ABSOLUTE = "absolute"
    FIXED = "fixed"
    STICKY = "sticky"


class FlexDirection(str, Enum):
    """CSS flex-direction values."""
    ROW = "row"
    ROW_REVERSE = "row-reverse"
    COLUMN = "column"
    COLUMN_REVERSE = "column-reverse"


class FlexWrap(str, Enum):
    """CSS flex-wrap values."""
    NOWRAP = "nowrap"
    WRAP = "wrap"
    WRAP_REVERSE = "wrap-reverse"


class JustifyContent(str, Enum):
    """CSS justify-content values."""
    FLEX_START = "flex-start"
    FLEX_END = "flex-end"
    CENTER = "center"
    SPACE_BETWEEN = "space-between"
    SPACE_AROUND = "space-around"
    SPACE_EVENLY = "space-evenly"


class AlignItems(str, Enum):
    """CSS align-items values."""
    FLEX_START = "flex-start"
    FLEX_END = "flex-end"
    CENTER = "center"
    BASELINE = "baseline"
    STRETCH = "stretch"


class Overflow(str, Enum):
    """CSS overflow values."""
    VISIBLE = "visible"
    HIDDEN = "hidden"
    SCROLL = "scroll"
    AUTO = "auto"


# =============================================================================
# Layout Specification
# =============================================================================

@dataclass
class FlexSpec:
    """Flexbox layout properties."""
    direction: FlexDirection = FlexDirection.ROW
    wrap: FlexWrap = FlexWrap.NOWRAP
    justify: JustifyContent = JustifyContent.FLEX_START
    align: AlignItems = AlignItems.STRETCH
    gap: DimensionValue | None = None

    def to_css(self) -> dict[str, str]:
        """Convert to CSS properties."""
        result = {
            "display": "flex",
            "flex-direction": self.direction.value,
            "flex-wrap": self.wrap.value,
            "justify-content": self.justify.value,
            "align-items": self.align.value,
        }
        if self.gap:
            result["gap"] = self.gap.to_css()
        return result


@dataclass
class GridSpec:
    """CSS Grid layout properties."""
    columns: str = "1fr"  # grid-template-columns
    rows: str = "auto"    # grid-template-rows
    gap: DimensionValue | None = None
    auto_flow: Literal["row", "column", "dense", "row dense", "column dense"] = "row"

    def to_css(self) -> dict[str, str]:
        """Convert to CSS properties."""
        result = {
            "display": "grid",
            "grid-template-columns": self.columns,
            "grid-template-rows": self.rows,
            "grid-auto-flow": self.auto_flow,
        }
        if self.gap:
            result["gap"] = self.gap.to_css()
        return result


@dataclass
class SpacingSpec:
    """Spacing (padding/margin) specification."""
    top: DimensionValue | None = None
    right: DimensionValue | None = None
    bottom: DimensionValue | None = None
    left: DimensionValue | None = None

    @classmethod
    def all(cls, value: DimensionValue) -> "SpacingSpec":
        """Create uniform spacing."""
        return cls(top=value, right=value, bottom=value, left=value)

    @classmethod
    def symmetric(cls, vertical: DimensionValue, horizontal: DimensionValue) -> "SpacingSpec":
        """Create symmetric spacing."""
        return cls(top=vertical, right=horizontal, bottom=vertical, left=horizontal)

    def to_css(self, property: str = "padding") -> dict[str, str]:
        """Convert to CSS properties."""
        result = {}
        if self.top:
            result[f"{property}-top"] = self.top.to_css()
        if self.right:
            result[f"{property}-right"] = self.right.to_css()
        if self.bottom:
            result[f"{property}-bottom"] = self.bottom.to_css()
        if self.left:
            result[f"{property}-left"] = self.left.to_css()
        return result

    def to_shorthand(self) -> str:
        """Convert to CSS shorthand value."""
        values = [
            self.top.to_css() if self.top else "0",
            self.right.to_css() if self.right else "0",
            self.bottom.to_css() if self.bottom else "0",
            self.left.to_css() if self.left else "0",
        ]
        # Simplify if possible
        if values[0] == values[1] == values[2] == values[3]:
            return values[0]
        if values[0] == values[2] and values[1] == values[3]:
            return f"{values[0]} {values[1]}"
        return " ".join(values)


@dataclass
class LayoutSpec:
    """Complete layout specification for a component."""
    display: Display = Display.BLOCK
    position: Position = Position.STATIC

    # Dimensions
    width: DimensionValue | Literal["auto", "100%", "fit-content"] | None = None
    height: DimensionValue | Literal["auto", "100%", "fit-content"] | None = None
    min_width: DimensionValue | None = None
    max_width: DimensionValue | None = None
    min_height: DimensionValue | None = None
    max_height: DimensionValue | None = None

    # Spacing
    padding: SpacingSpec | None = None
    margin: SpacingSpec | None = None

    # Flexbox (when display=flex)
    flex: FlexSpec | None = None

    # Grid (when display=grid)
    grid: GridSpec | None = None

    # Flex item properties
    flex_grow: float | None = None
    flex_shrink: float | None = None
    flex_basis: DimensionValue | Literal["auto"] | None = None

    # Grid item properties
    grid_column: str | None = None
    grid_row: str | None = None

    # Positioning
    top: DimensionValue | None = None
    right: DimensionValue | None = None
    bottom: DimensionValue | None = None
    left: DimensionValue | None = None
    z_index: int | None = None

    # Overflow
    overflow: Overflow = Overflow.VISIBLE
    overflow_x: Overflow | None = None
    overflow_y: Overflow | None = None

    def to_css(self) -> dict[str, str]:
        """Convert to CSS properties dict."""
        result: dict[str, str] = {}

        # Display & position
        if self.display != Display.BLOCK:
            result["display"] = self.display.value
        if self.position != Position.STATIC:
            result["position"] = self.position.value

        # Dimensions
        if self.width:
            result["width"] = self.width.to_css() if isinstance(self.width, DimensionValue) else self.width
        if self.height:
            result["height"] = self.height.to_css() if isinstance(self.height, DimensionValue) else self.height
        if self.min_width:
            result["min-width"] = self.min_width.to_css()
        if self.max_width:
            result["max-width"] = self.max_width.to_css()
        if self.min_height:
            result["min-height"] = self.min_height.to_css()
        if self.max_height:
            result["max-height"] = self.max_height.to_css()

        # Spacing
        if self.padding:
            result.update(self.padding.to_css("padding"))
        if self.margin:
            result.update(self.margin.to_css("margin"))

        # Flexbox
        if self.flex and self.display in (Display.FLEX, Display.INLINE_FLEX):
            result.update(self.flex.to_css())
            result.pop("display", None)  # Already set

        # Grid
        if self.grid and self.display in (Display.GRID, Display.INLINE_GRID):
            result.update(self.grid.to_css())
            result.pop("display", None)  # Already set

        # Flex item
        if self.flex_grow is not None:
            result["flex-grow"] = str(self.flex_grow)
        if self.flex_shrink is not None:
            result["flex-shrink"] = str(self.flex_shrink)
        if self.flex_basis:
            result["flex-basis"] = self.flex_basis.to_css() if isinstance(self.flex_basis, DimensionValue) else self.flex_basis

        # Grid item
        if self.grid_column:
            result["grid-column"] = self.grid_column
        if self.grid_row:
            result["grid-row"] = self.grid_row

        # Positioning
        if self.top:
            result["top"] = self.top.to_css()
        if self.right:
            result["right"] = self.right.to_css()
        if self.bottom:
            result["bottom"] = self.bottom.to_css()
        if self.left:
            result["left"] = self.left.to_css()
        if self.z_index is not None:
            result["z-index"] = str(self.z_index)

        # Overflow
        if self.overflow != Overflow.VISIBLE:
            result["overflow"] = self.overflow.value
        if self.overflow_x:
            result["overflow-x"] = self.overflow_x.value
        if self.overflow_y:
            result["overflow-y"] = self.overflow_y.value

        return result


# =============================================================================
# Responsive Breakpoints
# =============================================================================

@dataclass
class Breakpoint:
    """A responsive breakpoint definition."""
    name: str
    min_width: int | None = None  # px
    max_width: int | None = None  # px

    def to_media_query(self) -> str:
        """Convert to CSS media query."""
        conditions = []
        if self.min_width:
            conditions.append(f"(min-width: {self.min_width}px)")
        if self.max_width:
            conditions.append(f"(max-width: {self.max_width}px)")
        return " and ".join(conditions) if conditions else "all"


# Common breakpoint presets (Tailwind-style)
BREAKPOINTS = {
    "sm": Breakpoint("sm", min_width=640),
    "md": Breakpoint("md", min_width=768),
    "lg": Breakpoint("lg", min_width=1024),
    "xl": Breakpoint("xl", min_width=1280),
    "2xl": Breakpoint("2xl", min_width=1536),
}


@dataclass
class ResponsiveValue:
    """A value that changes at different breakpoints."""
    default: Any
    overrides: dict[str, Any] = field(default_factory=dict)  # breakpoint -> value

    def at(self, breakpoint: str, value: Any) -> "ResponsiveValue":
        """Set value at a breakpoint."""
        self.overrides[breakpoint] = value
        return self


# =============================================================================
# Animation / Keyframes
# =============================================================================

@dataclass
class KeyframeStep:
    """A single keyframe step."""
    offset: float  # 0.0 to 1.0 (or "from"=0, "to"=1)
    properties: dict[str, str]  # CSS property -> value


@dataclass
class KeyframeAnimation:
    """A CSS keyframe animation definition."""
    name: str
    steps: list[KeyframeStep]
    duration: DurationValue = field(default_factory=lambda: DurationValue(300, "ms"))
    timing_function: str = "ease"
    delay: DurationValue = field(default_factory=lambda: DurationValue(0, "ms"))
    iteration_count: int | Literal["infinite"] = 1
    direction: Literal["normal", "reverse", "alternate", "alternate-reverse"] = "normal"
    fill_mode: Literal["none", "forwards", "backwards", "both"] = "none"

    def to_css_keyframes(self) -> str:
        """Generate @keyframes CSS."""
        lines = [f"@keyframes {self.name} {{"]
        for step in self.steps:
            percent = f"{int(step.offset * 100)}%"
            if step.offset == 0:
                percent = "from"
            elif step.offset == 1:
                percent = "to"
            props = "; ".join(f"{k}: {v}" for k, v in step.properties.items())
            lines.append(f"  {percent} {{ {props} }}")
        lines.append("}")
        return "\n".join(lines)

    def to_css_animation(self) -> str:
        """Generate animation property value."""
        iteration = "infinite" if self.iteration_count == "infinite" else str(self.iteration_count)
        return f"{self.name} {self.duration.to_css()} {self.timing_function} {self.delay.to_css()} {iteration} {self.direction} {self.fill_mode}"


# =============================================================================
# Asset Specification
# =============================================================================

@dataclass
class IconSpec:
    """Icon specification."""
    name: str  # Icon name/identifier
    library: str = "heroicons"  # Icon library (heroicons, lucide, material, etc.)
    size: DimensionValue = field(default_factory=lambda: DimensionValue(24, "px"))
    color: ColorValue | str | None = None  # None = inherit

    def to_svg_url(self) -> str | None:
        """Generate SVG URL if using a known library."""
        # This could integrate with icon CDNs
        return None


@dataclass
class ImageSpec:
    """Image specification."""
    src: str
    alt: str
    width: DimensionValue | None = None
    height: DimensionValue | None = None
    object_fit: Literal["contain", "cover", "fill", "none", "scale-down"] = "cover"
    loading: Literal["eager", "lazy"] = "lazy"


# =============================================================================
# Component Visual Specification
# =============================================================================

@dataclass
class StateStyles:
    """Styles for different component states."""
    default: dict[str, DesignToken] = field(default_factory=dict)
    hover: dict[str, DesignToken] = field(default_factory=dict)
    focus: dict[str, DesignToken] = field(default_factory=dict)
    active: dict[str, DesignToken] = field(default_factory=dict)
    disabled: dict[str, DesignToken] = field(default_factory=dict)

    def to_css(self, selector: str) -> str:
        """Generate CSS for all states."""
        lines = []

        # Default state
        if self.default:
            props = "; ".join(f"{k}: {v.to_css()}" for k, v in self.default.items())
            lines.append(f"{selector} {{ {props} }}")

        # Hover
        if self.hover:
            props = "; ".join(f"{k}: {v.to_css()}" for k, v in self.hover.items())
            lines.append(f"{selector}:hover {{ {props} }}")

        # Focus
        if self.focus:
            props = "; ".join(f"{k}: {v.to_css()}" for k, v in self.focus.items())
            lines.append(f"{selector}:focus {{ {props} }}")

        # Active
        if self.active:
            props = "; ".join(f"{k}: {v.to_css()}" for k, v in self.active.items())
            lines.append(f"{selector}:active {{ {props} }}")

        # Disabled
        if self.disabled:
            props = "; ".join(f"{k}: {v.to_css()}" for k, v in self.disabled.items())
            lines.append(f"{selector}:disabled {{ {props} }}")

        return "\n".join(lines)


@dataclass
class VisualSpec:
    """
    Complete visual specification for a component.

    This captures everything tests can't:
    - Colors, typography, spacing
    - Layout and positioning
    - Responsive behavior
    - Animations and transitions
    - Icons and images
    """
    # Identity
    component_id: str
    component_type: str = ""

    # Design tokens
    tokens: dict[str, DesignToken] = field(default_factory=dict)

    # State-based styling
    states: StateStyles | None = None

    # Layout
    layout: LayoutSpec = field(default_factory=LayoutSpec)

    # Responsive overrides
    responsive: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Animations
    transitions: list[TransitionValue] = field(default_factory=list)
    animations: list[KeyframeAnimation] = field(default_factory=list)

    # Assets
    icon: IconSpec | None = None
    image: ImageSpec | None = None

    # Component hierarchy
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)

    # Linked test file (for the test <-> spec mapping)
    test_file: str | None = None
    test_line: int | None = None

    def add_token(self, name: str, token: DesignToken) -> "VisualSpec":
        """Add a design token."""
        self.tokens[name] = token
        return self

    def set_colors(
        self,
        background: str | ColorValue | None = None,
        text: str | ColorValue | None = None,
        border: str | ColorValue | None = None,
    ) -> "VisualSpec":
        """Convenience method to set common color tokens."""
        if background:
            self.tokens["background"] = DesignToken.color(background)
        if text:
            self.tokens["color"] = DesignToken.color(text)
        if border:
            self.tokens["border-color"] = DesignToken.color(border)
        return self

    def set_spacing(
        self,
        padding: SpacingSpec | DimensionValue | None = None,
        margin: SpacingSpec | DimensionValue | None = None,
    ) -> "VisualSpec":
        """Convenience method to set spacing."""
        if padding:
            if isinstance(padding, DimensionValue):
                self.layout.padding = SpacingSpec.all(padding)
            else:
                self.layout.padding = padding
        if margin:
            if isinstance(margin, DimensionValue):
                self.layout.margin = SpacingSpec.all(margin)
            else:
                self.layout.margin = margin
        return self

    def add_transition(self, property: str = "all", duration_ms: int = 200) -> "VisualSpec":
        """Add a transition."""
        self.transitions.append(TransitionValue(
            duration=DurationValue(duration_ms, "ms"),
        ))
        return self

    def to_css(self, selector: str | None = None) -> str:
        """Generate CSS for this component."""
        if selector is None:
            selector = f"#{self.component_id}"

        lines = []

        # Keyframe animations
        for anim in self.animations:
            lines.append(anim.to_css_keyframes())

        # Main styles
        props: list[str] = []

        # Tokens
        for name, token in self.tokens.items():
            props.append(f"{name}: {token.to_css()}")

        # Layout
        for name, value in self.layout.to_css().items():
            props.append(f"{name}: {value}")

        # Transitions
        if self.transitions:
            trans_css = ", ".join(t.to_css() for t in self.transitions)
            props.append(f"transition: {trans_css}")

        # Animations
        if self.animations:
            anim_css = ", ".join(a.to_css_animation() for a in self.animations)
            props.append(f"animation: {anim_css}")

        if props:
            lines.append(f"{selector} {{ {'; '.join(props)} }}")

        # State styles
        if self.states:
            lines.append(self.states.to_css(selector))

        # Responsive styles
        for breakpoint_name, overrides in self.responsive.items():
            bp = BREAKPOINTS.get(breakpoint_name)
            if bp:
                override_props = "; ".join(f"{k}: {v}" for k, v in overrides.items())
                lines.append(f"@media {bp.to_media_query()} {{ {selector} {{ {override_props} }} }}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Export to JSON-serializable dict."""
        result: dict[str, Any] = {
            "component_id": self.component_id,
            "component_type": self.component_type,
            "tokens": {k: v.to_dtcg() for k, v in self.tokens.items()},
            "layout": self.layout.to_css(),
        }

        if self.transitions:
            result["transitions"] = [t.to_dtcg() for t in self.transitions]

        if self.responsive:
            result["responsive"] = self.responsive

        if self.icon:
            result["icon"] = {
                "name": self.icon.name,
                "library": self.icon.library,
                "size": self.icon.size.to_css(),
            }

        if self.image:
            result["image"] = {
                "src": self.image.src,
                "alt": self.image.alt,
            }

        if self.parent_id:
            result["parent_id"] = self.parent_id

        if self.children_ids:
            result["children_ids"] = self.children_ids

        if self.test_file:
            result["test_file"] = self.test_file
            result["test_line"] = self.test_line

        return result


# =============================================================================
# Page Visual Specification
# =============================================================================

@dataclass
class PageSpec:
    """Visual specification for an entire page/view."""
    name: str
    route: str
    components: dict[str, VisualSpec] = field(default_factory=dict)

    # Page-level tokens (inherited by components)
    tokens: TokenGroup = field(default_factory=TokenGroup)

    # Page layout
    layout: Literal["sidebar-main", "centered", "dashboard-grid", "full-width", "split"] = "full-width"

    # Breakpoints for this page
    breakpoints: dict[str, Breakpoint] = field(default_factory=lambda: dict(BREAKPOINTS))

    def add_component(self, spec: VisualSpec) -> "PageSpec":
        """Add a component visual spec."""
        self.components[spec.component_id] = spec
        return self

    def get_component(self, component_id: str) -> VisualSpec | None:
        """Get a component by ID."""
        return self.components.get(component_id)

    def to_css(self) -> str:
        """Generate CSS for the entire page."""
        lines = []

        # CSS custom properties from tokens
        flat_tokens = self.tokens.flatten()
        if flat_tokens:
            vars_css = "; ".join(
                f"--{k.replace('.', '-')}: {v.to_css()}"
                for k, v in flat_tokens.items()
            )
            lines.append(f":root {{ {vars_css} }}")

        # Component styles
        for spec in self.components.values():
            lines.append(spec.to_css())

        return "\n\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Export to JSON-serializable dict."""
        return {
            "name": self.name,
            "route": self.route,
            "layout": self.layout,
            "tokens": self.tokens.to_dtcg(),
            "components": {k: v.to_dict() for k, v in self.components.items()},
            "breakpoints": {k: {"min_width": v.min_width, "max_width": v.max_width} for k, v in self.breakpoints.items()},
        }


# =============================================================================
# UI Specification (Full App)
# =============================================================================

@dataclass
class UISpec:
    """
    Complete visual specification for an application.

    This is the top-level container that holds:
    - Global design tokens
    - Page specifications
    - Theme variants
    """
    name: str
    version: str = "1.0.0"

    # Global design tokens
    tokens: TokenGroup = field(default_factory=TokenGroup)

    # Pages
    pages: dict[str, PageSpec] = field(default_factory=dict)

    # Theme variants (e.g., light, dark)
    themes: dict[str, TokenGroup] = field(default_factory=dict)

    # Global breakpoints
    breakpoints: dict[str, Breakpoint] = field(default_factory=lambda: dict(BREAKPOINTS))

    def add_page(self, page: PageSpec) -> "UISpec":
        """Add a page specification."""
        self.pages[page.route] = page
        return self

    def add_theme(self, name: str, tokens: TokenGroup) -> "UISpec":
        """Add a theme variant."""
        self.themes[name] = tokens
        return self

    def to_dict(self) -> dict[str, Any]:
        """Export to JSON-serializable dict."""
        return {
            "name": self.name,
            "version": self.version,
            "tokens": self.tokens.to_dtcg(),
            "pages": {k: v.to_dict() for k, v in self.pages.items()},
            "themes": {k: v.to_dtcg() for k, v in self.themes.items()},
            "breakpoints": {k: {"min_width": v.min_width, "max_width": v.max_width} for k, v in self.breakpoints.items()},
        }

    def save(self, path: str | Path) -> None:
        """Save specification to JSON file."""
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "UISpec":
        """Load specification from JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        # TODO: Implement full deserialization
        return cls(name=data.get("name", ""), version=data.get("version", "1.0.0"))

    def to_css(self, theme: str | None = None) -> str:
        """Generate CSS for the entire application."""
        lines = []

        # Global tokens as CSS custom properties
        flat_tokens = self.tokens.flatten()
        if flat_tokens:
            vars_css = "; ".join(
                f"--{k.replace('.', '-')}: {v.to_css()}"
                for k, v in flat_tokens.items()
            )
            lines.append(f":root {{ {vars_css} }}")

        # Theme tokens
        if theme and theme in self.themes:
            theme_tokens = self.themes[theme].flatten()
            if theme_tokens:
                vars_css = "; ".join(
                    f"--{k.replace('.', '-')}: {v.to_css()}"
                    for k, v in theme_tokens.items()
                )
                lines.append(f"[data-theme=\"{theme}\"] {{ {vars_css} }}")

        # Page styles
        for page in self.pages.values():
            lines.append(f"/* Page: {page.name} ({page.route}) */")
            lines.append(page.to_css())

        return "\n\n".join(lines)
