"""
Tests for Visual Specification module (spec.py).

Tests:
- Layout enums (Display, Position, FlexDirection, etc.)
- FlexSpec and GridSpec CSS generation
- SpacingSpec with shorthand conversion
- LayoutSpec comprehensive CSS output
- Breakpoint media query generation
- ResponsiveValue overrides
- KeyframeAnimation CSS generation
- IconSpec and ImageSpec
- StateStyles CSS generation
- VisualSpec comprehensive testing
- PageSpec component management
- UISpec save/load and CSS generation
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch

from integradio.visual.spec import (
    # Enums
    Display,
    Position,
    FlexDirection,
    FlexWrap,
    JustifyContent,
    AlignItems,
    Overflow,
    # Layout specs
    FlexSpec,
    GridSpec,
    SpacingSpec,
    LayoutSpec,
    # Responsive
    Breakpoint,
    ResponsiveValue,
    BREAKPOINTS,
    # Animation
    KeyframeStep,
    KeyframeAnimation,
    # Assets
    IconSpec,
    ImageSpec,
    # Component specs
    StateStyles,
    VisualSpec,
    PageSpec,
    UISpec,
)
from integradio.visual.tokens import (
    DesignToken,
    TokenGroup,
    TokenType,
    DimensionValue,
    DurationValue,
    ColorValue,
    TransitionValue,
)


# =============================================================================
# Enum Tests
# =============================================================================

class TestEnums:
    """Tests for layout enums."""

    def test_display_enum_values(self):
        """Test Display enum has all expected values."""
        assert Display.BLOCK.value == "block"
        assert Display.INLINE.value == "inline"
        assert Display.INLINE_BLOCK.value == "inline-block"
        assert Display.FLEX.value == "flex"
        assert Display.INLINE_FLEX.value == "inline-flex"
        assert Display.GRID.value == "grid"
        assert Display.INLINE_GRID.value == "inline-grid"
        assert Display.NONE.value == "none"

    def test_position_enum_values(self):
        """Test Position enum has all expected values."""
        assert Position.STATIC.value == "static"
        assert Position.RELATIVE.value == "relative"
        assert Position.ABSOLUTE.value == "absolute"
        assert Position.FIXED.value == "fixed"
        assert Position.STICKY.value == "sticky"

    def test_flex_direction_enum_values(self):
        """Test FlexDirection enum values."""
        assert FlexDirection.ROW.value == "row"
        assert FlexDirection.ROW_REVERSE.value == "row-reverse"
        assert FlexDirection.COLUMN.value == "column"
        assert FlexDirection.COLUMN_REVERSE.value == "column-reverse"

    def test_flex_wrap_enum_values(self):
        """Test FlexWrap enum values."""
        assert FlexWrap.NOWRAP.value == "nowrap"
        assert FlexWrap.WRAP.value == "wrap"
        assert FlexWrap.WRAP_REVERSE.value == "wrap-reverse"

    def test_justify_content_enum_values(self):
        """Test JustifyContent enum values."""
        assert JustifyContent.FLEX_START.value == "flex-start"
        assert JustifyContent.FLEX_END.value == "flex-end"
        assert JustifyContent.CENTER.value == "center"
        assert JustifyContent.SPACE_BETWEEN.value == "space-between"
        assert JustifyContent.SPACE_AROUND.value == "space-around"
        assert JustifyContent.SPACE_EVENLY.value == "space-evenly"

    def test_align_items_enum_values(self):
        """Test AlignItems enum values."""
        assert AlignItems.FLEX_START.value == "flex-start"
        assert AlignItems.FLEX_END.value == "flex-end"
        assert AlignItems.CENTER.value == "center"
        assert AlignItems.BASELINE.value == "baseline"
        assert AlignItems.STRETCH.value == "stretch"

    def test_overflow_enum_values(self):
        """Test Overflow enum values."""
        assert Overflow.VISIBLE.value == "visible"
        assert Overflow.HIDDEN.value == "hidden"
        assert Overflow.SCROLL.value == "scroll"
        assert Overflow.AUTO.value == "auto"


# =============================================================================
# FlexSpec Tests
# =============================================================================

class TestFlexSpec:
    """Tests for FlexSpec dataclass."""

    def test_default_values(self):
        """Test FlexSpec default values."""
        spec = FlexSpec()
        assert spec.direction == FlexDirection.ROW
        assert spec.wrap == FlexWrap.NOWRAP
        assert spec.justify == JustifyContent.FLEX_START
        assert spec.align == AlignItems.STRETCH
        assert spec.gap is None

    def test_to_css_basic(self):
        """Test FlexSpec to_css without gap."""
        spec = FlexSpec()
        css = spec.to_css()

        assert css["display"] == "flex"
        assert css["flex-direction"] == "row"
        assert css["flex-wrap"] == "nowrap"
        assert css["justify-content"] == "flex-start"
        assert css["align-items"] == "stretch"
        assert "gap" not in css

    def test_to_css_with_gap(self):
        """Test FlexSpec to_css with gap."""
        spec = FlexSpec(gap=DimensionValue(16, "px"))
        css = spec.to_css()

        assert css["gap"] == "16px"

    def test_to_css_custom_values(self):
        """Test FlexSpec to_css with custom values."""
        spec = FlexSpec(
            direction=FlexDirection.COLUMN,
            wrap=FlexWrap.WRAP,
            justify=JustifyContent.CENTER,
            align=AlignItems.CENTER,
            gap=DimensionValue(1, "rem"),
        )
        css = spec.to_css()

        assert css["flex-direction"] == "column"
        assert css["flex-wrap"] == "wrap"
        assert css["justify-content"] == "center"
        assert css["align-items"] == "center"
        assert css["gap"] == "1rem"


# =============================================================================
# GridSpec Tests
# =============================================================================

class TestGridSpec:
    """Tests for GridSpec dataclass."""

    def test_default_values(self):
        """Test GridSpec default values."""
        spec = GridSpec()
        assert spec.columns == "1fr"
        assert spec.rows == "auto"
        assert spec.gap is None
        assert spec.auto_flow == "row"

    def test_to_css_basic(self):
        """Test GridSpec to_css without gap."""
        spec = GridSpec()
        css = spec.to_css()

        assert css["display"] == "grid"
        assert css["grid-template-columns"] == "1fr"
        assert css["grid-template-rows"] == "auto"
        assert css["grid-auto-flow"] == "row"
        assert "gap" not in css

    def test_to_css_with_gap(self):
        """Test GridSpec to_css with gap."""
        spec = GridSpec(
            columns="repeat(3, 1fr)",
            rows="100px auto",
            gap=DimensionValue(20, "px"),
            auto_flow="column dense",
        )
        css = spec.to_css()

        assert css["grid-template-columns"] == "repeat(3, 1fr)"
        assert css["grid-template-rows"] == "100px auto"
        assert css["gap"] == "20px"
        assert css["grid-auto-flow"] == "column dense"


# =============================================================================
# SpacingSpec Tests
# =============================================================================

class TestSpacingSpec:
    """Tests for SpacingSpec dataclass."""

    def test_default_values(self):
        """Test SpacingSpec default values (all None)."""
        spec = SpacingSpec()
        assert spec.top is None
        assert spec.right is None
        assert spec.bottom is None
        assert spec.left is None

    def test_all_classmethod(self):
        """Test SpacingSpec.all() creates uniform spacing."""
        value = DimensionValue(16, "px")
        spec = SpacingSpec.all(value)

        assert spec.top == value
        assert spec.right == value
        assert spec.bottom == value
        assert spec.left == value

    def test_symmetric_classmethod(self):
        """Test SpacingSpec.symmetric() creates symmetric spacing."""
        vertical = DimensionValue(8, "px")
        horizontal = DimensionValue(16, "px")
        spec = SpacingSpec.symmetric(vertical, horizontal)

        assert spec.top == vertical
        assert spec.bottom == vertical
        assert spec.left == horizontal
        assert spec.right == horizontal

    def test_to_css_with_all_sides(self):
        """Test to_css with all sides specified."""
        spec = SpacingSpec(
            top=DimensionValue(10, "px"),
            right=DimensionValue(20, "px"),
            bottom=DimensionValue(30, "px"),
            left=DimensionValue(40, "px"),
        )
        css = spec.to_css("padding")

        assert css["padding-top"] == "10px"
        assert css["padding-right"] == "20px"
        assert css["padding-bottom"] == "30px"
        assert css["padding-left"] == "40px"

    def test_to_css_with_margin(self):
        """Test to_css with margin property."""
        spec = SpacingSpec.all(DimensionValue(16, "px"))
        css = spec.to_css("margin")

        assert "margin-top" in css
        assert "margin-left" in css

    def test_to_css_partial(self):
        """Test to_css with partial sides."""
        spec = SpacingSpec(top=DimensionValue(8, "px"), left=DimensionValue(16, "px"))
        css = spec.to_css("padding")

        assert css["padding-top"] == "8px"
        assert css["padding-left"] == "16px"
        assert "padding-right" not in css
        assert "padding-bottom" not in css

    def test_to_shorthand_uniform(self):
        """Test to_shorthand with uniform spacing."""
        spec = SpacingSpec.all(DimensionValue(16, "px"))
        shorthand = spec.to_shorthand()

        assert shorthand == "16px"

    def test_to_shorthand_symmetric(self):
        """Test to_shorthand with symmetric spacing."""
        spec = SpacingSpec.symmetric(
            DimensionValue(8, "px"),
            DimensionValue(16, "px"),
        )
        shorthand = spec.to_shorthand()

        assert shorthand == "8px 16px"

    def test_to_shorthand_all_different(self):
        """Test to_shorthand with all different values."""
        spec = SpacingSpec(
            top=DimensionValue(1, "px"),
            right=DimensionValue(2, "px"),
            bottom=DimensionValue(3, "px"),
            left=DimensionValue(4, "px"),
        )
        shorthand = spec.to_shorthand()

        assert shorthand == "1px 2px 3px 4px"

    def test_to_shorthand_with_none_values(self):
        """Test to_shorthand substitutes 0 for None."""
        spec = SpacingSpec(top=DimensionValue(16, "px"))
        shorthand = spec.to_shorthand()

        # None values become "0"
        assert "16px" in shorthand
        assert "0" in shorthand


# =============================================================================
# LayoutSpec Tests
# =============================================================================

class TestLayoutSpec:
    """Tests for LayoutSpec dataclass."""

    def test_default_values(self):
        """Test LayoutSpec default values."""
        spec = LayoutSpec()
        assert spec.display == Display.BLOCK
        assert spec.position == Position.STATIC
        assert spec.overflow == Overflow.VISIBLE

    def test_to_css_minimal(self):
        """Test to_css with defaults (minimal output)."""
        spec = LayoutSpec()
        css = spec.to_css()

        # Default display=block and position=static are not included
        assert "display" not in css
        assert "position" not in css

    def test_to_css_with_display(self):
        """Test to_css with non-default display."""
        spec = LayoutSpec(display=Display.FLEX)
        css = spec.to_css()

        assert css["display"] == "flex"

    def test_to_css_with_position(self):
        """Test to_css with non-default position."""
        spec = LayoutSpec(position=Position.ABSOLUTE)
        css = spec.to_css()

        assert css["position"] == "absolute"

    def test_to_css_with_dimensions(self):
        """Test to_css with dimension values."""
        spec = LayoutSpec(
            width=DimensionValue(100, "%"),
            height=DimensionValue(50, "vh"),
            min_width=DimensionValue(320, "px"),
            max_width=DimensionValue(1200, "px"),
            min_height=DimensionValue(400, "px"),
            max_height=DimensionValue(800, "px"),
        )
        css = spec.to_css()

        assert css["width"] == "100%"
        assert css["height"] == "50vh"
        assert css["min-width"] == "320px"
        assert css["max-width"] == "1200px"
        assert css["min-height"] == "400px"
        assert css["max-height"] == "800px"

    def test_to_css_with_string_dimensions(self):
        """Test to_css with string literal dimensions."""
        spec = LayoutSpec(
            width="100%",
            height="auto",
        )
        css = spec.to_css()

        assert css["width"] == "100%"
        assert css["height"] == "auto"

    def test_to_css_with_padding_margin(self):
        """Test to_css with padding and margin."""
        spec = LayoutSpec(
            padding=SpacingSpec.all(DimensionValue(16, "px")),
            margin=SpacingSpec.symmetric(
                DimensionValue(0, "px"),
                DimensionValue(8, "px"),
            ),
        )
        css = spec.to_css()

        assert css["padding-top"] == "16px"
        assert css["margin-left"] == "8px"

    def test_to_css_with_flex(self):
        """Test to_css with flex layout."""
        spec = LayoutSpec(
            display=Display.FLEX,
            flex=FlexSpec(
                direction=FlexDirection.COLUMN,
                justify=JustifyContent.CENTER,
            ),
        )
        css = spec.to_css()

        # Flex properties should be merged
        assert css["flex-direction"] == "column"
        assert css["justify-content"] == "center"
        # display should not be duplicated from FlexSpec
        assert css.get("display") is None or css["display"] == "flex"

    def test_to_css_with_grid(self):
        """Test to_css with grid layout."""
        spec = LayoutSpec(
            display=Display.GRID,
            grid=GridSpec(
                columns="repeat(3, 1fr)",
                gap=DimensionValue(16, "px"),
            ),
        )
        css = spec.to_css()

        assert css["grid-template-columns"] == "repeat(3, 1fr)"
        assert css["gap"] == "16px"

    def test_to_css_with_flex_item_properties(self):
        """Test to_css with flex item properties."""
        spec = LayoutSpec(
            flex_grow=1,
            flex_shrink=0,
            flex_basis=DimensionValue(200, "px"),
        )
        css = spec.to_css()

        assert css["flex-grow"] == "1"
        assert css["flex-shrink"] == "0"
        assert css["flex-basis"] == "200px"

    def test_to_css_with_flex_basis_auto(self):
        """Test to_css with flex-basis: auto."""
        spec = LayoutSpec(flex_basis="auto")
        css = spec.to_css()

        assert css["flex-basis"] == "auto"

    def test_to_css_with_grid_item_properties(self):
        """Test to_css with grid item properties."""
        spec = LayoutSpec(
            grid_column="1 / 3",
            grid_row="span 2",
        )
        css = spec.to_css()

        assert css["grid-column"] == "1 / 3"
        assert css["grid-row"] == "span 2"

    def test_to_css_with_positioning(self):
        """Test to_css with position offsets."""
        spec = LayoutSpec(
            position=Position.ABSOLUTE,
            top=DimensionValue(0, "px"),
            right=DimensionValue(0, "px"),
            bottom=DimensionValue(0, "px"),
            left=DimensionValue(0, "px"),
            z_index=100,
        )
        css = spec.to_css()

        assert css["position"] == "absolute"
        assert css["top"] == "0px"
        assert css["right"] == "0px"
        assert css["bottom"] == "0px"
        assert css["left"] == "0px"
        assert css["z-index"] == "100"

    def test_to_css_with_overflow(self):
        """Test to_css with overflow settings."""
        spec = LayoutSpec(
            overflow=Overflow.HIDDEN,
            overflow_x=Overflow.AUTO,
            overflow_y=Overflow.SCROLL,
        )
        css = spec.to_css()

        assert css["overflow"] == "hidden"
        assert css["overflow-x"] == "auto"
        assert css["overflow-y"] == "scroll"


# =============================================================================
# Breakpoint Tests
# =============================================================================

class TestBreakpoint:
    """Tests for Breakpoint dataclass."""

    def test_min_width_only(self):
        """Test media query with min-width only."""
        bp = Breakpoint("md", min_width=768)
        query = bp.to_media_query()

        assert query == "(min-width: 768px)"

    def test_max_width_only(self):
        """Test media query with max-width only."""
        bp = Breakpoint("mobile", max_width=767)
        query = bp.to_media_query()

        assert query == "(max-width: 767px)"

    def test_both_widths(self):
        """Test media query with both min and max width."""
        bp = Breakpoint("tablet", min_width=768, max_width=1023)
        query = bp.to_media_query()

        assert query == "(min-width: 768px) and (max-width: 1023px)"

    def test_neither_width(self):
        """Test media query with no width constraints."""
        bp = Breakpoint("all")
        query = bp.to_media_query()

        assert query == "all"

    def test_predefined_breakpoints(self):
        """Test BREAKPOINTS constant has expected values."""
        assert "sm" in BREAKPOINTS
        assert "md" in BREAKPOINTS
        assert "lg" in BREAKPOINTS
        assert "xl" in BREAKPOINTS
        assert "2xl" in BREAKPOINTS

        assert BREAKPOINTS["sm"].min_width == 640
        assert BREAKPOINTS["md"].min_width == 768
        assert BREAKPOINTS["lg"].min_width == 1024
        assert BREAKPOINTS["xl"].min_width == 1280
        assert BREAKPOINTS["2xl"].min_width == 1536


# =============================================================================
# ResponsiveValue Tests
# =============================================================================

class TestResponsiveValue:
    """Tests for ResponsiveValue dataclass."""

    def test_default_value(self):
        """Test ResponsiveValue with default only."""
        rv = ResponsiveValue(default="16px")
        assert rv.default == "16px"
        assert rv.overrides == {}

    def test_at_method(self):
        """Test at() method for setting breakpoint values."""
        rv = ResponsiveValue(default="16px")
        result = rv.at("md", "24px").at("lg", "32px")

        assert result is rv  # Returns self for chaining
        assert rv.overrides["md"] == "24px"
        assert rv.overrides["lg"] == "32px"


# =============================================================================
# KeyframeAnimation Tests
# =============================================================================

class TestKeyframeAnimation:
    """Tests for KeyframeAnimation dataclass."""

    def test_basic_creation(self):
        """Test basic KeyframeAnimation creation."""
        anim = KeyframeAnimation(
            name="fadeIn",
            steps=[
                KeyframeStep(offset=0.0, properties={"opacity": "0"}),
                KeyframeStep(offset=1.0, properties={"opacity": "1"}),
            ],
        )
        assert anim.name == "fadeIn"
        assert len(anim.steps) == 2

    def test_to_css_keyframes_basic(self):
        """Test basic keyframes CSS generation."""
        anim = KeyframeAnimation(
            name="fadeIn",
            steps=[
                KeyframeStep(offset=0.0, properties={"opacity": "0"}),
                KeyframeStep(offset=1.0, properties={"opacity": "1"}),
            ],
        )
        css = anim.to_css_keyframes()

        assert "@keyframes fadeIn" in css
        assert "from { opacity: 0 }" in css
        assert "to { opacity: 1 }" in css

    def test_to_css_keyframes_percentage(self):
        """Test keyframes with percentage offset."""
        anim = KeyframeAnimation(
            name="bounce",
            steps=[
                KeyframeStep(offset=0.0, properties={"transform": "translateY(0)"}),
                KeyframeStep(offset=0.5, properties={"transform": "translateY(-20px)"}),
                KeyframeStep(offset=1.0, properties={"transform": "translateY(0)"}),
            ],
        )
        css = anim.to_css_keyframes()

        assert "50% { transform: translateY(-20px) }" in css

    def test_to_css_animation(self):
        """Test animation property value generation."""
        anim = KeyframeAnimation(
            name="slideIn",
            steps=[KeyframeStep(0, {"transform": "translateX(-100%)"}), KeyframeStep(1, {"transform": "translateX(0)"})],
            duration=DurationValue(300, "ms"),
            timing_function="ease-out",
            delay=DurationValue(100, "ms"),
            iteration_count=1,
            direction="normal",
            fill_mode="forwards",
        )
        css = anim.to_css_animation()

        assert "slideIn" in css
        assert "300ms" in css
        assert "ease-out" in css
        assert "100ms" in css
        assert "1" in css
        assert "normal" in css
        assert "forwards" in css

    def test_to_css_animation_infinite(self):
        """Test animation with infinite iterations."""
        anim = KeyframeAnimation(
            name="spin",
            steps=[
                KeyframeStep(0, {"transform": "rotate(0deg)"}),
                KeyframeStep(1, {"transform": "rotate(360deg)"}),
            ],
            iteration_count="infinite",
        )
        css = anim.to_css_animation()

        assert "infinite" in css


# =============================================================================
# IconSpec and ImageSpec Tests
# =============================================================================

class TestIconSpec:
    """Tests for IconSpec dataclass."""

    def test_default_values(self):
        """Test IconSpec default values."""
        icon = IconSpec(name="home")
        assert icon.name == "home"
        assert icon.library == "heroicons"
        assert icon.size.value == 24
        assert icon.color is None

    def test_to_svg_url_returns_none(self):
        """Test to_svg_url returns None (stub implementation)."""
        icon = IconSpec(name="settings")
        assert icon.to_svg_url() is None


class TestImageSpec:
    """Tests for ImageSpec dataclass."""

    def test_basic_creation(self):
        """Test basic ImageSpec creation."""
        img = ImageSpec(
            src="/images/hero.jpg",
            alt="Hero image",
        )
        assert img.src == "/images/hero.jpg"
        assert img.alt == "Hero image"
        assert img.object_fit == "cover"
        assert img.loading == "lazy"

    def test_custom_values(self):
        """Test ImageSpec with custom values."""
        img = ImageSpec(
            src="logo.png",
            alt="Logo",
            width=DimensionValue(200, "px"),
            height=DimensionValue(100, "px"),
            object_fit="contain",
            loading="eager",
        )
        assert img.width.value == 200
        assert img.height.value == 100
        assert img.object_fit == "contain"
        assert img.loading == "eager"


# =============================================================================
# StateStyles Tests
# =============================================================================

class TestStateStyles:
    """Tests for StateStyles dataclass."""

    def test_empty_states(self):
        """Test StateStyles with all empty states."""
        styles = StateStyles()
        css = styles.to_css(".btn")
        assert css == ""

    def test_default_state(self):
        """Test StateStyles with default state only."""
        styles = StateStyles(
            default={"background": DesignToken.color("#3b82f6")},
        )
        css = styles.to_css(".btn")

        assert ".btn {" in css
        assert "background: rgb(59, 130, 246)" in css

    def test_hover_state(self):
        """Test StateStyles with hover state."""
        styles = StateStyles(
            hover={"background": DesignToken.color("#2563eb")},
        )
        css = styles.to_css(".btn")

        assert ".btn:hover {" in css

    def test_focus_state(self):
        """Test StateStyles with focus state."""
        styles = StateStyles(
            focus={"outline": DesignToken.dimension(2, "px")},
        )
        css = styles.to_css(".input")

        assert ".input:focus {" in css

    def test_active_state(self):
        """Test StateStyles with active state."""
        styles = StateStyles(
            active={"transform": DesignToken.number(0.95)},
        )
        css = styles.to_css(".btn")

        assert ".btn:active {" in css

    def test_disabled_state(self):
        """Test StateStyles with disabled state."""
        styles = StateStyles(
            disabled={"opacity": DesignToken.number(0.5)},
        )
        css = styles.to_css(".btn")

        assert ".btn:disabled {" in css

    def test_all_states(self):
        """Test StateStyles with all states."""
        styles = StateStyles(
            default={"background": DesignToken.color("#3b82f6")},
            hover={"background": DesignToken.color("#2563eb")},
            focus={"outline": DesignToken.dimension(2, "px")},
            active={"transform": DesignToken.number(0.95)},
            disabled={"opacity": DesignToken.number(0.5)},
        )
        css = styles.to_css(".btn")

        assert ".btn {" in css
        assert ".btn:hover {" in css
        assert ".btn:focus {" in css
        assert ".btn:active {" in css
        assert ".btn:disabled {" in css


# =============================================================================
# VisualSpec Tests
# =============================================================================

class TestVisualSpec:
    """Tests for VisualSpec dataclass."""

    def test_basic_creation(self):
        """Test basic VisualSpec creation."""
        spec = VisualSpec(component_id="btn-primary")
        assert spec.component_id == "btn-primary"
        assert spec.component_type == ""
        assert len(spec.tokens) == 0

    def test_add_token(self):
        """Test add_token method."""
        spec = VisualSpec(component_id="card")
        result = spec.add_token("background", DesignToken.color("#ffffff"))

        assert result is spec  # Returns self for chaining
        assert "background" in spec.tokens

    def test_set_colors(self):
        """Test set_colors convenience method."""
        spec = VisualSpec(component_id="button")
        result = spec.set_colors(
            background="#3b82f6",
            text="#ffffff",
            border="#2563eb",
        )

        assert result is spec
        assert "background" in spec.tokens
        assert "color" in spec.tokens
        assert "border-color" in spec.tokens

    def test_set_colors_with_color_value(self):
        """Test set_colors with ColorValue objects."""
        spec = VisualSpec(component_id="button")
        spec.set_colors(
            background=ColorValue.from_hex("#ff0000"),
        )

        assert "background" in spec.tokens

    def test_set_spacing_with_dimension(self):
        """Test set_spacing with DimensionValue."""
        spec = VisualSpec(component_id="card")
        result = spec.set_spacing(
            padding=DimensionValue(16, "px"),
            margin=DimensionValue(8, "px"),
        )

        assert result is spec
        assert spec.layout.padding is not None
        assert spec.layout.margin is not None

    def test_set_spacing_with_spacing_spec(self):
        """Test set_spacing with SpacingSpec objects."""
        spec = VisualSpec(component_id="card")
        spec.set_spacing(
            padding=SpacingSpec.all(DimensionValue(20, "px")),
        )

        assert spec.layout.padding.top.value == 20

    def test_add_transition(self):
        """Test add_transition method."""
        spec = VisualSpec(component_id="button")
        result = spec.add_transition("all", 200)

        assert result is spec
        assert len(spec.transitions) == 1
        assert spec.transitions[0].duration.value == 200

    def test_to_css_basic(self):
        """Test to_css with basic spec."""
        spec = VisualSpec(component_id="test-btn")
        spec.set_colors(background="#3b82f6")
        css = spec.to_css()

        assert "#test-btn {" in css
        assert "background" in css

    def test_to_css_custom_selector(self):
        """Test to_css with custom selector."""
        spec = VisualSpec(component_id="btn")
        spec.set_colors(background="#ff0000")
        css = spec.to_css(".my-button")

        assert ".my-button {" in css

    def test_to_css_with_layout(self):
        """Test to_css includes layout properties."""
        spec = VisualSpec(component_id="card")
        spec.layout = LayoutSpec(
            display=Display.FLEX,
            width=DimensionValue(100, "%"),
        )
        css = spec.to_css()

        assert "display: flex" in css
        assert "width: 100%" in css

    def test_to_css_with_transitions(self):
        """Test to_css includes transitions."""
        spec = VisualSpec(component_id="btn")
        spec.add_transition("all", 300)
        css = spec.to_css()

        assert "transition:" in css

    def test_to_css_with_animations(self):
        """Test to_css includes animations."""
        spec = VisualSpec(component_id="spinner")
        spec.animations = [
            KeyframeAnimation(
                name="spin",
                steps=[
                    KeyframeStep(0, {"transform": "rotate(0deg)"}),
                    KeyframeStep(1, {"transform": "rotate(360deg)"}),
                ],
                iteration_count="infinite",
            )
        ]
        css = spec.to_css()

        assert "@keyframes spin" in css
        assert "animation:" in css

    def test_to_css_with_states(self):
        """Test to_css includes state styles."""
        spec = VisualSpec(component_id="btn")
        spec.states = StateStyles(
            hover={"background": DesignToken.color("#2563eb")},
        )
        css = spec.to_css()

        assert ":hover {" in css

    def test_to_css_with_responsive(self):
        """Test to_css includes responsive overrides."""
        spec = VisualSpec(component_id="grid")
        spec.responsive = {
            "md": {"display": "grid", "grid-template-columns": "repeat(2, 1fr)"},
            "lg": {"grid-template-columns": "repeat(3, 1fr)"},
        }
        css = spec.to_css()

        assert "@media (min-width: 768px)" in css
        assert "@media (min-width: 1024px)" in css

    def test_to_dict_basic(self):
        """Test to_dict with basic spec."""
        spec = VisualSpec(
            component_id="card",
            component_type="Card",
        )
        data = spec.to_dict()

        assert data["component_id"] == "card"
        assert data["component_type"] == "Card"
        assert "tokens" in data
        assert "layout" in data

    def test_to_dict_with_optional_fields(self):
        """Test to_dict includes optional fields when present."""
        spec = VisualSpec(component_id="btn")
        spec.add_transition("all", 200)
        spec.responsive = {"md": {"width": "50%"}}
        spec.icon = IconSpec(name="home")
        spec.image = ImageSpec(src="img.png", alt="Image")
        spec.parent_id = "container"
        spec.children_ids = ["child1", "child2"]
        spec.test_file = "test_btn.py"
        spec.test_line = 42

        data = spec.to_dict()

        assert "transitions" in data
        assert "responsive" in data
        assert "icon" in data
        assert "image" in data
        assert data["parent_id"] == "container"
        assert data["children_ids"] == ["child1", "child2"]
        assert data["test_file"] == "test_btn.py"
        assert data["test_line"] == 42


# =============================================================================
# PageSpec Tests
# =============================================================================

class TestPageSpec:
    """Tests for PageSpec dataclass."""

    def test_basic_creation(self):
        """Test basic PageSpec creation."""
        page = PageSpec(name="Home", route="/")
        assert page.name == "Home"
        assert page.route == "/"
        assert page.layout == "full-width"

    def test_add_component(self):
        """Test add_component method."""
        page = PageSpec(name="Home", route="/")
        component = VisualSpec(component_id="header")

        result = page.add_component(component)

        assert result is page
        assert "header" in page.components
        assert page.components["header"] is component

    def test_get_component(self):
        """Test get_component method."""
        page = PageSpec(name="Home", route="/")
        component = VisualSpec(component_id="nav")
        page.add_component(component)

        result = page.get_component("nav")
        assert result is component

        missing = page.get_component("nonexistent")
        assert missing is None

    def test_to_css_with_tokens(self):
        """Test to_css generates CSS custom properties from tokens."""
        page = PageSpec(name="Theme", route="/")
        page.tokens.add("primary", DesignToken.color("#3b82f6"))
        page.tokens.add("spacing", DesignToken.dimension(16, "px"))

        css = page.to_css()

        assert ":root {" in css
        assert "--primary" in css
        assert "--spacing" in css

    def test_to_css_with_components(self):
        """Test to_css includes component styles."""
        page = PageSpec(name="Home", route="/")
        btn = VisualSpec(component_id="btn")
        btn.set_colors(background="#ff0000")
        page.add_component(btn)

        css = page.to_css()

        assert "#btn {" in css

    def test_to_dict(self):
        """Test to_dict exports complete structure."""
        page = PageSpec(name="Dashboard", route="/dashboard", layout="dashboard-grid")
        page.add_component(VisualSpec(component_id="widget"))

        data = page.to_dict()

        assert data["name"] == "Dashboard"
        assert data["route"] == "/dashboard"
        assert data["layout"] == "dashboard-grid"
        assert "widget" in data["components"]
        assert "breakpoints" in data


# =============================================================================
# UISpec Tests
# =============================================================================

class TestUISpec:
    """Tests for UISpec dataclass."""

    def test_basic_creation(self):
        """Test basic UISpec creation."""
        ui = UISpec(name="MyApp")
        assert ui.name == "MyApp"
        assert ui.version == "1.0.0"

    def test_add_page(self):
        """Test add_page method."""
        ui = UISpec(name="App")
        page = PageSpec(name="Home", route="/")

        result = ui.add_page(page)

        assert result is ui
        assert "/" in ui.pages
        assert ui.pages["/"] is page

    def test_add_theme(self):
        """Test add_theme method."""
        ui = UISpec(name="App")
        dark_theme = TokenGroup()
        dark_theme.add("background", DesignToken.color("#1a1a1a"))

        result = ui.add_theme("dark", dark_theme)

        assert result is ui
        assert "dark" in ui.themes

    def test_to_dict(self):
        """Test to_dict exports complete structure."""
        ui = UISpec(name="MyApp", version="2.0.0")
        ui.add_page(PageSpec(name="Home", route="/"))
        ui.tokens.add("primary", DesignToken.color("#3b82f6"))

        dark = TokenGroup()
        dark.add("bg", DesignToken.color("#000000"))
        ui.add_theme("dark", dark)

        data = ui.to_dict()

        assert data["name"] == "MyApp"
        assert data["version"] == "2.0.0"
        assert "/" in data["pages"]
        assert "dark" in data["themes"]
        assert "breakpoints" in data

    def test_save_and_load(self, tmp_path):
        """Test save and load methods."""
        ui = UISpec(name="TestApp", version="1.0.0")
        ui.add_page(PageSpec(name="Home", route="/home"))

        file_path = tmp_path / "ui-spec.json"
        ui.save(file_path)

        assert file_path.exists()

        loaded = UISpec.load(file_path)

        assert loaded.name == "TestApp"
        assert loaded.version == "1.0.0"
        assert "/home" in loaded.pages
        assert loaded.pages["/home"].name == "Home"

    def test_load_with_partial_data(self, tmp_path):
        """Test load handles partial/minimal JSON."""
        file_path = tmp_path / "minimal.json"
        file_path.write_text('{"name": "Minimal"}')

        loaded = UISpec.load(file_path)

        assert loaded.name == "Minimal"
        assert loaded.version == "1.0.0"  # Default

    def test_to_css_basic(self):
        """Test to_css generates CSS for app."""
        ui = UISpec(name="App")
        ui.tokens.add("primary", DesignToken.color("#3b82f6"))

        page = PageSpec(name="Home", route="/")
        btn = VisualSpec(component_id="btn")
        btn.set_colors(background="#ff0000")
        page.add_component(btn)
        ui.add_page(page)

        css = ui.to_css()

        assert ":root {" in css
        assert "--primary" in css
        assert "#btn {" in css

    def test_to_css_with_theme(self):
        """Test to_css with theme generates themed CSS."""
        ui = UISpec(name="App")

        dark = TokenGroup()
        dark.add("bg", DesignToken.color("#1a1a1a"))
        dark.add("text", DesignToken.color("#ffffff"))
        ui.add_theme("dark", dark)

        css = ui.to_css(theme="dark")

        assert '[data-theme="dark"]' in css
        assert "--bg" in css
        assert "--text" in css

    def test_to_css_without_theme(self):
        """Test to_css without theme only uses global tokens."""
        ui = UISpec(name="App")
        ui.tokens.add("spacing", DesignToken.dimension(8, "px"))

        css = ui.to_css()

        assert "--spacing" in css
        assert "data-theme" not in css

    def test_to_css_with_nonexistent_theme(self):
        """Test to_css with non-existent theme ignores theme."""
        ui = UISpec(name="App")
        ui.tokens.add("primary", DesignToken.color("#3b82f6"))

        css = ui.to_css(theme="nonexistent")

        assert ":root {" in css
        assert "data-theme" not in css


# =============================================================================
# Integration Tests
# =============================================================================

class TestSpecIntegration:
    """Integration tests for spec module."""

    def test_complete_button_spec(self):
        """Test creating a complete button specification."""
        btn = VisualSpec(
            component_id="primary-button",
            component_type="Button",
        )

        # Set colors
        btn.set_colors(
            background="#3b82f6",
            text="#ffffff",
            border="#2563eb",
        )

        # Set spacing
        btn.set_spacing(
            padding=SpacingSpec.symmetric(
                DimensionValue(8, "px"),
                DimensionValue(16, "px"),
            ),
        )

        # Set layout
        btn.layout = LayoutSpec(
            display=Display.INLINE_FLEX,
            flex=FlexSpec(
                justify=JustifyContent.CENTER,
                align=AlignItems.CENTER,
            ),
        )

        # Add states
        btn.states = StateStyles(
            hover={"background": DesignToken.color("#2563eb")},
            focus={"outline": DesignToken.dimension(2, "px")},
            disabled={"opacity": DesignToken.number(0.5)},
        )

        # Add transition
        btn.add_transition("all", 200)

        # Generate CSS
        css = btn.to_css()

        assert "#primary-button {" in css
        # Display is inline-flex, flex properties should be present
        assert "justify-content: center" in css
        assert "align-items: center" in css
        assert ":hover {" in css
        assert ":focus {" in css
        assert ":disabled {" in css
        assert "transition:" in css

        # Export to dict
        data = btn.to_dict()
        assert data["component_id"] == "primary-button"
        assert data["component_type"] == "Button"

    def test_complete_page_with_components(self):
        """Test creating a complete page with multiple components."""
        page = PageSpec(name="Dashboard", route="/dashboard", layout="dashboard-grid")

        # Add page-level tokens
        page.tokens.add("primary", DesignToken.color("#3b82f6"))
        page.tokens.add("spacing-md", DesignToken.dimension(16, "px"))

        # Add header component
        header = VisualSpec(component_id="header", component_type="Header")
        header.layout = LayoutSpec(
            display=Display.FLEX,
            flex=FlexSpec(justify=JustifyContent.SPACE_BETWEEN),
            height=DimensionValue(64, "px"),
        )
        page.add_component(header)

        # Add sidebar component
        sidebar = VisualSpec(component_id="sidebar", component_type="Sidebar")
        sidebar.layout = LayoutSpec(
            width=DimensionValue(250, "px"),
            height="100vh",
        )
        sidebar.parent_id = "layout"
        page.add_component(sidebar)

        # Generate CSS
        css = page.to_css()

        assert ":root {" in css
        assert "--primary" in css
        assert "#header {" in css
        assert "#sidebar {" in css

        # Export
        data = page.to_dict()
        assert data["name"] == "Dashboard"
        assert "header" in data["components"]
        assert "sidebar" in data["components"]

    def test_complete_ui_spec(self, tmp_path):
        """Test creating a complete UI specification."""
        ui = UISpec(name="MyApp", version="1.0.0")

        # Global tokens
        ui.tokens.add("font-base", DesignToken.font_family(["Inter", "sans-serif"]))
        ui.tokens.add("primary", DesignToken.color("#3b82f6"))
        ui.tokens.add("spacing-unit", DesignToken.dimension(4, "px"))

        # Dark theme
        dark = TokenGroup()
        dark.add("bg", DesignToken.color("#0f172a"))
        dark.add("text", DesignToken.color("#f1f5f9"))
        ui.add_theme("dark", dark)

        # Home page
        home = PageSpec(name="Home", route="/")
        hero = VisualSpec(component_id="hero")
        hero.layout = LayoutSpec(
            display=Display.FLEX,
            flex=FlexSpec(direction=FlexDirection.COLUMN, align=AlignItems.CENTER),
            padding=SpacingSpec.all(DimensionValue(64, "px")),
        )
        home.add_component(hero)
        ui.add_page(home)

        # Save and reload
        path = tmp_path / "complete-spec.json"
        ui.save(path)

        loaded = UISpec.load(path)
        assert loaded.name == "MyApp"
        assert "/" in loaded.pages

        # Generate CSS
        css = ui.to_css(theme="dark")
        assert ":root {" in css
        assert '[data-theme="dark"]' in css
        assert "#hero {" in css
