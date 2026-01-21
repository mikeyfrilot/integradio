"""
Tests for Design Tokens module (tokens.py).

Tests:
- TokenType enum values
- ColorValue creation, conversion, and CSS generation
- DimensionValue and DurationValue CSS output
- CubicBezierValue presets and CSS generation
- StrokeStyleValue keyword and custom styles
- Composite values (Shadow, Border, Transition, Gradient, Typography)
- DesignToken class methods and conversions
- TokenGroup hierarchy and flattening
"""

import pytest
import json

from integradio.visual.tokens import (
    # Enums and types
    TokenType,
    # Value classes
    ColorValue,
    DimensionValue,
    DurationValue,
    CubicBezierValue,
    StrokeStyleValue,
    ShadowValue,
    BorderValue,
    TransitionValue,
    GradientStop,
    GradientValue,
    TypographyValue,
    # Token classes
    DesignToken,
    TokenGroup,
)


# =============================================================================
# TokenType Enum Tests
# =============================================================================

class TestTokenType:
    """Tests for TokenType enum."""

    def test_basic_types(self):
        """Test basic token type values."""
        assert TokenType.COLOR.value == "color"
        assert TokenType.DIMENSION.value == "dimension"
        assert TokenType.FONT_FAMILY.value == "fontFamily"
        assert TokenType.FONT_WEIGHT.value == "fontWeight"
        assert TokenType.FONT_STYLE.value == "fontStyle"
        assert TokenType.DURATION.value == "duration"
        assert TokenType.CUBIC_BEZIER.value == "cubicBezier"
        assert TokenType.NUMBER.value == "number"
        assert TokenType.STRING.value == "string"
        assert TokenType.BOOLEAN.value == "boolean"

    def test_composite_types(self):
        """Test composite token type values."""
        assert TokenType.STROKE_STYLE.value == "strokeStyle"
        assert TokenType.BORDER.value == "border"
        assert TokenType.TRANSITION.value == "transition"
        assert TokenType.SHADOW.value == "shadow"
        assert TokenType.GRADIENT.value == "gradient"
        assert TokenType.TYPOGRAPHY.value == "typography"


# =============================================================================
# ColorValue Tests
# =============================================================================

class TestColorValue:
    """Tests for ColorValue dataclass."""

    def test_default_values(self):
        """Test ColorValue default values."""
        color = ColorValue()
        assert color.color_space == "srgb"
        assert color.components == (0.0, 0.0, 0.0)
        assert color.alpha == 1.0

    def test_from_hex_6_chars(self):
        """Test from_hex with 6-character hex."""
        color = ColorValue.from_hex("#ff0000")
        assert color.components == (1.0, 0.0, 0.0)
        assert color.alpha == 1.0

    def test_from_hex_without_hash(self):
        """Test from_hex without # prefix."""
        color = ColorValue.from_hex("00ff00")
        assert color.components == (0.0, 1.0, 0.0)

    def test_from_hex_8_chars(self):
        """Test from_hex with 8-character hex (with alpha)."""
        color = ColorValue.from_hex("#ff000080")  # 50% alpha
        assert color.components[0] == 1.0
        assert 0.49 < color.alpha < 0.51

    def test_from_hex_invalid(self):
        """Test from_hex with invalid hex string."""
        with pytest.raises(ValueError, match="Invalid hex color"):
            ColorValue.from_hex("#fff")  # Too short

    def test_to_hex_no_alpha(self):
        """Test to_hex without alpha."""
        color = ColorValue.from_hex("#3b82f6")
        assert color.to_hex() == "#3b82f6"

    def test_to_hex_with_alpha(self):
        """Test to_hex with alpha channel."""
        color = ColorValue(
            components=(1.0, 0.0, 0.0),
            alpha=0.5,
        )
        hex_str = color.to_hex()
        assert len(hex_str) == 9  # #rrggbbaa
        assert hex_str.startswith("#ff0000")

    def test_to_css_rgb(self):
        """Test to_css for RGB color."""
        color = ColorValue.from_hex("#3b82f6")
        css = color.to_css()
        assert css == "rgb(59, 130, 246)"

    def test_to_css_rgba(self):
        """Test to_css for RGBA color."""
        color = ColorValue(
            components=(1.0, 0.0, 0.0),
            alpha=0.5,
        )
        css = color.to_css()
        assert css == "rgba(255, 0, 0, 0.5)"

    def test_to_css_modern_color_space(self):
        """Test to_css for modern color spaces."""
        color = ColorValue(
            color_space="display-p3",
            components=(0.5, 0.3, 0.8),
            alpha=1.0,
        )
        css = color.to_css()
        assert "color(display-p3" in css
        assert "0.5 0.3 0.8" in css

    def test_to_dtcg(self):
        """Test to_dtcg export."""
        color = ColorValue.from_hex("#3b82f6")
        dtcg = color.to_dtcg()

        assert dtcg["colorSpace"] == "srgb"
        assert len(dtcg["components"]) == 3
        assert dtcg["alpha"] == 1.0


# =============================================================================
# DimensionValue Tests
# =============================================================================

class TestDimensionValue:
    """Tests for DimensionValue dataclass."""

    def test_default_unit(self):
        """Test default unit is px."""
        dim = DimensionValue(16)
        assert dim.unit == "px"

    def test_to_css_integer(self):
        """Test to_css for integer values."""
        dim = DimensionValue(16, "px")
        assert dim.to_css() == "16px"

    def test_to_css_float(self):
        """Test to_css for float values."""
        dim = DimensionValue(1.5, "rem")
        assert dim.to_css() == "1.5rem"

    def test_to_css_various_units(self):
        """Test to_css with various units."""
        assert DimensionValue(100, "%").to_css() == "100%"
        assert DimensionValue(2, "em").to_css() == "2em"
        assert DimensionValue(50, "vw").to_css() == "50vw"
        assert DimensionValue(100, "vh").to_css() == "100vh"

    def test_to_dtcg(self):
        """Test to_dtcg export."""
        dim = DimensionValue(16, "px")
        dtcg = dim.to_dtcg()

        assert dtcg["value"] == 16
        assert dtcg["unit"] == "px"


# =============================================================================
# DurationValue Tests
# =============================================================================

class TestDurationValue:
    """Tests for DurationValue dataclass."""

    def test_default_unit(self):
        """Test default unit is ms."""
        dur = DurationValue(200)
        assert dur.unit == "ms"

    def test_to_css_integer(self):
        """Test to_css for integer values."""
        dur = DurationValue(300, "ms")
        assert dur.to_css() == "300ms"

    def test_to_css_float(self):
        """Test to_css for float values."""
        dur = DurationValue(0.5, "s")
        assert dur.to_css() == "0.5s"

    def test_to_dtcg(self):
        """Test to_dtcg export."""
        dur = DurationValue(200, "ms")
        dtcg = dur.to_dtcg()

        assert dtcg["value"] == 200
        assert dtcg["unit"] == "ms"


# =============================================================================
# CubicBezierValue Tests
# =============================================================================

class TestCubicBezierValue:
    """Tests for CubicBezierValue dataclass."""

    def test_custom_values(self):
        """Test custom cubic bezier values."""
        bezier = CubicBezierValue(0.1, 0.2, 0.3, 0.4)
        assert bezier.p1x == 0.1
        assert bezier.p1y == 0.2
        assert bezier.p2x == 0.3
        assert bezier.p2y == 0.4

    def test_ease_preset(self):
        """Test ease preset."""
        bezier = CubicBezierValue.ease()
        assert bezier.p1x == 0.25
        assert bezier.p1y == 0.1
        assert bezier.p2x == 0.25
        assert bezier.p2y == 1.0

    def test_ease_in_preset(self):
        """Test ease-in preset."""
        bezier = CubicBezierValue.ease_in()
        assert bezier.p1x == 0.42
        assert bezier.p1y == 0.0
        assert bezier.p2x == 1.0
        assert bezier.p2y == 1.0

    def test_ease_out_preset(self):
        """Test ease-out preset."""
        bezier = CubicBezierValue.ease_out()
        assert bezier.p1x == 0.0
        assert bezier.p2x == 0.58

    def test_ease_in_out_preset(self):
        """Test ease-in-out preset."""
        bezier = CubicBezierValue.ease_in_out()
        assert bezier.p1x == 0.42
        assert bezier.p2x == 0.58

    def test_linear_preset(self):
        """Test linear preset."""
        bezier = CubicBezierValue.linear()
        assert bezier.p1x == 0.0
        assert bezier.p1y == 0.0
        assert bezier.p2x == 1.0
        assert bezier.p2y == 1.0

    def test_to_css(self):
        """Test to_css output."""
        bezier = CubicBezierValue(0.42, 0.0, 0.58, 1.0)
        css = bezier.to_css()
        assert css == "cubic-bezier(0.42, 0.0, 0.58, 1.0)"

    def test_to_dtcg(self):
        """Test to_dtcg export."""
        bezier = CubicBezierValue.ease()
        dtcg = bezier.to_dtcg()
        assert dtcg == [0.25, 0.1, 0.25, 1.0]


# =============================================================================
# StrokeStyleValue Tests
# =============================================================================

class TestStrokeStyleValue:
    """Tests for StrokeStyleValue dataclass."""

    def test_keyword_style(self):
        """Test keyword stroke style."""
        stroke = StrokeStyleValue(style="solid")
        assert stroke.to_css() == "solid"

    def test_dashed_style(self):
        """Test dashed stroke style."""
        stroke = StrokeStyleValue(style="dashed")
        assert stroke.to_css() == "dashed"

    def test_custom_dash_array(self):
        """Test custom dash array (falls back to dashed in CSS)."""
        stroke = StrokeStyleValue(
            style=None,
            dash_array=[DimensionValue(4, "px"), DimensionValue(2, "px")],
            line_cap="round",
        )
        assert stroke.to_css() == "dashed"

    def test_to_dtcg_keyword(self):
        """Test to_dtcg for keyword style."""
        stroke = StrokeStyleValue(style="dotted")
        assert stroke.to_dtcg() == "dotted"

    def test_to_dtcg_custom(self):
        """Test to_dtcg for custom style."""
        stroke = StrokeStyleValue(
            style=None,
            dash_array=[DimensionValue(5, "px")],
            line_cap="butt",
        )
        dtcg = stroke.to_dtcg()
        assert isinstance(dtcg, dict)
        assert "dashArray" in dtcg
        assert dtcg["lineCap"] == "butt"


# =============================================================================
# ShadowValue Tests
# =============================================================================

class TestShadowValue:
    """Tests for ShadowValue dataclass."""

    def test_default_values(self):
        """Test ShadowValue default values."""
        shadow = ShadowValue(color=ColorValue.from_hex("#000000"))
        assert shadow.offset_x.value == 0
        assert shadow.offset_y.value == 4
        assert shadow.blur.value == 8
        assert shadow.spread.value == 0
        assert shadow.inset is False

    def test_to_css_basic(self):
        """Test to_css for basic shadow."""
        shadow = ShadowValue(
            color=ColorValue.from_hex("#000000"),
            offset_x=DimensionValue(0, "px"),
            offset_y=DimensionValue(4, "px"),
            blur=DimensionValue(8, "px"),
            spread=DimensionValue(0, "px"),
        )
        css = shadow.to_css()

        assert "0px" in css
        assert "4px" in css
        assert "8px" in css
        assert "rgb(0, 0, 0)" in css
        assert "inset" not in css

    def test_to_css_inset(self):
        """Test to_css for inset shadow."""
        shadow = ShadowValue(
            color=ColorValue.from_hex("#000000"),
            inset=True,
        )
        css = shadow.to_css()
        assert css.startswith("inset ")

    def test_to_dtcg(self):
        """Test to_dtcg export."""
        shadow = ShadowValue(color=ColorValue.from_hex("#000000"))
        dtcg = shadow.to_dtcg()

        assert "color" in dtcg
        assert "offsetX" in dtcg
        assert "offsetY" in dtcg
        assert "blur" in dtcg
        assert "spread" in dtcg
        assert dtcg["inset"] is False


# =============================================================================
# BorderValue Tests
# =============================================================================

class TestBorderValue:
    """Tests for BorderValue dataclass."""

    def test_default_values(self):
        """Test BorderValue default values."""
        border = BorderValue(color=ColorValue.from_hex("#cccccc"))
        assert border.width.value == 1
        assert border.style.style == "solid"

    def test_to_css(self):
        """Test to_css output."""
        border = BorderValue(
            color=ColorValue.from_hex("#3b82f6"),
            width=DimensionValue(2, "px"),
            style=StrokeStyleValue("dashed"),
        )
        css = border.to_css()

        assert "2px" in css
        assert "dashed" in css
        assert "rgb(59, 130, 246)" in css

    def test_to_dtcg(self):
        """Test to_dtcg export."""
        border = BorderValue(color=ColorValue.from_hex("#000000"))
        dtcg = border.to_dtcg()

        assert "color" in dtcg
        assert "width" in dtcg
        assert "style" in dtcg


# =============================================================================
# TransitionValue Tests
# =============================================================================

class TestTransitionValue:
    """Tests for TransitionValue dataclass."""

    def test_default_values(self):
        """Test TransitionValue default values."""
        trans = TransitionValue()
        assert trans.duration.value == 200
        assert trans.delay.value == 0

    def test_to_css(self):
        """Test to_css output."""
        trans = TransitionValue(
            duration=DurationValue(300, "ms"),
            delay=DurationValue(100, "ms"),
            timing_function=CubicBezierValue.ease_out(),
        )
        css = trans.to_css("opacity")

        assert "opacity" in css
        assert "300ms" in css
        assert "cubic-bezier" in css
        assert "100ms" in css

    def test_to_css_default_property(self):
        """Test to_css with default property 'all'."""
        trans = TransitionValue()
        css = trans.to_css()
        assert "all" in css

    def test_to_dtcg(self):
        """Test to_dtcg export."""
        trans = TransitionValue()
        dtcg = trans.to_dtcg()

        assert "duration" in dtcg
        assert "delay" in dtcg
        assert "timingFunction" in dtcg


# =============================================================================
# GradientValue Tests
# =============================================================================

class TestGradientValue:
    """Tests for GradientValue dataclass."""

    def test_empty_gradient(self):
        """Test empty gradient."""
        grad = GradientValue()
        assert len(grad.stops) == 0

    def test_with_stops(self):
        """Test gradient with stops."""
        grad = GradientValue(
            stops=[
                GradientStop(ColorValue.from_hex("#ff0000"), 0.0),
                GradientStop(ColorValue.from_hex("#0000ff"), 1.0),
            ]
        )
        assert len(grad.stops) == 2

    def test_to_css(self):
        """Test to_css output."""
        grad = GradientValue(
            stops=[
                GradientStop(ColorValue.from_hex("#ffffff"), 0.0),
                GradientStop(ColorValue.from_hex("#000000"), 1.0),
            ]
        )
        css = grad.to_css()

        assert "linear-gradient" in css
        assert "to bottom" in css
        # Position format may be 0% or 0.0%, 100% or 100.0%
        assert ("0%" in css or "0.0%" in css)
        assert ("100%" in css or "100.0%" in css)

    def test_to_css_custom_direction(self):
        """Test to_css with custom direction."""
        grad = GradientValue(
            stops=[
                GradientStop(ColorValue.from_hex("#ff0000"), 0.0),
                GradientStop(ColorValue.from_hex("#0000ff"), 1.0),
            ]
        )
        css = grad.to_css("to right")
        assert "to right" in css

    def test_to_dtcg(self):
        """Test to_dtcg export."""
        grad = GradientValue(
            stops=[
                GradientStop(ColorValue.from_hex("#ffffff"), 0.0),
                GradientStop(ColorValue.from_hex("#000000"), 1.0),
            ]
        )
        dtcg = grad.to_dtcg()

        assert len(dtcg) == 2
        assert "color" in dtcg[0]
        assert "position" in dtcg[0]

    def test_gradient_stop_to_dtcg(self):
        """Test GradientStop to_dtcg."""
        stop = GradientStop(ColorValue.from_hex("#3b82f6"), 0.5)
        dtcg = stop.to_dtcg()

        assert dtcg["position"] == 0.5
        assert "color" in dtcg


# =============================================================================
# TypographyValue Tests
# =============================================================================

class TestTypographyValue:
    """Tests for TypographyValue dataclass."""

    def test_default_values(self):
        """Test TypographyValue default values."""
        typo = TypographyValue()
        assert typo.font_family == ["system-ui", "sans-serif"]
        assert typo.font_size.value == 16
        assert typo.font_weight == 400
        assert typo.font_style == "normal"
        assert typo.line_height == 1.5

    def test_to_css(self):
        """Test to_css output (returns dict)."""
        typo = TypographyValue(
            font_family=["Inter", "sans-serif"],
            font_size=DimensionValue(18, "px"),
            font_weight=600,
            line_height=1.6,
        )
        css = typo.to_css()

        assert isinstance(css, dict)
        assert css["font-size"] == "18px"
        assert css["font-weight"] == "600"
        assert css["line-height"] == "1.6"
        assert "Inter" in css["font-family"]

    def test_to_css_font_family_quoting(self):
        """Test font-family quoting for names with spaces."""
        typo = TypographyValue(font_family=["Open Sans", "Arial"])
        css = typo.to_css()

        # Names with spaces should be quoted
        assert '"Open Sans"' in css["font-family"]
        # Names without spaces should not be quoted
        assert "Arial" in css["font-family"]
        assert '"Arial"' not in css["font-family"]

    def test_to_dtcg(self):
        """Test to_dtcg export."""
        typo = TypographyValue()
        dtcg = typo.to_dtcg()

        assert "fontFamily" in dtcg
        assert "fontSize" in dtcg
        assert "fontWeight" in dtcg
        assert "fontStyle" in dtcg
        assert "letterSpacing" in dtcg
        assert "lineHeight" in dtcg


# =============================================================================
# DesignToken Tests
# =============================================================================

class TestDesignToken:
    """Tests for DesignToken dataclass."""

    def test_color_factory(self):
        """Test color class method."""
        token = DesignToken.color("#3b82f6")
        assert token.type == TokenType.COLOR
        assert isinstance(token.value, ColorValue)

    def test_color_factory_with_color_value(self):
        """Test color factory with ColorValue object."""
        color_val = ColorValue.from_hex("#ff0000")
        token = DesignToken.color(color_val)
        assert token.value is color_val

    def test_dimension_factory(self):
        """Test dimension class method."""
        token = DesignToken.dimension(16, "px")
        assert token.type == TokenType.DIMENSION
        assert token.value.value == 16
        assert token.value.unit == "px"

    def test_duration_factory(self):
        """Test duration class method."""
        token = DesignToken.duration(300, "ms")
        assert token.type == TokenType.DURATION
        assert token.value.value == 300

    def test_font_family_factory_string(self):
        """Test font_family factory with string."""
        token = DesignToken.font_family("Inter")
        assert token.type == TokenType.FONT_FAMILY
        assert token.value == ["Inter"]

    def test_font_family_factory_list(self):
        """Test font_family factory with list."""
        token = DesignToken.font_family(["Inter", "sans-serif"])
        assert token.value == ["Inter", "sans-serif"]

    def test_font_weight_factory(self):
        """Test font_weight factory."""
        token = DesignToken.font_weight(700)
        assert token.type == TokenType.FONT_WEIGHT
        assert token.value == 700

    def test_number_factory(self):
        """Test number factory."""
        token = DesignToken.number(1.5)
        assert token.type == TokenType.NUMBER
        assert token.value == 1.5

    def test_shadow_factory(self):
        """Test shadow factory."""
        shadow = ShadowValue(color=ColorValue.from_hex("#000000"))
        token = DesignToken.shadow(shadow)
        assert token.type == TokenType.SHADOW

    def test_border_factory(self):
        """Test border factory."""
        border = BorderValue(color=ColorValue.from_hex("#cccccc"))
        token = DesignToken.border(border)
        assert token.type == TokenType.BORDER

    def test_transition_factory(self):
        """Test transition factory."""
        trans = TransitionValue()
        token = DesignToken.transition(trans)
        assert token.type == TokenType.TRANSITION

    def test_gradient_factory(self):
        """Test gradient factory."""
        grad = GradientValue()
        token = DesignToken.gradient(grad)
        assert token.type == TokenType.GRADIENT

    def test_typography_factory(self):
        """Test typography factory."""
        typo = TypographyValue()
        token = DesignToken.typography(typo)
        assert token.type == TokenType.TYPOGRAPHY

    def test_reference_factory(self):
        """Test reference (alias) factory."""
        token = DesignToken.reference("colors.primary", TokenType.COLOR)
        assert token.is_alias
        assert token._reference == "colors.primary"

    def test_is_alias_false(self):
        """Test is_alias for non-alias token."""
        token = DesignToken.color("#ff0000")
        assert not token.is_alias

    def test_to_css_basic(self):
        """Test to_css for basic token."""
        token = DesignToken.color("#3b82f6")
        css = token.to_css()
        assert "rgb(59, 130, 246)" in css

    def test_to_css_alias(self):
        """Test to_css for alias token."""
        token = DesignToken.reference("colors.primary", TokenType.COLOR)
        css = token.to_css()
        assert css == "var(--colors-primary)"

    def test_to_css_typography(self):
        """Test to_css for typography token (returns joined dict)."""
        token = DesignToken.typography(TypographyValue())
        css = token.to_css()
        assert "font-family:" in css
        assert "font-size:" in css

    def test_to_css_primitive(self):
        """Test to_css for primitive value."""
        token = DesignToken.number(42)
        assert token.to_css() == "42"

    def test_to_dtcg_basic(self):
        """Test to_dtcg for basic token."""
        token = DesignToken.color("#ff0000", description="Primary red")
        dtcg = token.to_dtcg()

        assert dtcg["$type"] == "color"
        assert "$value" in dtcg
        assert dtcg["$description"] == "Primary red"

    def test_to_dtcg_without_description(self):
        """Test to_dtcg without description."""
        token = DesignToken.dimension(16, "px")
        dtcg = token.to_dtcg()

        assert "$description" not in dtcg

    def test_to_dtcg_with_extensions(self):
        """Test to_dtcg with extensions."""
        token = DesignToken.color("#ff0000")
        token.extensions = {"figma": {"styleId": "123"}}
        dtcg = token.to_dtcg()

        assert "$extensions" in dtcg
        assert dtcg["$extensions"]["figma"]["styleId"] == "123"

    def test_to_dtcg_alias(self):
        """Test to_dtcg for alias token."""
        token = DesignToken.reference("spacing.md", TokenType.DIMENSION)
        dtcg = token.to_dtcg()

        assert dtcg["$value"] == "{spacing.md}"

    def test_to_dtcg_primitive_value(self):
        """Test to_dtcg with primitive value (no to_dtcg method)."""
        token = DesignToken.font_weight(700)
        dtcg = token.to_dtcg()

        assert dtcg["$value"] == 700


# =============================================================================
# TokenGroup Tests
# =============================================================================

class TestTokenGroup:
    """Tests for TokenGroup dataclass."""

    def test_empty_group(self):
        """Test empty TokenGroup."""
        group = TokenGroup()
        assert len(group.tokens) == 0
        assert group.type is None
        assert group.description == ""

    def test_add_token(self):
        """Test add method with token."""
        group = TokenGroup()
        token = DesignToken.color("#ff0000")
        group.add("primary", token)

        assert "primary" in group.tokens
        assert group.tokens["primary"] is token

    def test_add_subgroup(self):
        """Test add method with subgroup."""
        parent = TokenGroup()
        child = TokenGroup()
        child.add("sm", DesignToken.dimension(8, "px"))

        parent.add("spacing", child)

        assert "spacing" in parent.tokens
        assert isinstance(parent.tokens["spacing"], TokenGroup)

    def test_get_direct(self):
        """Test get with direct path."""
        group = TokenGroup()
        group.add("primary", DesignToken.color("#ff0000"))

        result = group.get("primary")
        assert result is not None
        assert result.type == TokenType.COLOR

    def test_get_nested(self):
        """Test get with nested path."""
        root = TokenGroup()
        colors = TokenGroup()
        colors.add("primary", DesignToken.color("#3b82f6"))
        colors.add("secondary", DesignToken.color("#6366f1"))
        root.add("colors", colors)

        result = root.get("colors.primary")
        assert result is not None
        assert result.type == TokenType.COLOR

    def test_get_deeply_nested(self):
        """Test get with deeply nested path."""
        root = TokenGroup()
        theme = TokenGroup()
        colors = TokenGroup()
        colors.add("blue", DesignToken.color("#0000ff"))
        theme.add("colors", colors)
        root.add("theme", theme)

        result = root.get("theme.colors.blue")
        assert result is not None

    def test_get_nonexistent(self):
        """Test get with non-existent path."""
        group = TokenGroup()
        assert group.get("nonexistent") is None
        assert group.get("a.b.c") is None

    def test_get_path_to_non_group(self):
        """Test get when path goes through non-group."""
        group = TokenGroup()
        group.add("primary", DesignToken.color("#ff0000"))

        # Try to traverse through a token (not a group)
        result = group.get("primary.variant")
        assert result is None

    def test_to_dtcg_empty(self):
        """Test to_dtcg for empty group."""
        group = TokenGroup()
        dtcg = group.to_dtcg()
        assert dtcg == {}

    def test_to_dtcg_with_type(self):
        """Test to_dtcg with group type."""
        group = TokenGroup(type=TokenType.COLOR)
        group.add("primary", DesignToken.color("#ff0000"))
        dtcg = group.to_dtcg()

        assert dtcg["$type"] == "color"

    def test_to_dtcg_with_description(self):
        """Test to_dtcg with description."""
        group = TokenGroup(description="Color palette")
        dtcg = group.to_dtcg()

        assert dtcg["$description"] == "Color palette"

    def test_to_dtcg_with_tokens(self):
        """Test to_dtcg with tokens."""
        group = TokenGroup()
        group.add("sm", DesignToken.dimension(8, "px"))
        group.add("md", DesignToken.dimension(16, "px"))
        dtcg = group.to_dtcg()

        assert "sm" in dtcg
        assert "md" in dtcg
        assert dtcg["sm"]["$type"] == "dimension"

    def test_flatten_empty(self):
        """Test flatten for empty group."""
        group = TokenGroup()
        flat = group.flatten()
        assert flat == {}

    def test_flatten_single_level(self):
        """Test flatten for single-level group."""
        group = TokenGroup()
        group.add("primary", DesignToken.color("#ff0000"))
        group.add("secondary", DesignToken.color("#00ff00"))

        flat = group.flatten()

        assert "primary" in flat
        assert "secondary" in flat
        assert flat["primary"].type == TokenType.COLOR

    def test_flatten_nested(self):
        """Test flatten for nested groups."""
        root = TokenGroup()

        colors = TokenGroup()
        colors.add("primary", DesignToken.color("#3b82f6"))
        colors.add("secondary", DesignToken.color("#6366f1"))

        spacing = TokenGroup()
        spacing.add("sm", DesignToken.dimension(8, "px"))
        spacing.add("md", DesignToken.dimension(16, "px"))

        root.add("colors", colors)
        root.add("spacing", spacing)

        flat = root.flatten()

        assert "colors.primary" in flat
        assert "colors.secondary" in flat
        assert "spacing.sm" in flat
        assert "spacing.md" in flat

    def test_flatten_with_prefix(self):
        """Test flatten with custom prefix."""
        group = TokenGroup()
        group.add("primary", DesignToken.color("#ff0000"))

        flat = group.flatten(prefix="theme")

        assert "theme.primary" in flat


# =============================================================================
# Integration Tests
# =============================================================================

class TestTokenIntegration:
    """Integration tests for token module."""

    def test_complete_token_system(self):
        """Test creating a complete token system."""
        # Create root group
        tokens = TokenGroup(description="Design System Tokens")

        # Colors
        colors = TokenGroup(type=TokenType.COLOR, description="Color palette")
        colors.add("primary", DesignToken.color("#3b82f6", "Primary brand color"))
        colors.add("secondary", DesignToken.color("#6366f1", "Secondary color"))
        colors.add("success", DesignToken.color("#22c55e"))
        colors.add("error", DesignToken.color("#ef4444"))
        tokens.add("colors", colors)

        # Spacing
        spacing = TokenGroup(type=TokenType.DIMENSION, description="Spacing scale")
        spacing.add("xs", DesignToken.dimension(4, "px"))
        spacing.add("sm", DesignToken.dimension(8, "px"))
        spacing.add("md", DesignToken.dimension(16, "px"))
        spacing.add("lg", DesignToken.dimension(24, "px"))
        spacing.add("xl", DesignToken.dimension(32, "px"))
        tokens.add("spacing", spacing)

        # Typography
        typo = TokenGroup(type=TokenType.TYPOGRAPHY)
        typo.add("body", DesignToken.typography(TypographyValue(
            font_family=["Inter", "sans-serif"],
            font_size=DimensionValue(16, "px"),
        )))
        typo.add("heading", DesignToken.typography(TypographyValue(
            font_family=["Inter", "sans-serif"],
            font_size=DimensionValue(24, "px"),
            font_weight=700,
        )))
        tokens.add("typography", typo)

        # Shadows
        shadows = TokenGroup(type=TokenType.SHADOW)
        shadows.add("sm", DesignToken.shadow(ShadowValue(
            color=ColorValue(components=(0, 0, 0), alpha=0.1),
            offset_y=DimensionValue(2, "px"),
            blur=DimensionValue(4, "px"),
        )))
        tokens.add("shadows", shadows)

        # Export to DTCG
        dtcg = tokens.to_dtcg()

        assert "$description" in dtcg
        assert "colors" in dtcg
        assert "spacing" in dtcg
        assert "typography" in dtcg
        assert "shadows" in dtcg

        # Flatten for CSS custom properties
        flat = tokens.flatten()

        assert "colors.primary" in flat
        assert "spacing.md" in flat
        assert "typography.body" in flat

    def test_token_alias_chain(self):
        """Test token alias references."""
        tokens = TokenGroup()

        # Define base colors
        base = TokenGroup()
        base.add("blue-500", DesignToken.color("#3b82f6"))
        tokens.add("base", base)

        # Create semantic aliases
        semantic = TokenGroup()
        semantic.add("primary", DesignToken.reference("base.blue-500", TokenType.COLOR))
        tokens.add("semantic", semantic)

        # Get the alias
        alias = tokens.get("semantic.primary")
        assert alias.is_alias
        assert alias.to_css() == "var(--base-blue-500)"

    def test_json_serialization(self):
        """Test full JSON serialization roundtrip."""
        tokens = TokenGroup()
        tokens.add("primary", DesignToken.color("#3b82f6"))
        tokens.add("spacing", DesignToken.dimension(16, "px"))

        dtcg = tokens.to_dtcg()
        json_str = json.dumps(dtcg, indent=2)
        parsed = json.loads(json_str)

        assert "primary" in parsed
        assert parsed["primary"]["$type"] == "color"
        assert "spacing" in parsed
        assert parsed["spacing"]["$type"] == "dimension"
