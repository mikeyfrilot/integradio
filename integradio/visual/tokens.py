"""
Design Tokens - W3C DTCG-compliant token definitions.

Implements the Design Tokens Format Module (2025.10) specification.
Tokens are the atomic visual values that make up a design system.

Reference: https://www.designtokens.org/TR/drafts/format/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Union
from enum import Enum


# =============================================================================
# Token Types (W3C DTCG)
# =============================================================================

class TokenType(str, Enum):
    """Supported token types per W3C DTCG specification."""
    COLOR = "color"
    DIMENSION = "dimension"
    FONT_FAMILY = "fontFamily"
    FONT_WEIGHT = "fontWeight"
    FONT_STYLE = "fontStyle"
    DURATION = "duration"
    CUBIC_BEZIER = "cubicBezier"
    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"
    # Composite types
    STROKE_STYLE = "strokeStyle"
    BORDER = "border"
    TRANSITION = "transition"
    SHADOW = "shadow"
    GRADIENT = "gradient"
    TYPOGRAPHY = "typography"


# =============================================================================
# Value Types
# =============================================================================

ColorSpace = Literal["srgb", "srgb-linear", "display-p3", "a98-rgb", "prophoto-rgb", "rec2020", "lab", "oklab", "xyz", "xyz-d50", "xyz-d65", "hsl", "hwb", "lch", "oklch"]
DimensionUnit = Literal["px", "rem", "em", "%", "vw", "vh", "vmin", "vmax"]
DurationUnit = Literal["ms", "s"]
FontWeightKeyword = Literal["thin", "hairline", "extra-light", "ultra-light", "light", "normal", "regular", "medium", "semi-bold", "demi-bold", "bold", "extra-bold", "ultra-bold", "black", "heavy"]
FontStyleKeyword = Literal["normal", "italic", "oblique"]
StrokeStyleKeyword = Literal["solid", "dashed", "dotted", "double", "groove", "ridge", "outset", "inset"]
LineCap = Literal["round", "butt", "square"]


@dataclass
class ColorValue:
    """
    DTCG Color value.

    Example:
        {"colorSpace": "srgb", "components": [0.2, 0.4, 0.9], "alpha": 1}
    """
    color_space: ColorSpace = "srgb"
    components: tuple[float, float, float] = (0.0, 0.0, 0.0)
    alpha: float = 1.0

    @classmethod
    def from_hex(cls, hex_color: str) -> "ColorValue":
        """Create ColorValue from hex string (#RRGGBB or #RRGGBBAA)."""
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 6:
            r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
            alpha = 1.0
        elif len(hex_color) == 8:
            r, g, b, a = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16), int(hex_color[6:8], 16)
            alpha = a / 255
        else:
            raise ValueError(f"Invalid hex color: {hex_color}")
        return cls(
            color_space="srgb",
            components=(r / 255, g / 255, b / 255),
            alpha=alpha,
        )

    def to_hex(self) -> str:
        """Convert to hex string."""
        r, g, b = self.components
        if self.alpha < 1.0:
            return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}{int(self.alpha*255):02x}"
        return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    def to_css(self) -> str:
        """Convert to CSS color string."""
        r, g, b = self.components
        if self.color_space == "srgb":
            if self.alpha < 1.0:
                return f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {self.alpha})"
            return f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
        # For modern color spaces, use CSS Color Level 4 syntax
        return f"color({self.color_space} {r} {g} {b} / {self.alpha})"

    def to_dtcg(self) -> dict:
        """Export to DTCG JSON format."""
        return {
            "colorSpace": self.color_space,
            "components": list(self.components),
            "alpha": self.alpha,
        }


@dataclass
class DimensionValue:
    """
    DTCG Dimension value.

    Example:
        {"value": 16, "unit": "px"}
    """
    value: float
    unit: DimensionUnit = "px"

    def to_css(self) -> str:
        """Convert to CSS dimension string."""
        # Handle integer values cleanly
        if self.value == int(self.value):
            return f"{int(self.value)}{self.unit}"
        return f"{self.value}{self.unit}"

    def to_dtcg(self) -> dict:
        """Export to DTCG JSON format."""
        return {"value": self.value, "unit": self.unit}


@dataclass
class DurationValue:
    """
    DTCG Duration value.

    Example:
        {"value": 200, "unit": "ms"}
    """
    value: float
    unit: DurationUnit = "ms"

    def to_css(self) -> str:
        """Convert to CSS duration string."""
        if self.value == int(self.value):
            return f"{int(self.value)}{self.unit}"
        return f"{self.value}{self.unit}"

    def to_dtcg(self) -> dict:
        """Export to DTCG JSON format."""
        return {"value": self.value, "unit": self.unit}


@dataclass
class CubicBezierValue:
    """
    DTCG Cubic Bezier value for easing functions.

    Example:
        [0.42, 0, 0.58, 1]  # ease-in-out
    """
    p1x: float
    p1y: float
    p2x: float
    p2y: float

    # Common presets
    EASE = (0.25, 0.1, 0.25, 1.0)
    EASE_IN = (0.42, 0.0, 1.0, 1.0)
    EASE_OUT = (0.0, 0.0, 0.58, 1.0)
    EASE_IN_OUT = (0.42, 0.0, 0.58, 1.0)
    LINEAR = (0.0, 0.0, 1.0, 1.0)

    @classmethod
    def ease(cls) -> "CubicBezierValue":
        return cls(*cls.EASE)

    @classmethod
    def ease_in(cls) -> "CubicBezierValue":
        return cls(*cls.EASE_IN)

    @classmethod
    def ease_out(cls) -> "CubicBezierValue":
        return cls(*cls.EASE_OUT)

    @classmethod
    def ease_in_out(cls) -> "CubicBezierValue":
        return cls(*cls.EASE_IN_OUT)

    @classmethod
    def linear(cls) -> "CubicBezierValue":
        return cls(*cls.LINEAR)

    def to_css(self) -> str:
        """Convert to CSS cubic-bezier string."""
        return f"cubic-bezier({self.p1x}, {self.p1y}, {self.p2x}, {self.p2y})"

    def to_dtcg(self) -> list:
        """Export to DTCG JSON format."""
        return [self.p1x, self.p1y, self.p2x, self.p2y]


@dataclass
class StrokeStyleValue:
    """
    DTCG Stroke Style value.

    Can be a keyword or object with dashArray/lineCap.
    """
    style: StrokeStyleKeyword | None = "solid"
    dash_array: list[DimensionValue] | None = None
    line_cap: LineCap | None = None

    def to_css(self) -> str:
        """Convert to CSS border-style string."""
        if self.style:
            return self.style
        # Custom dash patterns aren't directly supported in CSS border-style
        return "dashed"

    def to_dtcg(self) -> str | dict:
        """Export to DTCG JSON format."""
        if self.style:
            return self.style
        return {
            "dashArray": [d.to_dtcg() for d in (self.dash_array or [])],
            "lineCap": self.line_cap,
        }


# =============================================================================
# Composite Token Values
# =============================================================================

@dataclass
class ShadowValue:
    """
    DTCG Shadow value.

    Example:
        {
            "color": {...},
            "offsetX": {"value": 0, "unit": "px"},
            "offsetY": {"value": 4, "unit": "px"},
            "blur": {"value": 8, "unit": "px"},
            "spread": {"value": 0, "unit": "px"},
            "inset": false
        }
    """
    color: ColorValue
    offset_x: DimensionValue = field(default_factory=lambda: DimensionValue(0, "px"))
    offset_y: DimensionValue = field(default_factory=lambda: DimensionValue(4, "px"))
    blur: DimensionValue = field(default_factory=lambda: DimensionValue(8, "px"))
    spread: DimensionValue = field(default_factory=lambda: DimensionValue(0, "px"))
    inset: bool = False

    def to_css(self) -> str:
        """Convert to CSS box-shadow string."""
        inset_str = "inset " if self.inset else ""
        return f"{inset_str}{self.offset_x.to_css()} {self.offset_y.to_css()} {self.blur.to_css()} {self.spread.to_css()} {self.color.to_css()}"

    def to_dtcg(self) -> dict:
        """Export to DTCG JSON format."""
        return {
            "color": self.color.to_dtcg(),
            "offsetX": self.offset_x.to_dtcg(),
            "offsetY": self.offset_y.to_dtcg(),
            "blur": self.blur.to_dtcg(),
            "spread": self.spread.to_dtcg(),
            "inset": self.inset,
        }


@dataclass
class BorderValue:
    """
    DTCG Border value.

    Example:
        {
            "color": {...},
            "width": {"value": 1, "unit": "px"},
            "style": "solid"
        }
    """
    color: ColorValue
    width: DimensionValue = field(default_factory=lambda: DimensionValue(1, "px"))
    style: StrokeStyleValue = field(default_factory=lambda: StrokeStyleValue("solid"))

    def to_css(self) -> str:
        """Convert to CSS border string."""
        return f"{self.width.to_css()} {self.style.to_css()} {self.color.to_css()}"

    def to_dtcg(self) -> dict:
        """Export to DTCG JSON format."""
        return {
            "color": self.color.to_dtcg(),
            "width": self.width.to_dtcg(),
            "style": self.style.to_dtcg(),
        }


@dataclass
class TransitionValue:
    """
    DTCG Transition value.

    Example:
        {
            "duration": {"value": 200, "unit": "ms"},
            "delay": {"value": 0, "unit": "ms"},
            "timingFunction": [0.42, 0, 0.58, 1]
        }
    """
    duration: DurationValue = field(default_factory=lambda: DurationValue(200, "ms"))
    delay: DurationValue = field(default_factory=lambda: DurationValue(0, "ms"))
    timing_function: CubicBezierValue = field(default_factory=CubicBezierValue.ease)

    def to_css(self, property: str = "all") -> str:
        """Convert to CSS transition string."""
        return f"{property} {self.duration.to_css()} {self.timing_function.to_css()} {self.delay.to_css()}"

    def to_dtcg(self) -> dict:
        """Export to DTCG JSON format."""
        return {
            "duration": self.duration.to_dtcg(),
            "delay": self.delay.to_dtcg(),
            "timingFunction": self.timing_function.to_dtcg(),
        }


@dataclass
class GradientStop:
    """A single stop in a gradient."""
    color: ColorValue
    position: float  # 0.0 to 1.0

    def to_dtcg(self) -> dict:
        """Export to DTCG JSON format."""
        return {
            "color": self.color.to_dtcg(),
            "position": self.position,
        }


@dataclass
class GradientValue:
    """
    DTCG Gradient value.

    Example:
        [
            {"color": {...}, "position": 0},
            {"color": {...}, "position": 1}
        ]
    """
    stops: list[GradientStop] = field(default_factory=list)

    def to_css(self, direction: str = "to bottom") -> str:
        """Convert to CSS linear-gradient string."""
        stops_css = ", ".join(
            f"{stop.color.to_css()} {stop.position * 100}%"
            for stop in self.stops
        )
        return f"linear-gradient({direction}, {stops_css})"

    def to_dtcg(self) -> list:
        """Export to DTCG JSON format."""
        return [stop.to_dtcg() for stop in self.stops]


@dataclass
class TypographyValue:
    """
    DTCG Typography composite value.

    Example:
        {
            "fontFamily": ["Inter", "sans-serif"],
            "fontSize": {"value": 16, "unit": "px"},
            "fontWeight": 400,
            "fontStyle": "normal",
            "letterSpacing": {"value": 0, "unit": "px"},
            "lineHeight": 1.5
        }
    """
    font_family: list[str] = field(default_factory=lambda: ["system-ui", "sans-serif"])
    font_size: DimensionValue = field(default_factory=lambda: DimensionValue(16, "px"))
    font_weight: int | FontWeightKeyword = 400
    font_style: FontStyleKeyword = "normal"
    letter_spacing: DimensionValue = field(default_factory=lambda: DimensionValue(0, "px"))
    line_height: float = 1.5

    def to_css(self) -> dict[str, str]:
        """Convert to CSS properties dict."""
        return {
            "font-family": ", ".join(f'"{f}"' if " " in f else f for f in self.font_family),
            "font-size": self.font_size.to_css(),
            "font-weight": str(self.font_weight),
            "font-style": self.font_style,
            "letter-spacing": self.letter_spacing.to_css(),
            "line-height": str(self.line_height),
        }

    def to_dtcg(self) -> dict:
        """Export to DTCG JSON format."""
        return {
            "fontFamily": self.font_family,
            "fontSize": self.font_size.to_dtcg(),
            "fontWeight": self.font_weight,
            "fontStyle": self.font_style,
            "letterSpacing": self.letter_spacing.to_dtcg(),
            "lineHeight": self.line_height,
        }


# =============================================================================
# Design Token
# =============================================================================

# All possible token value types
TokenValue = Union[
    ColorValue,
    DimensionValue,
    DurationValue,
    CubicBezierValue,
    StrokeStyleValue,
    ShadowValue,
    BorderValue,
    TransitionValue,
    GradientValue,
    TypographyValue,
    str,    # For fontFamily (single), fontWeight keyword, string tokens
    int,    # For number tokens, fontWeight numeric
    float,  # For number tokens
    bool,   # For boolean tokens
    list,   # For fontFamily (array), shadow (array)
]


@dataclass
class DesignToken:
    """
    A W3C DTCG-compliant design token.

    Example DTCG JSON:
        {
            "$type": "color",
            "$value": {"colorSpace": "srgb", "components": [0.2, 0.4, 0.9], "alpha": 1},
            "$description": "Primary brand color"
        }
    """
    value: TokenValue
    type: TokenType
    description: str = ""
    extensions: dict[str, Any] = field(default_factory=dict)

    # Reference to another token (alias)
    _reference: str | None = None

    @classmethod
    def color(cls, value: ColorValue | str, description: str = "") -> "DesignToken":
        """Create a color token."""
        if isinstance(value, str):
            value = ColorValue.from_hex(value)
        return cls(value=value, type=TokenType.COLOR, description=description)

    @classmethod
    def dimension(cls, value: float, unit: DimensionUnit = "px", description: str = "") -> "DesignToken":
        """Create a dimension token."""
        return cls(value=DimensionValue(value, unit), type=TokenType.DIMENSION, description=description)

    @classmethod
    def duration(cls, value: float, unit: DurationUnit = "ms", description: str = "") -> "DesignToken":
        """Create a duration token."""
        return cls(value=DurationValue(value, unit), type=TokenType.DURATION, description=description)

    @classmethod
    def font_family(cls, value: str | list[str], description: str = "") -> "DesignToken":
        """Create a font family token."""
        if isinstance(value, str):
            value = [value]
        return cls(value=value, type=TokenType.FONT_FAMILY, description=description)

    @classmethod
    def font_weight(cls, value: int | FontWeightKeyword, description: str = "") -> "DesignToken":
        """Create a font weight token."""
        return cls(value=value, type=TokenType.FONT_WEIGHT, description=description)

    @classmethod
    def number(cls, value: float | int, description: str = "") -> "DesignToken":
        """Create a number token."""
        return cls(value=value, type=TokenType.NUMBER, description=description)

    @classmethod
    def shadow(cls, value: ShadowValue | list[ShadowValue], description: str = "") -> "DesignToken":
        """Create a shadow token."""
        return cls(value=value, type=TokenType.SHADOW, description=description)

    @classmethod
    def border(cls, value: BorderValue, description: str = "") -> "DesignToken":
        """Create a border token."""
        return cls(value=value, type=TokenType.BORDER, description=description)

    @classmethod
    def transition(cls, value: TransitionValue, description: str = "") -> "DesignToken":
        """Create a transition token."""
        return cls(value=value, type=TokenType.TRANSITION, description=description)

    @classmethod
    def gradient(cls, value: GradientValue, description: str = "") -> "DesignToken":
        """Create a gradient token."""
        return cls(value=value, type=TokenType.GRADIENT, description=description)

    @classmethod
    def typography(cls, value: TypographyValue, description: str = "") -> "DesignToken":
        """Create a typography token."""
        return cls(value=value, type=TokenType.TYPOGRAPHY, description=description)

    @classmethod
    def reference(cls, path: str, type: TokenType, description: str = "") -> "DesignToken":
        """Create a token that references another token (alias)."""
        token = cls(value=path, type=type, description=description)
        token._reference = path
        return token

    @property
    def is_alias(self) -> bool:
        """Check if this token is an alias to another token."""
        return self._reference is not None

    def to_dtcg(self) -> dict:
        """Export to DTCG JSON format."""
        result: dict[str, Any] = {
            "$type": self.type.value,
        }

        # Handle aliases
        if self.is_alias:
            result["$value"] = f"{{{self._reference}}}"
        elif hasattr(self.value, "to_dtcg"):
            result["$value"] = self.value.to_dtcg()
        else:
            result["$value"] = self.value

        if self.description:
            result["$description"] = self.description

        if self.extensions:
            result["$extensions"] = self.extensions

        return result

    def to_css(self) -> str:
        """Convert to CSS value string."""
        if self.is_alias:
            # Convert path to CSS variable reference
            var_name = self._reference.replace(".", "-").replace("/", "-")
            return f"var(--{var_name})"

        if hasattr(self.value, "to_css"):
            result = self.value.to_css()
            # Handle dict returns (typography)
            if isinstance(result, dict):
                return "; ".join(f"{k}: {v}" for k, v in result.items())
            return result

        return str(self.value)


# =============================================================================
# Token Group (hierarchical organization)
# =============================================================================

@dataclass
class TokenGroup:
    """
    A group of tokens with optional shared type.

    Example DTCG JSON:
        {
            "colors": {
                "$type": "color",
                "primary": {"$value": {...}},
                "secondary": {"$value": {...}}
            }
        }
    """
    tokens: dict[str, "DesignToken | TokenGroup"] = field(default_factory=dict)
    type: TokenType | None = None
    description: str = ""

    def add(self, name: str, token: "DesignToken | TokenGroup") -> None:
        """Add a token or subgroup."""
        self.tokens[name] = token

    def get(self, path: str) -> "DesignToken | TokenGroup | None":
        """Get a token by dot-separated path."""
        parts = path.split(".")
        current: DesignToken | TokenGroup | None = self

        for part in parts:
            if isinstance(current, TokenGroup):
                current = current.tokens.get(part)
            else:
                return None

        return current

    def to_dtcg(self) -> dict:
        """Export to DTCG JSON format."""
        result: dict[str, Any] = {}

        if self.type:
            result["$type"] = self.type.value

        if self.description:
            result["$description"] = self.description

        for name, item in self.tokens.items():
            result[name] = item.to_dtcg()

        return result

    def flatten(self, prefix: str = "") -> dict[str, DesignToken]:
        """Flatten the group into a dict of path -> token."""
        result: dict[str, DesignToken] = {}

        for name, item in self.tokens.items():
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(item, DesignToken):
                result[path] = item
            else:
                result.update(item.flatten(path))

        return result
