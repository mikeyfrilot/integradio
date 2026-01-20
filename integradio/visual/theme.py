"""
Theme Generator - Auto-generate theme variants from a base palette.

Features:
- Dark mode generation from light palette
- Color shade/tint generation (50-950 scale)
- Semantic color mapping (primary -> variants)
- Accessibility-aware contrast adjustments
- CSS custom property output with prefers-color-scheme
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, field
from typing import Literal

from .tokens import (
    DesignToken,
    TokenType,
    TokenGroup,
    ColorValue,
)
from .spec import UISpec


# =============================================================================
# Color Utilities
# =============================================================================

def hex_to_hsl(hex_color: str) -> tuple[float, float, float]:
    """Convert hex to HSL (h: 0-360, s: 0-100, l: 0-100)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return h * 360, s * 100, l * 100


def hsl_to_hex(h: float, s: float, l: float) -> str:
    """Convert HSL to hex (h: 0-360, s: 0-100, l: 0-100)."""
    h, s, l = h / 360, s / 100, l / 100
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def adjust_lightness(hex_color: str, amount: float) -> str:
    """
    Adjust color lightness.

    Args:
        hex_color: Color to adjust
        amount: -100 to +100 (negative = darker, positive = lighter)
    """
    h, s, l = hex_to_hsl(hex_color)
    new_l = max(0, min(100, l + amount))
    return hsl_to_hex(h, s, new_l)


def adjust_saturation(hex_color: str, amount: float) -> str:
    """Adjust color saturation (-100 to +100)."""
    h, s, l = hex_to_hsl(hex_color)
    new_s = max(0, min(100, s + amount))
    return hsl_to_hex(h, new_s, l)


def mix_colors(color1: str, color2: str, ratio: float = 0.5) -> str:
    """
    Mix two colors.

    Args:
        color1: First color
        color2: Second color
        ratio: 0.0 = all color1, 1.0 = all color2
    """
    c1 = color1.lstrip("#")
    c2 = color2.lstrip("#")

    r1, g1, b1 = (int(c1[i:i+2], 16) for i in (0, 2, 4))
    r2, g2, b2 = (int(c2[i:i+2], 16) for i in (0, 2, 4))

    r = int(r1 * (1 - ratio) + r2 * ratio)
    g = int(g1 * (1 - ratio) + g2 * ratio)
    b = int(b1 * (1 - ratio) + b2 * ratio)

    return f"#{r:02x}{g:02x}{b:02x}"


def get_luminance(hex_color: str) -> float:
    """Get relative luminance (0-1) per WCAG 2.1."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i+2], 16) / 255 for i in (0, 2, 4))

    def adjust(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * adjust(r) + 0.7152 * adjust(g) + 0.0722 * adjust(b)


def is_light(hex_color: str) -> bool:
    """Check if a color is considered light (for contrast calculations)."""
    return get_luminance(hex_color) > 0.5


def get_contrast_color(hex_color: str) -> str:
    """Get black or white depending on which has better contrast."""
    return "#000000" if is_light(hex_color) else "#ffffff"


# =============================================================================
# Shade Generation
# =============================================================================

# Tailwind-style lightness values for shade scale
SHADE_LIGHTNESS = {
    50: 97,
    100: 94,
    200: 86,
    300: 77,
    400: 66,
    500: 55,
    600: 44,
    700: 35,
    800: 26,
    900: 18,
    950: 10,
}


def generate_shade_scale(base_color: str) -> dict[int, str]:
    """
    Generate Tailwind-style color scale from base color.

    Args:
        base_color: The base color (will be mapped to closest shade)

    Returns:
        Dict of shade level (50-950) to hex color
    """
    h, s, l = hex_to_hsl(base_color)

    # Generate each shade
    shades = {}
    for level, target_l in SHADE_LIGHTNESS.items():
        shades[level] = hsl_to_hex(h, s, target_l)

    return shades


def generate_shade_tokens(name: str, base_color: str) -> TokenGroup:
    """
    Generate a TokenGroup with full shade scale.

    Args:
        name: Base name (e.g., "blue")
        base_color: Base hex color

    Returns:
        TokenGroup with tokens for each shade
    """
    shades = generate_shade_scale(base_color)
    group = TokenGroup(type=TokenType.COLOR)

    for level, color in shades.items():
        group.add(str(level), DesignToken.color(color, f"{name.title()} {level}"))

    return group


# =============================================================================
# Theme Generation
# =============================================================================

@dataclass
class ThemeColors:
    """Core theme colors."""
    primary: str
    secondary: str
    background: str
    surface: str
    text: str
    text_muted: str
    border: str
    success: str = "#22c55e"
    warning: str = "#f59e0b"
    error: str = "#ef4444"
    info: str = "#3b82f6"


@dataclass
class ThemeConfig:
    """Configuration for theme generation."""
    # Base colors
    primary: str = "#3b82f6"
    secondary: str = "#64748b"
    success: str = "#22c55e"
    warning: str = "#f59e0b"
    error: str = "#ef4444"

    # Light theme specifics
    light_background: str = "#ffffff"
    light_surface: str = "#f8fafc"
    light_text: str = "#0f172a"
    light_text_muted: str = "#64748b"
    light_border: str = "#e2e8f0"

    # Dark theme adjustments (relative to light)
    dark_background: str = "#0f172a"
    dark_surface: str = "#1e293b"
    dark_text: str = "#f8fafc"
    dark_text_muted: str = "#94a3b8"
    dark_border: str = "#334155"


class ThemeGenerator:
    """
    Generate complete theme systems from base colors.

    Usage:
        gen = ThemeGenerator(ThemeConfig(primary="#3b82f6"))
        light, dark = gen.generate_themes()
    """

    def __init__(self, config: ThemeConfig):
        self.config = config

    def generate_light_theme(self) -> TokenGroup:
        """Generate light theme tokens."""
        theme = TokenGroup()

        # Core colors
        colors = TokenGroup(type=TokenType.COLOR)
        colors.add("primary", DesignToken.color(self.config.primary, "Primary brand color"))
        colors.add("secondary", DesignToken.color(self.config.secondary, "Secondary color"))
        colors.add("success", DesignToken.color(self.config.success, "Success state"))
        colors.add("warning", DesignToken.color(self.config.warning, "Warning state"))
        colors.add("error", DesignToken.color(self.config.error, "Error state"))

        # Background/surface
        colors.add("background", DesignToken.color(self.config.light_background, "Page background"))
        colors.add("surface", DesignToken.color(self.config.light_surface, "Card/surface background"))

        # Text
        colors.add("text", DesignToken.color(self.config.light_text, "Primary text"))
        colors.add("text-muted", DesignToken.color(self.config.light_text_muted, "Muted text"))

        # Border
        colors.add("border", DesignToken.color(self.config.light_border, "Border color"))

        # Generate shade scales for primary/secondary
        colors.add("primary-shades", generate_shade_tokens("primary", self.config.primary))
        colors.add("secondary-shades", generate_shade_tokens("secondary", self.config.secondary))

        theme.add("colors", colors)

        # Add contrast colors (auto-generated)
        contrasts = TokenGroup(type=TokenType.COLOR)
        contrasts.add("on-primary", DesignToken.color(
            get_contrast_color(self.config.primary),
            "Text on primary color"
        ))
        contrasts.add("on-secondary", DesignToken.color(
            get_contrast_color(self.config.secondary),
            "Text on secondary color"
        ))
        theme.add("contrast", contrasts)

        return theme

    def generate_dark_theme(self) -> TokenGroup:
        """Generate dark theme tokens."""
        theme = TokenGroup()

        colors = TokenGroup(type=TokenType.COLOR)

        # Core colors (slightly adjusted for dark mode)
        primary_dark = adjust_lightness(self.config.primary, 5)
        secondary_dark = adjust_lightness(self.config.secondary, 5)

        colors.add("primary", DesignToken.color(primary_dark, "Primary brand color"))
        colors.add("secondary", DesignToken.color(secondary_dark, "Secondary color"))
        colors.add("success", DesignToken.color(adjust_lightness(self.config.success, 5), "Success state"))
        colors.add("warning", DesignToken.color(adjust_lightness(self.config.warning, 5), "Warning state"))
        colors.add("error", DesignToken.color(adjust_lightness(self.config.error, 5), "Error state"))

        # Dark backgrounds
        colors.add("background", DesignToken.color(self.config.dark_background, "Page background"))
        colors.add("surface", DesignToken.color(self.config.dark_surface, "Card/surface background"))

        # Light text for dark mode
        colors.add("text", DesignToken.color(self.config.dark_text, "Primary text"))
        colors.add("text-muted", DesignToken.color(self.config.dark_text_muted, "Muted text"))

        # Border
        colors.add("border", DesignToken.color(self.config.dark_border, "Border color"))

        # Shade scales (inverted for dark mode)
        colors.add("primary-shades", generate_shade_tokens("primary", primary_dark))
        colors.add("secondary-shades", generate_shade_tokens("secondary", secondary_dark))

        theme.add("colors", colors)

        # Contrast colors
        contrasts = TokenGroup(type=TokenType.COLOR)
        contrasts.add("on-primary", DesignToken.color(
            get_contrast_color(primary_dark),
            "Text on primary color"
        ))
        contrasts.add("on-secondary", DesignToken.color(
            get_contrast_color(secondary_dark),
            "Text on secondary color"
        ))
        theme.add("contrast", contrasts)

        return theme

    def generate_themes(self) -> tuple[TokenGroup, TokenGroup]:
        """Generate both light and dark themes."""
        return self.generate_light_theme(), self.generate_dark_theme()

    def apply_to_spec(self, spec: UISpec) -> None:
        """
        Apply generated themes to a UISpec.

        Creates "light" and "dark" theme variants.
        """
        light, dark = self.generate_themes()

        # Set light theme as base tokens
        for name, item in light.tokens.items():
            spec.tokens.add(name, item)

        # Add dark as theme variant
        spec.add_theme("dark", dark)


# =============================================================================
# CSS Output
# =============================================================================

def generate_theme_css(
    spec: UISpec,
    include_system_preference: bool = True,
) -> str:
    """
    Generate CSS with theme support.

    Includes:
    - :root for light theme (default)
    - [data-theme="dark"] for dark theme
    - @media (prefers-color-scheme: dark) for system preference

    Args:
        spec: UISpec with themes
        include_system_preference: Include prefers-color-scheme media query

    Returns:
        Complete CSS string
    """
    lines = []

    # Light theme (default)
    light_vars = _tokens_to_css_vars(spec.tokens)
    if light_vars:
        lines.append(f":root {{\n{light_vars}\n}}")

    # Dark theme (explicit)
    if "dark" in spec.themes:
        dark_vars = _tokens_to_css_vars(spec.themes["dark"])
        if dark_vars:
            lines.append(f'[data-theme="dark"] {{\n{dark_vars}\n}}')

            # System preference
            if include_system_preference:
                lines.append(f"""
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
{dark_vars}
  }}
}}""".strip())

    return "\n\n".join(lines)


def _tokens_to_css_vars(group: TokenGroup, prefix: str = "") -> str:
    """Convert token group to CSS custom properties."""
    flat = group.flatten()
    lines = []

    for path, token in flat.items():
        var_name = f"--{path.replace('.', '-')}"
        value = token.to_css()
        lines.append(f"  {var_name}: {value};")

    return "\n".join(lines)


def generate_theme_toggle_script() -> str:
    """Generate JavaScript for theme toggling."""
    return '''
<script>
(function() {
  // Check for saved preference or system preference
  const saved = localStorage.getItem("theme");
  const systemDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const theme = saved || (systemDark ? "dark" : "light");

  document.documentElement.setAttribute("data-theme", theme);

  // Listen for system preference changes
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
    if (!localStorage.getItem("theme")) {
      document.documentElement.setAttribute("data-theme", e.matches ? "dark" : "light");
    }
  });

  // Toggle function
  window.toggleTheme = function() {
    const current = document.documentElement.getAttribute("data-theme");
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("theme", next);
  };
})();
</script>
'''.strip()


# =============================================================================
# Palette Presets
# =============================================================================

@dataclass
class PalettePreset:
    """A predefined color palette."""
    name: str
    primary: str
    secondary: str
    accent: str | None = None
    description: str = ""


# Popular palettes
PALETTES = {
    "blue": PalettePreset(
        name="Blue",
        primary="#3b82f6",
        secondary="#64748b",
        accent="#0ea5e9",
        description="Classic professional blue",
    ),
    "purple": PalettePreset(
        name="Purple",
        primary="#8b5cf6",
        secondary="#6b7280",
        accent="#a855f7",
        description="Modern creative purple",
    ),
    "green": PalettePreset(
        name="Green",
        primary="#22c55e",
        secondary="#64748b",
        accent="#10b981",
        description="Fresh natural green",
    ),
    "orange": PalettePreset(
        name="Orange",
        primary="#f97316",
        secondary="#64748b",
        accent="#fb923c",
        description="Energetic warm orange",
    ),
    "rose": PalettePreset(
        name="Rose",
        primary="#f43f5e",
        secondary="#64748b",
        accent="#fb7185",
        description="Bold modern rose",
    ),
    "teal": PalettePreset(
        name="Teal",
        primary="#14b8a6",
        secondary="#64748b",
        accent="#2dd4bf",
        description="Calm professional teal",
    ),
    "slate": PalettePreset(
        name="Slate",
        primary="#475569",
        secondary="#64748b",
        accent="#94a3b8",
        description="Neutral corporate slate",
    ),
}


def get_palette(name: str) -> PalettePreset | None:
    """Get a palette preset by name."""
    return PALETTES.get(name.lower())


def list_palettes() -> list[PalettePreset]:
    """List all available palette presets."""
    return list(PALETTES.values())


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_theme_from_primary(
    primary: str,
    spec: UISpec | None = None,
) -> UISpec:
    """
    Generate a complete themed UISpec from just a primary color.

    Args:
        primary: Primary brand color (hex)
        spec: Existing spec to add themes to (creates new if None)

    Returns:
        UISpec with light/dark themes
    """
    if spec is None:
        spec = UISpec(name="Generated Theme")

    # Derive secondary from primary (desaturated, shifted)
    h, s, l = hex_to_hsl(primary)
    secondary = hsl_to_hex(h, max(0, s - 30), l)

    config = ThemeConfig(primary=primary, secondary=secondary)
    generator = ThemeGenerator(config)
    generator.apply_to_spec(spec)

    return spec


def generate_theme_from_palette(
    palette: str | PalettePreset,
    spec: UISpec | None = None,
) -> UISpec:
    """
    Generate a themed UISpec from a palette preset.

    Args:
        palette: Palette name or PalettePreset object
        spec: Existing spec to add themes to

    Returns:
        UISpec with themes
    """
    if isinstance(palette, str):
        preset = get_palette(palette)
        if preset is None:
            raise ValueError(f"Unknown palette: {palette}. Available: {list(PALETTES.keys())}")
    else:
        preset = palette

    if spec is None:
        spec = UISpec(name=f"{preset.name} Theme")

    config = ThemeConfig(primary=preset.primary, secondary=preset.secondary)
    generator = ThemeGenerator(config)
    generator.apply_to_spec(spec)

    return spec


def quick_dark_mode(light_color: str) -> str:
    """
    Quick conversion of a single color for dark mode.

    - Light colors get darker
    - Dark colors get lighter
    - Maintains hue and saturation
    """
    h, s, l = hex_to_hsl(light_color)

    # Invert lightness around 50%
    dark_l = 100 - l

    # Clamp to reasonable range
    dark_l = max(15, min(85, dark_l))

    return hsl_to_hex(h, s, dark_l)
