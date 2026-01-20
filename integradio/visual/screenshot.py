"""
Screenshot to Spec - Extract visual specifications from UI screenshots.

Analyzes images to extract:
- Color palettes (dominant, accent, background, text colors)
- Layout structure (regions, hierarchy, spacing)
- Typography estimates (sizes, weights from visual analysis)
- Component detection (buttons, inputs, cards, etc.)

This module provides a foundation for screenshot analysis that can be
extended with AI vision models for more accurate component detection.

Usage:
    from integradio.visual.screenshot import (
        analyze_screenshot,
        extract_colors,
        detect_regions,
        ScreenshotAnalyzer,
    )

    # Quick analysis
    spec = analyze_screenshot("path/to/screenshot.png")

    # Detailed extraction
    analyzer = ScreenshotAnalyzer()
    result = analyzer.analyze("screenshot.png")
    print(result.colors)  # Extracted color palette
    print(result.regions)  # Detected UI regions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Callable
from enum import Enum
from pathlib import Path
import json
import colorsys
import math

from .tokens import (
    ColorValue,
    DimensionValue,
    TypographyValue,
    TokenGroup,
    TokenType,
    DesignToken,
)
from .spec import VisualSpec, SpacingSpec, LayoutSpec, Display


# =============================================================================
# Types and Enums
# =============================================================================

class RegionType(str, Enum):
    """Types of UI regions that can be detected."""
    HEADER = "header"
    FOOTER = "footer"
    SIDEBAR = "sidebar"
    CONTENT = "content"
    CARD = "card"
    BUTTON = "button"
    INPUT = "input"
    IMAGE = "image"
    TEXT = "text"
    ICON = "icon"
    NAVIGATION = "navigation"
    MODAL = "modal"
    LIST = "list"
    TABLE = "table"
    FORM = "form"
    UNKNOWN = "unknown"


class ColorRole(str, Enum):
    """Semantic roles for extracted colors."""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    ACCENT = "accent"
    BACKGROUND = "background"
    SURFACE = "surface"
    TEXT_PRIMARY = "text-primary"
    TEXT_SECONDARY = "text-secondary"
    BORDER = "border"
    ERROR = "error"
    WARNING = "warning"
    SUCCESS = "success"
    INFO = "info"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BoundingBox:
    """A rectangular region in the image."""
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        """Right edge x coordinate."""
        return self.x + self.width

    @property
    def y2(self) -> int:
        """Bottom edge y coordinate."""
        return self.y + self.height

    @property
    def center(self) -> tuple[int, int]:
        """Center point of the box."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def area(self) -> int:
        """Area of the box in pixels."""
        return self.width * self.height

    def contains(self, other: "BoundingBox") -> bool:
        """Check if this box fully contains another box."""
        return (
            self.x <= other.x and
            self.y <= other.y and
            self.x2 >= other.x2 and
            self.y2 >= other.y2
        )

    def overlaps(self, other: "BoundingBox") -> bool:
        """Check if this box overlaps with another box."""
        return not (
            self.x2 < other.x or
            other.x2 < self.x or
            self.y2 < other.y or
            other.y2 < self.y
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class ExtractedColor:
    """A color extracted from the image with metadata."""
    color: ColorValue
    frequency: float  # 0-1, how much of image this color covers
    role: ColorRole | None = None
    source_regions: list[str] = field(default_factory=list)

    @property
    def hex(self) -> str:
        """Get hex representation."""
        return self.color.to_hex()

    @property
    def luminance(self) -> float:
        """Calculate relative luminance for contrast calculations."""
        r, g, b = self.color.components
        # sRGB to linear
        def linearize(c):
            return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

        r_lin = linearize(r)
        g_lin = linearize(g)
        b_lin = linearize(b)
        return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "hex": self.hex,
            "frequency": self.frequency,
            "role": self.role.value if self.role else None,
            "luminance": self.luminance,
        }


@dataclass
class DetectedRegion:
    """A detected UI region in the screenshot."""
    region_type: RegionType
    bounds: BoundingBox
    confidence: float  # 0-1
    colors: list[ExtractedColor] = field(default_factory=list)
    children: list["DetectedRegion"] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def dominant_color(self) -> ExtractedColor | None:
        """Get the most frequent color in this region."""
        if not self.colors:
            return None
        return max(self.colors, key=lambda c: c.frequency)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "type": self.region_type.value,
            "bounds": self.bounds.to_dict(),
            "confidence": self.confidence,
            "colors": [c.to_dict() for c in self.colors],
            "children": [c.to_dict() for c in self.children],
            "attributes": self.attributes,
        }


@dataclass
class TypographyEstimate:
    """Estimated typography from visual analysis."""
    size_px: int
    weight: Literal["normal", "medium", "bold"] = "normal"
    style: Literal["normal", "italic"] = "normal"
    line_height: float = 1.5
    region: BoundingBox | None = None

    def to_typography_value(self) -> TypographyValue:
        """Convert to TypographyValue token."""
        weight_map = {"normal": 400, "medium": 500, "bold": 700}
        return TypographyValue(
            font_family=["system-ui", "sans-serif"],
            font_size=DimensionValue(self.size_px, "px"),
            font_weight=weight_map.get(self.weight, 400),
            font_style=self.style,
            line_height=self.line_height,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "size_px": self.size_px,
            "weight": self.weight,
            "style": self.style,
            "line_height": self.line_height,
        }


@dataclass
class SpacingEstimate:
    """Estimated spacing values from visual analysis."""
    base: int = 8  # Base spacing unit
    xs: int = 4
    sm: int = 8
    md: int = 16
    lg: int = 24
    xl: int = 32

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "base": self.base,
            "xs": self.xs,
            "sm": self.sm,
            "md": self.md,
            "lg": self.lg,
            "xl": self.xl,
        }


@dataclass
class AnalysisResult:
    """Complete result of screenshot analysis."""
    image_path: str
    width: int
    height: int
    colors: list[ExtractedColor] = field(default_factory=list)
    regions: list[DetectedRegion] = field(default_factory=list)
    typography: list[TypographyEstimate] = field(default_factory=list)
    spacing: SpacingEstimate | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def color_palette(self) -> dict[str, str]:
        """Get named color palette from extracted colors."""
        palette = {}
        for color in self.colors:
            if color.role:
                palette[color.role.value] = color.hex
        return palette

    @property
    def dominant_colors(self) -> list[str]:
        """Get list of dominant colors by frequency."""
        sorted_colors = sorted(self.colors, key=lambda c: c.frequency, reverse=True)
        return [c.hex for c in sorted_colors[:5]]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "image_path": self.image_path,
            "dimensions": {"width": self.width, "height": self.height},
            "colors": [c.to_dict() for c in self.colors],
            "regions": [r.to_dict() for r in self.regions],
            "typography": [t.to_dict() for t in self.typography],
            "spacing": self.spacing.to_dict() if self.spacing else None,
            "color_palette": self.color_palette,
            "dominant_colors": self.dominant_colors,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Export to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> None:
        """Save analysis result to JSON file."""
        path = Path(path)
        path.write_text(self.to_json())


# =============================================================================
# Color Analysis
# =============================================================================

def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB to hex string."""
    return f"#{r:02x}{g:02x}{b:02x}"


def color_distance(c1: tuple[int, int, int], c2: tuple[int, int, int]) -> float:
    """Calculate Euclidean distance between two RGB colors."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))


def rgb_to_hsl(r: int, g: int, b: int) -> tuple[float, float, float]:
    """Convert RGB (0-255) to HSL (h: 0-360, s: 0-1, l: 0-1)."""
    r_norm, g_norm, b_norm = r / 255, g / 255, b / 255
    h, l, s = colorsys.rgb_to_hls(r_norm, g_norm, b_norm)
    return (h * 360, s, l)


def is_grayscale(r: int, g: int, b: int, threshold: int = 15) -> bool:
    """Check if a color is grayscale."""
    return max(abs(r - g), abs(g - b), abs(r - b)) <= threshold


def classify_color_role(
    color: ExtractedColor,
    all_colors: list[ExtractedColor],
    image_width: int,
    image_height: int,
) -> ColorRole:
    """
    Classify a color's semantic role based on its properties and context.

    Heuristics:
    - High frequency + low saturation = background/surface
    - High frequency + high saturation = primary
    - Low frequency + high saturation = accent
    - Very dark = text-primary
    - Medium dark = text-secondary
    """
    r, g, b = color.color.components
    r_int, g_int, b_int = int(r * 255), int(g * 255), int(b * 255)
    h, s, l = rgb_to_hsl(r_int, g_int, b_int)

    # Grayscale colors
    if is_grayscale(r_int, g_int, b_int):
        if l < 0.2:
            return ColorRole.TEXT_PRIMARY
        elif l < 0.5:
            return ColorRole.TEXT_SECONDARY
        elif l > 0.9:
            return ColorRole.BACKGROUND
        else:
            return ColorRole.SURFACE

    # Chromatic colors
    if color.frequency > 0.3:
        if s < 0.3:
            return ColorRole.BACKGROUND
        else:
            return ColorRole.PRIMARY
    elif color.frequency > 0.1:
        return ColorRole.SECONDARY
    else:
        # Check for semantic colors
        if 0 <= h < 30 or h > 330:  # Red-ish
            return ColorRole.ERROR
        elif 30 <= h < 60:  # Orange-ish
            return ColorRole.WARNING
        elif 90 <= h < 150:  # Green-ish
            return ColorRole.SUCCESS
        elif 180 <= h < 240:  # Blue-ish
            return ColorRole.INFO
        else:
            return ColorRole.ACCENT


def quantize_colors(
    pixels: list[tuple[int, int, int]],
    num_colors: int = 8,
) -> list[tuple[tuple[int, int, int], int]]:
    """
    Reduce colors to a representative palette using median cut algorithm.

    Returns list of (color, count) tuples.
    """
    if not pixels:
        return []

    # Simple k-means-like clustering
    from collections import Counter

    # Count exact colors first
    color_counts = Counter(pixels)

    # If few unique colors, return them directly
    if len(color_counts) <= num_colors:
        return list(color_counts.most_common())

    # Group similar colors
    clusters: list[list[tuple[int, int, int]]] = []
    centroids: list[tuple[int, int, int]] = []

    # Initialize with most frequent colors
    initial_colors = [c for c, _ in color_counts.most_common(num_colors)]
    centroids = initial_colors.copy()

    # Assign pixels to nearest centroid
    for _ in range(5):  # Few iterations
        clusters = [[] for _ in range(num_colors)]

        for pixel in color_counts.keys():
            min_dist = float("inf")
            min_idx = 0
            for i, centroid in enumerate(centroids):
                dist = color_distance(pixel, centroid)
                if dist < min_dist:
                    min_dist = dist
                    min_idx = i
            clusters[min_idx].append(pixel)

        # Update centroids
        for i, cluster in enumerate(clusters):
            if cluster:
                r = sum(c[0] for c in cluster) // len(cluster)
                g = sum(c[1] for c in cluster) // len(cluster)
                b = sum(c[2] for c in cluster) // len(cluster)
                centroids[i] = (r, g, b)

    # Calculate cluster sizes
    result = []
    for i, cluster in enumerate(clusters):
        if cluster:
            count = sum(color_counts[c] for c in cluster)
            result.append((centroids[i], count))

    return sorted(result, key=lambda x: x[1], reverse=True)


def extract_colors_from_pixels(
    pixels: list[tuple[int, int, int]],
    num_colors: int = 8,
) -> list[ExtractedColor]:
    """
    Extract a color palette from pixel data.

    Args:
        pixels: List of RGB tuples
        num_colors: Number of colors to extract

    Returns:
        List of ExtractedColor objects
    """
    if not pixels:
        return []

    total_pixels = len(pixels)
    quantized = quantize_colors(pixels, num_colors)

    colors = []
    for rgb, count in quantized:
        color_value = ColorValue(
            color_space="srgb",
            components=(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255),
            alpha=1.0,
        )
        colors.append(ExtractedColor(
            color=color_value,
            frequency=count / total_pixels,
        ))

    return colors


# =============================================================================
# Region Detection
# =============================================================================

def detect_horizontal_regions(
    height: int,
    row_brightness: list[float],
    threshold: float = 0.1,
) -> list[tuple[int, int]]:
    """
    Detect horizontal regions based on brightness changes.

    Returns list of (start_y, end_y) tuples.
    """
    if not row_brightness:
        return [(0, height)]

    regions = []
    region_start = 0
    prev_brightness = row_brightness[0]

    for y, brightness in enumerate(row_brightness[1:], 1):
        if abs(brightness - prev_brightness) > threshold:
            if y - region_start > 10:  # Minimum region height
                regions.append((region_start, y))
            region_start = y
        prev_brightness = brightness

    # Add final region
    if height - region_start > 10:
        regions.append((region_start, height))

    return regions


def classify_region_type(
    bounds: BoundingBox,
    image_width: int,
    image_height: int,
    colors: list[ExtractedColor],
) -> RegionType:
    """
    Classify a region's type based on position and size.

    Heuristics:
    - Top of image, full width = header
    - Bottom of image, full width = footer
    - Left/right edge, tall = sidebar
    - Small, centered = button
    - Wide, short = input
    """
    # Handle zero dimensions
    if image_width == 0 or image_height == 0:
        return RegionType.UNKNOWN

    # Position ratios
    x_ratio = bounds.x / image_width
    y_ratio = bounds.y / image_height
    width_ratio = bounds.width / image_width
    height_ratio = bounds.height / image_height

    # Full-width top region = header
    if y_ratio < 0.15 and width_ratio > 0.8:
        return RegionType.HEADER

    # Full-width bottom region = footer
    if y_ratio > 0.85 and width_ratio > 0.8:
        return RegionType.FOOTER

    # Tall narrow side region = sidebar
    if height_ratio > 0.5 and width_ratio < 0.3:
        if x_ratio < 0.1:
            return RegionType.SIDEBAR
        elif x_ratio > 0.7:
            return RegionType.SIDEBAR

    # Small region = could be button or icon
    if width_ratio < 0.2 and height_ratio < 0.1:
        aspect = bounds.width / bounds.height if bounds.height > 0 else 1
        if 0.8 < aspect < 5:
            return RegionType.BUTTON
        elif aspect < 1.2:
            return RegionType.ICON

    # Wide short region = input
    if width_ratio > 0.3 and height_ratio < 0.08:
        return RegionType.INPUT

    # Medium-sized region = card
    if 0.2 < width_ratio < 0.6 and 0.1 < height_ratio < 0.4:
        return RegionType.CARD

    # Large central region = content
    if width_ratio > 0.5 and height_ratio > 0.3:
        return RegionType.CONTENT

    return RegionType.UNKNOWN


# =============================================================================
# Screenshot Analyzer
# =============================================================================

class ScreenshotAnalyzer:
    """
    Analyzes screenshots to extract visual specifications.

    This is the main class for screenshot analysis. It can work with:
    - Raw pixel data (list of RGB tuples)
    - Image paths (requires PIL)
    - Mock data for testing

    Example:
        analyzer = ScreenshotAnalyzer()
        result = analyzer.analyze("screenshot.png")

        # Or with raw pixels
        result = analyzer.analyze_pixels(pixels, width=1920, height=1080)
    """

    def __init__(
        self,
        num_colors: int = 8,
        min_region_size: int = 50,
        confidence_threshold: float = 0.5,
    ):
        """
        Initialize the analyzer.

        Args:
            num_colors: Number of colors to extract
            min_region_size: Minimum region size in pixels
            confidence_threshold: Minimum confidence for region detection
        """
        self.num_colors = num_colors
        self.min_region_size = min_region_size
        self.confidence_threshold = confidence_threshold
        self._has_pil = self._check_pil()

    def _check_pil(self) -> bool:
        """Check if PIL is available."""
        try:
            from PIL import Image
            return True
        except ImportError:
            return False

    def analyze(self, image_path: str | Path) -> AnalysisResult:
        """
        Analyze a screenshot image file.

        Args:
            image_path: Path to the image file

        Returns:
            AnalysisResult with extracted specifications

        Raises:
            ImportError: If PIL is not installed
            FileNotFoundError: If image file doesn't exist
        """
        if not self._has_pil:
            raise ImportError(
                "PIL (Pillow) is required for image analysis. "
                "Install with: pip install Pillow"
            )

        from PIL import Image

        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        # Load image
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        width, height = img.size
        pixels = list(img.getdata())

        return self.analyze_pixels(
            pixels=pixels,
            width=width,
            height=height,
            image_path=str(path),
        )

    def analyze_pixels(
        self,
        pixels: list[tuple[int, int, int]],
        width: int,
        height: int,
        image_path: str = "",
    ) -> AnalysisResult:
        """
        Analyze raw pixel data.

        Args:
            pixels: List of RGB tuples
            width: Image width
            height: Image height
            image_path: Optional path for metadata

        Returns:
            AnalysisResult with extracted specifications
        """
        # Extract colors
        colors = extract_colors_from_pixels(pixels, self.num_colors)

        # Classify color roles
        for color in colors:
            color.role = classify_color_role(color, colors, width, height)

        # Detect regions (simplified without edge detection)
        regions = self._detect_regions_simple(pixels, width, height, colors)

        # Estimate typography
        typography = self._estimate_typography(regions, height)

        # Estimate spacing
        spacing = self._estimate_spacing(regions, width, height)

        return AnalysisResult(
            image_path=image_path,
            width=width,
            height=height,
            colors=colors,
            regions=regions,
            typography=typography,
            spacing=spacing,
            metadata={
                "analyzer_version": "1.0.0",
                "num_colors": self.num_colors,
            },
        )

    def _detect_regions_simple(
        self,
        pixels: list[tuple[int, int, int]],
        width: int,
        height: int,
        colors: list[ExtractedColor],
    ) -> list[DetectedRegion]:
        """
        Simple region detection based on color changes.

        This is a basic implementation. For production use, consider
        integrating computer vision libraries like OpenCV.
        """
        regions = []

        # Calculate row brightness
        row_brightness = []
        for y in range(height):
            row_start = y * width
            row_pixels = pixels[row_start:row_start + width]
            avg_brightness = sum(
                (r + g + b) / 3 for r, g, b in row_pixels
            ) / width / 255
            row_brightness.append(avg_brightness)

        # Find horizontal regions
        h_regions = detect_horizontal_regions(height, row_brightness)

        for start_y, end_y in h_regions:
            bounds = BoundingBox(
                x=0,
                y=start_y,
                width=width,
                height=end_y - start_y,
            )

            # Get colors in this region
            region_pixels = []
            for y in range(start_y, min(end_y, height)):
                row_start = y * width
                region_pixels.extend(pixels[row_start:row_start + width])

            region_colors = extract_colors_from_pixels(region_pixels, 3)

            # Classify region type
            region_type = classify_region_type(bounds, width, height, region_colors)

            regions.append(DetectedRegion(
                region_type=region_type,
                bounds=bounds,
                confidence=0.7,  # Basic detection confidence
                colors=region_colors,
            ))

        return regions

    def _estimate_typography(
        self,
        regions: list[DetectedRegion],
        image_height: int,
    ) -> list[TypographyEstimate]:
        """
        Estimate typography based on region analysis.

        This is heuristic-based. For accurate typography detection,
        use OCR libraries like Tesseract.
        """
        typography = []

        # Common typography sizes based on region type
        for region in regions:
            if region.region_type == RegionType.HEADER:
                # Headers typically have larger text
                typography.append(TypographyEstimate(
                    size_px=24,
                    weight="bold",
                    region=region.bounds,
                ))
            elif region.region_type == RegionType.CONTENT:
                # Body text
                typography.append(TypographyEstimate(
                    size_px=16,
                    weight="normal",
                    region=region.bounds,
                ))
            elif region.region_type == RegionType.BUTTON:
                # Button text
                typography.append(TypographyEstimate(
                    size_px=14,
                    weight="medium",
                    region=region.bounds,
                ))

        return typography

    def _estimate_spacing(
        self,
        regions: list[DetectedRegion],
        width: int,
        height: int,
    ) -> SpacingEstimate:
        """
        Estimate spacing values from region gaps.
        """
        if len(regions) < 2:
            return SpacingEstimate()

        # Calculate gaps between regions
        gaps = []
        sorted_regions = sorted(regions, key=lambda r: r.bounds.y)

        for i in range(len(sorted_regions) - 1):
            gap = sorted_regions[i + 1].bounds.y - sorted_regions[i].bounds.y2
            if gap > 0:
                gaps.append(gap)

        if not gaps:
            return SpacingEstimate()

        # Estimate base unit from smallest gap
        min_gap = min(gaps)
        base = max(4, (min_gap // 4) * 4)  # Round to multiple of 4

        return SpacingEstimate(
            base=base,
            xs=base // 2,
            sm=base,
            md=base * 2,
            lg=base * 3,
            xl=base * 4,
        )

    def to_visual_spec(
        self,
        result: AnalysisResult,
        component_id: str = "extracted",
    ) -> VisualSpec:
        """
        Convert analysis result to a VisualSpec.

        Args:
            result: Analysis result to convert
            component_id: ID for the generated spec

        Returns:
            VisualSpec with extracted visual properties
        """
        spec = VisualSpec(
            component_id=component_id,
            component_type="Extracted",
        )

        # Set colors from palette
        palette = result.color_palette
        if palette.get("background"):
            spec.set_colors(background=palette["background"])
        if palette.get("text-primary"):
            spec.set_colors(text=palette["text-primary"])
        if palette.get("primary"):
            spec.set_colors(background=palette["primary"])
        if palette.get("border"):
            spec.set_colors(border=palette["border"])

        # Set spacing
        if result.spacing:
            spec.spacing = SpacingSpec(
                top=DimensionValue(result.spacing.md, "px"),
                right=DimensionValue(result.spacing.md, "px"),
                bottom=DimensionValue(result.spacing.md, "px"),
                left=DimensionValue(result.spacing.md, "px"),
            )

        return spec

    def to_token_group(self, result: AnalysisResult) -> TokenGroup:
        """
        Convert analysis result to a TokenGroup for the design system.

        Args:
            result: Analysis result to convert

        Returns:
            TokenGroup with extracted tokens
        """
        group = TokenGroup(type=TokenType.COLOR)

        for color in result.colors:
            if color.role:
                token = DesignToken(
                    value=color.color,
                    type=TokenType.COLOR,
                    description=f"Extracted {color.role.value} color",
                )
                group.add(color.role.value, token)

        return group


# =============================================================================
# Convenience Functions
# =============================================================================

def analyze_screenshot(
    image_path: str | Path,
    num_colors: int = 8,
) -> AnalysisResult:
    """
    Analyze a screenshot and extract visual specifications.

    Args:
        image_path: Path to the screenshot image
        num_colors: Number of colors to extract

    Returns:
        AnalysisResult with colors, regions, typography, and spacing
    """
    analyzer = ScreenshotAnalyzer(num_colors=num_colors)
    return analyzer.analyze(image_path)


def extract_colors(
    image_path: str | Path,
    num_colors: int = 8,
) -> list[ExtractedColor]:
    """
    Extract color palette from an image.

    Args:
        image_path: Path to the image
        num_colors: Number of colors to extract

    Returns:
        List of ExtractedColor objects
    """
    result = analyze_screenshot(image_path, num_colors)
    return result.colors


def detect_regions(
    image_path: str | Path,
) -> list[DetectedRegion]:
    """
    Detect UI regions in a screenshot.

    Args:
        image_path: Path to the screenshot

    Returns:
        List of DetectedRegion objects
    """
    result = analyze_screenshot(image_path)
    return result.regions


def screenshot_to_spec(
    image_path: str | Path,
    component_id: str = "extracted",
) -> VisualSpec:
    """
    Convert a screenshot directly to a VisualSpec.

    Args:
        image_path: Path to the screenshot
        component_id: ID for the generated spec

    Returns:
        VisualSpec with extracted visual properties
    """
    analyzer = ScreenshotAnalyzer()
    result = analyzer.analyze(image_path)
    return analyzer.to_visual_spec(result, component_id)


def screenshot_to_tokens(
    image_path: str | Path,
) -> TokenGroup:
    """
    Extract design tokens from a screenshot.

    Args:
        image_path: Path to the screenshot

    Returns:
        TokenGroup with extracted color tokens
    """
    analyzer = ScreenshotAnalyzer()
    result = analyzer.analyze(image_path)
    return analyzer.to_token_group(result)


# =============================================================================
# Mock Data for Testing
# =============================================================================

def create_mock_pixels(
    width: int = 100,
    height: int = 100,
    pattern: Literal["solid", "gradient", "regions"] = "solid",
    base_color: tuple[int, int, int] = (255, 255, 255),
) -> list[tuple[int, int, int]]:
    """
    Create mock pixel data for testing.

    Args:
        width: Image width
        height: Image height
        pattern: Type of pattern to generate
        base_color: Base color for the pattern

    Returns:
        List of RGB tuples
    """
    pixels = []

    if pattern == "solid":
        pixels = [base_color] * (width * height)

    elif pattern == "gradient":
        for y in range(height):
            for x in range(width):
                factor = x / width
                r = int(base_color[0] * (1 - factor))
                g = int(base_color[1] * (1 - factor))
                b = int(base_color[2] * (1 - factor))
                pixels.append((r, g, b))

    elif pattern == "regions":
        # Header (dark)
        header_height = height // 6
        for y in range(header_height):
            for x in range(width):
                pixels.append((50, 50, 50))

        # Content (light)
        content_height = height * 4 // 6
        for y in range(content_height):
            for x in range(width):
                pixels.append((245, 245, 245))

        # Footer (medium)
        footer_height = height - header_height - content_height
        for y in range(footer_height):
            for x in range(width):
                pixels.append((100, 100, 100))

    return pixels


def create_mock_result(
    width: int = 1920,
    height: int = 1080,
) -> AnalysisResult:
    """
    Create a mock AnalysisResult for testing.

    Args:
        width: Image width
        height: Image height

    Returns:
        AnalysisResult with mock data
    """
    colors = [
        ExtractedColor(
            color=ColorValue.from_hex("#3b82f6"),
            frequency=0.15,
            role=ColorRole.PRIMARY,
        ),
        ExtractedColor(
            color=ColorValue.from_hex("#f8fafc"),
            frequency=0.60,
            role=ColorRole.BACKGROUND,
        ),
        ExtractedColor(
            color=ColorValue.from_hex("#1e293b"),
            frequency=0.10,
            role=ColorRole.TEXT_PRIMARY,
        ),
        ExtractedColor(
            color=ColorValue.from_hex("#64748b"),
            frequency=0.05,
            role=ColorRole.TEXT_SECONDARY,
        ),
        ExtractedColor(
            color=ColorValue.from_hex("#22c55e"),
            frequency=0.02,
            role=ColorRole.SUCCESS,
        ),
    ]

    regions = [
        DetectedRegion(
            region_type=RegionType.HEADER,
            bounds=BoundingBox(0, 0, width, 64),
            confidence=0.9,
            colors=[colors[0], colors[2]],
        ),
        DetectedRegion(
            region_type=RegionType.CONTENT,
            bounds=BoundingBox(0, 64, width, height - 128),
            confidence=0.85,
            colors=[colors[1], colors[2]],
        ),
        DetectedRegion(
            region_type=RegionType.FOOTER,
            bounds=BoundingBox(0, height - 64, width, 64),
            confidence=0.9,
            colors=[colors[0], colors[3]],
        ),
    ]

    typography = [
        TypographyEstimate(size_px=24, weight="bold"),
        TypographyEstimate(size_px=16, weight="normal"),
        TypographyEstimate(size_px=14, weight="medium"),
    ]

    return AnalysisResult(
        image_path="mock://screenshot.png",
        width=width,
        height=height,
        colors=colors,
        regions=regions,
        typography=typography,
        spacing=SpacingEstimate(base=8),
        metadata={"mock": True},
    )
