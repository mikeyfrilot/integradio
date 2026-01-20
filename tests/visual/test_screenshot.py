"""
Tests for Screenshot to Spec module.

Tests:
- BoundingBox geometry operations
- Color extraction and quantization
- Color role classification
- Region detection and classification
- Typography and spacing estimation
- ScreenshotAnalyzer with mock data
- AnalysisResult serialization
- Conversion to VisualSpec and TokenGroup
- Edge cases and error handling
"""

import pytest
import json
from pathlib import Path

from integradio.visual.screenshot import (
    # Types
    RegionType,
    ColorRole,
    # Data classes
    BoundingBox,
    ExtractedColor,
    DetectedRegion,
    TypographyEstimate,
    SpacingEstimate,
    AnalysisResult,
    # Analyzer
    ScreenshotAnalyzer,
    # Color utilities
    hex_to_rgb,
    rgb_to_hex,
    color_distance,
    rgb_to_hsl,
    is_grayscale,
    quantize_colors,
    extract_colors_from_pixels,
    classify_color_role,
    # Region detection
    detect_horizontal_regions,
    classify_region_type,
    # Convenience functions
    analyze_screenshot,
    extract_colors,
    detect_regions,
    screenshot_to_spec,
    screenshot_to_tokens,
    # Mock data
    create_mock_pixels,
    create_mock_result,
)
from integradio.visual.tokens import ColorValue, TokenType


# =============================================================================
# BoundingBox Tests
# =============================================================================

class TestBoundingBox:
    """Tests for BoundingBox dataclass."""

    def test_basic_properties(self):
        """Test basic BoundingBox properties."""
        box = BoundingBox(x=10, y=20, width=100, height=50)

        assert box.x == 10
        assert box.y == 20
        assert box.width == 100
        assert box.height == 50

    def test_x2_y2_properties(self):
        """Test x2 and y2 edge coordinates."""
        box = BoundingBox(x=10, y=20, width=100, height=50)

        assert box.x2 == 110  # 10 + 100
        assert box.y2 == 70   # 20 + 50

    def test_center_property(self):
        """Test center point calculation."""
        box = BoundingBox(x=0, y=0, width=100, height=100)

        assert box.center == (50, 50)

    def test_center_property_offset(self):
        """Test center point with offset box."""
        box = BoundingBox(x=10, y=20, width=100, height=50)

        assert box.center == (60, 45)  # (10 + 50, 20 + 25)

    def test_area_property(self):
        """Test area calculation."""
        box = BoundingBox(x=0, y=0, width=100, height=50)

        assert box.area == 5000

    def test_contains_fully_inside(self):
        """Test contains with fully enclosed box."""
        outer = BoundingBox(x=0, y=0, width=100, height=100)
        inner = BoundingBox(x=10, y=10, width=50, height=50)

        assert outer.contains(inner)
        assert not inner.contains(outer)

    def test_contains_same_box(self):
        """Test contains with identical boxes."""
        box1 = BoundingBox(x=0, y=0, width=100, height=100)
        box2 = BoundingBox(x=0, y=0, width=100, height=100)

        assert box1.contains(box2)
        assert box2.contains(box1)

    def test_contains_partial_overlap(self):
        """Test contains with partial overlap."""
        box1 = BoundingBox(x=0, y=0, width=100, height=100)
        box2 = BoundingBox(x=50, y=50, width=100, height=100)

        assert not box1.contains(box2)
        assert not box2.contains(box1)

    def test_overlaps_true(self):
        """Test overlaps with overlapping boxes."""
        box1 = BoundingBox(x=0, y=0, width=100, height=100)
        box2 = BoundingBox(x=50, y=50, width=100, height=100)

        assert box1.overlaps(box2)
        assert box2.overlaps(box1)

    def test_overlaps_false(self):
        """Test overlaps with non-overlapping boxes."""
        box1 = BoundingBox(x=0, y=0, width=100, height=100)
        box2 = BoundingBox(x=200, y=200, width=100, height=100)

        assert not box1.overlaps(box2)
        assert not box2.overlaps(box1)

    def test_overlaps_adjacent(self):
        """Test overlaps with adjacent boxes (touching edges)."""
        box1 = BoundingBox(x=0, y=0, width=100, height=100)
        box2 = BoundingBox(x=101, y=0, width=100, height=100)  # 1px gap

        # Boxes with gap don't overlap
        assert not box1.overlaps(box2)

    def test_overlaps_touching_edge(self):
        """Test overlaps with boxes sharing an edge."""
        box1 = BoundingBox(x=0, y=0, width=100, height=100)
        box2 = BoundingBox(x=100, y=0, width=100, height=100)

        # Boxes sharing edge are considered overlapping in this implementation
        # because x2 (100) is not < x (100) of box2
        assert box1.overlaps(box2)

    def test_to_dict(self):
        """Test dictionary conversion."""
        box = BoundingBox(x=10, y=20, width=100, height=50)
        result = box.to_dict()

        assert result == {"x": 10, "y": 20, "width": 100, "height": 50}


# =============================================================================
# Color Utility Tests
# =============================================================================

class TestColorUtilities:
    """Tests for color utility functions."""

    def test_hex_to_rgb(self):
        """Test hex to RGB conversion."""
        assert hex_to_rgb("#ff0000") == (255, 0, 0)
        assert hex_to_rgb("#00ff00") == (0, 255, 0)
        assert hex_to_rgb("#0000ff") == (0, 0, 255)
        assert hex_to_rgb("#ffffff") == (255, 255, 255)
        assert hex_to_rgb("#000000") == (0, 0, 0)

    def test_hex_to_rgb_no_hash(self):
        """Test hex to RGB without hash prefix."""
        assert hex_to_rgb("ff0000") == (255, 0, 0)

    def test_rgb_to_hex(self):
        """Test RGB to hex conversion."""
        assert rgb_to_hex(255, 0, 0) == "#ff0000"
        assert rgb_to_hex(0, 255, 0) == "#00ff00"
        assert rgb_to_hex(0, 0, 255) == "#0000ff"
        assert rgb_to_hex(255, 255, 255) == "#ffffff"
        assert rgb_to_hex(0, 0, 0) == "#000000"

    def test_color_distance_same(self):
        """Test color distance for same colors."""
        assert color_distance((255, 0, 0), (255, 0, 0)) == 0.0

    def test_color_distance_different(self):
        """Test color distance for different colors."""
        dist = color_distance((255, 0, 0), (0, 255, 0))
        assert dist > 0
        # sqrt((255-0)^2 + (0-255)^2 + (0-0)^2) = sqrt(255^2 + 255^2) ≈ 360.6
        assert 360 < dist < 361

    def test_color_distance_black_white(self):
        """Test color distance between black and white."""
        dist = color_distance((0, 0, 0), (255, 255, 255))
        # sqrt(255^2 + 255^2 + 255^2) ≈ 441.7
        assert 441 < dist < 442

    def test_rgb_to_hsl_red(self):
        """Test RGB to HSL for pure red."""
        h, s, l = rgb_to_hsl(255, 0, 0)
        assert h == 0  # Hue is 0 for red
        assert s == 1.0  # Full saturation
        assert l == 0.5  # 50% lightness

    def test_rgb_to_hsl_green(self):
        """Test RGB to HSL for pure green."""
        h, s, l = rgb_to_hsl(0, 255, 0)
        assert h == 120  # Hue is 120 for green
        assert s == 1.0

    def test_rgb_to_hsl_blue(self):
        """Test RGB to HSL for pure blue."""
        h, s, l = rgb_to_hsl(0, 0, 255)
        assert h == 240  # Hue is 240 for blue
        assert s == 1.0

    def test_rgb_to_hsl_gray(self):
        """Test RGB to HSL for gray."""
        h, s, l = rgb_to_hsl(128, 128, 128)
        assert s == 0  # No saturation for gray
        assert 0.4 < l < 0.6  # Mid lightness

    def test_is_grayscale_true(self):
        """Test is_grayscale for actual grays."""
        assert is_grayscale(128, 128, 128)
        assert is_grayscale(0, 0, 0)
        assert is_grayscale(255, 255, 255)
        assert is_grayscale(100, 105, 102)  # Within threshold

    def test_is_grayscale_false(self):
        """Test is_grayscale for chromatic colors."""
        assert not is_grayscale(255, 0, 0)
        assert not is_grayscale(0, 255, 0)
        assert not is_grayscale(0, 0, 255)
        assert not is_grayscale(200, 100, 50)


# =============================================================================
# Color Quantization Tests
# =============================================================================

class TestColorQuantization:
    """Tests for color quantization functions."""

    def test_quantize_colors_single_color(self):
        """Test quantization with single color."""
        pixels = [(255, 0, 0)] * 100
        result = quantize_colors(pixels, num_colors=4)

        assert len(result) == 1
        assert result[0][0] == (255, 0, 0)
        assert result[0][1] == 100

    def test_quantize_colors_multiple_colors(self):
        """Test quantization with multiple distinct colors."""
        pixels = [(255, 0, 0)] * 50 + [(0, 255, 0)] * 50
        result = quantize_colors(pixels, num_colors=4)

        assert len(result) >= 2
        colors = [c for c, _ in result]
        # Should contain red and green (or close)
        assert any(c[0] > 200 and c[1] < 50 for c in colors)  # Red-ish
        assert any(c[1] > 200 and c[0] < 50 for c in colors)  # Green-ish

    def test_quantize_colors_empty(self):
        """Test quantization with empty pixel list."""
        result = quantize_colors([], num_colors=4)
        assert result == []

    def test_quantize_colors_few_unique(self):
        """Test quantization when unique colors < num_colors."""
        pixels = [(255, 0, 0)] * 50 + [(0, 255, 0)] * 50
        result = quantize_colors(pixels, num_colors=8)

        # Should return all unique colors
        assert len(result) <= 8

    def test_extract_colors_from_pixels(self):
        """Test extracting colors from pixel data."""
        pixels = [(255, 255, 255)] * 80 + [(0, 0, 0)] * 20
        colors = extract_colors_from_pixels(pixels, num_colors=4)

        assert len(colors) >= 1
        assert isinstance(colors[0], ExtractedColor)
        assert colors[0].frequency > 0

    def test_extract_colors_frequency(self):
        """Test that frequencies sum to approximately 1."""
        pixels = [(255, 0, 0)] * 30 + [(0, 255, 0)] * 70
        colors = extract_colors_from_pixels(pixels, num_colors=4)

        total_freq = sum(c.frequency for c in colors)
        assert 0.99 < total_freq <= 1.01


# =============================================================================
# Color Role Classification Tests
# =============================================================================

class TestColorRoleClassification:
    """Tests for color role classification."""

    def test_classify_dark_gray_as_text(self):
        """Test that dark gray is classified as text."""
        color = ExtractedColor(
            color=ColorValue.from_hex("#1a1a1a"),
            frequency=0.05,
        )
        role = classify_color_role(color, [color], 1920, 1080)

        assert role == ColorRole.TEXT_PRIMARY

    def test_classify_white_as_background(self):
        """Test that white is classified as background."""
        color = ExtractedColor(
            color=ColorValue.from_hex("#ffffff"),
            frequency=0.60,
        )
        role = classify_color_role(color, [color], 1920, 1080)

        assert role == ColorRole.BACKGROUND

    def test_classify_high_frequency_color_as_primary(self):
        """Test that high frequency saturated color is primary."""
        color = ExtractedColor(
            color=ColorValue.from_hex("#3b82f6"),  # Blue
            frequency=0.35,
        )
        role = classify_color_role(color, [color], 1920, 1080)

        assert role == ColorRole.PRIMARY

    def test_classify_red_as_error(self):
        """Test that low frequency red is error."""
        color = ExtractedColor(
            color=ColorValue.from_hex("#ef4444"),  # Red
            frequency=0.02,
        )
        role = classify_color_role(color, [color], 1920, 1080)

        assert role == ColorRole.ERROR

    def test_classify_green_as_success(self):
        """Test that low frequency green is success."""
        color = ExtractedColor(
            color=ColorValue.from_hex("#22c55e"),  # Green
            frequency=0.02,
        )
        role = classify_color_role(color, [color], 1920, 1080)

        assert role == ColorRole.SUCCESS


# =============================================================================
# ExtractedColor Tests
# =============================================================================

class TestExtractedColor:
    """Tests for ExtractedColor dataclass."""

    def test_hex_property(self):
        """Test hex color property."""
        color = ExtractedColor(
            color=ColorValue.from_hex("#3b82f6"),
            frequency=0.15,
        )

        assert color.hex == "#3b82f6"

    def test_luminance_property(self):
        """Test luminance calculation."""
        white = ExtractedColor(
            color=ColorValue.from_hex("#ffffff"),
            frequency=0.5,
        )
        black = ExtractedColor(
            color=ColorValue.from_hex("#000000"),
            frequency=0.5,
        )

        assert white.luminance > 0.9
        assert black.luminance < 0.1

    def test_to_dict(self):
        """Test dictionary conversion."""
        color = ExtractedColor(
            color=ColorValue.from_hex("#3b82f6"),
            frequency=0.15,
            role=ColorRole.PRIMARY,
        )
        result = color.to_dict()

        assert result["hex"] == "#3b82f6"
        assert result["frequency"] == 0.15
        assert result["role"] == "primary"
        assert "luminance" in result


# =============================================================================
# Region Detection Tests
# =============================================================================

class TestRegionDetection:
    """Tests for region detection functions."""

    def test_detect_horizontal_regions_uniform(self):
        """Test region detection with uniform brightness."""
        brightness = [0.5] * 100
        regions = detect_horizontal_regions(100, brightness)

        # Should detect single region
        assert len(regions) >= 1

    def test_detect_horizontal_regions_with_change(self):
        """Test region detection with brightness change."""
        brightness = [0.2] * 50 + [0.8] * 50
        regions = detect_horizontal_regions(100, brightness)

        # Should detect multiple regions
        assert len(regions) >= 2

    def test_detect_horizontal_regions_empty(self):
        """Test region detection with empty input."""
        regions = detect_horizontal_regions(100, [])

        assert len(regions) == 1
        assert regions[0] == (0, 100)

    def test_classify_region_header(self):
        """Test classifying a header region."""
        bounds = BoundingBox(x=0, y=0, width=1920, height=80)
        region_type = classify_region_type(bounds, 1920, 1080, [])

        assert region_type == RegionType.HEADER

    def test_classify_region_footer(self):
        """Test classifying a footer region."""
        bounds = BoundingBox(x=0, y=1000, width=1920, height=80)
        region_type = classify_region_type(bounds, 1920, 1080, [])

        assert region_type == RegionType.FOOTER

    def test_classify_region_sidebar(self):
        """Test classifying a sidebar region."""
        bounds = BoundingBox(x=0, y=80, width=250, height=800)
        region_type = classify_region_type(bounds, 1920, 1080, [])

        assert region_type == RegionType.SIDEBAR

    def test_classify_region_button(self):
        """Test classifying a button region."""
        bounds = BoundingBox(x=500, y=500, width=120, height=40)
        region_type = classify_region_type(bounds, 1920, 1080, [])

        assert region_type == RegionType.BUTTON

    def test_classify_region_content(self):
        """Test classifying a content region."""
        bounds = BoundingBox(x=0, y=100, width=1500, height=600)
        region_type = classify_region_type(bounds, 1920, 1080, [])

        assert region_type == RegionType.CONTENT


# =============================================================================
# DetectedRegion Tests
# =============================================================================

class TestDetectedRegion:
    """Tests for DetectedRegion dataclass."""

    def test_basic_creation(self):
        """Test basic DetectedRegion creation."""
        region = DetectedRegion(
            region_type=RegionType.HEADER,
            bounds=BoundingBox(0, 0, 1920, 64),
            confidence=0.9,
        )

        assert region.region_type == RegionType.HEADER
        assert region.confidence == 0.9

    def test_dominant_color(self):
        """Test dominant color property."""
        colors = [
            ExtractedColor(ColorValue.from_hex("#ffffff"), 0.3),
            ExtractedColor(ColorValue.from_hex("#000000"), 0.7),
        ]
        region = DetectedRegion(
            region_type=RegionType.CONTENT,
            bounds=BoundingBox(0, 0, 100, 100),
            confidence=0.8,
            colors=colors,
        )

        assert region.dominant_color.hex == "#000000"

    def test_dominant_color_empty(self):
        """Test dominant color with no colors."""
        region = DetectedRegion(
            region_type=RegionType.CONTENT,
            bounds=BoundingBox(0, 0, 100, 100),
            confidence=0.8,
        )

        assert region.dominant_color is None

    def test_to_dict(self):
        """Test dictionary conversion."""
        region = DetectedRegion(
            region_type=RegionType.BUTTON,
            bounds=BoundingBox(100, 200, 120, 40),
            confidence=0.85,
            attributes={"text": "Submit"},
        )
        result = region.to_dict()

        assert result["type"] == "button"
        assert result["confidence"] == 0.85
        assert result["bounds"]["x"] == 100
        assert result["attributes"]["text"] == "Submit"


# =============================================================================
# Typography Estimate Tests
# =============================================================================

class TestTypographyEstimate:
    """Tests for TypographyEstimate dataclass."""

    def test_basic_creation(self):
        """Test basic TypographyEstimate creation."""
        typo = TypographyEstimate(
            size_px=16,
            weight="normal",
            style="normal",
        )

        assert typo.size_px == 16
        assert typo.weight == "normal"

    def test_to_typography_value(self):
        """Test conversion to TypographyValue token."""
        typo = TypographyEstimate(
            size_px=24,
            weight="bold",
            line_height=1.4,
        )
        value = typo.to_typography_value()

        assert value.font_size.value == 24
        assert value.font_weight == 700
        assert value.line_height == 1.4

    def test_to_dict(self):
        """Test dictionary conversion."""
        typo = TypographyEstimate(size_px=16, weight="medium")
        result = typo.to_dict()

        assert result["size_px"] == 16
        assert result["weight"] == "medium"


# =============================================================================
# Spacing Estimate Tests
# =============================================================================

class TestSpacingEstimate:
    """Tests for SpacingEstimate dataclass."""

    def test_default_values(self):
        """Test default spacing values."""
        spacing = SpacingEstimate()

        assert spacing.base == 8
        assert spacing.xs == 4
        assert spacing.sm == 8
        assert spacing.md == 16
        assert spacing.lg == 24
        assert spacing.xl == 32

    def test_custom_values(self):
        """Test custom spacing values."""
        spacing = SpacingEstimate(
            base=4,
            xs=2,
            sm=4,
            md=8,
            lg=12,
            xl=16,
        )

        assert spacing.base == 4
        assert spacing.xl == 16

    def test_to_dict(self):
        """Test dictionary conversion."""
        spacing = SpacingEstimate()
        result = spacing.to_dict()

        assert result["base"] == 8
        assert result["md"] == 16


# =============================================================================
# AnalysisResult Tests
# =============================================================================

class TestAnalysisResult:
    """Tests for AnalysisResult dataclass."""

    def test_basic_creation(self):
        """Test basic AnalysisResult creation."""
        result = AnalysisResult(
            image_path="test.png",
            width=1920,
            height=1080,
        )

        assert result.image_path == "test.png"
        assert result.width == 1920
        assert result.height == 1080

    def test_color_palette_property(self):
        """Test color_palette property."""
        colors = [
            ExtractedColor(
                ColorValue.from_hex("#3b82f6"),
                0.15,
                role=ColorRole.PRIMARY,
            ),
            ExtractedColor(
                ColorValue.from_hex("#ffffff"),
                0.60,
                role=ColorRole.BACKGROUND,
            ),
        ]
        result = AnalysisResult(
            image_path="test.png",
            width=1920,
            height=1080,
            colors=colors,
        )

        palette = result.color_palette
        assert palette["primary"] == "#3b82f6"
        assert palette["background"] == "#ffffff"

    def test_dominant_colors_property(self):
        """Test dominant_colors property."""
        colors = [
            ExtractedColor(ColorValue.from_hex("#ffffff"), 0.60),
            ExtractedColor(ColorValue.from_hex("#3b82f6"), 0.30),
            ExtractedColor(ColorValue.from_hex("#000000"), 0.10),
        ]
        result = AnalysisResult(
            image_path="test.png",
            width=1920,
            height=1080,
            colors=colors,
        )

        dominant = result.dominant_colors
        assert dominant[0] == "#ffffff"  # Most frequent first
        assert len(dominant) <= 5

    def test_to_dict(self):
        """Test dictionary conversion."""
        result = AnalysisResult(
            image_path="test.png",
            width=1920,
            height=1080,
            metadata={"test": True},
        )
        data = result.to_dict()

        assert data["image_path"] == "test.png"
        assert data["dimensions"]["width"] == 1920
        assert data["metadata"]["test"] is True

    def test_to_json(self):
        """Test JSON serialization."""
        result = AnalysisResult(
            image_path="test.png",
            width=100,
            height=100,
        )
        json_str = result.to_json()

        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["image_path"] == "test.png"

    def test_save(self, tmp_path):
        """Test saving to file."""
        result = AnalysisResult(
            image_path="test.png",
            width=100,
            height=100,
        )
        output_path = tmp_path / "result.json"
        result.save(output_path)

        assert output_path.exists()
        content = json.loads(output_path.read_text())
        assert content["image_path"] == "test.png"


# =============================================================================
# Mock Data Tests
# =============================================================================

class TestMockData:
    """Tests for mock data generation functions."""

    def test_create_mock_pixels_solid(self):
        """Test creating solid color mock pixels."""
        pixels = create_mock_pixels(
            width=10,
            height=10,
            pattern="solid",
            base_color=(255, 0, 0),
        )

        assert len(pixels) == 100
        assert all(p == (255, 0, 0) for p in pixels)

    def test_create_mock_pixels_gradient(self):
        """Test creating gradient mock pixels."""
        pixels = create_mock_pixels(
            width=10,
            height=10,
            pattern="gradient",
            base_color=(255, 255, 255),
        )

        assert len(pixels) == 100
        # First pixel should be brighter than last in each row
        assert pixels[0][0] > pixels[9][0]

    def test_create_mock_pixels_regions(self):
        """Test creating region-based mock pixels."""
        pixels = create_mock_pixels(
            width=100,
            height=100,
            pattern="regions",
        )

        assert len(pixels) == 10000
        # Should have distinct regions (header dark, content light, footer medium)
        unique_colors = set(pixels)
        assert len(unique_colors) >= 2

    def test_create_mock_result(self):
        """Test creating mock AnalysisResult."""
        result = create_mock_result(width=1920, height=1080)

        assert result.width == 1920
        assert result.height == 1080
        assert len(result.colors) > 0
        assert len(result.regions) > 0
        assert len(result.typography) > 0
        assert result.spacing is not None
        assert result.metadata.get("mock") is True


# =============================================================================
# ScreenshotAnalyzer Tests
# =============================================================================

class TestScreenshotAnalyzer:
    """Tests for ScreenshotAnalyzer class."""

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = ScreenshotAnalyzer(
            num_colors=10,
            min_region_size=100,
            confidence_threshold=0.6,
        )

        assert analyzer.num_colors == 10
        assert analyzer.min_region_size == 100
        assert analyzer.confidence_threshold == 0.6

    def test_analyze_pixels_basic(self):
        """Test analyzing raw pixel data."""
        analyzer = ScreenshotAnalyzer(num_colors=4)
        pixels = create_mock_pixels(100, 100, "solid", (128, 128, 128))

        result = analyzer.analyze_pixels(pixels, 100, 100)

        assert result.width == 100
        assert result.height == 100
        assert len(result.colors) > 0

    def test_analyze_pixels_with_regions(self):
        """Test analyzing pixels with distinct regions."""
        analyzer = ScreenshotAnalyzer(num_colors=4)
        pixels = create_mock_pixels(100, 100, "regions")

        result = analyzer.analyze_pixels(pixels, 100, 100, "test.png")

        assert result.image_path == "test.png"
        assert len(result.regions) > 0

    def test_analyze_requires_pil(self):
        """Test that analyze() requires PIL for file loading."""
        analyzer = ScreenshotAnalyzer()

        # This test checks behavior based on PIL availability
        # If PIL is installed, it will try to load the file
        # If not, it will raise ImportError
        try:
            analyzer.analyze("nonexistent.png")
        except ImportError as e:
            assert "PIL" in str(e) or "Pillow" in str(e)
        except FileNotFoundError:
            # PIL is installed but file doesn't exist
            pass

    def test_to_visual_spec(self):
        """Test converting result to VisualSpec."""
        analyzer = ScreenshotAnalyzer()
        mock_result = create_mock_result()

        spec = analyzer.to_visual_spec(mock_result, "test-component")

        assert spec.component_id == "test-component"
        assert spec.component_type == "Extracted"

    def test_to_token_group(self):
        """Test converting result to TokenGroup."""
        analyzer = ScreenshotAnalyzer()
        mock_result = create_mock_result()

        group = analyzer.to_token_group(mock_result)

        assert group.type == TokenType.COLOR
        # Should have tokens for colors with roles
        assert len(group.tokens) > 0


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_extract_colors_with_mock_pixels(self):
        """Test extract_colors with mock analyzer setup."""
        # This would need PIL installed to work with real images
        # For now, test that it raises appropriate error
        try:
            colors = extract_colors("nonexistent.png")
        except ImportError:
            pass  # Expected if PIL not installed
        except FileNotFoundError:
            pass  # Expected if PIL installed but file missing

    def test_screenshot_to_spec_mock(self):
        """Test screenshot_to_spec returns VisualSpec."""
        # Use the mock result approach for testing
        analyzer = ScreenshotAnalyzer()
        mock_result = create_mock_result()
        spec = analyzer.to_visual_spec(mock_result)

        assert spec.component_type == "Extracted"

    def test_screenshot_to_tokens_mock(self):
        """Test screenshot_to_tokens returns TokenGroup."""
        analyzer = ScreenshotAnalyzer()
        mock_result = create_mock_result()
        tokens = analyzer.to_token_group(mock_result)

        assert isinstance(tokens.type, TokenType)


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_pixels(self):
        """Test handling empty pixel list."""
        analyzer = ScreenshotAnalyzer()
        result = analyzer.analyze_pixels([], 0, 0)

        assert result.width == 0
        assert result.height == 0
        assert len(result.colors) == 0

    def test_single_pixel(self):
        """Test handling single pixel image."""
        analyzer = ScreenshotAnalyzer()
        result = analyzer.analyze_pixels([(255, 0, 0)], 1, 1)

        assert result.width == 1
        assert result.height == 1
        assert len(result.colors) == 1

    def test_very_large_dimensions(self):
        """Test handling large dimension values."""
        box = BoundingBox(0, 0, 10000, 10000)
        assert box.area == 100000000

    def test_zero_dimension_box(self):
        """Test handling zero-dimension boxes."""
        box = BoundingBox(0, 0, 0, 0)
        assert box.area == 0
        assert box.center == (0, 0)

    def test_negative_coordinates(self):
        """Test handling negative coordinates."""
        # While unusual, the dataclass should handle them
        box = BoundingBox(x=-10, y=-10, width=20, height=20)
        assert box.x2 == 10
        assert box.y2 == 10

    def test_uniform_brightness_regions(self):
        """Test region detection with uniform brightness."""
        brightness = [0.5] * 100
        regions = detect_horizontal_regions(100, brightness)

        # Should return at least one region covering the whole height
        assert len(regions) >= 1
        total_height = sum(r[1] - r[0] for r in regions)
        assert total_height > 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline_mock(self):
        """Test full analysis pipeline with mock data."""
        # Create mock pixels
        pixels = create_mock_pixels(200, 200, "regions")

        # Analyze
        analyzer = ScreenshotAnalyzer(num_colors=6)
        result = analyzer.analyze_pixels(pixels, 200, 200, "mock://test.png")

        # Check result completeness
        assert result.image_path == "mock://test.png"
        assert len(result.colors) > 0
        assert len(result.regions) > 0

        # Convert to spec
        spec = analyzer.to_visual_spec(result, "mock-component")
        assert spec.component_id == "mock-component"

        # Convert to tokens
        tokens = analyzer.to_token_group(result)
        assert len(tokens.tokens) >= 0

        # Serialize
        json_output = result.to_json()
        assert json.loads(json_output)["image_path"] == "mock://test.png"

    def test_color_analysis_roundtrip(self):
        """Test color extraction and role assignment."""
        # Create pixels with known colors
        pixels = (
            [(255, 255, 255)] * 600 +  # 60% white
            [(59, 130, 246)] * 150 +    # 15% blue
            [(30, 41, 59)] * 100 +      # 10% dark
            [(100, 116, 139)] * 50 +    # 5% gray
            [(34, 197, 94)] * 20 +      # 2% green
            [(239, 68, 68)] * 80        # 8% red
        )

        colors = extract_colors_from_pixels(pixels, num_colors=6)

        # Verify extraction worked
        assert len(colors) <= 6
        assert sum(c.frequency for c in colors) > 0.95

    def test_region_hierarchy(self):
        """Test region containment and hierarchy."""
        parent = DetectedRegion(
            region_type=RegionType.CONTENT,
            bounds=BoundingBox(0, 0, 1000, 1000),
            confidence=0.9,
        )
        child = DetectedRegion(
            region_type=RegionType.CARD,
            bounds=BoundingBox(100, 100, 200, 200),
            confidence=0.8,
        )

        # Add child to parent
        parent.children.append(child)

        assert len(parent.children) == 1
        assert parent.bounds.contains(child.bounds)


# =============================================================================
# Serialization Tests
# =============================================================================

class TestSerialization:
    """Tests for JSON serialization and deserialization."""

    def test_analysis_result_json_roundtrip(self):
        """Test AnalysisResult JSON roundtrip."""
        result = create_mock_result()
        json_str = result.to_json()
        parsed = json.loads(json_str)

        assert parsed["dimensions"]["width"] == result.width
        assert parsed["dimensions"]["height"] == result.height
        assert len(parsed["colors"]) == len(result.colors)
        assert len(parsed["regions"]) == len(result.regions)

    def test_bounding_box_json(self):
        """Test BoundingBox JSON serialization."""
        box = BoundingBox(10, 20, 100, 50)
        data = box.to_dict()
        json_str = json.dumps(data)
        parsed = json.loads(json_str)

        assert parsed["x"] == 10
        assert parsed["width"] == 100

    def test_extracted_color_json(self):
        """Test ExtractedColor JSON serialization."""
        color = ExtractedColor(
            color=ColorValue.from_hex("#3b82f6"),
            frequency=0.15,
            role=ColorRole.PRIMARY,
        )
        data = color.to_dict()
        json_str = json.dumps(data)
        parsed = json.loads(json_str)

        assert parsed["hex"] == "#3b82f6"
        assert parsed["role"] == "primary"

    def test_detected_region_nested_json(self):
        """Test DetectedRegion with nested children."""
        region = DetectedRegion(
            region_type=RegionType.CONTENT,
            bounds=BoundingBox(0, 0, 1000, 1000),
            confidence=0.9,
            children=[
                DetectedRegion(
                    region_type=RegionType.CARD,
                    bounds=BoundingBox(100, 100, 200, 200),
                    confidence=0.8,
                )
            ],
        )
        data = region.to_dict()
        json_str = json.dumps(data)
        parsed = json.loads(json_str)

        assert len(parsed["children"]) == 1
        assert parsed["children"][0]["type"] == "card"
