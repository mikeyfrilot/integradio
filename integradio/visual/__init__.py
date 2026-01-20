"""
Visual Module - Complete visual specification system for Gradio components.

This module provides:
- W3C DTCG-compliant design tokens (colors, dimensions, typography, etc.)
- Visual specifications for components (layout, spacing, animations)
- Gradio-based viewer/editor UI
- Export to Style Dictionary, CSS, Tailwind, and Figma tokens
- Validation for specs, templates, and CSS (0% test score solution!)
- Test file parsing to link behavior tests to visual specs
- Figma bidirectional sync
- Theme generation from base palettes

Basic Usage:
    from integradio.visual import VisualSpec, DesignToken, UISpec

    # Create a visual spec for a component
    spec = VisualSpec(
        component_id="search-button",
        component_type="Button",
    )
    spec.set_colors(background="#3b82f6", text="#ffffff")
    spec.add_transition("background", 200)

    # Export to CSS
    print(spec.to_css())

Design Tokens:
    from integradio.visual import DesignToken, TokenGroup, ColorValue

    # Create tokens using the W3C DTCG format
    primary = DesignToken.color("#3b82f6", "Primary brand color")
    spacing = DesignToken.dimension(16, "px", "Base spacing")

    # Create token groups
    colors = TokenGroup(type=TokenType.COLOR)
    colors.add("primary", primary)

Viewer UI:
    from integradio.visual import VisualSpecViewer, view_spec

    # Launch the viewer
    view_spec("path/to/spec.json")

    # Or with a UISpec object
    viewer = VisualSpecViewer(spec=my_spec)
    viewer.launch()

Export:
    from integradio.visual import (
        export_to_style_dictionary,
        export_to_css,
        export_to_tailwind,
        export_to_dtcg,
    )

    export_to_css(spec, "output.css")
    export_to_style_dictionary(spec, "output/")
"""

# Core token types
from .tokens import (
    # Enums
    TokenType,
    # Value types
    ColorValue,
    DimensionValue,
    DurationValue,
    CubicBezierValue,
    StrokeStyleValue,
    # Composite values
    ShadowValue,
    BorderValue,
    TransitionValue,
    GradientValue,
    GradientStop,
    TypographyValue,
    # Token classes
    DesignToken,
    TokenGroup,
    # Type aliases
    TokenValue,
    ColorSpace,
    DimensionUnit,
    DurationUnit,
    FontWeightKeyword,
    FontStyleKeyword,
    StrokeStyleKeyword,
    LineCap,
)

# Specification classes
from .spec import (
    # Layout
    Display,
    Position,
    FlexDirection,
    FlexWrap,
    JustifyContent,
    AlignItems,
    Overflow,
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
    # Component spec
    StateStyles,
    VisualSpec,
    # Page & App spec
    PageSpec,
    UISpec,
)

# Viewer
from .viewer import (
    VisualSpecViewer,
    view_spec,
    create_viewer_demo,
)

# Export
from .export import (
    # Exporters
    StyleDictionaryExporter,
    StyleDictionaryConfig,
    CSSExporter,
    TailwindExporter,
    DTCGExporter,
    # Convenience functions
    export_to_style_dictionary,
    export_to_css,
    export_to_tailwind,
    export_to_dtcg,
)

# Validation (the 0% test score solution!)
from .validation import (
    # Types
    Severity,
    ValidationIssue,
    ValidationReport,
    # Validators
    SpecValidator,
    TemplateValidator,
    CSSValidator,
    # Utilities
    contrast_ratio,
    is_color_accessible,
    # Convenience functions
    validate_spec,
    validate_templates,
    validate_css,
    validate_all,
    get_validation_score,
)

# Test bridge (link tests to visual specs)
from .bridge import (
    # Types
    ComponentReference,
    TestAssertion,
    DataFlowEdge,
    TestExtraction,
    LinkReport,
    # Parsers
    TestFileParser,
    TestSuiteScanner,
    SpecGenerator,
    TestSpecLinker,
    # Convenience functions
    parse_test_file,
    scan_tests,
    generate_spec_from_tests,
    link_tests_to_spec,
    auto_fill_spec_from_tests,
)

# Theme generation
from .theme import (
    # Color utilities
    hex_to_hsl,
    hsl_to_hex,
    adjust_lightness,
    adjust_saturation,
    mix_colors,
    get_contrast_color,
    is_light,
    # Shade generation
    generate_shade_scale,
    generate_shade_tokens,
    SHADE_LIGHTNESS,
    # Theme generation
    ThemeColors,
    ThemeConfig,
    ThemeGenerator,
    # CSS output
    generate_theme_css,
    generate_theme_toggle_script,
    # Palettes
    PalettePreset,
    PALETTES,
    get_palette,
    list_palettes,
    # Convenience functions
    generate_theme_from_primary,
    generate_theme_from_palette,
    quick_dark_mode,
)

# Figma integration (optional - requires httpx)
try:
    from .figma import (
        # Types
        FigmaConfig,
        FigmaAPIError,
        FigmaColor,
        FigmaStyle,
        FigmaNode,
        DriftItem,
        DriftReport,
        # Client
        FigmaClient,
        # Import/Export
        FigmaImporter,
        FigmaExporter,
        FigmaSyncer,
        # Convenience functions
        import_from_figma,
        export_to_figma_tokens,
        check_figma_drift,
    )
    _HAS_FIGMA = True
except ImportError:
    _HAS_FIGMA = False

# Spec diffing
from .diff import (
    # Types
    ChangeType,
    ChangeLevel,
    ChangeCategory,
    Change,
    DiffReport,
    # Differ
    SpecDiffer,
    # Changelog
    generate_changelog,
    generate_json_changelog,
    # Version utilities
    parse_version,
    bump_version,
    suggest_version,
    # Convenience functions
    diff_specs,
    diff_visual_specs,
    diff_ui_specs,
)

# Component Library Generator
from .library import (
    # Types
    ComponentStatus,
    PageType,
    # Data classes
    ComponentVariant,
    ComponentEntry,
    CategoryEntry,
    GuidePage,
    LibraryConfig,
    # Library
    ComponentLibrary,
    # HTML generation
    generate_css,
    generate_component_preview_html,
    generate_header_html,
    generate_sidebar_html,
    generate_props_table_html,
    generate_variants_html,
    generate_page_html,
    # Site generator
    LibrarySiteGenerator,
    # Convenience functions
    generate_library_site,
    create_library_from_specs,
    create_library_from_ui_spec,
    quick_library,
)

# Screenshot to Spec
from .screenshot import (
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

__all__ = [
    # Token types
    "TokenType",
    "ColorValue",
    "DimensionValue",
    "DurationValue",
    "CubicBezierValue",
    "StrokeStyleValue",
    "ShadowValue",
    "BorderValue",
    "TransitionValue",
    "GradientValue",
    "GradientStop",
    "TypographyValue",
    "DesignToken",
    "TokenGroup",
    "TokenValue",
    "ColorSpace",
    "DimensionUnit",
    "DurationUnit",
    "FontWeightKeyword",
    "FontStyleKeyword",
    "StrokeStyleKeyword",
    "LineCap",
    # Layout
    "Display",
    "Position",
    "FlexDirection",
    "FlexWrap",
    "JustifyContent",
    "AlignItems",
    "Overflow",
    "FlexSpec",
    "GridSpec",
    "SpacingSpec",
    "LayoutSpec",
    # Responsive
    "Breakpoint",
    "ResponsiveValue",
    "BREAKPOINTS",
    # Animation
    "KeyframeStep",
    "KeyframeAnimation",
    # Assets
    "IconSpec",
    "ImageSpec",
    # Specs
    "StateStyles",
    "VisualSpec",
    "PageSpec",
    "UISpec",
    # Viewer
    "VisualSpecViewer",
    "view_spec",
    "create_viewer_demo",
    # Export
    "StyleDictionaryExporter",
    "StyleDictionaryConfig",
    "CSSExporter",
    "TailwindExporter",
    "DTCGExporter",
    "export_to_style_dictionary",
    "export_to_css",
    "export_to_tailwind",
    "export_to_dtcg",
    # Validation
    "Severity",
    "ValidationIssue",
    "ValidationReport",
    "SpecValidator",
    "TemplateValidator",
    "CSSValidator",
    "contrast_ratio",
    "is_color_accessible",
    "validate_spec",
    "validate_templates",
    "validate_css",
    "validate_all",
    "get_validation_score",
    # Test bridge
    "ComponentReference",
    "TestAssertion",
    "DataFlowEdge",
    "TestExtraction",
    "LinkReport",
    "TestFileParser",
    "TestSuiteScanner",
    "SpecGenerator",
    "TestSpecLinker",
    "parse_test_file",
    "scan_tests",
    "generate_spec_from_tests",
    "link_tests_to_spec",
    "auto_fill_spec_from_tests",
    # Theme generation
    "hex_to_hsl",
    "hsl_to_hex",
    "adjust_lightness",
    "adjust_saturation",
    "mix_colors",
    "get_contrast_color",
    "is_light",
    "generate_shade_scale",
    "generate_shade_tokens",
    "SHADE_LIGHTNESS",
    "ThemeColors",
    "ThemeConfig",
    "ThemeGenerator",
    "generate_theme_css",
    "generate_theme_toggle_script",
    "PalettePreset",
    "PALETTES",
    "get_palette",
    "list_palettes",
    "generate_theme_from_primary",
    "generate_theme_from_palette",
    "quick_dark_mode",
    # Figma (optional)
    "FigmaConfig",
    "FigmaAPIError",
    "FigmaColor",
    "FigmaStyle",
    "FigmaNode",
    "DriftItem",
    "DriftReport",
    "FigmaClient",
    "FigmaImporter",
    "FigmaExporter",
    "FigmaSyncer",
    "import_from_figma",
    "export_to_figma_tokens",
    "check_figma_drift",
    "_HAS_FIGMA",
    # Spec diffing
    "ChangeType",
    "ChangeLevel",
    "ChangeCategory",
    "Change",
    "DiffReport",
    "SpecDiffer",
    "generate_changelog",
    "generate_json_changelog",
    "parse_version",
    "bump_version",
    "suggest_version",
    "diff_specs",
    "diff_visual_specs",
    "diff_ui_specs",
    # Component Library Generator
    "ComponentStatus",
    "PageType",
    "ComponentVariant",
    "ComponentEntry",
    "CategoryEntry",
    "GuidePage",
    "LibraryConfig",
    "ComponentLibrary",
    "generate_css",
    "generate_component_preview_html",
    "generate_header_html",
    "generate_sidebar_html",
    "generate_props_table_html",
    "generate_variants_html",
    "generate_page_html",
    "LibrarySiteGenerator",
    "generate_library_site",
    "create_library_from_specs",
    "create_library_from_ui_spec",
    "quick_library",
    # Screenshot to Spec
    "RegionType",
    "ColorRole",
    "BoundingBox",
    "ExtractedColor",
    "DetectedRegion",
    "TypographyEstimate",
    "SpacingEstimate",
    "AnalysisResult",
    "ScreenshotAnalyzer",
    "hex_to_rgb",
    "rgb_to_hex",
    "color_distance",
    "rgb_to_hsl",
    "is_grayscale",
    "quantize_colors",
    "extract_colors_from_pixels",
    "detect_horizontal_regions",
    "classify_region_type",
    "analyze_screenshot",
    "extract_colors",
    "detect_regions",
    "screenshot_to_spec",
    "screenshot_to_tokens",
    "create_mock_pixels",
    "create_mock_result",
]
