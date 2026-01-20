"""
Component Library Generator - Generate documentation sites for UI components.

Creates Storybook-like documentation from VisualSpecs including:
- Component catalogs with interactive examples
- Design token documentation
- Variant showcases with different states
- API documentation from specs
- Markdown/MDX content support
- Static HTML site generation
- Theme previews (light/dark mode)

This module provides tools to:
1. Organize components into categories
2. Generate HTML documentation pages
3. Create searchable component catalogs
4. Export to static sites for deployment

Usage:
    from integradio.visual.library import (
        ComponentLibrary,
        generate_library_site,
        ComponentEntry,
    )

    # Create library
    library = ComponentLibrary("My Design System")
    library.add_component(button_spec, category="Inputs")
    library.add_component(card_spec, category="Layout")

    # Generate static site
    generate_library_site(library, "output/docs")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Callable
from enum import Enum
from pathlib import Path
import json
import html
from datetime import datetime

from .tokens import (
    TokenGroup,
    TokenType,
    DesignToken,
    ColorValue,
    DimensionValue,
)
from .spec import VisualSpec, UISpec, StateStyles


# =============================================================================
# Types and Enums
# =============================================================================

class ComponentStatus(str, Enum):
    """Status of a component in the library."""
    DRAFT = "draft"
    BETA = "beta"
    STABLE = "stable"
    DEPRECATED = "deprecated"


class PageType(str, Enum):
    """Type of documentation page."""
    COMPONENT = "component"
    CATEGORY = "category"
    TOKEN = "token"
    GUIDE = "guide"
    INDEX = "index"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ComponentVariant:
    """A variant of a component (e.g., primary, secondary, outline)."""
    name: str
    description: str = ""
    spec: VisualSpec | None = None
    props: dict[str, Any] = field(default_factory=dict)
    code_example: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "props": self.props,
            "code_example": self.code_example,
        }


@dataclass
class ComponentEntry:
    """
    An entry in the component library.

    Represents a single component with its documentation.
    """
    name: str
    description: str
    spec: VisualSpec
    category: str = "Uncategorized"
    status: ComponentStatus = ComponentStatus.STABLE
    tags: list[str] = field(default_factory=list)
    variants: list[ComponentVariant] = field(default_factory=list)
    props: dict[str, dict[str, Any]] = field(default_factory=dict)
    usage_notes: str = ""
    code_example: str = ""
    related_components: list[str] = field(default_factory=list)
    changelog: list[dict[str, str]] = field(default_factory=list)

    @property
    def slug(self) -> str:
        """URL-safe slug for the component."""
        import re
        # Remove special characters that are invalid in filenames
        slug = self.name.lower()
        slug = slug.replace(" ", "-").replace("_", "-")
        # Remove any characters not allowed in URLs/filenames
        slug = re.sub(r'[<>:"/\\|?*]', '', slug)
        # Remove any remaining non-alphanumeric except hyphens
        slug = re.sub(r'[^a-z0-9\-]', '', slug)
        # Collapse multiple hyphens
        slug = re.sub(r'-+', '-', slug)
        # Strip leading/trailing hyphens
        slug = slug.strip('-')
        return slug or 'component'

    def add_variant(
        self,
        name: str,
        description: str = "",
        spec: VisualSpec | None = None,
        **props,
    ) -> None:
        """Add a variant to the component."""
        self.variants.append(ComponentVariant(
            name=name,
            description=description,
            spec=spec,
            props=props,
        ))

    def add_prop(
        self,
        name: str,
        type: str,
        description: str,
        default: Any = None,
        required: bool = False,
    ) -> None:
        """Document a prop for the component."""
        self.props[name] = {
            "type": type,
            "description": description,
            "default": default,
            "required": required,
        }

    def add_changelog_entry(
        self,
        version: str,
        description: str,
        date: str | None = None,
    ) -> None:
        """Add a changelog entry."""
        self.changelog.append({
            "version": version,
            "description": description,
            "date": date or datetime.now().strftime("%Y-%m-%d"),
        })

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "slug": self.slug,
            "description": self.description,
            "category": self.category,
            "status": self.status.value,
            "tags": self.tags,
            "variants": [v.to_dict() for v in self.variants],
            "props": self.props,
            "usage_notes": self.usage_notes,
            "code_example": self.code_example,
            "related_components": self.related_components,
            "changelog": self.changelog,
        }


@dataclass
class CategoryEntry:
    """A category in the component library."""
    name: str
    description: str = ""
    icon: str = ""
    components: list[str] = field(default_factory=list)
    order: int = 0

    @property
    def slug(self) -> str:
        """URL-safe slug for the category."""
        return self.name.lower().replace(" ", "-").replace("_", "-")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "slug": self.slug,
            "description": self.description,
            "icon": self.icon,
            "components": self.components,
            "order": self.order,
        }


@dataclass
class GuidePage:
    """A documentation guide page."""
    title: str
    slug: str
    content: str  # Markdown or HTML content
    category: str = "Guides"
    order: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "slug": self.slug,
            "category": self.category,
            "order": self.order,
        }


@dataclass
class LibraryConfig:
    """Configuration for the component library."""
    name: str = "Component Library"
    version: str = "1.0.0"
    description: str = ""
    logo_url: str = ""
    primary_color: str = "#3b82f6"
    footer_text: str = ""
    github_url: str = ""
    figma_url: str = ""
    show_theme_toggle: bool = True
    show_search: bool = True
    custom_css: str = ""
    custom_head: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "logo_url": self.logo_url,
            "primary_color": self.primary_color,
            "footer_text": self.footer_text,
            "github_url": self.github_url,
            "figma_url": self.figma_url,
            "show_theme_toggle": self.show_theme_toggle,
            "show_search": self.show_search,
        }


# =============================================================================
# Component Library
# =============================================================================

class ComponentLibrary:
    """
    A component library for organizing and documenting UI components.

    Example:
        library = ComponentLibrary("My Design System")

        # Add components
        library.add_component(button_spec, category="Inputs")
        library.add_component(card_spec, category="Layout")

        # Add design tokens
        library.add_token_group("colors", color_tokens)

        # Add guide
        library.add_guide("Getting Started", getting_started_content)

        # Generate site
        library.generate_site("output/")
    """

    def __init__(
        self,
        name: str = "Component Library",
        config: LibraryConfig | None = None,
    ):
        """
        Initialize the component library.

        Args:
            name: Name of the library
            config: Optional configuration
        """
        self.config = config or LibraryConfig(name=name)
        self.components: dict[str, ComponentEntry] = {}
        self.categories: dict[str, CategoryEntry] = {}
        self.guides: dict[str, GuidePage] = {}
        self.token_groups: dict[str, TokenGroup] = {}

    @property
    def name(self) -> str:
        """Get library name."""
        return self.config.name

    def add_component(
        self,
        spec: VisualSpec,
        name: str | None = None,
        description: str = "",
        category: str = "Components",
        status: ComponentStatus = ComponentStatus.STABLE,
        tags: list[str] | None = None,
        **kwargs,
    ) -> ComponentEntry:
        """
        Add a component to the library.

        Args:
            spec: The VisualSpec for the component
            name: Display name (defaults to component_id)
            description: Component description
            category: Category to organize under
            status: Component status
            tags: Search/filter tags
            **kwargs: Additional ComponentEntry fields

        Returns:
            The created ComponentEntry
        """
        component_name = name or spec.component_id

        entry = ComponentEntry(
            name=component_name,
            description=description,
            spec=spec,
            category=category,
            status=status,
            tags=tags or [],
            **kwargs,
        )

        self.components[entry.slug] = entry

        # Add to category
        if category not in self.categories:
            self.categories[category] = CategoryEntry(name=category)
        if entry.slug not in self.categories[category].components:
            self.categories[category].components.append(entry.slug)

        return entry

    def add_category(
        self,
        name: str,
        description: str = "",
        icon: str = "",
        order: int = 0,
    ) -> CategoryEntry:
        """
        Add or update a category.

        Args:
            name: Category name
            description: Category description
            icon: Icon (emoji or URL)
            order: Sort order

        Returns:
            The CategoryEntry
        """
        if name in self.categories:
            entry = self.categories[name]
            entry.description = description or entry.description
            entry.icon = icon or entry.icon
            entry.order = order
        else:
            entry = CategoryEntry(
                name=name,
                description=description,
                icon=icon,
                order=order,
            )
            self.categories[name] = entry

        return entry

    def add_guide(
        self,
        title: str,
        content: str,
        slug: str | None = None,
        category: str = "Guides",
        order: int = 0,
    ) -> GuidePage:
        """
        Add a documentation guide.

        Args:
            title: Guide title
            content: Markdown or HTML content
            slug: URL slug (defaults from title)
            category: Guide category
            order: Sort order

        Returns:
            The GuidePage
        """
        guide_slug = slug or title.lower().replace(" ", "-")

        guide = GuidePage(
            title=title,
            slug=guide_slug,
            content=content,
            category=category,
            order=order,
        )

        self.guides[guide_slug] = guide
        return guide

    def add_token_group(
        self,
        name: str,
        group: TokenGroup,
    ) -> None:
        """
        Add a design token group to document.

        Args:
            name: Group name (e.g., "colors", "spacing")
            group: The TokenGroup
        """
        self.token_groups[name] = group

    def get_component(self, slug: str) -> ComponentEntry | None:
        """Get a component by slug."""
        return self.components.get(slug)

    def get_components_by_category(self, category: str) -> list[ComponentEntry]:
        """Get all components in a category."""
        return [
            self.components[slug]
            for slug in self.categories.get(category, CategoryEntry(name=category)).components
            if slug in self.components
        ]

    def search_components(self, query: str) -> list[ComponentEntry]:
        """
        Search components by name, description, or tags.

        Args:
            query: Search query

        Returns:
            Matching components
        """
        query_lower = query.lower()
        results = []

        for entry in self.components.values():
            if (
                query_lower in entry.name.lower() or
                query_lower in entry.description.lower() or
                any(query_lower in tag.lower() for tag in entry.tags)
            ):
                results.append(entry)

        return results

    def to_dict(self) -> dict:
        """Export library to dictionary."""
        return {
            "config": self.config.to_dict(),
            "components": {
                slug: entry.to_dict()
                for slug, entry in self.components.items()
            },
            "categories": {
                name: cat.to_dict()
                for name, cat in self.categories.items()
            },
            "guides": {
                slug: guide.to_dict()
                for slug, guide in self.guides.items()
            },
            "token_groups": list(self.token_groups.keys()),
        }

    def to_json(self, indent: int = 2) -> str:
        """Export library to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> None:
        """Save library metadata to JSON file."""
        path = Path(path)
        path.write_text(self.to_json())


# =============================================================================
# HTML Generation
# =============================================================================

def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return html.escape(str(text))


def generate_css(config: LibraryConfig) -> str:
    """Generate CSS for the documentation site."""
    return f"""
:root {{
    --primary: {config.primary_color};
    --primary-dark: color-mix(in srgb, {config.primary_color} 80%, black);
    --primary-light: color-mix(in srgb, {config.primary_color} 20%, white);
    --bg: #ffffff;
    --bg-secondary: #f8fafc;
    --text: #1e293b;
    --text-secondary: #64748b;
    --border: #e2e8f0;
    --radius: 8px;
    --shadow: 0 1px 3px rgba(0,0,0,0.1);
}}

[data-theme="dark"] {{
    --bg: #0f172a;
    --bg-secondary: #1e293b;
    --text: #f1f5f9;
    --text-secondary: #94a3b8;
    --border: #334155;
}}

* {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}}

body {{
    font-family: system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
}}

.container {{
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 24px;
}}

/* Header */
.header {{
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
    padding: 16px 0;
    position: sticky;
    top: 0;
    z-index: 100;
}}

.header-content {{
    display: flex;
    align-items: center;
    justify-content: space-between;
}}

.logo {{
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--primary);
    text-decoration: none;
}}

.nav {{
    display: flex;
    gap: 24px;
}}

.nav a {{
    color: var(--text-secondary);
    text-decoration: none;
    transition: color 0.2s;
}}

.nav a:hover {{
    color: var(--primary);
}}

/* Sidebar */
.layout {{
    display: grid;
    grid-template-columns: 250px 1fr;
    gap: 48px;
    padding: 32px 0;
}}

.sidebar {{
    position: sticky;
    top: 80px;
    height: fit-content;
}}

.sidebar-section {{
    margin-bottom: 24px;
}}

.sidebar-title {{
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
    margin-bottom: 8px;
}}

.sidebar-links {{
    list-style: none;
}}

.sidebar-links a {{
    display: block;
    padding: 6px 12px;
    color: var(--text);
    text-decoration: none;
    border-radius: var(--radius);
    transition: background 0.2s;
}}

.sidebar-links a:hover {{
    background: var(--bg-secondary);
}}

.sidebar-links a.active {{
    background: var(--primary-light);
    color: var(--primary);
}}

/* Content */
.content {{
    min-width: 0;
}}

.page-title {{
    font-size: 2rem;
    margin-bottom: 8px;
}}

.page-description {{
    color: var(--text-secondary);
    margin-bottom: 32px;
}}

/* Component Card */
.component-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 24px;
}}

.component-card {{
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 24px;
    transition: box-shadow 0.2s;
}}

.component-card:hover {{
    box-shadow: var(--shadow);
}}

.component-card h3 {{
    margin-bottom: 8px;
}}

.component-card p {{
    color: var(--text-secondary);
    font-size: 0.875rem;
}}

.component-card a {{
    color: inherit;
    text-decoration: none;
}}

/* Status Badge */
.status-badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 500;
}}

.status-stable {{ background: #dcfce7; color: #166534; }}
.status-beta {{ background: #fef3c7; color: #92400e; }}
.status-draft {{ background: #e2e8f0; color: #475569; }}
.status-deprecated {{ background: #fee2e2; color: #991b1b; }}

[data-theme="dark"] .status-stable {{ background: #166534; color: #dcfce7; }}
[data-theme="dark"] .status-beta {{ background: #92400e; color: #fef3c7; }}
[data-theme="dark"] .status-draft {{ background: #475569; color: #e2e8f0; }}
[data-theme="dark"] .status-deprecated {{ background: #991b1b; color: #fee2e2; }}

/* Props Table */
.props-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 24px 0;
}}

.props-table th,
.props-table td {{
    padding: 12px;
    text-align: left;
    border-bottom: 1px solid var(--border);
}}

.props-table th {{
    background: var(--bg-secondary);
    font-weight: 600;
}}

.props-table code {{
    background: var(--bg-secondary);
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.875rem;
}}

/* Code Block */
.code-block {{
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px;
    overflow-x: auto;
    font-family: ui-monospace, monospace;
    font-size: 0.875rem;
    margin: 16px 0;
}}

/* Preview */
.preview-container {{
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 32px;
    margin: 24px 0;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 120px;
}}

/* Token Swatch */
.token-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    gap: 16px;
}}

.token-swatch {{
    text-align: center;
}}

.token-swatch .color {{
    width: 100%;
    height: 60px;
    border-radius: var(--radius);
    border: 1px solid var(--border);
    margin-bottom: 8px;
}}

.token-swatch .name {{
    font-size: 0.75rem;
    font-weight: 500;
}}

.token-swatch .value {{
    font-size: 0.75rem;
    color: var(--text-secondary);
}}

/* Search */
.search-box {{
    position: relative;
    margin-bottom: 24px;
}}

.search-input {{
    width: 100%;
    padding: 12px 16px;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--bg);
    color: var(--text);
    font-size: 1rem;
}}

.search-input:focus {{
    outline: none;
    border-color: var(--primary);
}}

/* Theme Toggle */
.theme-toggle {{
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 8px 12px;
    cursor: pointer;
    color: var(--text);
}}

/* Footer */
.footer {{
    border-top: 1px solid var(--border);
    padding: 24px 0;
    margin-top: 48px;
    text-align: center;
    color: var(--text-secondary);
    font-size: 0.875rem;
}}

/* Responsive */
@media (max-width: 768px) {{
    .layout {{
        grid-template-columns: 1fr;
    }}
    .sidebar {{
        display: none;
    }}
}}

{config.custom_css}
"""


def generate_component_preview_html(spec: VisualSpec) -> str:
    """Generate HTML preview for a component based on its spec."""
    # Create a basic preview based on component type
    comp_type = spec.component_type.lower()
    preview_id = f"preview-{spec.component_id}"

    # Get colors from tokens
    bg_color = "#3b82f6"
    text_color = "#ffffff"

    if spec.tokens:
        if "background" in spec.tokens:
            bg_token = spec.tokens["background"]
            bg_val = bg_token.value if hasattr(bg_token, 'value') else bg_token
            if hasattr(bg_val, 'to_hex'):
                bg_color = bg_val.to_hex()
            elif hasattr(bg_val, 'to_css'):
                bg_color = bg_val.to_css()
        if "color" in spec.tokens:
            text_token = spec.tokens["color"]
            text_val = text_token.value if hasattr(text_token, 'value') else text_token
            if hasattr(text_val, 'to_hex'):
                text_color = text_val.to_hex()
            elif hasattr(text_val, 'to_css'):
                text_color = text_val.to_css()

    styles = f"background: {bg_color}; color: {text_color};"

    if comp_type == "button":
        return f'''<button id="{preview_id}" style="{styles} padding: 12px 24px; border: none; border-radius: 6px; font-weight: 500; cursor: pointer;">{escape_html(spec.component_id)}</button>'''
    elif comp_type == "textbox":
        return f'''<input id="{preview_id}" type="text" placeholder="Enter text..." style="padding: 12px; border: 1px solid #e2e8f0; border-radius: 6px; width: 200px;">'''
    elif comp_type == "card":
        return f'''<div id="{preview_id}" style="{styles} padding: 24px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">Card Content</div>'''
    else:
        return f'''<div id="{preview_id}" style="{styles} padding: 16px; border-radius: 6px;">{escape_html(spec.component_type)}</div>'''


def generate_header_html(config: LibraryConfig, current_page: str = "") -> str:
    """Generate header HTML."""
    theme_toggle = ""
    if config.show_theme_toggle:
        theme_toggle = '''<button class="theme-toggle" onclick="toggleTheme()">🌓</button>'''

    links = ""
    if config.github_url:
        links += f'<a href="{escape_html(config.github_url)}" target="_blank">GitHub</a>'
    if config.figma_url:
        links += f'<a href="{escape_html(config.figma_url)}" target="_blank">Figma</a>'

    return f'''
<header class="header">
    <div class="container header-content">
        <a href="index.html" class="logo">{escape_html(config.name)}</a>
        <nav class="nav">
            <a href="index.html">Components</a>
            <a href="tokens.html">Tokens</a>
            <a href="guides.html">Guides</a>
            {links}
            {theme_toggle}
        </nav>
    </div>
</header>
'''


def generate_sidebar_html(
    library: ComponentLibrary,
    current_slug: str = "",
) -> str:
    """Generate sidebar navigation HTML."""
    sections = []

    # Categories
    sorted_categories = sorted(
        library.categories.values(),
        key=lambda c: (c.order, c.name),
    )

    for category in sorted_categories:
        links = []
        for comp_slug in category.components:
            comp = library.components.get(comp_slug)
            if comp:
                active = "active" if comp_slug == current_slug else ""
                links.append(
                    f'<li><a href="{comp_slug}.html" class="{active}">{escape_html(comp.name)}</a></li>'
                )

        if links:
            icon = f"{category.icon} " if category.icon else ""
            sections.append(f'''
<div class="sidebar-section">
    <h4 class="sidebar-title">{icon}{escape_html(category.name)}</h4>
    <ul class="sidebar-links">
        {"".join(links)}
    </ul>
</div>
''')

    return f'<nav class="sidebar">{"".join(sections)}</nav>'


def generate_props_table_html(props: dict[str, dict]) -> str:
    """Generate props documentation table."""
    if not props:
        return ""

    rows = []
    for name, info in props.items():
        required = "Required" if info.get("required") else "Optional"
        default = f"<code>{escape_html(str(info.get('default', '-')))}</code>"
        rows.append(f'''
<tr>
    <td><code>{escape_html(name)}</code></td>
    <td><code>{escape_html(info.get('type', 'any'))}</code></td>
    <td>{default}</td>
    <td>{required}</td>
    <td>{escape_html(info.get('description', ''))}</td>
</tr>
''')

    return f'''
<h3>Props</h3>
<table class="props-table">
    <thead>
        <tr>
            <th>Name</th>
            <th>Type</th>
            <th>Default</th>
            <th>Required</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        {"".join(rows)}
    </tbody>
</table>
'''


def generate_variants_html(variants: list[ComponentVariant]) -> str:
    """Generate variants section HTML."""
    if not variants:
        return ""

    items = []
    for variant in variants:
        items.append(f'''
<div class="component-card">
    <h4>{escape_html(variant.name)}</h4>
    <p>{escape_html(variant.description)}</p>
</div>
''')

    return f'''
<h3>Variants</h3>
<div class="component-grid">
    {"".join(items)}
</div>
'''


def generate_page_html(
    config: LibraryConfig,
    title: str,
    content: str,
    sidebar: str = "",
) -> str:
    """Generate a full page HTML."""
    css = generate_css(config)
    header = generate_header_html(config)

    layout_content = f'''
<div class="layout">
    {sidebar}
    <main class="content">
        {content}
    </main>
</div>
''' if sidebar else f'<main class="content container" style="padding: 32px 0;">{content}</main>'

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{escape_html(title)} - {escape_html(config.name)}</title>
    <style>{css}</style>
    {config.custom_head}
</head>
<body>
    {header}
    <div class="container">
        {layout_content}
    </div>
    <footer class="footer">
        {escape_html(config.footer_text or f"Built with integradio • {config.name} v{config.version}")}
    </footer>
    <script>
        function toggleTheme() {{
            const html = document.documentElement;
            const currentTheme = html.getAttribute('data-theme');
            html.setAttribute('data-theme', currentTheme === 'dark' ? 'light' : 'dark');
            localStorage.setItem('theme', html.getAttribute('data-theme'));
        }}
        // Load saved theme
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme) {{
            document.documentElement.setAttribute('data-theme', savedTheme);
        }}
    </script>
</body>
</html>
'''


# =============================================================================
# Site Generator
# =============================================================================

class LibrarySiteGenerator:
    """
    Generates a static documentation site from a ComponentLibrary.

    Example:
        generator = LibrarySiteGenerator(library)
        generator.generate("output/")
    """

    def __init__(self, library: ComponentLibrary):
        """
        Initialize the generator.

        Args:
            library: The component library to generate docs for
        """
        self.library = library
        self.config = library.config

    def generate(self, output_dir: str | Path) -> Path:
        """
        Generate the static site.

        Args:
            output_dir: Directory to output files to

        Returns:
            Path to the output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate pages
        self._generate_index_page(output_path)
        self._generate_component_pages(output_path)
        self._generate_tokens_page(output_path)
        self._generate_guides_page(output_path)

        # Save library data as JSON for search
        data_path = output_path / "library-data.json"
        data_path.write_text(self.library.to_json())

        return output_path

    def _generate_index_page(self, output_path: Path) -> None:
        """Generate the index/home page."""
        # Component cards
        cards = []
        for entry in self.library.components.values():
            status_class = f"status-{entry.status.value}"
            cards.append(f'''
<div class="component-card">
    <a href="{entry.slug}.html">
        <span class="status-badge {status_class}">{entry.status.value}</span>
        <h3>{escape_html(entry.name)}</h3>
        <p>{escape_html(entry.description[:100])}{"..." if len(entry.description) > 100 else ""}</p>
    </a>
</div>
''')

        search_box = ""
        if self.config.show_search:
            search_box = '''
<div class="search-box">
    <input type="text" class="search-input" placeholder="Search components..." id="search-input" onkeyup="filterComponents()">
</div>
<script>
function filterComponents() {
    const query = document.getElementById('search-input').value.toLowerCase();
    document.querySelectorAll('.component-card').forEach(card => {
        const text = card.textContent.toLowerCase();
        card.style.display = text.includes(query) ? '' : 'none';
    });
}
</script>
'''

        content = f'''
<h1 class="page-title">{escape_html(self.config.name)}</h1>
<p class="page-description">{escape_html(self.config.description)}</p>

{search_box}

<div class="component-grid" id="component-grid">
    {"".join(cards)}
</div>
'''

        sidebar = generate_sidebar_html(self.library)
        html = generate_page_html(self.config, "Home", content, sidebar)

        (output_path / "index.html").write_text(html, encoding="utf-8")

    def _generate_component_pages(self, output_path: Path) -> None:
        """Generate individual component pages."""
        for entry in self.library.components.values():
            preview = generate_component_preview_html(entry.spec)
            props_table = generate_props_table_html(entry.props)
            variants = generate_variants_html(entry.variants)

            code_block = ""
            if entry.code_example:
                code_block = f'''
<h3>Usage</h3>
<pre class="code-block"><code>{escape_html(entry.code_example)}</code></pre>
'''

            usage_notes = ""
            if entry.usage_notes:
                usage_notes = f'''
<h3>Notes</h3>
<p>{escape_html(entry.usage_notes)}</p>
'''

            related = ""
            if entry.related_components:
                links = ", ".join(
                    f'<a href="{slug}.html">{slug}</a>'
                    for slug in entry.related_components
                )
                related = f"<h3>Related Components</h3><p>{links}</p>"

            status_class = f"status-{entry.status.value}"
            tags_html = ""
            if entry.tags:
                tags_html = f'''<p style="margin-top: 8px;">Tags: {", ".join(escape_html(t) for t in entry.tags)}</p>'''

            content = f'''
<span class="status-badge {status_class}">{entry.status.value}</span>
<h1 class="page-title">{escape_html(entry.name)}</h1>
<p class="page-description">{escape_html(entry.description)}</p>
{tags_html}

<h3>Preview</h3>
<div class="preview-container">
    {preview}
</div>

{code_block}
{props_table}
{variants}
{usage_notes}
{related}
'''

            sidebar = generate_sidebar_html(self.library, entry.slug)
            html = generate_page_html(self.config, entry.name, content, sidebar)

            (output_path / f"{entry.slug}.html").write_text(html, encoding="utf-8")

    def _generate_tokens_page(self, output_path: Path) -> None:
        """Generate design tokens page."""
        sections = []

        for name, group in self.library.token_groups.items():
            swatches = []
            for token_name, token in group.tokens.items():
                if group.type == TokenType.COLOR:
                    value = token.value
                    if hasattr(value, 'to_hex'):
                        hex_color = value.to_hex()
                    elif hasattr(value, 'to_css'):
                        hex_color = value.to_css()
                    else:
                        hex_color = str(value)

                    swatches.append(f'''
<div class="token-swatch">
    <div class="color" style="background: {hex_color};"></div>
    <div class="name">{escape_html(token_name)}</div>
    <div class="value">{escape_html(hex_color)}</div>
</div>
''')
                else:
                    # Non-color tokens
                    value_str = str(token.value)
                    if hasattr(token.value, 'to_css'):
                        value_str = token.value.to_css()

                    swatches.append(f'''
<div class="token-swatch">
    <div class="name">{escape_html(token_name)}</div>
    <div class="value">{escape_html(value_str)}</div>
</div>
''')

            sections.append(f'''
<h2>{escape_html(name.title())}</h2>
<div class="token-grid">
    {"".join(swatches)}
</div>
''')

        if not sections:
            sections.append("<p>No design tokens documented yet.</p>")

        content = f'''
<h1 class="page-title">Design Tokens</h1>
<p class="page-description">The foundational values that power the design system.</p>

{"".join(sections)}
'''

        html = generate_page_html(self.config, "Design Tokens", content)
        (output_path / "tokens.html").write_text(html, encoding="utf-8")

    def _generate_guides_page(self, output_path: Path) -> None:
        """Generate guides page."""
        if not self.library.guides:
            content = '''
<h1 class="page-title">Guides</h1>
<p class="page-description">Documentation and usage guides.</p>
<p>No guides have been added yet.</p>
'''
        else:
            guide_links = []
            for guide in sorted(
                self.library.guides.values(),
                key=lambda g: (g.order, g.title),
            ):
                guide_links.append(f'''
<div class="component-card">
    <a href="guide-{guide.slug}.html">
        <h3>{escape_html(guide.title)}</h3>
    </a>
</div>
''')

            content = f'''
<h1 class="page-title">Guides</h1>
<p class="page-description">Documentation and usage guides.</p>

<div class="component-grid">
    {"".join(guide_links)}
</div>
'''

            # Generate individual guide pages
            for guide in self.library.guides.values():
                guide_content = f'''
<h1 class="page-title">{escape_html(guide.title)}</h1>
<div class="guide-content">
    {guide.content}
</div>
'''
                guide_html = generate_page_html(self.config, guide.title, guide_content)
                (output_path / f"guide-{guide.slug}.html").write_text(guide_html, encoding="utf-8")

        html = generate_page_html(self.config, "Guides", content)
        (output_path / "guides.html").write_text(html, encoding="utf-8")


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_library_site(
    library: ComponentLibrary,
    output_dir: str | Path,
) -> Path:
    """
    Generate a static documentation site from a component library.

    Args:
        library: The component library
        output_dir: Output directory path

    Returns:
        Path to the generated site
    """
    generator = LibrarySiteGenerator(library)
    return generator.generate(output_dir)


def create_library_from_specs(
    specs: list[VisualSpec],
    name: str = "Component Library",
    **config_kwargs,
) -> ComponentLibrary:
    """
    Create a component library from a list of VisualSpecs.

    Args:
        specs: List of VisualSpec objects
        name: Library name
        **config_kwargs: Additional LibraryConfig options

    Returns:
        ComponentLibrary with all specs added
    """
    config = LibraryConfig(name=name, **config_kwargs)
    library = ComponentLibrary(config=config)

    for spec in specs:
        library.add_component(spec)

    return library


def create_library_from_ui_spec(
    ui_spec: UISpec,
    name: str | None = None,
) -> ComponentLibrary:
    """
    Create a component library from a UISpec.

    Args:
        ui_spec: The UISpec containing components
        name: Library name (defaults to UISpec name)

    Returns:
        ComponentLibrary with all components from UISpec
    """
    library_name = name or ui_spec.name
    config = LibraryConfig(
        name=library_name,
        version=ui_spec.version,
    )
    library = ComponentLibrary(config=config)

    # Add components from all pages
    for page in ui_spec.pages.values():
        for comp in page.components.values():
            library.add_component(
                comp,
                category=page.name,
            )

    # Add design tokens
    if ui_spec.tokens:
        library.add_token_group("tokens", ui_spec.tokens)

    return library


def quick_library(
    *specs: VisualSpec,
    name: str = "Component Library",
) -> ComponentLibrary:
    """
    Quickly create a component library from specs.

    Args:
        *specs: VisualSpec objects to include
        name: Library name

    Returns:
        ComponentLibrary
    """
    return create_library_from_specs(list(specs), name=name)
