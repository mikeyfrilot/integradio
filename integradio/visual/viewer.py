"""
Visual Spec Viewer - Gradio UI for browsing and editing visual specifications.

This provides an interactive interface to:
- Browse components and their visual properties
- Edit design tokens with live preview
- View responsive breakpoint behavior
- Export to CSS/Style Dictionary
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

try:
    import gradio as gr
except ImportError:
    gr = None  # type: ignore

from .tokens import (
    DesignToken,
    TokenType,
    TokenGroup,
    ColorValue,
    DimensionValue,
    TypographyValue,
)
from .spec import (
    VisualSpec,
    PageSpec,
    UISpec,
    LayoutSpec,
    SpacingSpec,
    Display,
    Position,
    FlexSpec,
    BREAKPOINTS,
)


class VisualSpecViewer:
    """
    Gradio-based viewer for visual specifications.

    Usage:
        viewer = VisualSpecViewer(spec)
        viewer.launch()
    """

    def __init__(
        self,
        spec: UISpec | None = None,
        spec_path: str | Path | None = None,
    ):
        """
        Initialize the viewer.

        Args:
            spec: UISpec to view/edit
            spec_path: Path to load/save spec JSON
        """
        if gr is None:
            raise ImportError("Gradio is required for VisualSpecViewer. Install with: pip install gradio")

        self.spec_path = Path(spec_path) if spec_path else None

        if spec:
            self.spec = spec
        elif self.spec_path and self.spec_path.exists():
            self.spec = UISpec.load(self.spec_path)
        else:
            self.spec = UISpec(name="New UI Spec")

        self._app: gr.Blocks | None = None

    def _get_pages_list(self) -> list[str]:
        """Get list of page names."""
        return list(self.spec.pages.keys()) or ["(no pages)"]

    def _get_components_for_page(self, page_route: str) -> list[str]:
        """Get component IDs for a page."""
        page = self.spec.pages.get(page_route)
        if page:
            return list(page.components.keys()) or ["(no components)"]
        return ["(no components)"]

    def _get_component_spec(self, page_route: str, component_id: str) -> VisualSpec | None:
        """Get a component's visual spec."""
        page = self.spec.pages.get(page_route)
        if page:
            return page.components.get(component_id)
        return None

    def _render_preview(self, page_route: str, component_id: str) -> str:
        """Generate HTML preview for a component."""
        spec = self._get_component_spec(page_route, component_id)
        if not spec:
            return "<p>Select a component to preview</p>"

        # Generate CSS
        css = spec.to_css(f"#preview-{component_id}")

        # Generate placeholder HTML based on component type
        comp_type = spec.component_type.lower()
        inner_html = self._generate_placeholder_html(comp_type, component_id)

        return f"""
        <style>
            {css}
            #preview-container {{
                padding: 20px;
                background: #f5f5f5;
                border-radius: 8px;
                min-height: 100px;
            }}
        </style>
        <div id="preview-container">
            <div id="preview-{component_id}">
                {inner_html}
            </div>
        </div>
        """

    def _generate_placeholder_html(self, comp_type: str, component_id: str) -> str:
        """Generate placeholder HTML for different component types."""
        placeholders = {
            "button": '<button type="button">Button</button>',
            "textbox": '<input type="text" placeholder="Enter text..." />',
            "textarea": '<textarea placeholder="Enter text..."></textarea>',
            "dropdown": '<select><option>Option 1</option><option>Option 2</option></select>',
            "checkbox": '<label><input type="checkbox" /> Checkbox</label>',
            "radio": '<label><input type="radio" name="r" /> Radio</label>',
            "slider": '<input type="range" min="0" max="100" />',
            "markdown": '<div class="markdown"><h3>Heading</h3><p>Paragraph text</p></div>',
            "image": '<div style="width:100px;height:100px;background:#ddd;display:flex;align-items:center;justify-content:center;">Image</div>',
            "chatbot": '<div class="chat"><div class="msg user">User message</div><div class="msg bot">Bot response</div></div>',
        }
        return placeholders.get(comp_type, f'<div>{comp_type or "Component"}</div>')

    def _tokens_to_json(self, page_route: str, component_id: str) -> str:
        """Get component tokens as JSON string."""
        spec = self._get_component_spec(page_route, component_id)
        if spec:
            return json.dumps({k: v.to_dtcg() for k, v in spec.tokens.items()}, indent=2)
        return "{}"

    def _layout_to_json(self, page_route: str, component_id: str) -> str:
        """Get component layout as JSON string."""
        spec = self._get_component_spec(page_route, component_id)
        if spec:
            return json.dumps(spec.layout.to_css(), indent=2)
        return "{}"

    def _update_token_color(
        self,
        page_route: str,
        component_id: str,
        token_name: str,
        hex_color: str,
    ) -> str:
        """Update a color token and return new preview."""
        spec = self._get_component_spec(page_route, component_id)
        if spec and hex_color:
            spec.tokens[token_name] = DesignToken.color(hex_color)
        return self._render_preview(page_route, component_id)

    def _generate_full_css(self, theme: str | None = None) -> str:
        """Generate CSS for the entire spec."""
        return self.spec.to_css(theme if theme != "(none)" else None)

    def _export_style_dictionary(self) -> str:
        """Export as Style Dictionary JSON."""
        return json.dumps(self.spec.to_dict(), indent=2)

    def _save_spec(self) -> str:
        """Save the spec to file."""
        if self.spec_path:
            self.spec.save(self.spec_path)
            return f"Saved to {self.spec_path}"
        return "No save path configured"

    def _add_page(self, name: str, route: str) -> list[str]:
        """Add a new page."""
        page = PageSpec(name=name, route=route)
        self.spec.add_page(page)
        return self._get_pages_list()

    def _add_component(
        self,
        page_route: str,
        component_id: str,
        component_type: str,
    ) -> list[str]:
        """Add a new component to a page."""
        page = self.spec.pages.get(page_route)
        if page:
            spec = VisualSpec(
                component_id=component_id,
                component_type=component_type,
            )
            page.add_component(spec)
        return self._get_components_for_page(page_route)

    def build(self) -> gr.Blocks:
        """Build the Gradio interface."""
        with gr.Blocks(
            title="Visual Spec Viewer",
            theme=gr.themes.Soft(),
            css="""
                .token-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 1rem; }
                .color-swatch { width: 40px; height: 40px; border-radius: 4px; border: 1px solid #ccc; }
            """,
        ) as app:
            gr.Markdown("# 🎨 Visual Spec Viewer")
            gr.Markdown("Browse and edit visual specifications for your Gradio components.")

            with gr.Tabs():
                # =============================================================
                # Tab 1: Component Browser
                # =============================================================
                with gr.Tab("📦 Components"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            page_dropdown = gr.Dropdown(
                                choices=self._get_pages_list(),
                                label="Page",
                                value=self._get_pages_list()[0] if self._get_pages_list() else None,
                            )
                            component_dropdown = gr.Dropdown(
                                choices=[],
                                label="Component",
                            )

                            gr.Markdown("### Quick Edit")
                            bg_color = gr.ColorPicker(label="Background", value="#ffffff")
                            text_color = gr.ColorPicker(label="Text", value="#000000")
                            border_color = gr.ColorPicker(label="Border", value="#cccccc")

                        with gr.Column(scale=2):
                            gr.Markdown("### Preview")
                            preview_html = gr.HTML(
                                value="<p>Select a component to preview</p>",
                                label="Live Preview",
                            )

                            with gr.Accordion("Tokens (JSON)", open=False):
                                tokens_json = gr.Code(language="json", label="Design Tokens")

                            with gr.Accordion("Layout (CSS)", open=False):
                                layout_json = gr.Code(language="json", label="Layout Properties")

                    # Event handlers
                    def update_components(page_route):
                        choices = self._get_components_for_page(page_route)
                        return gr.update(choices=choices, value=choices[0] if choices else None)

                    def update_preview(page_route, component_id):
                        return (
                            self._render_preview(page_route, component_id),
                            self._tokens_to_json(page_route, component_id),
                            self._layout_to_json(page_route, component_id),
                        )

                    def update_bg(page_route, component_id, color):
                        return self._update_token_color(page_route, component_id, "background", color)

                    def update_text(page_route, component_id, color):
                        return self._update_token_color(page_route, component_id, "color", color)

                    def update_border(page_route, component_id, color):
                        return self._update_token_color(page_route, component_id, "border-color", color)

                    page_dropdown.change(
                        update_components,
                        inputs=[page_dropdown],
                        outputs=[component_dropdown],
                    )

                    component_dropdown.change(
                        update_preview,
                        inputs=[page_dropdown, component_dropdown],
                        outputs=[preview_html, tokens_json, layout_json],
                    )

                    bg_color.change(
                        update_bg,
                        inputs=[page_dropdown, component_dropdown, bg_color],
                        outputs=[preview_html],
                    )

                    text_color.change(
                        update_text,
                        inputs=[page_dropdown, component_dropdown, text_color],
                        outputs=[preview_html],
                    )

                    border_color.change(
                        update_border,
                        inputs=[page_dropdown, component_dropdown, border_color],
                        outputs=[preview_html],
                    )

                # =============================================================
                # Tab 2: Design Tokens
                # =============================================================
                with gr.Tab("🎨 Design Tokens"):
                    gr.Markdown("### Global Design Tokens")
                    gr.Markdown("These tokens are inherited by all components.")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### Colors")
                            global_tokens_json = gr.Code(
                                value=json.dumps(self.spec.tokens.to_dtcg(), indent=2),
                                language="json",
                                label="Token Definitions (DTCG Format)",
                                interactive=True,
                            )

                        with gr.Column():
                            gr.Markdown("#### Preview")
                            token_preview = gr.HTML(
                                value=self._generate_token_preview(),
                            )

                    def update_global_tokens(tokens_str):
                        # TODO: Parse and update tokens
                        return self._generate_token_preview()

                    global_tokens_json.change(
                        update_global_tokens,
                        inputs=[global_tokens_json],
                        outputs=[token_preview],
                    )

                # =============================================================
                # Tab 3: Responsive
                # =============================================================
                with gr.Tab("📱 Responsive"):
                    gr.Markdown("### Breakpoint Preview")
                    gr.Markdown("See how components look at different screen sizes.")

                    breakpoint_select = gr.Radio(
                        choices=list(BREAKPOINTS.keys()),
                        value="md",
                        label="Breakpoint",
                    )

                    with gr.Row():
                        for bp_name, bp in list(BREAKPOINTS.items())[:3]:
                            with gr.Column():
                                gr.Markdown(f"**{bp_name}** ({bp.min_width}px+)")
                                gr.HTML(
                                    value=f'<div style="border:1px dashed #ccc;padding:10px;min-height:200px;">Preview at {bp_name}</div>',
                                )

                # =============================================================
                # Tab 4: Export
                # =============================================================
                with gr.Tab("📤 Export"):
                    gr.Markdown("### Export Options")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### CSS Output")
                            theme_select = gr.Dropdown(
                                choices=["(none)"] + list(self.spec.themes.keys()),
                                value="(none)",
                                label="Theme",
                            )
                            css_output = gr.Code(
                                value=self._generate_full_css(),
                                language="css",
                                label="Generated CSS",
                            )
                            css_download = gr.Button("📥 Download CSS")

                        with gr.Column():
                            gr.Markdown("#### Style Dictionary")
                            sd_output = gr.Code(
                                value=self._export_style_dictionary(),
                                language="json",
                                label="Style Dictionary JSON",
                            )
                            sd_download = gr.Button("📥 Download JSON")

                    theme_select.change(
                        self._generate_full_css,
                        inputs=[theme_select],
                        outputs=[css_output],
                    )

                # =============================================================
                # Tab 5: Add/Edit
                # =============================================================
                with gr.Tab("➕ Add"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Add Page")
                            new_page_name = gr.Textbox(label="Page Name", placeholder="Home")
                            new_page_route = gr.Textbox(label="Route", placeholder="/")
                            add_page_btn = gr.Button("Add Page", variant="primary")

                        with gr.Column():
                            gr.Markdown("### Add Component")
                            page_for_component = gr.Dropdown(
                                choices=self._get_pages_list(),
                                label="Page",
                            )
                            new_component_id = gr.Textbox(label="Component ID", placeholder="search-button")
                            new_component_type = gr.Dropdown(
                                choices=["Button", "Textbox", "Dropdown", "Chatbot", "Markdown", "Image", "Slider"],
                                label="Component Type",
                            )
                            add_component_btn = gr.Button("Add Component", variant="primary")

                    add_result = gr.Markdown()

                    def add_page_handler(name, route):
                        if name and route:
                            self._add_page(name, route)
                            return f"✅ Added page: {name} ({route})"
                        return "❌ Please fill in both fields"

                    def add_component_handler(page_route, comp_id, comp_type):
                        if page_route and comp_id and comp_type:
                            self._add_component(page_route, comp_id, comp_type)
                            return f"✅ Added component: {comp_id} to {page_route}"
                        return "❌ Please fill in all fields"

                    add_page_btn.click(
                        add_page_handler,
                        inputs=[new_page_name, new_page_route],
                        outputs=[add_result],
                    )

                    add_component_btn.click(
                        add_component_handler,
                        inputs=[page_for_component, new_component_id, new_component_type],
                        outputs=[add_result],
                    )

                # =============================================================
                # Tab 6: Settings
                # =============================================================
                with gr.Tab("⚙️ Settings"):
                    gr.Markdown("### Specification Settings")

                    spec_name = gr.Textbox(
                        value=self.spec.name,
                        label="Spec Name",
                    )
                    spec_version = gr.Textbox(
                        value=self.spec.version,
                        label="Version",
                    )

                    with gr.Row():
                        save_btn = gr.Button("💾 Save Spec", variant="primary")
                        load_btn = gr.Button("📂 Load Spec")

                    save_status = gr.Markdown()

                    save_btn.click(
                        self._save_spec,
                        outputs=[save_status],
                    )

        self._app = app
        return app

    def _generate_token_preview(self) -> str:
        """Generate HTML preview of global tokens."""
        flat = self.spec.tokens.flatten()
        if not flat:
            return "<p>No global tokens defined</p>"

        rows = []
        for path, token in flat.items():
            css_val = token.to_css()
            swatch = ""
            if token.type == TokenType.COLOR:
                swatch = f'<span class="color-swatch" style="background:{css_val};display:inline-block;"></span>'

            rows.append(f"""
                <tr>
                    <td><code>{path}</code></td>
                    <td>{swatch} {css_val}</td>
                    <td>{token.type.value}</td>
                </tr>
            """)

        return f"""
        <table style="width:100%;border-collapse:collapse;">
            <thead>
                <tr style="border-bottom:1px solid #ccc;">
                    <th style="text-align:left;padding:8px;">Token</th>
                    <th style="text-align:left;padding:8px;">Value</th>
                    <th style="text-align:left;padding:8px;">Type</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        """

    def launch(self, **kwargs) -> None:
        """Launch the Gradio interface."""
        if self._app is None:
            self.build()

        default_kwargs = {
            "server_port": 7861,
            "share": False,
        }
        default_kwargs.update(kwargs)

        self._app.launch(**default_kwargs)


# =============================================================================
# Convenience Functions
# =============================================================================

def view_spec(spec: UISpec | str | Path, **kwargs) -> None:
    """
    Quick way to view a visual specification.

    Args:
        spec: UISpec object or path to spec JSON file
        **kwargs: Passed to Gradio launch()
    """
    if isinstance(spec, (str, Path)):
        viewer = VisualSpecViewer(spec_path=spec)
    else:
        viewer = VisualSpecViewer(spec=spec)

    viewer.launch(**kwargs)


def create_viewer_demo() -> gr.Blocks:
    """Create a demo viewer with sample data."""
    # Create sample spec
    spec = UISpec(name="Demo App")

    # Add global tokens
    spec.tokens.add("colors", TokenGroup(type=TokenType.COLOR))
    spec.tokens.get("colors").add("primary", DesignToken.color("#3b82f6", "Primary brand color"))
    spec.tokens.get("colors").add("secondary", DesignToken.color("#64748b", "Secondary color"))
    spec.tokens.get("colors").add("success", DesignToken.color("#22c55e", "Success state"))
    spec.tokens.get("colors").add("error", DesignToken.color("#ef4444", "Error state"))

    spec.tokens.add("spacing", TokenGroup(type=TokenType.DIMENSION))
    spec.tokens.get("spacing").add("sm", DesignToken.dimension(8, "px", "Small spacing"))
    spec.tokens.get("spacing").add("md", DesignToken.dimension(16, "px", "Medium spacing"))
    spec.tokens.get("spacing").add("lg", DesignToken.dimension(24, "px", "Large spacing"))

    # Add a page
    home_page = PageSpec(name="Home", route="/")

    # Add components
    search_box = VisualSpec(
        component_id="search-input",
        component_type="Textbox",
    )
    search_box.set_colors(background="#ffffff", text="#1f2937", border="#d1d5db")
    search_box.set_spacing(padding=DimensionValue(12, "px"))
    search_box.add_transition("border-color", 150)
    home_page.add_component(search_box)

    search_btn = VisualSpec(
        component_id="search-button",
        component_type="Button",
    )
    search_btn.set_colors(background="#3b82f6", text="#ffffff")
    search_btn.set_spacing(padding=SpacingSpec.symmetric(
        vertical=DimensionValue(10, "px"),
        horizontal=DimensionValue(20, "px"),
    ))
    search_btn.add_transition("background", 200)
    home_page.add_component(search_btn)

    results_area = VisualSpec(
        component_id="results-markdown",
        component_type="Markdown",
    )
    results_area.set_colors(background="#f8fafc", text="#334155")
    results_area.set_spacing(padding=DimensionValue(16, "px"))
    home_page.add_component(results_area)

    spec.add_page(home_page)

    # Create viewer
    viewer = VisualSpecViewer(spec=spec)
    return viewer.build()


if __name__ == "__main__":
    # Run demo
    demo = create_viewer_demo()
    demo.launch(server_port=7861)
