"""
Full Integradio App - Demonstrates all 10 page templates.

Run with: python examples/full_app.py
"""

import gradio as gr
from integradio import SemanticBlocks, semantic
from integradio.pages import (
    ChatPage,
    SettingsPage,
    GalleryPage,
    HeroPage,
    HelpPage,
    DashboardPage,
    FormPage,
    DataTablePage,
    UploadPage,
    AnalyticsPage,
    # Factory functions for embedding
    create_chat_interface,
    create_settings_panel,
    create_gallery_grid,
    create_hero_section,
    create_help_center,
    create_dashboard,
    create_form_wizard,
    create_data_table,
    create_upload_center,
    create_analytics_dashboard,
)


def create_full_app():
    """Create a complete app with all page types in tabs."""

    with SemanticBlocks(title="Integradio Demo") as demo:

        gr.Markdown("# Integradio - Full Demo")
        gr.Markdown("All 10 page templates with semantic vector tracking")

        with gr.Tabs():
            # 1. Hero/Landing
            with gr.Tab("Home"):
                create_hero_section()

            # 2. Dashboard
            with gr.Tab("Dashboard"):
                create_dashboard()

            # 3. Chat
            with gr.Tab("Chat"):
                create_chat_interface()

            # 4. Gallery
            with gr.Tab("Gallery"):
                create_gallery_grid()

            # 5. Analytics
            with gr.Tab("Analytics"):
                create_analytics_dashboard()

            # 6. Data Table
            with gr.Tab("Data"):
                create_data_table()

            # 7. Upload
            with gr.Tab("Upload"):
                create_upload_center()

            # 8. Form
            with gr.Tab("Form"):
                create_form_wizard()

            # 9. Settings
            with gr.Tab("Settings"):
                create_settings_panel()

            # 10. Help
            with gr.Tab("Help"):
                create_help_center()

        # Registry info footer
        gr.Markdown("---")
        with gr.Accordion("Component Registry", open=False):
            info_btn = semantic(
                gr.Button("Show Registry Summary"),
                intent="displays semantic component registry info",
            )
            info_output = semantic(
                gr.Code(language=None, label="Registry"),
                intent="shows all registered components",
            )

            def show_registry():
                return demo.summary()

            info_btn.click(fn=show_registry, outputs=info_output)

            search_input = semantic(
                gr.Textbox(placeholder="Search components by intent...", label="Search"),
                intent="searches components by semantic query",
            )
            search_output = semantic(
                gr.JSON(label="Search Results"),
                intent="displays component search results",
            )

            def search_components(query):
                if not query:
                    return []
                results = demo.search(query, k=5)
                return [
                    {
                        "id": r.component_id,
                        "type": r.metadata.component_type,
                        "intent": r.metadata.intent,
                        "score": round(r.score, 3),
                    }
                    for r in results
                ]

            search_input.submit(fn=search_components, inputs=search_input, outputs=search_output)

    return demo


if __name__ == "__main__":
    demo = create_full_app()

    print("\n" + "=" * 70)
    print("INTEGRADIO - Full Demo")
    print("=" * 70)
    try:
        print(demo.summary())
    except UnicodeEncodeError:
        print("(Summary contains unicode - view in browser)")
    print("=" * 70)
    print("\nLaunching at http://localhost:7860")
    print("=" * 70 + "\n")

    demo.launch(server_port=7860, share=False, theme=gr.themes.Soft())
