"""
Basic Integradio Example - Search application with semantic component tracking.

Run with: python examples/basic_app.py
Then try:
    - Search for components: demo.search("where does user input go")
    - View the graph: open http://localhost:7860/semantic/graph
"""

import gradio as gr
from integradio import SemanticBlocks, semantic


def search_fn(query: str, num_results: int) -> str:
    """Simulate a search function."""
    if not query:
        return "Please enter a search query"
    return f"Found {num_results} results for: **{query}**\n\n" + "\n".join(
        f"- Result {i+1}: Match for '{query}'" for i in range(num_results)
    )


def clear_fn():
    """Clear all inputs."""
    return "", 5, ""


# Create the app with SemanticBlocks
with SemanticBlocks(
    title="Semantic Search Demo",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown("# Integradio Search Demo")
    gr.Markdown("Components are embedded with vectors for semantic search!")

    with gr.Row():
        with gr.Column(scale=3):
            # Wrap components with semantic() and provide intent
            query_input = semantic(
                gr.Textbox(
                    label="Search Query",
                    placeholder="Enter your search...",
                    elem_id="search-input",
                ),
                intent="user enters search query text",
                tags=["primary-input", "search"],
            )

        with gr.Column(scale=1):
            num_results = semantic(
                gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Number of Results",
                ),
                intent="controls how many search results to show",
            )

    with gr.Row():
        search_btn = semantic(
            gr.Button("Search", variant="primary"),
            intent="triggers the search operation",
            tags=["action", "primary"],
        )
        clear_btn = semantic(
            gr.Button("Clear", variant="secondary"),
            intent="clears all input fields",
            tags=["action", "reset"],
        )

    results_output = semantic(
        gr.Markdown(
            value="Results will appear here...",
            elem_id="results-display",
        ),
        intent="displays search results to user",
        tags=["output", "results"],
    )

    # Wire up events (these create dataflow relationships)
    search_btn.click(
        fn=search_fn,
        inputs=[query_input, num_results],
        outputs=results_output,
    )

    clear_btn.click(
        fn=clear_fn,
        outputs=[query_input, num_results, results_output],
    )

    # Also search on Enter
    query_input.submit(
        fn=search_fn,
        inputs=[query_input, num_results],
        outputs=results_output,
    )

    # Add a section showing component info
    with gr.Accordion("Component Registry Info", open=False):
        info_btn = semantic(
            gr.Button("Show Registered Components"),
            intent="displays component registry summary",
        )
        info_output = semantic(
            gr.Code(language="json", label="Registry"),
            intent="shows component metadata as JSON",
        )

        def show_info():
            return demo.summary()

        info_btn.click(fn=show_info, outputs=info_output)


if __name__ == "__main__":
    # Print summary before launch
    print("\n" + "=" * 60)
    print("INTEGRADIO - Component Registry")
    print("=" * 60)
    print(demo.summary())
    print("=" * 60 + "\n")

    # Launch the app
    demo.launch(
        server_port=7860,
        share=False,
    )
