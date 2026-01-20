# Integradio

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Gradio 4.0+](https://img.shields.io/badge/gradio-4.0+-orange.svg)](https://gradio.app/)

Vector-embedded Gradio components for semantic codebase navigation.

## Overview

Integradio extends [Gradio](https://gradio.app/) with semantic search capabilities powered by embeddings. Components carry vector representations that make them discoverable by intent rather than by ID or label alone.

**Key Features:**
- Non-invasive component wrapping (works with any Gradio component)
- Semantic search via Ollama/nomic-embed-text
- Automatic dataflow extraction from event listeners
- Multiple visualization formats (Mermaid, D3.js, ASCII)
- 10 pre-built page templates
- FastAPI integration for programmatic access

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai/) with `nomic-embed-text` model
- Gradio 4.0+ (compatible with Gradio 5.x and 6.x)

## Installation

```bash
# Basic installation
pip install integradio

# With all optional dependencies
pip install "integradio[all]"

# Development installation
pip install -e ".[dev]"
```

### Ollama Setup

Integradio requires Ollama for generating embeddings:

```bash
# Install Ollama (see https://ollama.ai/)
# Then pull the embedding model:
ollama pull nomic-embed-text

# Start Ollama server
ollama serve
```

## Quick Start

```python
import gradio as gr
from integradio import SemanticBlocks, semantic

with SemanticBlocks() as demo:
    # Wrap components with semantic intent
    query = semantic(
        gr.Textbox(label="Search Query"),
        intent="user enters search terms"
    )

    search_btn = semantic(
        gr.Button("Search"),
        intent="triggers the search operation"
    )

    results = semantic(
        gr.Markdown(),
        intent="displays search results"
    )

    search_btn.click(fn=search, inputs=query, outputs=results)

# Components are now searchable by semantic intent
results = demo.search("user input")  # Finds the Textbox
print(demo.summary())  # Shows all registered components

demo.launch()
```

## API Reference

### SemanticBlocks

Extended `gr.Blocks` with registry and embedder integration.

```python
with SemanticBlocks(
    db_path=None,           # SQLite path (None = in-memory)
    cache_dir=None,         # Embedding cache directory
    ollama_url="http://localhost:11434",
    embed_model="nomic-embed-text",
) as demo:
    ...

# Methods
demo.search(query, k=10)     # Semantic search
demo.find(query)             # Get single most relevant component
demo.trace(component)        # Get upstream/downstream flow
demo.map()                   # Export graph as D3.js JSON
demo.describe(component)     # Full metadata dump
demo.summary()               # Text report
```

### semantic()

Wrap any Gradio component with semantic metadata.

```python
component = semantic(
    gr.Textbox(label="Name"),
    intent="user enters their full name",
    tags=["form", "required"],
)
```

### Specialized Wrappers

For complex components, use specialized wrappers that provide richer semantic metadata:

```python
from integradio import (
    semantic_multimodal,      # MultimodalTextbox
    semantic_image_editor,    # ImageEditor
    semantic_annotated_image, # AnnotatedImage (object detection)
    semantic_highlighted_text,# HighlightedText (NER)
    semantic_chatbot,         # Chatbot
    semantic_plot,            # LinePlot, BarPlot, ScatterPlot
    semantic_model3d,         # Model3D
    semantic_dataframe,       # DataFrame
    semantic_file_explorer,   # FileExplorer
)

# AI Chat with persona and streaming support
chat = semantic_chatbot(
    gr.Chatbot(label="Assistant"),
    persona="coder",
    supports_streaming=True,
    supports_like=True,
)
# Auto-tags: ["io", "conversation", "ai", "streaming", "persona-coder", "code-assistant", "programming"]

# Image editor for inpainting with mask support
editor = semantic_image_editor(
    gr.ImageEditor(label="Edit"),
    use_case="inpainting",
    supports_masks=True,
    tools=["brush", "eraser"],
)
# Auto-tags: ["input", "media", "editor", "visual", "inpainting", "masking", "tool-brush", "tool-eraser"]

# Object detection output
detections = semantic_annotated_image(
    gr.AnnotatedImage(label="Detections"),
    annotation_type="bbox",
    entity_types=["person", "car", "dog"],
)
# Auto-tags: ["output", "media", "annotation", "bbox", "detection", "detects-person", "detects-car", "detects-dog"]

# NER visualization
entities = semantic_highlighted_text(
    gr.HighlightedText(label="Entities"),
    annotation_type="ner",
    entity_types=["PERSON", "ORG", "LOC"],
)
# Auto-tags: ["output", "text", "annotation", "nlp", "ner", "person-entity", "organization-entity", "location-entity"]

# Multimodal input for vision-language models
vlm_input = semantic_multimodal(
    gr.MultimodalTextbox(label="Ask about images"),
    use_case="image_analysis",
    accepts_images=True,
)
# Auto-tags: ["input", "text", "multimodal", "vision", "image-input", "image_analysis", "vlm"]

# Data visualization with domain context
metrics_chart = semantic_plot(
    gr.LinePlot(x="date", y="value"),
    chart_type="line",
    data_domain="metrics",
    axes=["date", "value"],
)
# Auto-tags: ["output", "visualization", "chart-line", "timeseries", "domain-metrics"]
```

### Page Templates

10 pre-built page templates for common UI patterns:

```python
from integradio.pages import (
    ChatPage,        # Conversational AI interface
    DashboardPage,   # KPI cards and activity feed
    HeroPage,        # Landing page with CTAs
    GalleryPage,     # Image grid with filtering
    AnalyticsPage,   # Charts and metrics
    DataTablePage,   # Editable data grid
    FormPage,        # Multi-step form wizard
    UploadPage,      # File upload with preview
    SettingsPage,    # Configuration panels
    HelpPage,        # FAQ accordion
)

# Use in your app
page = ChatPage()
page.launch()
```

## Visualization

```python
from integradio.viz import (
    generate_mermaid,      # Mermaid diagram
    generate_html_graph,   # Interactive D3.js
    generate_ascii_graph,  # ASCII art
)

# Generate Mermaid diagram
print(generate_mermaid(demo))

# Save interactive HTML visualization
html = generate_html_graph(demo)
with open("graph.html", "w") as f:
    f.write(html)
```

## FastAPI Integration

```python
from fastapi import FastAPI

app = FastAPI()
demo.add_api_routes(app)

# Endpoints:
# GET /semantic/search?q=<query>&k=<limit>
# GET /semantic/component/<id>
# GET /semantic/graph
# GET /semantic/trace/<id>
# GET /semantic/summary
```

## Examples

See the `examples/` directory:

- `basic_app.py` - Simple search demo
- `full_app.py` - All 10 page templates showcase

```bash
# Run basic example
python examples/basic_app.py
# Visit http://localhost:7860
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=integradio --cov-report=html

# Type checking
mypy integradio

# Linting
ruff check integradio
```

## Architecture

```
integradio/
├── components.py      # SemanticComponent wrapper
├── specialized.py     # Specialized wrappers (Chatbot, ImageEditor, etc.)
├── embedder.py        # Ollama embedding client with circuit breaker
├── registry.py        # HNSW + SQLite storage
├── blocks.py          # Extended gr.Blocks
├── introspect.py      # Source location extraction
├── api.py             # FastAPI routes
├── viz.py             # Graph visualization (Mermaid, D3.js, ASCII)
├── circuit_breaker.py # Resilience pattern for external services
├── exceptions.py      # Exception hierarchy
├── logging_config.py  # Structured logging
├── pages/             # 10 pre-built page templates
├── events/            # WebSocket event mesh with HMAC signing
├── visual/            # Design tokens, themes, Figma sync
├── agent/             # LangChain tools and MCP server
└── inspector/         # Component tree navigation
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read our contributing guidelines and submit PRs.

## Links

- [Gradio Documentation](https://gradio.app/docs/)
- [Ollama](https://ollama.ai/)
- [nomic-embed-text](https://ollama.ai/library/nomic-embed-text)
