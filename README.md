<div align="center">

# Integradio

**Vector-embedded Gradio components for semantic codebase navigation**

[![PyPI version](https://img.shields.io/pypi/v/integradio?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/integradio/)
[![Downloads](https://img.shields.io/pypi/dm/integradio?color=green&logo=pypi&logoColor=white)](https://pypi.org/project/integradio/)
[![Tests](https://github.com/mikeyfrilot/integradio/actions/workflows/test.yml/badge.svg)](https://github.com/mikeyfrilot/integradio/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/mikeyfrilot/integradio/graph/badge.svg)](https://codecov.io/gh/mikeyfrilot/integradio)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Gradio 5.0+](https://img.shields.io/badge/gradio-5.0+-orange.svg?logo=gradio&logoColor=white)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/mikeyfrilot/integradio?style=social)](https://github.com/mikeyfrilot/integradio)

*Find Gradio components by what they do, not what they're called*

[Installation](#installation) • [Quick Start](#quick-start-30-seconds) • [Examples](#examples) • [API Reference](#api-reference) • [Contributing](#contributing)

</div>

---

## Why Integradio?

| Problem | Solution |
|---------|----------|
| "Which textbox handles user input?" | `demo.find("user enters search terms")` |
| Complex UIs with dozens of components | Semantic search finds components by intent |
| Debugging dataflow between components | Automatic flow tracing and visualization |
| Building consistent Gradio apps | 10 pre-built page templates |

<!--
## Demo

<p align="center">
  <img src="docs/assets/demo.gif" alt="Integradio Demo" width="600">
</p>
-->

## Quick Start (30 seconds)

```bash
pip install integradio
ollama pull nomic-embed-text
```

```python
import gradio as gr
from integradio import SemanticBlocks, semantic

with SemanticBlocks() as demo:
    query = semantic(gr.Textbox(label="Search"), intent="user enters search terms")
    btn = semantic(gr.Button("Go"), intent="triggers search")
    results = semantic(gr.Markdown(), intent="displays results")
    btn.click(fn=lambda x: f"Results for: {x}", inputs=query, outputs=results)

# Find by intent, not by variable name
demo.find("user input")  # Returns the Textbox
demo.launch()
```

## Features

- **Semantic Search** - Find components by describing what they do
- **Non-Invasive** - Works with any existing Gradio component
- **Flow Tracing** - Automatic dataflow extraction from event listeners
- **Visualization** - Mermaid, D3.js, and ASCII graph exports
- **10 Page Templates** - Chat, Dashboard, Gallery, Analytics, and more
- **FastAPI Integration** - REST API for programmatic access
- **Local Embeddings** - Uses Ollama (no API keys needed)

## Installation

```bash
# Basic installation
pip install integradio

# With all optional dependencies
pip install "integradio[all]"

# Development installation
pip install -e ".[dev]"
```

### Requirements

- Python 3.10+
- [Ollama](https://ollama.ai/) with `nomic-embed-text` model
- Gradio 4.0+ (compatible with 5.x and 6.x)

```bash
# Install Ollama, then:
ollama pull nomic-embed-text
ollama serve
```

## Examples

### Semantic Component Wrapping

```python
from integradio import SemanticBlocks, semantic

with SemanticBlocks() as demo:
    # Wrap components with semantic intent
    query = semantic(
        gr.Textbox(label="Search Query"),
        intent="user enters search terms",
        tags=["input", "required"]
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

# Semantic operations
demo.search("user input")     # Find related components
demo.find("search trigger")   # Get single best match
demo.trace(results)           # Upstream/downstream flow
demo.summary()                # Text report of all components
```

### Specialized Wrappers

For complex components, use specialized wrappers with richer metadata:

```python
from integradio import (
    semantic_chatbot,
    semantic_image_editor,
    semantic_annotated_image,
    semantic_dataframe,
)

# AI Chat with persona
chat = semantic_chatbot(
    gr.Chatbot(label="Assistant"),
    persona="coder",
    supports_streaming=True,
)
# Auto-tags: ["conversation", "ai", "streaming", "persona-coder"]

# Image editor for inpainting
editor = semantic_image_editor(
    gr.ImageEditor(label="Edit"),
    use_case="inpainting",
    supports_masks=True,
)
# Auto-tags: ["editor", "visual", "inpainting", "masking"]

# Object detection output
detections = semantic_annotated_image(
    gr.AnnotatedImage(label="Detections"),
    annotation_type="bbox",
    entity_types=["person", "car"],
)
# Auto-tags: ["annotation", "bbox", "detection", "detects-person"]
```

### Page Templates

10 pre-built templates for common UI patterns:

```python
from integradio.pages import (
    ChatPage,        # Conversational AI
    DashboardPage,   # KPIs and activity
    HeroPage,        # Landing page
    GalleryPage,     # Image grid
    AnalyticsPage,   # Charts and metrics
    DataTablePage,   # Editable grid
    FormPage,        # Multi-step wizard
    UploadPage,      # File upload
    SettingsPage,    # Configuration
    HelpPage,        # FAQ accordion
)

page = ChatPage()
page.launch()
```

### Visualization

```python
from integradio.viz import generate_mermaid, generate_html_graph

# Mermaid diagram
print(generate_mermaid(demo))

# Interactive D3.js visualization
with open("graph.html", "w") as f:
    f.write(generate_html_graph(demo))
```

### FastAPI Integration

```python
from fastapi import FastAPI

app = FastAPI()
demo.add_api_routes(app)

# Endpoints:
# GET /semantic/search?q=<query>&k=<limit>
# GET /semantic/component/<id>
# GET /semantic/graph
# GET /semantic/trace/<id>
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
demo.find(query)             # Single most relevant component
demo.trace(component)        # Upstream/downstream flow
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

## Architecture

```
integradio/
├── components.py      # SemanticComponent wrapper
├── specialized.py     # Specialized wrappers (Chatbot, ImageEditor, etc.)
├── embedder.py        # Ollama client with circuit breaker
├── registry.py        # HNSW + SQLite storage
├── blocks.py          # Extended gr.Blocks
├── introspect.py      # Source location extraction
├── api.py             # FastAPI routes
├── viz.py             # Graph visualization
├── pages/             # 10 pre-built page templates
├── events/            # WebSocket event mesh
├── visual/            # Design tokens, themes
├── agent/             # LangChain tools and MCP server
└── inspector/         # Component tree navigation
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

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Related Projects

Part of the **Compass Suite** for AI-powered development:

- [Tool Compass](https://github.com/mikeyfrilot/tool-compass) - Semantic MCP tool discovery
- [File Compass](https://github.com/mikeyfrilot/file-compass) - Semantic file search
- [Backpropagate](https://github.com/mikeyfrilot/backpropagate) - Headless LLM fine-tuning
- [Comfy Headless](https://github.com/mikeyfrilot/comfy-headless) - ComfyUI without the complexity

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**[Documentation](https://github.com/mikeyfrilot/integradio#readme)** • **[Issues](https://github.com/mikeyfrilot/integradio/issues)** • **[Discussions](https://github.com/mikeyfrilot/integradio/discussions)**

</div>
