# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-01-20

### Added
- **Specialized semantic wrappers** for complex components (specialized.py):
  - `SemanticMultimodal` / `semantic_multimodal()` - Enhanced wrapper for MultimodalTextbox
    - Use case detection: chat, document_qa, image_analysis, code_review
    - Auto-tags: vision, image-input, audio-input, document-input
  - `SemanticImageEditor` / `semantic_image_editor()` - Enhanced wrapper for ImageEditor
    - Use cases: inpainting, annotation, segmentation, photo_editing
    - Tool-specific tags, mask support detection
  - `SemanticAnnotatedImage` / `semantic_annotated_image()` - Object detection outputs
    - Annotation types: bbox, segmentation, polygon, keypoint
    - Entity type tags (detects-person, detects-car, etc.)
  - `SemanticHighlightedText` / `semantic_highlighted_text()` - NLP annotation outputs
    - NLP types: ner, pos, sentiment, classification, highlight
    - NER entity mapping (PERSON, ORG, LOC, etc.)
  - `SemanticChatbot` / `semantic_chatbot()` - AI conversation interfaces
    - Persona support: assistant, coder, tutor, creative, analyst
    - Streaming, retry, like support detection
  - `SemanticPlot` / `semantic_plot()` - Data visualization components
    - Chart types: line, bar, scatter, pie, heatmap, histogram
    - Data domain tagging, axis detection
  - `SemanticModel3D` / `semantic_model3d()` - 3D model viewers
    - Use cases: mesh_generation, cad_viewer, game_asset, medical
    - Format and animation support tags
  - `SemanticDataFrame` / `semantic_dataframe()` - Tabular data components
    - Data domains: database, spreadsheet, metrics, logs
    - Column-based inference (temporal, financial, entity, etc.)
  - `SemanticFileExplorer` / `semantic_file_explorer()` - File navigation
    - Root types: code_project, documents, media, data, config
    - File type categorization
- **Metadata dataclasses** for extended component information:
  - `MultimodalMetadata`, `ImageEditorMetadata`, `AnnotationMetadata`
  - `ChatMetadata`, `VisualizationMetadata`, `Model3DMetadata`
- **Comprehensive Gradio 6 component support** in `infer_tags()` (introspect.py)
  - Text: MultimodalTextbox
  - Date/Time: DateTime
  - Buttons: DownloadButton, ClearButton, DuplicateButton, LoginButton, Timer
  - Files: FileExplorer
  - Media: ImageEditor, Model3D, AnnotatedImage, SimpleImage, ImageSlider
  - Text Output: HighlightedText
  - Data: Dataset, State, ParamViewer
  - Visualization: ScatterPlot, LinePlot, BarPlot
  - Conversation: ChatInterface
  - Layout: Accordion, Tab, Tabs, Row, Column, Group, Blocks
  - Gradio 6 Navigation: Sidebar, Navbar, Dialogue, Walkthrough
  - Interfaces: Interface, TabbedInterface
- **Attribute-based tag inference** for dynamic component properties:
  - `streaming` tag for streaming components
  - `filepath`/`binary` tags based on file type
  - `webcam`/`microphone`/`upload` tags based on media sources
  - `copyable` tag for components with copy buttons
  - `rtl` tag for right-to-left text components

## [0.1.1] - 2026-01-20

### Added
- README.md with comprehensive documentation, quick start guide, and API reference
- CHANGELOG.md following Keep a Changelog format
- Cache versioning for embeddings (CACHE_VERSION constant in embedder.py)
  - Prevents stale embeddings when model or prefix changes
  - Cache key now includes model name and version
- Python 3.13 support in classifiers
- `Typing :: Typed` classifier for PEP 561 compliance
- `maintainers` field in pyproject.toml
- Additional keywords: `ollama`, `nomic-embed-text`

### Changed
- **pyproject.toml**: Updated to 2026 Python packaging best practices
  - `setuptools>=70.0` (was >=61.0) per PEP 621 recommendations
  - `license = "MIT"` using SPDX expression format (was `{text = "MIT"}`)
  - `gradio>=4.0.0,<7.0.0` - explicit upper bound for Gradio 5.x/6.x compatibility
  - Updated dev dependencies to latest stable versions:
    - pytest>=8.0.0, pytest-asyncio>=0.23.0, pytest-cov>=4.1.0
    - ruff>=0.4.0, mypy>=1.10.0

### Fixed
- **Division by zero bug** in `registry.py` fallback search
  - Added epsilon (`np.finfo(np.float32).eps`) to denominator in cosine similarity
  - Prevents crash when Ollama unavailable and zero vectors are used
  - Explicit float conversion for score/distance values

### Security
- No security vulnerabilities identified in audit
- All SQL queries use parameterized statements
- HTTP requests use proper timeouts

## [0.1.0] - 2026-01-15

### Added
- Initial release
- `SemanticComponent` wrapper for Gradio components with semantic embeddings
- `SemanticBlocks` extended context manager with auto-registration
- `ComponentRegistry` with HNSW index and SQLite persistence
- `Embedder` client for Ollama/nomic-embed-text with caching
- Introspection utilities for source location and dataflow extraction
- Visualization generators (Mermaid, D3.js HTML, ASCII)
- FastAPI integration with 5 REST endpoints
- 10 pre-built page templates:
  - ChatPage, DashboardPage, HeroPage, GalleryPage
  - AnalyticsPage, DataTablePage, FormPage
  - UploadPage, SettingsPage, HelpPage
- Graceful degradation when Ollama unavailable
- Fallback brute-force search when hnswlib not installed
- Comprehensive test fixtures in conftest.py
- Test suite for registry module

### Dependencies
- gradio>=4.0.0
- numpy>=1.24.0
- httpx>=0.24.0
- pandas>=2.0.0
- Optional: hnswlib>=0.7.0, fastapi>=0.100.0, uvicorn>=0.23.0

[0.3.0]: https://github.com/mikeyfrilot/integradio/compare/v0.1.1...v0.3.0
[0.1.1]: https://github.com/mikeyfrilot/integradio/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/mikeyfrilot/integradio/releases/tag/v0.1.0
