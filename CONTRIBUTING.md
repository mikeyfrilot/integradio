# Contributing to Integradio

Thank you for your interest in contributing to Integradio! We welcome contributions from the community and are excited to have you here.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

By participating in this project, you agree to maintain a welcoming, inclusive, and harassment-free environment. Please be respectful and constructive in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/integradio.git
   cd integradio
   ```
3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/mikeyfrilot/integradio.git
   ```

## Development Setup

### Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.ai/) (optional, for embedding tests)
- Git

### Installation

```bash
# Create a virtual environment
python -m venv .venv

# Activate it
# On Windows:
.venv\Scripts\activate
# On Unix/macOS:
source .venv/bin/activate

# Install in development mode with all dependencies
pip install -e ".[dev,all]"

# Pull the embedding model (optional, for full test suite)
ollama pull nomic-embed-text
```

### Verify Installation

```bash
# Run the test suite
pytest tests/ -v

# Check linting
ruff check integradio/

# Type checking
mypy integradio/
```

## Making Changes

### Branch Naming

Create a descriptive branch for your changes:

```bash
git checkout -b feature/add-new-component
git checkout -b fix/embedding-cache-race
git checkout -b docs/improve-api-reference
```

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
feat: add semantic wrapper for AudioInput component
fix: prevent race condition in embedder cache
docs: update installation instructions for Windows
test: add tests for WebSocket reconnection
refactor: simplify event matching logic
chore: update dependencies to latest versions
```

### Keep Changes Focused

- One feature or fix per pull request
- Keep PRs reasonably sized (under 500 lines when possible)
- Break large changes into smaller, reviewable chunks

## Pull Request Process

1. **Update your fork** with the latest upstream changes:
   ```bash
   git fetch upstream
   git rebase upstream/master
   ```

2. **Run the full test suite** before submitting:
   ```bash
   pytest tests/ -v --cov=integradio
   ruff check integradio/
   ```

3. **Push your branch** and create a PR:
   ```bash
   git push origin feature/your-feature
   ```

4. **Fill out the PR template** with:
   - Description of what the PR does
   - Related issue numbers (if any)
   - Testing performed
   - Screenshots (for UI changes)

5. **Respond to review feedback** promptly

### PR Requirements

- All tests must pass
- Code must pass linting (`ruff check`)
- New features should include tests
- Public APIs should have docstrings
- CHANGELOG.md should be updated for user-facing changes

## Code Style

We follow [PEP 8](https://pep8.org/) with some modifications:

### Formatting

- **Line length**: 100 characters maximum
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Double quotes for strings
- **Imports**: Sorted with `isort` (integrated in Ruff)

### Tools

```bash
# Format code
ruff format integradio/

# Check linting
ruff check integradio/

# Auto-fix linting issues
ruff check integradio/ --fix
```

### Type Hints

Use type hints for all public APIs:

```python
def search(
    self,
    query: str,
    k: int = 10,
    threshold: float = 0.0,
) -> list[tuple[SemanticComponent, float]]:
    """
    Search for components by semantic similarity.

    Args:
        query: Natural language search query
        k: Maximum number of results to return
        threshold: Minimum similarity score (0.0-1.0)

    Returns:
        List of (component, score) tuples sorted by relevance
    """
    ...
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=integradio --cov-report=html

# Run specific test file
pytest tests/test_embedder.py -v

# Run tests matching a pattern
pytest tests/ -k "test_search" -v

# Skip integration tests (requires Ollama)
pytest tests/ -v -m "not integration"
```

### Writing Tests

- Place tests in `tests/` directory
- Mirror the source structure (e.g., `integradio/embedder.py` → `tests/test_embedder.py`)
- Use descriptive test names:
  ```python
  def test_search_returns_empty_list_when_no_components_registered():
      ...

  def test_embedder_gracefully_handles_ollama_timeout():
      ...
  ```

### Test Categories

- **Unit tests**: Test individual functions/methods in isolation
- **Integration tests**: Test component interactions (marked with `@pytest.mark.integration`)
- **Edge case tests**: Test boundary conditions and error handling

## Documentation

### Docstrings

Use Google-style docstrings:

```python
def semantic(
    component: gr.Component,
    intent: str,
    tags: list[str] | None = None,
) -> SemanticComponent:
    """
    Wrap a Gradio component with semantic metadata.

    This enables the component to be discovered via natural language
    search and included in dataflow visualizations.

    Args:
        component: Any Gradio component instance
        intent: Natural language description of what this component does
        tags: Optional list of categorization tags

    Returns:
        A SemanticComponent wrapper that proxies to the original

    Raises:
        InvalidComponentError: If component is None or invalid type

    Example:
        >>> import gradio as gr
        >>> from integradio import semantic
        >>> search_box = semantic(
        ...     gr.Textbox(label="Search"),
        ...     intent="user enters search query"
        ... )
    """
```

### Changelog

Update `CHANGELOG.md` for user-facing changes:

```markdown
## [Unreleased]

### Added
- New `semantic_audio` wrapper for Audio components (#123)

### Fixed
- Race condition in embedder cache when used from multiple threads (#456)

### Changed
- Improved error messages for invalid component types
```

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

1. **Python version**: `python --version`
2. **Package versions**: `pip show integradio gradio`
3. **Operating system**
4. **Steps to reproduce**
5. **Expected behavior**
6. **Actual behavior**
7. **Error messages/stack traces**

### Feature Requests

When requesting features:

1. **Describe the use case** - What problem are you trying to solve?
2. **Propose a solution** - How would you like it to work?
3. **Consider alternatives** - What other approaches exist?

### Security Vulnerabilities

**Do NOT report security vulnerabilities through public issues.**

Please see [SECURITY.md](SECURITY.md) for responsible disclosure instructions.

## Questions?

- Open a [GitHub Discussion](https://github.com/mikeyfrilot/integradio/discussions)
- Check existing issues and discussions first
- Be patient - maintainers are often volunteers

---

Thank you for contributing to Integradio!
