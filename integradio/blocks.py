"""
SemanticBlocks - Extended Gradio Blocks with vector registry integration.
"""

import logging
from pathlib import Path
from typing import Optional, Any, Callable
import json

import gradio as gr

from .registry import ComponentRegistry, SearchResult
from .embedder import Embedder
from .components import SemanticComponent, get_semantic
from .introspect import extract_dataflow

logger = logging.getLogger(__name__)


class SemanticBlocks(gr.Blocks):
    """
    Extended Gradio Blocks with semantic component registry.

    Automatically tracks and embeds components created within the block,
    enabling semantic search and visualization.
    """

    def __init__(
        self,
        *args,
        db_path: Optional[Path] = None,
        cache_dir: Optional[Path] = None,
        ollama_url: str = "http://localhost:11434",
        embed_model: str = "nomic-embed-text",
        auto_register: bool = True,
        theme: Any = None,
        **kwargs,
    ):
        """
        Initialize SemanticBlocks.

        Args:
            *args: Passed to gr.Blocks
            db_path: Path to SQLite database for registry
            cache_dir: Path to embedding cache directory
            ollama_url: Ollama API URL
            embed_model: Model name for embeddings
            auto_register: Auto-register components on exit
            theme: Gradio theme (stored for launch)
            **kwargs: Passed to gr.Blocks
        """
        super().__init__(*args, **kwargs)
        self._theme = theme  # Store for launch()

        self._db_path = db_path
        self._cache_dir = cache_dir
        self._ollama_url = ollama_url
        self._embed_model = embed_model
        self._auto_register = auto_register

        # Initialize registry and embedder
        self._registry = ComponentRegistry(db_path=db_path)
        self._embedder = Embedder(
            model=embed_model,
            base_url=ollama_url,
            cache_dir=cache_dir,
        )

        # Set global references for SemanticComponent
        SemanticComponent._registry = self._registry
        SemanticComponent._embedder = self._embedder

        # Track pending components
        self._pending_semantics: list[SemanticComponent] = []

    @property
    def registry(self) -> ComponentRegistry:
        """Access the component registry."""
        return self._registry

    @property
    def embedder(self) -> Embedder:
        """Access the embedder."""
        return self._embedder

    def __exit__(self, *args, **kwargs):
        """Register all components on context exit."""
        result = super().__exit__(*args, **kwargs)

        if self._auto_register:
            self._register_all_components()
            self._register_dataflow()

        return result

    def _register_all_components(self) -> None:
        """Register all semantic components in the blocks."""
        for comp_id, semantic_comp in SemanticComponent._instances.items():
            if not semantic_comp._semantic_meta.embedded:
                semantic_comp._register_to_registry()

    def _register_dataflow(self) -> None:
        """Extract and register dataflow relationships."""
        try:
            flows = extract_dataflow(self)

            for flow in flows:
                # Connect triggers to inputs
                for trigger_id in flow.get("triggers", []):
                    for input_id in flow.get("inputs", []):
                        self._registry.add_relationship(
                            trigger_id, input_id, "trigger"
                        )

                # Connect inputs to outputs
                for input_id in flow.get("inputs", []):
                    for output_id in flow.get("outputs", []):
                        self._registry.add_relationship(
                            input_id, output_id, "dataflow"
                        )
        except Exception as e:
            logger.error(f"Failed to extract dataflow relationships: {e}")

    def search(
        self,
        query: str,
        k: int = 10,
        component_type: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        """
        Search components by semantic query.

        Args:
            query: Natural language search query
            k: Maximum results to return
            component_type: Filter by component type
            tags: Filter by tags

        Returns:
            List of SearchResult ordered by relevance
        """
        query_vector = self._embedder.embed_query(query)
        return self._registry.search(
            query_vector,
            k=k,
            component_type=component_type,
            tags=tags,
        )

    def find(self, query: str) -> Optional[Any]:
        """
        Find a single component by query.

        Args:
            query: Natural language query

        Returns:
            The most relevant Gradio component or None
        """
        results = self.search(query, k=1)
        if not results:
            return None

        comp_id = results[0].component_id
        semantic_comp = get_semantic(comp_id)
        if semantic_comp is None:
            return None
        return getattr(semantic_comp, "component", None)

    def trace(self, component: Any) -> dict[str, Any]:
        """
        Trace dataflow through a component.

        Args:
            component: Gradio component or SemanticComponent

        Returns:
            Dict with upstream and downstream component info
        """
        comp_id = self._get_component_id(component)
        if comp_id is None:
            return {"error": "Component must have _id attribute", "upstream": [], "downstream": []}

        flow = self._registry.get_dataflow(comp_id)

        # Enrich with component info
        def enrich_ids(ids: list[int]) -> list[dict]:
            result = []
            for cid in ids:
                meta = self._registry.get(cid)
                if meta:
                    result.append({
                        "id": cid,
                        "type": meta.component_type,
                        "intent": meta.intent,
                        "label": meta.label,
                    })
            return result

        return {
            "upstream": enrich_ids(flow["upstream"]),
            "downstream": enrich_ids(flow["downstream"]),
        }

    def map(self) -> dict[str, Any]:
        """
        Get component graph for visualization.

        Returns:
            D3.js compatible graph with nodes and links
        """
        return self._registry.export_graph()

    def map_json(self) -> str:
        """Get component graph as JSON string."""
        return json.dumps(self.map(), indent=2)

    def _get_component_id(self, component: Any) -> Optional[int]:
        """
        Extract component ID from various component types.

        Args:
            component: Gradio component or SemanticComponent

        Returns:
            Component ID or None if not found
        """
        if isinstance(component, SemanticComponent):
            wrapped = component.component
            if hasattr(wrapped, "_id"):
                return wrapped._id
            return None
        elif hasattr(component, "_id"):
            return component._id
        return None

    def describe(self, component: Any) -> dict[str, Any]:
        """
        Get full description of a component.

        Args:
            component: Gradio component or SemanticComponent

        Returns:
            Dict with all component metadata
        """
        comp_id = self._get_component_id(component)
        if comp_id is None:
            return {"error": "Component must have _id attribute"}

        meta = self._registry.get(comp_id)
        if not meta:
            return {"error": "Component not registered"}

        relationships = self._registry.get_relationships(comp_id)

        return {
            "component_id": comp_id,
            "type": meta.component_type,
            "intent": meta.intent,
            "label": meta.label,
            "elem_id": meta.elem_id,
            "tags": meta.tags,
            "source": {
                "file": meta.file_path,
                "line": meta.line_number,
            },
            "relationships": relationships,
        }

    def add_api_routes(self, app: Any) -> None:
        """
        Add search API routes to a FastAPI/Starlette app.

        Args:
            app: FastAPI or Starlette application
        """
        from .api import create_api_routes
        create_api_routes(app, self)

    def summary(self) -> str:
        """Get a text summary of registered components."""
        components = self._registry.all_components()
        if not components:
            return "No components registered"

        lines = [f"SemanticBlocks: {len(components)} components registered\n"]

        # Group by type
        by_type: dict[str, list] = {}
        for meta in components:
            if meta.component_type not in by_type:
                by_type[meta.component_type] = []
            by_type[meta.component_type].append(meta)

        for comp_type, metas in sorted(by_type.items()):
            lines.append(f"\n{comp_type} ({len(metas)}):")
            for meta in metas:
                label = meta.label or meta.elem_id or f"id={meta.component_id}"
                lines.append(f"  - {label}: {meta.intent}")

        return "\n".join(lines)
