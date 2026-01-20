"""
API module - FastAPI endpoints for search and visualization.
"""

import logging
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .blocks import SemanticBlocks

logger = logging.getLogger(__name__)

# Constants for input validation
MAX_SEARCH_RESULTS = 1000
MIN_SEARCH_RESULTS = 1
DEFAULT_SEARCH_RESULTS = 10
MAX_QUERY_LENGTH = 1000


def create_api_routes(app: Any, blocks: "SemanticBlocks") -> None:
    """
    Add semantic API routes to a FastAPI/Starlette app.

    Endpoints:
        GET /semantic/search?q=<query>&k=<limit>&type=<component_type>
        GET /semantic/component/<id>
        GET /semantic/graph
        GET /semantic/trace/<id>
        GET /semantic/summary

    Args:
        app: FastAPI or Starlette application
        blocks: SemanticBlocks instance
    """
    try:
        from fastapi import Query
        from fastapi.responses import JSONResponse
    except ImportError:
        # Fallback for Starlette
        from starlette.responses import JSONResponse
        Query = None
        logger.debug("FastAPI not found, using Starlette fallback")

    @app.get("/semantic/search")
    async def search_components(
        q: str,
        k: int = DEFAULT_SEARCH_RESULTS,
        type: Optional[str] = None,
        tags: Optional[str] = None,
    ) -> JSONResponse:
        """Search components by semantic query."""
        # Validate query
        if not q or not q.strip():
            return JSONResponse(
                {"error": "Query parameter 'q' is required and cannot be empty"},
                status_code=400,
            )
        if len(q) > MAX_QUERY_LENGTH:
            return JSONResponse(
                {"error": f"Query too long. Maximum {MAX_QUERY_LENGTH} characters"},
                status_code=400,
            )

        # Validate k parameter
        if k < MIN_SEARCH_RESULTS:
            k = MIN_SEARCH_RESULTS
        elif k > MAX_SEARCH_RESULTS:
            k = MAX_SEARCH_RESULTS

        tag_list = tags.split(",") if tags else None
        results = blocks.search(q.strip(), k=k, component_type=type, tags=tag_list)

        return JSONResponse({
            "query": q,
            "count": len(results),
            "results": [
                {
                    "component_id": r.component_id,
                    "type": r.metadata.component_type,
                    "intent": r.metadata.intent,
                    "label": r.metadata.label,
                    "score": round(r.score, 4),
                    "tags": r.metadata.tags,
                    "source": {
                        "file": r.metadata.file_path,
                        "line": r.metadata.line_number,
                    },
                }
                for r in results
            ],
        })

    @app.get("/semantic/component/{component_id}")
    async def get_component(component_id: int) -> JSONResponse:
        """Get component details by ID."""
        # Validate component_id
        if component_id < 0:
            return JSONResponse(
                {"error": "Component ID must be a non-negative integer"},
                status_code=400,
            )

        meta = blocks.registry.get(component_id)
        if not meta:
            return JSONResponse(
                {"error": "Component not found"},
                status_code=404,
            )

        relationships = blocks.registry.get_relationships(component_id)

        return JSONResponse({
            "component_id": component_id,
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
            "inputs_from": meta.inputs_from,
            "outputs_to": meta.outputs_to,
        })

    @app.get("/semantic/graph")
    async def get_graph() -> JSONResponse:
        """Get component graph for visualization."""
        return JSONResponse(blocks.map())

    @app.get("/semantic/trace/{component_id}")
    async def trace_component(component_id: int) -> JSONResponse:
        """Trace dataflow through a component."""
        # Validate component_id
        if component_id < 0:
            return JSONResponse(
                {"error": "Component ID must be a non-negative integer"},
                status_code=400,
            )

        meta = blocks.registry.get(component_id)
        if not meta:
            return JSONResponse(
                {"error": "Component not found"},
                status_code=404,
            )

        flow = blocks.registry.get_dataflow(component_id)

        def enrich_ids(ids: list[int]) -> list[dict]:
            result = []
            for cid in ids:
                m = blocks.registry.get(cid)
                if m:
                    result.append({
                        "id": cid,
                        "type": m.component_type,
                        "intent": m.intent,
                        "label": m.label,
                    })
            return result

        return JSONResponse({
            "component_id": component_id,
            "type": meta.component_type,
            "intent": meta.intent,
            "upstream": enrich_ids(flow["upstream"]),
            "downstream": enrich_ids(flow["downstream"]),
        })

    @app.get("/semantic/summary")
    async def get_summary() -> JSONResponse:
        """Get registry summary."""
        components = blocks.registry.all_components()

        # Group by type
        by_type: dict[str, int] = {}
        for meta in components:
            by_type[meta.component_type] = by_type.get(meta.component_type, 0) + 1

        return JSONResponse({
            "total_components": len(components),
            "by_type": by_type,
            "components": [
                {
                    "id": m.component_id,
                    "type": m.component_type,
                    "intent": m.intent,
                    "label": m.label,
                }
                for m in components
            ],
        })


def create_gradio_api(blocks: "SemanticBlocks") -> dict[str, Any]:
    """
    Create Gradio API endpoint handlers.

    These can be added to a Blocks instance via .load() or custom routes.

    Returns:
        Dict of endpoint name -> handler function
    """

    def search_handler(query: str, k: int = 10) -> list[dict]:
        """Search components by query."""
        results = blocks.search(query, k=k)
        return [
            {
                "id": r.component_id,
                "type": r.metadata.component_type,
                "intent": r.metadata.intent,
                "score": round(r.score, 4),
            }
            for r in results
        ]

    def graph_handler() -> dict:
        """Get component graph."""
        return blocks.map()

    def summary_handler() -> str:
        """Get text summary."""
        return blocks.summary()

    return {
        "search": search_handler,
        "graph": graph_handler,
        "summary": summary_handler,
    }
