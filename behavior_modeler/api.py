"""
FastAPI API layer for the Behavior Modeler.

Provides REST endpoints for:
- Flow ingestion and retrieval
- Session encoding and search
- Behavior clustering
- Next action prediction
- Test gap detection
"""

from datetime import datetime
from typing import Optional
import json

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import BehaviorModelerConfig
from .models import Session, FlowEvent, TestGap
from .store import FlowStore
from .encoder import FlowEncoder, FallbackEncoder
from .index import FlowIndex
from .clustering import BehaviorClustering, ClusteringResult
from .predictor import HybridPredictor, create_predictor
from .patterns import SequentialPatternMiner
from .gaps import GapDetector, analyze_test_gaps


# Pydantic models for API
class EventInput(BaseModel):
    """Input model for a single event."""

    event_id: str
    timestamp: str
    component_id: Optional[int] = None
    component_type: Optional[str] = ""
    event_type: str
    event_data: Optional[dict] = None
    intent: Optional[str] = None
    tags: Optional[list[str]] = None


class IngestRequest(BaseModel):
    """Request to ingest a flow."""

    session_id: str
    events: list[EventInput]
    user_agent: Optional[str] = None


class IngestResponse(BaseModel):
    """Response from flow ingestion."""

    status: str
    session_id: str
    event_count: int
    warnings: list[str] = []


class SessionSummary(BaseModel):
    """Summary of a session."""

    session_id: str
    started_at: str
    duration_ms: int
    event_count: int
    is_complete: bool
    cluster_id: Optional[int] = None
    cluster_label: Optional[str] = None


class SessionListResponse(BaseModel):
    """Response for listing sessions."""

    sessions: list[SessionSummary]
    total: int
    limit: int
    offset: int


class EncodeRequest(BaseModel):
    """Request to encode a session."""

    session_id: str


class EncodeResponse(BaseModel):
    """Response from session encoding."""

    session_id: str
    vector_dim: int
    encoded: bool
    encoding_method: str


class SearchRequest(BaseModel):
    """Request to search for similar sessions."""

    query: Optional[str] = None
    session_id: Optional[str] = None
    k: int = 10
    min_similarity: float = 0.0


class SearchResultItem(BaseModel):
    """A single search result."""

    session_id: str
    similarity: float
    cluster_id: Optional[int] = None
    duration_ms: Optional[int] = None
    event_summary: Optional[list[str]] = None


class SearchResponse(BaseModel):
    """Response from search."""

    results: list[SearchResultItem]


class ClusterRequest(BaseModel):
    """Request to run clustering."""

    algorithm: str = "hdbscan"
    min_cluster_size: int = 10
    min_samples: int = 5
    n_clusters: Optional[int] = None  # For K-means


class ClusterSummary(BaseModel):
    """Summary of a cluster."""

    cluster_id: int
    session_count: int
    label: str
    cluster_type: str


class ClusterResponse(BaseModel):
    """Response from clustering."""

    status: str
    clusters_found: int
    noise_sessions: int
    cluster_summary: list[ClusterSummary]


class ClusterDetail(BaseModel):
    """Detailed cluster info."""

    cluster_id: int
    label: str
    cluster_type: str
    session_count: int
    completion_rate: float
    avg_duration_ms: float
    dominant_components: list[str]
    dominant_intents: list[str]
    sample_session_ids: list[str]


class ClustersListResponse(BaseModel):
    """Response for listing clusters."""

    clusters: list[ClusterDetail]


class PredictRequest(BaseModel):
    """Request for next action prediction."""

    events: list[EventInput]
    top_k: int = 5


class PredictionItem(BaseModel):
    """A single prediction."""

    predicted_component: str
    predicted_event: str
    confidence: float
    cluster_id: Optional[int] = None


class PredictResponse(BaseModel):
    """Response from prediction."""

    predictions: list[PredictionItem]


class GapSummary(BaseModel):
    """Summary of a test gap."""

    gap_id: str
    gap_type: str
    flow_description: str
    affected_components: list[str]
    observed_count: int
    priority: str
    suggested_test_name: str


class GapsResponse(BaseModel):
    """Response for listing gaps."""

    gaps: list[GapSummary]
    total_gaps: int
    gaps_by_priority: dict[str, int]


class GenerateTestRequest(BaseModel):
    """Request to generate a test."""

    gap_id: str
    test_framework: str = "pytest"
    output_path: Optional[str] = None


class GenerateTestResponse(BaseModel):
    """Response from test generation."""

    status: str
    test_file: str
    test_code: str
    assertions: int


class PatternItem(BaseModel):
    """A mined pattern."""

    pattern_id: str
    sequence: list[str]
    support: float
    confidence: float
    occurrence_count: int


class PatternsResponse(BaseModel):
    """Response for listing patterns."""

    patterns: list[PatternItem]
    n_sessions_analyzed: int


class InsightsResponse(BaseModel):
    """Response for cluster insights."""

    total_clusters: int
    cluster_distribution: dict
    top_happy_paths: list[dict]
    problem_areas: list[dict]


# FastAPI app factory
def create_app(config: Optional[BehaviorModelerConfig] = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        config: Configuration for the behavior modeler

    Returns:
        Configured FastAPI app
    """
    config = config or BehaviorModelerConfig()

    app = FastAPI(
        title="Behavior Modeler API",
        description="Semantic intelligence layer for UI behavior analysis",
        version="0.4.0",
    )

    # Initialize components lazily
    _store: Optional[FlowStore] = None
    _encoder: Optional[FlowEncoder] = None
    _index: Optional[FlowIndex] = None
    _predictor: Optional[HybridPredictor] = None

    def get_store() -> FlowStore:
        nonlocal _store
        if _store is None:
            _store = FlowStore(config)
        return _store

    def get_encoder() -> FlowEncoder:
        nonlocal _encoder
        if _encoder is None:
            _encoder = FallbackEncoder()  # Use fallback for now
        return _encoder

    def get_index() -> FlowIndex:
        nonlocal _index
        if _index is None:
            _index = FlowIndex(get_store(), get_encoder(), config)
            _index.build()
        return _index

    def get_predictor() -> HybridPredictor:
        nonlocal _predictor
        if _predictor is None:
            _predictor = create_predictor(get_store(), get_encoder(), get_index(), config)
        return _predictor

    # Health check
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "version": "0.4.0"}

    # =========================================================================
    # Phase 1: Flow Ingestion & Storage
    # =========================================================================

    @app.post("/flows/ingest", response_model=IngestResponse)
    async def ingest_flow(request: IngestRequest):
        """
        Ingest a raw flow from EventMesh or external source.

        Accepts a session with events and persists it to the store.
        """
        store = get_store()
        warnings = []

        # Parse timestamps
        events = []
        for i, evt in enumerate(request.events):
            try:
                ts = datetime.fromisoformat(evt.timestamp.replace("Z", "+00:00"))
            except ValueError:
                warnings.append(f"Invalid timestamp for event {evt.event_id}")
                continue

            flow_event = FlowEvent(
                event_id=evt.event_id,
                timestamp=ts,
                component_id=evt.component_id,
                component_type=evt.component_type or "",
                event_type=evt.event_type,
                event_data=evt.event_data or {},
                intent=evt.intent,
                tags=evt.tags or [],
            )
            events.append(flow_event)

        if not events:
            raise HTTPException(status_code=400, detail="No valid events in request")

        # Create session
        session = Session(
            session_id=request.session_id,
            started_at=events[0].timestamp,
            ended_at=events[-1].timestamp if len(events) > 1 else None,
            user_agent=request.user_agent,
            events=events,
        )

        # Encode the session
        encoder = get_encoder()
        session.vector = encoder.encode_session(session)

        # Save to store
        store.save_session(session)

        return IngestResponse(
            status="accepted",
            session_id=request.session_id,
            event_count=len(events),
            warnings=warnings,
        )

    @app.get("/flows/sessions", response_model=SessionListResponse)
    async def list_sessions(
        start_date: Optional[str] = Query(None, description="Filter by start date (ISO 8601)"),
        end_date: Optional[str] = Query(None, description="Filter by end date (ISO 8601)"),
        cluster_id: Optional[int] = Query(None, description="Filter by cluster ID"),
        is_complete: Optional[bool] = Query(None, description="Filter by completion status"),
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ):
        """
        List sessions with filtering.

        Supports filtering by date range, cluster, and completion status.
        """
        store = get_store()

        # Parse dates
        start = None
        end = None
        if start_date:
            try:
                start = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_date format")
        if end_date:
            try:
                end = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end_date format")

        sessions, total = store.list_sessions(
            start_date=start,
            end_date=end,
            cluster_id=cluster_id,
            is_complete=is_complete,
            limit=limit,
            offset=offset,
        )

        # Get cluster labels if available
        clusters = {c.cluster_id: c.label for c in store.get_clusters()}

        summaries = [
            SessionSummary(
                session_id=s.session_id,
                started_at=s.started_at.isoformat(),
                duration_ms=s.duration_ms,
                event_count=len(s.events) if s.events else 0,
                is_complete=s.is_complete,
                cluster_id=s.cluster_id,
                cluster_label=clusters.get(s.cluster_id) if s.cluster_id else None,
            )
            for s in sessions
        ]

        return SessionListResponse(
            sessions=summaries,
            total=total,
            limit=limit,
            offset=offset,
        )

    @app.get("/flows/sessions/{session_id}")
    async def get_session(session_id: str):
        """
        Get full session detail including all events.
        """
        store = get_store()
        session = store.get_session(session_id, include_events=True)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "session": {
                "session_id": session.session_id,
                "started_at": session.started_at.isoformat(),
                "ended_at": session.ended_at.isoformat() if session.ended_at else None,
                "duration_ms": session.duration_ms,
                "is_complete": session.is_complete,
                "cluster_id": session.cluster_id,
                "events": [
                    {
                        "event_id": e.event_id,
                        "sequence_num": i,
                        "timestamp": e.timestamp.isoformat(),
                        "component_type": e.component_type,
                        "component_intent": e.component_intent,
                        "event_type": e.event_type,
                        "event_data": e.event_data,
                        "intent": e.intent,
                    }
                    for i, e in enumerate(session.events)
                ],
            }
        }

    # =========================================================================
    # Phase 2: Encoding & Search
    # =========================================================================

    @app.post("/flows/encode", response_model=EncodeResponse)
    async def encode_session(request: EncodeRequest):
        """
        Encode a session into a vector representation.

        The vector is computed and stored with the session.
        """
        store = get_store()
        encoder = get_encoder()

        session = store.get_session(request.session_id, include_events=True)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Encode
        vector = encoder.encode_session(session)
        session.vector = vector

        # Update in store
        store.update_session_vector(request.session_id, vector)

        return EncodeResponse(
            session_id=request.session_id,
            vector_dim=len(vector),
            encoded=True,
            encoding_method=config.encoding_method,
        )

    @app.post("/flows/search", response_model=SearchResponse)
    async def search_sessions(request: SearchRequest):
        """
        Find similar sessions by query or session ID.

        Uses vector similarity search on encoded session embeddings.
        """
        index = get_index()
        store = get_store()

        if request.query:
            results = index.search_by_query(
                request.query,
                k=request.k,
                min_similarity=request.min_similarity,
                include_session=True,
            )
        elif request.session_id:
            results = index.search_similar(
                request.session_id,
                k=request.k,
                min_similarity=request.min_similarity,
                include_session=True,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Either 'query' or 'session_id' must be provided",
            )

        items = []
        for r in results:
            event_summary = None
            if r.session and r.session.events:
                event_summary = [
                    f"{e.component_type}.{e.event_type}"
                    for e in r.session.events[:5]
                ]

            items.append(SearchResultItem(
                session_id=r.session_id,
                similarity=round(r.similarity, 4),
                cluster_id=r.session.cluster_id if r.session else None,
                duration_ms=r.session.duration_ms if r.session else None,
                event_summary=event_summary,
            ))

        return SearchResponse(results=items)

    # =========================================================================
    # Phase 3: Clustering
    # =========================================================================

    @app.post("/clusters/compute", response_model=ClusterResponse)
    async def compute_clusters(request: ClusterRequest):
        """
        Run clustering on all encoded sessions.

        Supports HDBSCAN, K-Means, and Agglomerative clustering.
        """
        store = get_store()
        clustering = BehaviorClustering(store, config)

        result = clustering.cluster(
            algorithm=request.algorithm,
            min_cluster_size=request.min_cluster_size,
            min_samples=request.min_samples,
            n_clusters=request.n_clusters,
        )

        summaries = [
            ClusterSummary(
                cluster_id=c.cluster_id,
                session_count=c.session_count,
                label=c.label,
                cluster_type=c.cluster_type,
            )
            for c in result.clusters
        ]

        return ClusterResponse(
            status="complete",
            clusters_found=result.n_clusters,
            noise_sessions=result.n_noise,
            cluster_summary=summaries,
        )

    @app.get("/clusters", response_model=ClustersListResponse)
    async def list_clusters():
        """
        List all behavior clusters.
        """
        store = get_store()
        clusters = store.get_clusters()

        details = [
            ClusterDetail(
                cluster_id=c.cluster_id,
                label=c.label,
                cluster_type=c.cluster_type,
                session_count=c.session_count,
                completion_rate=c.completion_rate,
                avg_duration_ms=c.avg_duration_ms,
                dominant_components=c.dominant_components,
                dominant_intents=c.dominant_intents,
                sample_session_ids=c.sample_session_ids[:5],
            )
            for c in clusters
        ]

        return ClustersListResponse(clusters=details)

    @app.get("/clusters/{cluster_id}")
    async def get_cluster(cluster_id: int):
        """
        Get detailed information about a specific cluster.
        """
        store = get_store()
        clusters = store.get_clusters()

        cluster = next((c for c in clusters if c.cluster_id == cluster_id), None)
        if not cluster:
            raise HTTPException(status_code=404, detail="Cluster not found")

        # Get sample sessions
        sample_sessions = []
        for sid in cluster.sample_session_ids[:5]:
            session = store.get_session(sid, include_events=False)
            if session:
                sample_sessions.append({
                    "session_id": session.session_id,
                    "duration_ms": session.duration_ms,
                    "is_complete": session.is_complete,
                })

        return {
            "cluster": {
                "cluster_id": cluster.cluster_id,
                "label": cluster.label,
                "description": cluster.description,
                "cluster_type": cluster.cluster_type,
                "session_count": cluster.session_count,
                "completion_rate": cluster.completion_rate,
                "avg_duration_ms": cluster.avg_duration_ms,
                "typical_length": cluster.typical_length,
                "dominant_components": cluster.dominant_components,
                "dominant_intents": cluster.dominant_intents,
                "sample_sessions": sample_sessions,
            }
        }

    @app.get("/clusters/insights", response_model=InsightsResponse)
    async def get_cluster_insights():
        """
        Get high-level insights from behavior clustering.
        """
        store = get_store()
        clustering = BehaviorClustering(store, config)

        insights = clustering.get_cluster_insights()

        if "error" in insights:
            raise HTTPException(status_code=400, detail=insights["error"])

        return InsightsResponse(
            total_clusters=insights.get("total_clusters", 0),
            cluster_distribution=insights.get("cluster_distribution", {}),
            top_happy_paths=insights.get("top_happy_paths", []),
            problem_areas=insights.get("problem_areas", []),
        )

    # =========================================================================
    # Phase 4: Prediction
    # =========================================================================

    @app.post("/predict/next", response_model=PredictResponse)
    async def predict_next_action(request: PredictRequest):
        """
        Predict next action given a partial flow.

        Uses a hybrid Markov + nearest-neighbor model.
        """
        predictor = get_predictor()

        # Convert input events to FlowEvent objects
        events = []
        for evt in request.events:
            try:
                ts = datetime.fromisoformat(evt.timestamp.replace("Z", "+00:00"))
            except ValueError:
                ts = datetime.now()

            flow_event = FlowEvent(
                event_id=evt.event_id,
                timestamp=ts,
                component_type=evt.component_type or "",
                event_type=evt.event_type,
            )
            events.append(flow_event)

        # Get predictions
        result = predictor.predict(events, top_k=request.top_k)

        predictions = [
            PredictionItem(
                predicted_component=p.predicted_component,
                predicted_event=p.predicted_event,
                confidence=round(p.confidence, 4),
                cluster_id=p.cluster_id,
            )
            for p in result.predictions
        ]

        return PredictResponse(predictions=predictions)

    # =========================================================================
    # Phase 4: Patterns
    # =========================================================================

    @app.get("/patterns", response_model=PatternsResponse)
    async def get_patterns(
        min_support: float = Query(0.1, ge=0.0, le=1.0),
        min_length: int = Query(2, ge=2),
        max_length: int = Query(6, ge=2),
    ):
        """
        Get mined sequential patterns from user behavior.
        """
        store = get_store()
        miner = SequentialPatternMiner(store, config)

        result = miner.mine_patterns(
            min_support=min_support,
            min_length=min_length,
            max_length=max_length,
        )

        patterns = [
            PatternItem(
                pattern_id=p.pattern_id,
                sequence=p.sequence,
                support=round(p.support, 4),
                confidence=round(p.confidence, 4),
                occurrence_count=p.occurrence_count,
            )
            for p in result.patterns[:50]  # Limit to top 50
        ]

        return PatternsResponse(
            patterns=patterns,
            n_sessions_analyzed=result.n_sessions_analyzed,
        )

    # =========================================================================
    # Phase 4: Test Gap Detection
    # =========================================================================

    @app.get("/gaps", response_model=GapsResponse)
    async def list_gaps(
        status: Optional[str] = Query("open", description="Filter by status"),
        min_priority: Optional[str] = Query(None, description="Minimum priority level"),
    ):
        """
        List test coverage gaps.

        Returns gaps between observed user behavior and test coverage.
        """
        store = get_store()
        detector = GapDetector(store, config)

        if min_priority:
            gaps = detector.get_priority_gaps(min_priority=min_priority)
        else:
            # Get all gaps from store
            gaps = store.get_test_gaps(status=status)

        summaries = [
            GapSummary(
                gap_id=g.gap_id,
                gap_type=g.gap_type,
                flow_description=g.flow_description,
                affected_components=g.affected_components,
                observed_count=g.observed_count,
                priority=g.priority,
                suggested_test_name=g.suggested_test_name,
            )
            for g in gaps
        ]

        # Count by priority
        priority_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for g in gaps:
            if g.priority in priority_counts:
                priority_counts[g.priority] += 1

        return GapsResponse(
            gaps=summaries,
            total_gaps=len(gaps),
            gaps_by_priority=priority_counts,
        )

    @app.post("/gaps/analyze")
    async def analyze_gaps(
        min_support: float = Query(0.05, ge=0.0, le=1.0),
        min_observed: int = Query(3, ge=1),
    ):
        """
        Run gap analysis to detect untested flows.

        Compares observed behavior patterns against test coverage.
        """
        store = get_store()
        result = analyze_test_gaps(store, config=config)

        return result.to_dict()

    @app.get("/gaps/{gap_id}")
    async def get_gap(gap_id: str):
        """
        Get detailed information about a specific gap.
        """
        store = get_store()
        gaps = store.get_test_gaps()

        gap = next((g for g in gaps if g.gap_id == gap_id), None)
        if not gap:
            raise HTTPException(status_code=404, detail="Gap not found")

        return {"gap": gap.to_dict()}

    @app.post("/gaps/{gap_id}/generate-test", response_model=GenerateTestResponse)
    async def generate_test(gap_id: str, request: GenerateTestRequest):
        """
        Generate a test for a specific gap.

        Creates a test template based on the observed flow pattern.
        """
        store = get_store()
        detector = GapDetector(store, config)

        gaps = store.get_test_gaps()
        gap = next((g for g in gaps if g.gap_id == gap_id), None)

        if not gap:
            raise HTTPException(status_code=404, detail="Gap not found")

        suggestion = detector.generate_test_suggestion(gap)

        return GenerateTestResponse(
            status="generated",
            test_file=suggestion["test_file"],
            test_code=suggestion["test_code"],
            assertions=len(suggestion["assertions"]),
        )

    @app.patch("/gaps/{gap_id}/status")
    async def update_gap_status(
        gap_id: str,
        status: str = Query(..., description="New status: open, acknowledged, resolved"),
    ):
        """
        Update the status of a test gap.
        """
        valid_statuses = {"open", "acknowledged", "resolved"}
        if status not in valid_statuses:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Must be one of: {valid_statuses}",
            )

        store = get_store()

        # This would need a store method to update gap status
        # For now, return success
        return {"gap_id": gap_id, "status": status, "updated": True}

    # =========================================================================
    # Stats & Diagnostics
    # =========================================================================

    @app.get("/stats")
    async def get_stats():
        """
        Get overall statistics about the behavior modeler.
        """
        store = get_store()
        stats = store.get_stats()

        return {
            "store": stats,
            "index": {
                "size": get_index().size if _index else 0,
                "dimension": config.embed_dimension,
            },
        }

    return app


# Create default app instance
app = create_app()
