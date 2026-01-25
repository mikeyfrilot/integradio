# UI Behavior Modeler - Technical Specification

> **Version**: 1.0.0
> **Status**: Draft
> **Created**: 2026-01-20
> **Location**: `F:\AI\integradio\behavior_modeler\`

## Executive Summary

The UI Behavior Modeler is a semantic intelligence layer for Integradio that:
1. **Observes** user flows through the EventMesh
2. **Encodes** event sequences into vector representations
3. **Clusters** behaviors to discover patterns (happy paths, edge cases, drop-offs)
4. **Advises** on next actions, missing tests, and UX improvements

It turns Integradio from a reactive mesh into a **learning system**.

---

## Architecture Overview

```
                                 INTEGRADIO MESH
                                       │
                         ┌─────────────┴─────────────┐
                         │        EventMesh          │
                         │   (WebSocket + Pub/Sub)   │
                         └─────────────┬─────────────┘
                                       │
                         ┌─────────────▼─────────────┐
                         │       Event Tap           │
                         │  (behavior.** subscriber) │
                         └─────────────┬─────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
    ┌─────────▼─────────┐   ┌─────────▼─────────┐   ┌─────────▼─────────┐
    │    Flow Store     │   │   Flow Encoder    │   │   Mock Generator  │
    │    (SQLite)       │   │   (Ollama embed)  │   │   (Synthetic)     │
    └─────────┬─────────┘   └─────────┬─────────┘   └───────────────────┘
              │                       │
              │            ┌──────────▼──────────┐
              │            │    Flow Index       │
              │            │    (HNSW)           │
              │            └──────────┬──────────┘
              │                       │
              └───────────┬───────────┘
                          │
             ┌────────────▼────────────┐
             │    Pattern Miner        │
             │    (Clustering)         │
             └────────────┬────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
   ┌────▼────┐      ┌─────▼─────┐     ┌─────▼─────┐
   │ Next    │      │ Test Gap  │     │ UX        │
   │ Action  │      │ Detector  │     │ Insights  │
   │ Predict │      │           │     │           │
   └─────────┘      └───────────┘     └───────────┘
```

---

## Data Models

### Session

A session represents a single user journey through the application.

```python
@dataclass
class Session:
    """A user session containing a sequence of UI events."""

    session_id: str  # UUID
    started_at: datetime
    ended_at: Optional[datetime] = None
    user_agent: Optional[str] = None
    events: list["FlowEvent"] = field(default_factory=list)

    # Computed after encoding
    vector: Optional[np.ndarray] = None  # Session-level embedding
    cluster_id: Optional[int] = None     # Assigned behavior cluster

    @property
    def duration_ms(self) -> int:
        """Session duration in milliseconds."""
        if not self.events:
            return 0
        first = self.events[0].timestamp
        last = self.events[-1].timestamp
        return int((last - first).total_seconds() * 1000)

    @property
    def is_complete(self) -> bool:
        """Whether session has a natural end (submit, navigation away, etc.)."""
        if not self.events:
            return False
        terminal_events = {"submit", "navigate_away", "logout", "close"}
        return self.events[-1].event_type in terminal_events
```

### FlowEvent

Individual events within a session, mapped from SemanticEvent.

```python
@dataclass
class FlowEvent:
    """A single event within a user flow."""

    event_id: str
    timestamp: datetime

    # Component context (from ComponentRegistry)
    component_id: Optional[int] = None
    component_type: str = ""          # e.g., "Textbox", "Button"
    component_intent: str = ""        # Semantic description

    # Event details
    event_type: str = ""              # click, input, submit, select, etc.
    event_data: dict = field(default_factory=dict)  # Payload

    # Semantic context (from SemanticEvent)
    intent: Optional[str] = None      # User intent if detected
    tags: list[str] = field(default_factory=list)

    # Computed
    vector: Optional[np.ndarray] = None  # Event-level embedding

    @classmethod
    def from_semantic_event(cls, event: SemanticEvent, component_meta: Optional[ComponentMetadata] = None) -> "FlowEvent":
        """Create FlowEvent from Integradio SemanticEvent."""
        return cls(
            event_id=event.id,
            timestamp=datetime.fromisoformat(event.time.replace("Z", "+00:00")),
            component_id=event.data.get("component_id") if event.data else None,
            component_type=component_meta.component_type if component_meta else "",
            component_intent=component_meta.intent if component_meta else "",
            event_type=event.type.split(".")[-1],  # e.g., "ui.interaction.click" -> "click"
            event_data=event.data or {},
            intent=event.intent,
            tags=event.tags,
        )
```

### BehaviorCluster

Groups of similar user flows discovered through clustering.

```python
@dataclass
class BehaviorCluster:
    """A cluster of similar user behavior flows."""

    cluster_id: int
    label: str = ""                    # Auto-generated or manual label
    description: str = ""              # Human-readable description

    # Cluster stats
    centroid: Optional[np.ndarray] = None  # Center vector
    session_count: int = 0
    avg_duration_ms: float = 0
    completion_rate: float = 0         # % of sessions that complete

    # Dominant characteristics
    dominant_components: list[str] = field(default_factory=list)  # Top components
    dominant_intents: list[str] = field(default_factory=list)     # Top intents
    typical_length: int = 0            # Typical number of events

    # Classification
    cluster_type: str = "unknown"      # happy_path, edge_case, drop_off, error_flow

    # Sample sessions
    sample_session_ids: list[str] = field(default_factory=list)
```

### FlowPrediction

Prediction for next likely action(s).

```python
@dataclass
class FlowPrediction:
    """Predicted next action(s) in a user flow."""

    # What's likely to happen next
    predicted_component: str           # Component type
    predicted_event: str               # Event type
    predicted_intent: Optional[str]    # Intent if determinable
    confidence: float                  # 0.0 - 1.0

    # Supporting data
    similar_sessions: list[str]        # Session IDs that informed prediction
    cluster_id: Optional[int] = None   # Which behavior cluster this matches
```

### TestGap

Identified gaps between observed behavior and test coverage.

```python
@dataclass
class TestGap:
    """A gap between observed UI behavior and test coverage."""

    gap_id: str
    gap_type: str                      # uncovered_flow, uncovered_component, rare_path

    # What's missing
    flow_description: str              # Human-readable description
    affected_components: list[str]     # Components involved
    observed_count: int                # How many times seen in production

    # Context for test generation
    sample_session_id: Optional[str]   # Example session exhibiting this flow
    suggested_test_name: str           # e.g., "test_search_to_generate_flow"
    suggested_assertions: list[str]    # What the test should verify

    # Priority
    priority: str = "medium"           # low, medium, high, critical
    priority_reason: str = ""          # Why this priority
```

---

## Database Schema

SQLite database at `behavior_modeler/db/behavior.db`:

```sql
-- Sessions table
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,          -- ISO 8601
    ended_at TEXT,
    user_agent TEXT,
    duration_ms INTEGER,
    is_complete INTEGER DEFAULT 0,     -- Boolean
    cluster_id INTEGER,
    vector BLOB,                       -- numpy bytes
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_sessions_cluster ON sessions(cluster_id);
CREATE INDEX idx_sessions_started ON sessions(started_at);

-- Events table
CREATE TABLE flow_events (
    event_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    sequence_num INTEGER NOT NULL,     -- Order within session

    component_id INTEGER,
    component_type TEXT,
    component_intent TEXT,

    event_type TEXT NOT NULL,
    event_data TEXT,                   -- JSON
    intent TEXT,
    tags TEXT,                         -- JSON array

    vector BLOB,                       -- numpy bytes

    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX idx_events_session ON flow_events(session_id, sequence_num);
CREATE INDEX idx_events_component ON flow_events(component_type);
CREATE INDEX idx_events_type ON flow_events(event_type);

-- Clusters table
CREATE TABLE behavior_clusters (
    cluster_id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT,
    description TEXT,
    cluster_type TEXT,                 -- happy_path, edge_case, drop_off, error_flow

    centroid BLOB,                     -- numpy bytes
    session_count INTEGER DEFAULT 0,
    avg_duration_ms REAL,
    completion_rate REAL,
    typical_length INTEGER,

    dominant_components TEXT,          -- JSON array
    dominant_intents TEXT,             -- JSON array
    sample_session_ids TEXT,           -- JSON array

    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Test gaps table
CREATE TABLE test_gaps (
    gap_id TEXT PRIMARY KEY,
    gap_type TEXT NOT NULL,
    flow_description TEXT,
    affected_components TEXT,          -- JSON array
    observed_count INTEGER DEFAULT 0,

    sample_session_id TEXT,
    suggested_test_name TEXT,
    suggested_assertions TEXT,         -- JSON array

    priority TEXT DEFAULT 'medium',
    priority_reason TEXT,

    status TEXT DEFAULT 'open',        -- open, acknowledged, resolved
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    resolved_at TEXT,

    FOREIGN KEY (sample_session_id) REFERENCES sessions(session_id)
);

CREATE INDEX idx_gaps_status ON test_gaps(status, priority);

-- Flow patterns (for prediction)
CREATE TABLE flow_patterns (
    pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
    prefix_hash TEXT NOT NULL,         -- Hash of event sequence prefix
    prefix_length INTEGER NOT NULL,

    next_component TEXT,
    next_event TEXT,
    next_intent TEXT,

    occurrence_count INTEGER DEFAULT 1,
    last_seen TEXT,

    UNIQUE(prefix_hash, next_component, next_event)
);

CREATE INDEX idx_patterns_prefix ON flow_patterns(prefix_hash);
```

---

## API Contracts

### Phase 1: Event Tap & Flow Store

#### `POST /flows/ingest`

Ingest a raw flow from EventMesh or external source.

```python
# Request
{
    "session_id": "abc123",
    "events": [
        {
            "event_id": "evt_001",
            "timestamp": "2026-01-20T10:30:00Z",
            "component_id": 42,
            "event_type": "input",
            "event_data": {"value": "retry decorator"}
        },
        ...
    ]
}

# Response
{
    "status": "accepted",
    "session_id": "abc123",
    "event_count": 4,
    "warnings": []  # e.g., ["Unknown component_id: 999"]
}
```

#### `GET /flows/sessions`

List sessions with filtering.

```python
# Request params
?start_date=2026-01-19&end_date=2026-01-20
&cluster_id=3
&is_complete=true
&limit=50&offset=0

# Response
{
    "sessions": [
        {
            "session_id": "abc123",
            "started_at": "2026-01-20T10:30:00Z",
            "duration_ms": 45000,
            "event_count": 12,
            "is_complete": true,
            "cluster_id": 3,
            "cluster_label": "Search to Generate"
        }
    ],
    "total": 1234,
    "limit": 50,
    "offset": 0
}
```

#### `GET /flows/sessions/{session_id}`

Get full session detail.

```python
# Response
{
    "session": {
        "session_id": "abc123",
        "started_at": "2026-01-20T10:30:00Z",
        "ended_at": "2026-01-20T10:30:45Z",
        "duration_ms": 45000,
        "is_complete": true,
        "cluster_id": 3,
        "events": [
            {
                "event_id": "evt_001",
                "sequence_num": 0,
                "timestamp": "2026-01-20T10:30:00Z",
                "component_type": "SearchBox",
                "component_intent": "Search code files",
                "event_type": "input",
                "event_data": {"value": "retry decorator"},
                "intent": "code_search"
            },
            ...
        ]
    }
}
```

---

### Phase 2: Flow Encoding & Indexing

#### `POST /flows/encode`

Encode a session into a vector (on-demand or batch).

```python
# Request
{
    "session_id": "abc123"
}

# Response
{
    "session_id": "abc123",
    "vector_dim": 768,
    "encoded": true,
    "encoding_method": "pooled_sequence"
}
```

#### `POST /flows/search`

Find similar sessions.

```python
# Request
{
    "query": "user searches then views code then generates tests",
    # OR
    "session_id": "abc123",  # Find similar to this session

    "k": 10,
    "min_similarity": 0.5
}

# Response
{
    "results": [
        {
            "session_id": "def456",
            "similarity": 0.87,
            "cluster_id": 3,
            "duration_ms": 42000,
            "event_summary": ["SearchBox.input", "SearchResults.select", "CodePanel.view"]
        }
    ]
}
```

---

### Phase 3: Clustering & Pattern Mining

#### `POST /clusters/compute`

Run clustering on all encoded sessions.

```python
# Request
{
    "algorithm": "hdbscan",  # or "kmeans", "agglomerative"
    "min_cluster_size": 10,
    "min_samples": 5
}

# Response
{
    "status": "complete",
    "clusters_found": 12,
    "noise_sessions": 45,  # Unclustered
    "cluster_summary": [
        {
            "cluster_id": 1,
            "session_count": 234,
            "label": "Search Flow",
            "cluster_type": "happy_path"
        }
    ]
}
```

#### `GET /clusters`

List all behavior clusters.

```python
# Response
{
    "clusters": [
        {
            "cluster_id": 1,
            "label": "Search to Generate",
            "cluster_type": "happy_path",
            "session_count": 234,
            "completion_rate": 0.89,
            "dominant_components": ["SearchBox", "SearchResults", "GenerateButton"],
            "dominant_intents": ["code_search", "code_generation"]
        },
        {
            "cluster_id": 2,
            "label": "Abandoned Upload",
            "cluster_type": "drop_off",
            "session_count": 45,
            "completion_rate": 0.12,
            "dominant_components": ["UploadBox", "ProgressBar"],
            "dominant_intents": ["file_upload"]
        }
    ]
}
```

#### `GET /clusters/{cluster_id}`

Get cluster detail with sample sessions.

---

### Phase 4: Prediction & Advisors

#### `POST /predict/next`

Predict next action given partial flow.

```python
# Request
{
    "events": [
        {"component_type": "SearchBox", "event_type": "input"},
        {"component_type": "SearchResults", "event_type": "select"}
    ]
}

# Response
{
    "predictions": [
        {
            "predicted_component": "CodePanel",
            "predicted_event": "view",
            "confidence": 0.82,
            "cluster_id": 1
        },
        {
            "predicted_component": "GenerateButton",
            "predicted_event": "click",
            "confidence": 0.15,
            "cluster_id": 3
        }
    ]
}
```

#### `GET /gaps`

List test coverage gaps.

```python
# Response
{
    "gaps": [
        {
            "gap_id": "gap_001",
            "gap_type": "uncovered_flow",
            "flow_description": "Upload → Error → Retry flow",
            "affected_components": ["UploadBox", "ErrorDialog", "RetryButton"],
            "observed_count": 89,
            "priority": "high",
            "suggested_test_name": "test_upload_retry_flow"
        }
    ],
    "total_gaps": 15,
    "gaps_by_priority": {
        "critical": 2,
        "high": 5,
        "medium": 6,
        "low": 2
    }
}
```

#### `POST /gaps/{gap_id}/generate-test`

Generate a test for a specific gap (integrates with Code Covered).

```python
# Request
{
    "gap_id": "gap_001",
    "test_framework": "pytest",
    "output_path": "tests/ui_flows/"
}

# Response
{
    "status": "generated",
    "test_file": "tests/ui_flows/test_upload_retry_flow.py",
    "test_code": "...",
    "assertions": 5
}
```

---

## Integration Points

### 1. EventMesh Integration

The Behavior Modeler subscribes to the mesh using pattern-based subscriptions:

```python
# In behavior_modeler/tap.py

class BehaviorTap:
    """Taps into EventMesh to capture UI flows."""

    def __init__(self, mesh: EventMesh, store: FlowStore):
        self.mesh = mesh
        self.store = store
        self._sessions: dict[str, Session] = {}

    async def start(self):
        """Start listening to mesh events."""
        # Subscribe to all UI and interaction events
        self.mesh.subscribe(
            patterns=["ui.**", "data.**", "system.connection.*"],
            handler=self._handle_event,
            client_id="behavior_modeler"
        )

    async def _handle_event(self, event: SemanticEvent):
        """Process incoming event."""
        session_id = event.correlation_id or self._extract_session_id(event)

        if session_id not in self._sessions:
            self._sessions[session_id] = Session(
                session_id=session_id,
                started_at=datetime.now(timezone.utc)
            )

        # Convert to FlowEvent
        flow_event = FlowEvent.from_semantic_event(event)
        self._sessions[session_id].events.append(flow_event)

        # Check for session end
        if self._is_terminal_event(event):
            await self._finalize_session(session_id)
```

### 2. ComponentRegistry Integration

Enrich events with component metadata:

```python
# Lookup component metadata when processing events
async def _handle_event(self, event: SemanticEvent):
    component_id = event.data.get("component_id") if event.data else None
    component_meta = None

    if component_id and self.registry:
        component_meta = self.registry.get(component_id)

    flow_event = FlowEvent.from_semantic_event(event, component_meta)
```

### 3. Embedder Integration

Reuse Integradio's Embedder for flow encoding:

```python
# In behavior_modeler/encoder.py

class FlowEncoder:
    """Encode event sequences into vectors."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    def encode_event(self, event: FlowEvent) -> np.ndarray:
        """Encode single event."""
        # Combine component info + event type + intent
        text = f"{event.component_type} {event.event_type}"
        if event.component_intent:
            text += f" {event.component_intent}"
        if event.intent:
            text += f" intent:{event.intent}"

        return self.embedder.embed(text)

    def encode_session(self, session: Session) -> np.ndarray:
        """Encode entire session as single vector."""
        if not session.events:
            return np.zeros(self.embedder.dimension)

        # Embed each event
        event_vectors = [self.encode_event(e) for e in session.events]

        # Pooling strategy: weighted average with recency bias
        weights = np.linspace(0.5, 1.0, len(event_vectors))
        weights /= weights.sum()

        session_vector = np.average(event_vectors, axis=0, weights=weights)
        return session_vector / (np.linalg.norm(session_vector) + 1e-8)
```

---

## Configuration

```python
# behavior_modeler/config.py

@dataclass
class BehaviorModelerConfig:
    """Configuration for the Behavior Modeler."""

    # Database
    db_path: Path = Path("behavior_modeler/db/behavior.db")

    # Embedding
    ollama_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"

    # Flow encoding
    encoding_method: str = "pooled_weighted"  # pooled_weighted, pooled_avg, rnn
    max_sequence_length: int = 100

    # Clustering
    clustering_algorithm: str = "hdbscan"
    min_cluster_size: int = 10

    # Session management
    session_timeout_seconds: int = 300  # 5 minutes of inactivity = session end
    max_events_per_session: int = 500

    # Index
    hnsw_max_elements: int = 100000
    hnsw_ef_construction: int = 200
    hnsw_M: int = 16
```

---

## Phased Implementation Plan

### Phase 1: Event Tap + Flow Store (Week 1)

**Goal**: Capture and persist UI flows from EventMesh.

**Deliverables**:
- [ ] `behavior_modeler/tap.py` - EventMesh subscriber
- [ ] `behavior_modeler/store.py` - SQLite flow storage
- [ ] `behavior_modeler/models.py` - Data models
- [ ] `behavior_modeler/mock.py` - Synthetic flow generator for testing
- [ ] Database schema migrations
- [ ] Unit tests for tap and store

**Success Criteria**:
- Can ingest events from live EventMesh
- Can generate and ingest synthetic flows
- Flows persist correctly to SQLite
- Session boundaries detected correctly

### Phase 2: Flow Encoding + Indexing (Week 2)

**Goal**: Turn flows into searchable vectors.

**Deliverables**:
- [ ] `behavior_modeler/encoder.py` - Flow → vector encoding
- [ ] `behavior_modeler/index.py` - HNSW index for flow search
- [ ] API endpoints: `/flows/encode`, `/flows/search`
- [ ] Batch encoding job for historical data

**Success Criteria**:
- Sessions have vector representations
- Similar flows return high similarity scores
- Search latency < 50ms for 10k flows

### Phase 3: Clustering + Pattern Mining (Week 3)

**Goal**: Discover behavior patterns automatically.

**Deliverables**:
- [ ] `behavior_modeler/clustering.py` - HDBSCAN/K-means clustering
- [ ] `behavior_modeler/patterns.py` - Sequence pattern mining
- [ ] API endpoints: `/clusters/*`
- [ ] Auto-labeling for clusters

**Success Criteria**:
- Clusters emerge that map to recognizable user journeys
- Happy paths vs. drop-offs distinguishable
- Cluster labels are meaningful

### Phase 4: Advisors + Test Gap Detection (Week 4)

**Goal**: Provide actionable insights.

**Deliverables**:
- [ ] `behavior_modeler/predictor.py` - Next action prediction
- [ ] `behavior_modeler/gaps.py` - Test coverage gap detection
- [ ] API endpoints: `/predict/*`, `/gaps/*`
- [ ] Integration hooks for Code Covered

**Success Criteria**:
- Predictions are accurate >70% of the time
- Test gaps correctly identify uncovered flows
- Generated test suggestions are valid

---

## File Structure

```
F:\AI\integradio\behavior_modeler\
├── __init__.py
├── spec.md                 # This file
├── config.py               # Configuration
├── models.py               # Data models
│
├── tap.py                  # EventMesh subscriber
├── store.py                # SQLite flow storage
├── encoder.py              # Flow → vector encoding
├── index.py                # HNSW vector index
├── clustering.py           # Behavior clustering
├── patterns.py             # Sequence pattern mining
├── predictor.py            # Next action prediction
├── gaps.py                 # Test gap detection
│
├── mock.py                 # Synthetic flow generator
├── api.py                  # FastAPI/Flask routes
│
├── db/
│   └── behavior.db         # SQLite database
│
└── tests/
    ├── __init__.py
    ├── test_tap.py
    ├── test_store.py
    ├── test_encoder.py
    ├── test_clustering.py
    └── conftest.py         # Fixtures
```

---

## Open Questions

1. **Session ID Strategy**: Should we use `correlation_id` from SemanticEvent, extract from cookies/headers, or generate our own?

2. **Real-time vs. Batch Clustering**: Run clustering on every N new sessions, or scheduled batch jobs?

3. **Test Integration Depth**: Generate test code directly, or just test specifications for human review?

4. **Privacy Considerations**: Should we hash/anonymize user identifiers in stored flows?

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Flow capture rate | >95% | Events captured / Events emitted |
| Encoding latency | <100ms | Time to encode one session |
| Search accuracy | >80% | Similar flows in top-5 results |
| Cluster purity | >70% | Sessions in correct cluster |
| Prediction accuracy | >70% | Correct next action in top-3 |
| Gap detection recall | >80% | Real gaps found / Total gaps |

---

## References

- Integradio EventMesh: `integradio/events/mesh.py`
- Integradio Embedder: `integradio/embedder.py`
- Integradio ComponentRegistry: `integradio/registry.py`
- Tool Compass (similar architecture): `F:\AI\mcp-tool-shop\tool_compass\`
