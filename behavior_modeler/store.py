"""
Flow Store - SQLite persistence for user flows and sessions.

Handles storage, retrieval, and querying of behavioral data.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Iterator
import numpy as np

from .config import BehaviorModelerConfig
from .models import Session, FlowEvent, BehaviorCluster, TestGap

logger = logging.getLogger(__name__)


class FlowStore:
    """SQLite-backed storage for user flows."""

    def __init__(self, config: Optional[BehaviorModelerConfig] = None):
        """
        Initialize the flow store.

        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or BehaviorModelerConfig()
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database connection and schema."""
        db_path = self.config.db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        self._create_schema()
        logger.info(f"FlowStore initialized: {db_path}")

    def _create_schema(self) -> None:
        """Create database schema if not exists."""
        schema = """
        -- Sessions table
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            user_agent TEXT,
            duration_ms INTEGER,
            is_complete INTEGER DEFAULT 0,
            cluster_id INTEGER,
            vector BLOB,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_sessions_cluster ON sessions(cluster_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at);
        CREATE INDEX IF NOT EXISTS idx_sessions_complete ON sessions(is_complete);

        -- Events table
        CREATE TABLE IF NOT EXISTS flow_events (
            event_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            sequence_num INTEGER NOT NULL,

            component_id INTEGER,
            component_type TEXT,
            component_intent TEXT,

            event_type TEXT NOT NULL,
            event_data TEXT,
            intent TEXT,
            tags TEXT,

            vector BLOB,

            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        );

        CREATE INDEX IF NOT EXISTS idx_events_session ON flow_events(session_id, sequence_num);
        CREATE INDEX IF NOT EXISTS idx_events_component ON flow_events(component_type);
        CREATE INDEX IF NOT EXISTS idx_events_type ON flow_events(event_type);

        -- Clusters table
        CREATE TABLE IF NOT EXISTS behavior_clusters (
            cluster_id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT,
            description TEXT,
            cluster_type TEXT,

            centroid BLOB,
            session_count INTEGER DEFAULT 0,
            avg_duration_ms REAL,
            completion_rate REAL,
            typical_length INTEGER,

            dominant_components TEXT,
            dominant_intents TEXT,
            sample_session_ids TEXT,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        -- Test gaps table
        CREATE TABLE IF NOT EXISTS test_gaps (
            gap_id TEXT PRIMARY KEY,
            gap_type TEXT NOT NULL,
            flow_description TEXT,
            affected_components TEXT,
            observed_count INTEGER DEFAULT 0,

            sample_session_id TEXT,
            suggested_test_name TEXT,
            suggested_assertions TEXT,

            priority TEXT DEFAULT 'medium',
            priority_reason TEXT,

            status TEXT DEFAULT 'open',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            resolved_at TEXT,

            FOREIGN KEY (sample_session_id) REFERENCES sessions(session_id)
        );

        CREATE INDEX IF NOT EXISTS idx_gaps_status ON test_gaps(status, priority);

        -- Flow patterns for prediction
        CREATE TABLE IF NOT EXISTS flow_patterns (
            pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
            prefix_hash TEXT NOT NULL,
            prefix_length INTEGER NOT NULL,

            next_component TEXT,
            next_event TEXT,
            next_intent TEXT,

            occurrence_count INTEGER DEFAULT 1,
            last_seen TEXT,

            UNIQUE(prefix_hash, next_component, next_event)
        );

        CREATE INDEX IF NOT EXISTS idx_patterns_prefix ON flow_patterns(prefix_hash);
        """
        try:
            self._conn.executescript(schema)
            self._conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to create schema: {e}")
            raise

    def save_session(self, session: Session) -> bool:
        """
        Save a session and its events.

        Args:
            session: Session to save

        Returns:
            True if successful
        """
        try:
            # Convert vector to bytes if present
            vector_bytes = session.vector.tobytes() if session.vector is not None else None

            # Insert/update session
            self._conn.execute(
                """
                INSERT OR REPLACE INTO sessions
                (session_id, started_at, ended_at, user_agent, duration_ms,
                 is_complete, cluster_id, vector)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.session_id,
                    session.started_at.isoformat(),
                    session.ended_at.isoformat() if session.ended_at else None,
                    session.user_agent,
                    session.duration_ms,
                    1 if session.is_complete else 0,
                    session.cluster_id,
                    vector_bytes,
                ),
            )

            # Delete existing events for this session (for updates)
            self._conn.execute(
                "DELETE FROM flow_events WHERE session_id = ?",
                (session.session_id,),
            )

            # Insert events
            for i, event in enumerate(session.events):
                event_vector = event.vector.tobytes() if event.vector is not None else None

                self._conn.execute(
                    """
                    INSERT INTO flow_events
                    (event_id, session_id, timestamp, sequence_num, component_id,
                     component_type, component_intent, event_type, event_data,
                     intent, tags, vector)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id,
                        session.session_id,
                        event.timestamp.isoformat(),
                        i,
                        event.component_id,
                        event.component_type,
                        event.component_intent,
                        event.event_type,
                        json.dumps(event.event_data),
                        event.intent,
                        json.dumps(event.tags),
                        event_vector,
                    ),
                )

            self._conn.commit()
            logger.debug(f"Saved session {session.session_id} with {len(session.events)} events")
            return True

        except sqlite3.Error as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")
            self._conn.rollback()
            return False

    def get_session(self, session_id: str, include_events: bool = True) -> Optional[Session]:
        """
        Retrieve a session by ID.

        Args:
            session_id: Session ID
            include_events: Whether to load events

        Returns:
            Session or None if not found
        """
        try:
            cursor = self._conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            session = self._row_to_session(row)

            if include_events:
                session.events = self._load_events(session_id)

            return session

        except sqlite3.Error as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None

    def _row_to_session(self, row: sqlite3.Row) -> Session:
        """Convert database row to Session object."""
        started_at = datetime.fromisoformat(row["started_at"])
        ended_at = datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None

        vector = None
        if row["vector"]:
            vector = np.frombuffer(row["vector"], dtype=np.float32)

        return Session(
            session_id=row["session_id"],
            started_at=started_at,
            ended_at=ended_at,
            user_agent=row["user_agent"],
            vector=vector,
            cluster_id=row["cluster_id"],
            events=[],  # Loaded separately if needed
        )

    def _load_events(self, session_id: str) -> list[FlowEvent]:
        """Load events for a session."""
        cursor = self._conn.execute(
            """
            SELECT * FROM flow_events
            WHERE session_id = ?
            ORDER BY sequence_num
            """,
            (session_id,),
        )

        events = []
        for row in cursor:
            timestamp = datetime.fromisoformat(row["timestamp"])

            vector = None
            if row["vector"]:
                vector = np.frombuffer(row["vector"], dtype=np.float32)

            event = FlowEvent(
                event_id=row["event_id"],
                timestamp=timestamp,
                component_id=row["component_id"],
                component_type=row["component_type"] or "",
                component_intent=row["component_intent"] or "",
                event_type=row["event_type"],
                event_data=json.loads(row["event_data"]) if row["event_data"] else {},
                intent=row["intent"],
                tags=json.loads(row["tags"]) if row["tags"] else [],
                vector=vector,
            )
            events.append(event)

        return events

    def list_sessions(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        cluster_id: Optional[int] = None,
        is_complete: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[Session], int]:
        """
        List sessions with filtering.

        Args:
            start_date: Filter by start date (inclusive)
            end_date: Filter by end date (inclusive)
            cluster_id: Filter by cluster
            is_complete: Filter by completion status
            limit: Max results
            offset: Pagination offset

        Returns:
            Tuple of (sessions, total_count)
        """
        conditions = []
        params = []

        if start_date:
            conditions.append("started_at >= ?")
            params.append(start_date.isoformat())

        if end_date:
            conditions.append("started_at <= ?")
            params.append(end_date.isoformat())

        if cluster_id is not None:
            conditions.append("cluster_id = ?")
            params.append(cluster_id)

        if is_complete is not None:
            conditions.append("is_complete = ?")
            params.append(1 if is_complete else 0)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # Get total count
        count_cursor = self._conn.execute(
            f"SELECT COUNT(*) FROM sessions WHERE {where_clause}",
            params,
        )
        total = count_cursor.fetchone()[0]

        # Get paginated results
        query = f"""
            SELECT * FROM sessions
            WHERE {where_clause}
            ORDER BY started_at DESC
            LIMIT ? OFFSET ?
        """
        cursor = self._conn.execute(query, params + [limit, offset])

        sessions = [self._row_to_session(row) for row in cursor]
        return sessions, total

    def iter_sessions(
        self,
        include_events: bool = False,
        batch_size: int = 100,
    ) -> Iterator[Session]:
        """
        Iterate over all sessions.

        Args:
            include_events: Whether to load events
            batch_size: Batch size for memory efficiency

        Yields:
            Session objects
        """
        offset = 0
        while True:
            sessions, total = self.list_sessions(limit=batch_size, offset=offset)
            if not sessions:
                break

            for session in sessions:
                if include_events:
                    session.events = self._load_events(session.session_id)
                yield session

            offset += batch_size

    def update_session_vector(self, session_id: str, vector: np.ndarray) -> bool:
        """Update session's vector embedding."""
        try:
            self._conn.execute(
                "UPDATE sessions SET vector = ? WHERE session_id = ?",
                (vector.tobytes(), session_id),
            )
            self._conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Failed to update vector for {session_id}: {e}")
            return False

    def update_session_cluster(self, session_id: str, cluster_id: int) -> bool:
        """Update session's cluster assignment."""
        try:
            self._conn.execute(
                "UPDATE sessions SET cluster_id = ? WHERE session_id = ?",
                (cluster_id, session_id),
            )
            self._conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Failed to update cluster for {session_id}: {e}")
            return False

    # ========== Cluster Operations ==========

    def save_cluster(self, cluster: BehaviorCluster) -> int:
        """Save or update a behavior cluster. Returns cluster_id."""
        try:
            centroid_bytes = cluster.centroid.tobytes() if cluster.centroid is not None else None

            if cluster.cluster_id:
                # Update existing
                self._conn.execute(
                    """
                    UPDATE behavior_clusters SET
                        label = ?, description = ?, cluster_type = ?,
                        centroid = ?, session_count = ?, avg_duration_ms = ?,
                        completion_rate = ?, typical_length = ?,
                        dominant_components = ?, dominant_intents = ?,
                        sample_session_ids = ?, updated_at = ?
                    WHERE cluster_id = ?
                    """,
                    (
                        cluster.label,
                        cluster.description,
                        cluster.cluster_type,
                        centroid_bytes,
                        cluster.session_count,
                        cluster.avg_duration_ms,
                        cluster.completion_rate,
                        cluster.typical_length,
                        json.dumps(cluster.dominant_components),
                        json.dumps(cluster.dominant_intents),
                        json.dumps(cluster.sample_session_ids),
                        datetime.now(timezone.utc).isoformat(),
                        cluster.cluster_id,
                    ),
                )
                self._conn.commit()
                return cluster.cluster_id
            else:
                # Insert new
                cursor = self._conn.execute(
                    """
                    INSERT INTO behavior_clusters
                    (label, description, cluster_type, centroid, session_count,
                     avg_duration_ms, completion_rate, typical_length,
                     dominant_components, dominant_intents, sample_session_ids)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cluster.label,
                        cluster.description,
                        cluster.cluster_type,
                        centroid_bytes,
                        cluster.session_count,
                        cluster.avg_duration_ms,
                        cluster.completion_rate,
                        cluster.typical_length,
                        json.dumps(cluster.dominant_components),
                        json.dumps(cluster.dominant_intents),
                        json.dumps(cluster.sample_session_ids),
                    ),
                )
                self._conn.commit()
                return cursor.lastrowid

        except sqlite3.Error as e:
            logger.error(f"Failed to save cluster: {e}")
            self._conn.rollback()
            return -1

    def get_clusters(self) -> list[BehaviorCluster]:
        """Get all behavior clusters."""
        try:
            cursor = self._conn.execute(
                "SELECT * FROM behavior_clusters ORDER BY session_count DESC"
            )

            clusters = []
            for row in cursor:
                centroid = None
                if row["centroid"]:
                    centroid = np.frombuffer(row["centroid"], dtype=np.float32)

                cluster = BehaviorCluster(
                    cluster_id=row["cluster_id"],
                    label=row["label"] or "",
                    description=row["description"] or "",
                    cluster_type=row["cluster_type"] or "unknown",
                    centroid=centroid,
                    session_count=row["session_count"] or 0,
                    avg_duration_ms=row["avg_duration_ms"] or 0.0,
                    completion_rate=row["completion_rate"] or 0.0,
                    typical_length=row["typical_length"] or 0,
                    dominant_components=json.loads(row["dominant_components"]) if row["dominant_components"] else [],
                    dominant_intents=json.loads(row["dominant_intents"]) if row["dominant_intents"] else [],
                    sample_session_ids=json.loads(row["sample_session_ids"]) if row["sample_session_ids"] else [],
                )
                clusters.append(cluster)

            return clusters

        except sqlite3.Error as e:
            logger.error(f"Failed to get clusters: {e}")
            return []

    # ========== Test Gap Operations ==========

    def save_test_gap(self, gap: TestGap) -> bool:
        """Save or update a test gap."""
        try:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO test_gaps
                (gap_id, gap_type, flow_description, affected_components,
                 observed_count, sample_session_id, suggested_test_name,
                 suggested_assertions, priority, priority_reason, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    gap.gap_id,
                    gap.gap_type,
                    gap.flow_description,
                    json.dumps(gap.affected_components),
                    gap.observed_count,
                    gap.sample_session_id,
                    gap.suggested_test_name,
                    json.dumps(gap.suggested_assertions),
                    gap.priority,
                    gap.priority_reason,
                    gap.status,
                ),
            )
            self._conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Failed to save test gap: {e}")
            return False

    def get_test_gaps(self, status: Optional[str] = None) -> list[TestGap]:
        """Get test gaps, optionally filtered by status."""
        try:
            if status:
                cursor = self._conn.execute(
                    "SELECT * FROM test_gaps WHERE status = ? ORDER BY priority, created_at",
                    (status,),
                )
            else:
                cursor = self._conn.execute(
                    "SELECT * FROM test_gaps ORDER BY status, priority, created_at"
                )

            gaps = []
            for row in cursor:
                gap = TestGap(
                    gap_id=row["gap_id"],
                    gap_type=row["gap_type"],
                    flow_description=row["flow_description"] or "",
                    affected_components=json.loads(row["affected_components"]) if row["affected_components"] else [],
                    observed_count=row["observed_count"] or 0,
                    sample_session_id=row["sample_session_id"],
                    suggested_test_name=row["suggested_test_name"] or "",
                    suggested_assertions=json.loads(row["suggested_assertions"]) if row["suggested_assertions"] else [],
                    priority=row["priority"] or "medium",
                    priority_reason=row["priority_reason"] or "",
                    status=row["status"] or "open",
                )
                gaps.append(gap)

            return gaps

        except sqlite3.Error as e:
            logger.error(f"Failed to get test gaps: {e}")
            return []

    # ========== Statistics ==========

    def get_stats(self) -> dict:
        """Get store statistics."""
        try:
            stats = {}

            # Session counts
            cursor = self._conn.execute("SELECT COUNT(*) FROM sessions")
            stats["total_sessions"] = cursor.fetchone()[0]

            cursor = self._conn.execute("SELECT COUNT(*) FROM sessions WHERE is_complete = 1")
            stats["complete_sessions"] = cursor.fetchone()[0]

            cursor = self._conn.execute("SELECT COUNT(*) FROM sessions WHERE vector IS NOT NULL")
            stats["encoded_sessions"] = cursor.fetchone()[0]

            cursor = self._conn.execute("SELECT COUNT(*) FROM sessions WHERE cluster_id IS NOT NULL")
            stats["clustered_sessions"] = cursor.fetchone()[0]

            # Event counts
            cursor = self._conn.execute("SELECT COUNT(*) FROM flow_events")
            stats["total_events"] = cursor.fetchone()[0]

            # Cluster counts
            cursor = self._conn.execute("SELECT COUNT(*) FROM behavior_clusters")
            stats["total_clusters"] = cursor.fetchone()[0]

            # Gap counts
            cursor = self._conn.execute("SELECT COUNT(*) FROM test_gaps WHERE status = 'open'")
            stats["open_gaps"] = cursor.fetchone()[0]

            # Average session length
            cursor = self._conn.execute(
                "SELECT AVG(cnt) FROM (SELECT COUNT(*) as cnt FROM flow_events GROUP BY session_id)"
            )
            result = cursor.fetchone()[0]
            stats["avg_events_per_session"] = round(result, 1) if result else 0

            return stats

        except sqlite3.Error as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "FlowStore":
        return self

    def __exit__(self, *args) -> None:
        self.close()
