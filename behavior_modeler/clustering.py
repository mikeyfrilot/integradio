"""
Behavior Clustering - Discover patterns in user flows using HDBSCAN.

Based on best practices from:
- PMC research on UEBA clustering (HDBSCAN optimal for behavior analytics)
- Feature engineering with temporal behavioral features
- Auto-labeling clusters using dominant characteristics

Key insights applied:
- HDBSCAN handles varying density clusters (natural for user behavior)
- min_cluster_size 4-10 optimal for behavioral data
- Noise points are valuable (anomalies, edge cases)
- Combine with dimensionality reduction for high-dimensional data
"""

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Optional
import numpy as np

from .config import BehaviorModelerConfig
from .models import Session, BehaviorCluster
from .store import FlowStore

logger = logging.getLogger(__name__)

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    logger.warning("hdbscan not available, install with: pip install hdbscan")

try:
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class ClusteringResult:
    """Result from clustering operation."""
    n_clusters: int
    n_noise: int  # Sessions not assigned to any cluster (-1 label)
    labels: list[int]  # Cluster label for each session
    session_ids: list[str]  # Corresponding session IDs
    clusters: list[BehaviorCluster]  # Generated cluster objects


class BehaviorClustering:
    """
    Cluster user sessions to discover behavior patterns.

    Uses HDBSCAN by default (best for behavioral data with varying densities),
    with fallback to K-Means if HDBSCAN unavailable.
    """

    # Cluster type classification thresholds
    HAPPY_PATH_COMPLETION_THRESHOLD = 0.7
    DROP_OFF_COMPLETION_THRESHOLD = 0.3
    EDGE_CASE_SIZE_PERCENTILE = 10  # Clusters below this percentile are edge cases

    def __init__(
        self,
        store: FlowStore,
        config: Optional[BehaviorModelerConfig] = None,
    ):
        """
        Initialize clustering.

        Args:
            store: FlowStore containing encoded sessions
            config: Configuration options
        """
        self.store = store
        self.config = config or BehaviorModelerConfig()

    def cluster(
        self,
        algorithm: Optional[str] = None,
        min_cluster_size: Optional[int] = None,
        min_samples: Optional[int] = None,
        n_clusters: Optional[int] = None,  # For K-Means
    ) -> ClusteringResult:
        """
        Cluster all encoded sessions.

        Args:
            algorithm: 'hdbscan', 'kmeans', or 'agglomerative'
            min_cluster_size: Minimum cluster size (HDBSCAN)
            min_samples: Minimum samples for core point (HDBSCAN)
            n_clusters: Number of clusters (K-Means/Agglomerative)

        Returns:
            ClusteringResult with labels and cluster objects
        """
        algorithm = algorithm or self.config.clustering_algorithm
        min_cluster_size = min_cluster_size or self.config.min_cluster_size
        min_samples = min_samples or self.config.min_samples

        # Load all encoded sessions
        sessions, vectors, session_ids = self._load_encoded_sessions()

        if len(vectors) < min_cluster_size:
            logger.warning(f"Not enough sessions ({len(vectors)}) for clustering")
            return ClusteringResult(
                n_clusters=0,
                n_noise=len(vectors),
                labels=[-1] * len(vectors),
                session_ids=session_ids,
                clusters=[],
            )

        # Stack vectors
        X = np.stack(vectors)

        # Run clustering
        if algorithm == "hdbscan" and HAS_HDBSCAN:
            labels = self._cluster_hdbscan(X, min_cluster_size, min_samples)
        elif algorithm == "kmeans" and HAS_SKLEARN:
            n_clusters = n_clusters or self._estimate_n_clusters(len(vectors))
            labels = self._cluster_kmeans(X, n_clusters)
        elif algorithm == "agglomerative" and HAS_SKLEARN:
            n_clusters = n_clusters or self._estimate_n_clusters(len(vectors))
            labels = self._cluster_agglomerative(X, n_clusters)
        else:
            logger.warning(f"Algorithm {algorithm} not available, using simple clustering")
            labels = self._cluster_simple(X, min_cluster_size)

        # Build cluster objects
        clusters = self._build_clusters(sessions, labels, vectors)

        # Update session cluster assignments in store
        for session_id, label in zip(session_ids, labels):
            if label >= 0:  # Skip noise
                self.store.update_session_cluster(session_id, label)

        # Save clusters to store
        for cluster in clusters:
            self.store.save_cluster(cluster)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = sum(1 for l in labels if l == -1)

        logger.info(f"Clustering complete: {n_clusters} clusters, {n_noise} noise points")

        return ClusteringResult(
            n_clusters=n_clusters,
            n_noise=n_noise,
            labels=list(labels),
            session_ids=session_ids,
            clusters=clusters,
        )

    def _load_encoded_sessions(self) -> tuple[list[Session], list[np.ndarray], list[str]]:
        """Load all sessions with vectors."""
        sessions = []
        vectors = []
        session_ids = []

        for session in self.store.iter_sessions(include_events=True):
            if session.vector is not None:
                sessions.append(session)
                vectors.append(session.vector)
                session_ids.append(session.session_id)

        return sessions, vectors, session_ids

    def _cluster_hdbscan(
        self,
        X: np.ndarray,
        min_cluster_size: int,
        min_samples: int,
    ) -> np.ndarray:
        """
        Cluster using HDBSCAN.

        Best for behavioral data:
        - Handles varying density clusters
        - Identifies noise/anomalies
        - No need to specify n_clusters
        """
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method="eom",  # Excess of Mass
            prediction_data=True,
        )
        return clusterer.fit_predict(X)

    def _cluster_kmeans(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        """Cluster using K-Means."""
        # Normalize for K-Means
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10,
        )
        return kmeans.fit_predict(X_scaled)

    def _cluster_agglomerative(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        """Cluster using Agglomerative Clustering."""
        agg = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="euclidean",
            linkage="ward",
        )
        return agg.fit_predict(X)

    def _cluster_simple(self, X: np.ndarray, min_cluster_size: int) -> np.ndarray:
        """
        Simple clustering fallback when no libraries available.

        Uses greedy nearest-neighbor grouping.
        """
        n = len(X)
        labels = np.full(n, -1)
        cluster_id = 0
        assigned = set()

        for i in range(n):
            if i in assigned:
                continue

            # Find nearest neighbors
            distances = np.linalg.norm(X - X[i], axis=1)
            neighbors = np.argsort(distances)

            cluster_members = []
            for j in neighbors:
                if j not in assigned and distances[j] < 0.5:  # Distance threshold
                    cluster_members.append(j)
                    if len(cluster_members) >= min_cluster_size * 2:
                        break

            if len(cluster_members) >= min_cluster_size:
                for j in cluster_members:
                    labels[j] = cluster_id
                    assigned.add(j)
                cluster_id += 1

        return labels

    def _estimate_n_clusters(self, n_samples: int) -> int:
        """Estimate reasonable number of clusters."""
        # Rule of thumb: sqrt(n/2) or between 5-20
        estimate = int(np.sqrt(n_samples / 2))
        return max(5, min(20, estimate))

    def _build_clusters(
        self,
        sessions: list[Session],
        labels: np.ndarray,
        vectors: list[np.ndarray],
    ) -> list[BehaviorCluster]:
        """
        Build BehaviorCluster objects with auto-generated labels.

        Implements best practices:
        - Extract dominant components and intents
        - Classify cluster type based on completion rate
        - Calculate centroids for each cluster
        """
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label

        clusters = []
        cluster_sizes = []

        for cluster_id in sorted(unique_labels):
            mask = labels == cluster_id
            cluster_sessions = [s for s, m in zip(sessions, mask) if m]
            cluster_vectors = [v for v, m in zip(vectors, mask) if m]

            if not cluster_sessions:
                continue

            cluster = self._build_single_cluster(
                cluster_id=int(cluster_id),
                sessions=cluster_sessions,
                vectors=cluster_vectors,
            )
            clusters.append(cluster)
            cluster_sizes.append(cluster.session_count)

        # Classify edge cases based on size
        if cluster_sizes:
            size_threshold = np.percentile(cluster_sizes, self.EDGE_CASE_SIZE_PERCENTILE)
            for cluster in clusters:
                if cluster.session_count <= size_threshold and cluster.cluster_type == "unknown":
                    cluster.cluster_type = "edge_case"

        return clusters

    def _build_single_cluster(
        self,
        cluster_id: int,
        sessions: list[Session],
        vectors: list[np.ndarray],
    ) -> BehaviorCluster:
        """Build a single cluster with auto-labeling."""
        # Calculate centroid
        centroid = np.mean(np.stack(vectors), axis=0).astype(np.float32)

        # Gather statistics
        durations = [s.duration_ms for s in sessions]
        completions = [s.is_complete for s in sessions]
        lengths = [len(s.events) for s in sessions]

        # Extract dominant characteristics
        all_components = []
        all_intents = []
        for session in sessions:
            for event in session.events:
                if event.component_type:
                    all_components.append(event.component_type)
                if event.intent:
                    all_intents.append(event.intent)

        # Top 5 most common
        component_counts = Counter(all_components)
        intent_counts = Counter(all_intents)
        dominant_components = [c for c, _ in component_counts.most_common(5)]
        dominant_intents = [i for i, _ in intent_counts.most_common(5)]

        # Auto-generate label from dominant components
        label = self._generate_label(dominant_components, dominant_intents)

        # Classify cluster type
        completion_rate = sum(completions) / len(completions) if completions else 0
        cluster_type = self._classify_cluster_type(completion_rate, sessions)

        # Sample session IDs
        sample_ids = [s.session_id for s in sessions[:5]]

        return BehaviorCluster(
            cluster_id=cluster_id,
            label=label,
            description=f"Cluster of {len(sessions)} sessions with {completion_rate:.0%} completion",
            centroid=centroid,
            session_count=len(sessions),
            avg_duration_ms=float(np.mean(durations)) if durations else 0,
            completion_rate=completion_rate,
            typical_length=int(np.median(lengths)) if lengths else 0,
            dominant_components=dominant_components,
            dominant_intents=dominant_intents,
            cluster_type=cluster_type,
            sample_session_ids=sample_ids,
        )

    def _generate_label(
        self,
        components: list[str],
        intents: list[str],
    ) -> str:
        """
        Auto-generate human-readable cluster label.

        Strategy: Use top intent if available, otherwise top 2 components.
        """
        if intents:
            # Clean up intent name
            label = intents[0].replace("_", " ").title()
            return f"{label} Flow"

        if len(components) >= 2:
            # "SearchBox → Results" style
            return f"{components[0]} → {components[1]}"

        if components:
            return f"{components[0]} Flow"

        return "Unknown Flow"

    def _classify_cluster_type(
        self,
        completion_rate: float,
        sessions: list[Session],
    ) -> str:
        """
        Classify cluster as happy_path, drop_off, edge_case, or error_flow.

        Based on:
        - Completion rate thresholds
        - Presence of error-related events
        """
        # Check for error indicators
        error_keywords = {"error", "fail", "retry", "exception"}
        has_errors = False
        for session in sessions[:10]:  # Sample
            for event in session.events:
                event_text = f"{event.component_type} {event.event_type} {event.intent or ''}".lower()
                if any(kw in event_text for kw in error_keywords):
                    has_errors = True
                    break

        if has_errors:
            return "error_flow"

        if completion_rate >= self.HAPPY_PATH_COMPLETION_THRESHOLD:
            return "happy_path"

        if completion_rate <= self.DROP_OFF_COMPLETION_THRESHOLD:
            return "drop_off"

        return "unknown"

    def get_cluster_insights(self) -> dict:
        """
        Get high-level insights from clustering results.

        Returns dict with:
        - cluster_distribution: type -> count
        - top_happy_paths: highest completion clusters
        - problem_areas: drop-offs and error flows
        """
        clusters = self.store.get_clusters()

        if not clusters:
            return {"error": "No clusters found. Run cluster() first."}

        # Distribution by type
        type_counts = Counter(c.cluster_type for c in clusters)

        # Sort by completion rate
        by_completion = sorted(clusters, key=lambda c: c.completion_rate, reverse=True)

        # Problem areas (low completion or errors)
        problems = [c for c in clusters if c.cluster_type in ("drop_off", "error_flow")]

        return {
            "total_clusters": len(clusters),
            "cluster_distribution": dict(type_counts),
            "top_happy_paths": [
                {"label": c.label, "completion_rate": c.completion_rate, "sessions": c.session_count}
                for c in by_completion[:5]
                if c.cluster_type == "happy_path"
            ],
            "problem_areas": [
                {"label": c.label, "type": c.cluster_type, "completion_rate": c.completion_rate, "sessions": c.session_count}
                for c in problems
            ],
            "avg_completion_rate": np.mean([c.completion_rate for c in clusters]),
        }


def cluster_sessions(
    store: FlowStore,
    config: Optional[BehaviorModelerConfig] = None,
) -> ClusteringResult:
    """
    Convenience function to cluster all sessions in store.

    Args:
        store: FlowStore with encoded sessions
        config: Optional configuration

    Returns:
        ClusteringResult
    """
    clustering = BehaviorClustering(store, config)
    return clustering.cluster()
