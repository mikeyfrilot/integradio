"""
Configuration for the UI Behavior Modeler.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class BehaviorModelerConfig:
    """Configuration for the Behavior Modeler."""

    # Database
    db_path: Path = field(default_factory=lambda: Path("behavior_modeler/db/behavior.db"))

    # Embedding
    ollama_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    embed_dimension: int = 768

    # Flow encoding
    encoding_method: Literal["pooled_weighted", "pooled_avg"] = "pooled_weighted"
    max_sequence_length: int = 100

    # Clustering
    clustering_algorithm: Literal["hdbscan", "kmeans", "agglomerative"] = "hdbscan"
    min_cluster_size: int = 10
    min_samples: int = 5

    # Session management
    session_timeout_seconds: int = 300  # 5 minutes of inactivity = session end
    max_events_per_session: int = 500

    # HNSW index parameters
    hnsw_max_elements: int = 100000
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 50
    hnsw_M: int = 16

    # Prediction
    prediction_min_prefix: int = 2  # Minimum events before predicting
    prediction_top_k: int = 5       # Number of predictions to return

    # Test gap detection
    gap_min_observations: int = 5   # Minimum times a flow must be seen
    gap_coverage_threshold: float = 0.8  # Below this = gap

    def __post_init__(self):
        """Ensure db_path parent directories exist."""
        if isinstance(self.db_path, str):
            self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
