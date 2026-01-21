"""
UI Behavior Modeler - Semantic intelligence layer for Integradio.

Observes user flows through the EventMesh, encodes them into vectors,
clusters behaviors to discover patterns, and advises on next actions,
missing tests, and UX improvements.

Usage:
    from behavior_modeler import (
        BehaviorModelerConfig, FlowStore, FlowEncoder, FlowIndex,
        BehaviorClustering, HybridPredictor, SequentialPatternMiner
    )

    config = BehaviorModelerConfig(db_path=Path("db/behavior.db"))

    # Set up core components
    store = FlowStore(config)
    encoder = FlowEncoder(config=config)
    index = FlowIndex(store, encoder, config)

    # Ingest mock data for testing
    from behavior_modeler.mock import generate_sample_flows
    for session in generate_sample_flows(100):
        vector = encoder.encode_session(session)
        session.vector = vector
        store.save_session(session)
    index.build()

    # Cluster behaviors
    clustering = BehaviorClustering(store, config)
    result = clustering.cluster()

    # Predict next actions
    predictor = HybridPredictor(store, encoder, index, config)
    predictor.fit()
    prediction = predictor.predict(recent_events)

    # Mine patterns
    miner = SequentialPatternMiner(store, config)
    patterns = miner.mine_patterns()
"""

from .config import BehaviorModelerConfig
from .models import (
    Session,
    FlowEvent,
    BehaviorCluster,
    FlowPrediction,
    TestGap,
)
from .store import FlowStore
from .encoder import FlowEncoder, FallbackEncoder
from .index import FlowIndex, SearchResult
from .clustering import BehaviorClustering, ClusteringResult, cluster_sessions
from .predictor import (
    MarkovPredictor,
    NearestNeighborPredictor,
    HybridPredictor,
    PredictionResult,
    create_predictor,
)
from .patterns import (
    SequentialPattern,
    SequentialPatternMiner,
    PatternMiningResult,
    mine_patterns,
)
from .gaps import (
    GapDetector,
    CoverageInfo,
    GapAnalysisResult,
    analyze_test_gaps,
)
from .api import create_app, app

__version__ = "0.4.0"
__all__ = [
    # Config
    "BehaviorModelerConfig",
    # Models
    "Session",
    "FlowEvent",
    "BehaviorCluster",
    "FlowPrediction",
    "TestGap",
    # Core components
    "FlowStore",
    "FlowEncoder",
    "FallbackEncoder",
    "FlowIndex",
    "SearchResult",
    # Clustering
    "BehaviorClustering",
    "ClusteringResult",
    "cluster_sessions",
    # Prediction
    "MarkovPredictor",
    "NearestNeighborPredictor",
    "HybridPredictor",
    "PredictionResult",
    "create_predictor",
    # Pattern Mining
    "SequentialPattern",
    "SequentialPatternMiner",
    "PatternMiningResult",
    "mine_patterns",
    # Gap Detection
    "GapDetector",
    "CoverageInfo",
    "GapAnalysisResult",
    "analyze_test_gaps",
    # API
    "create_app",
    "app",
]
