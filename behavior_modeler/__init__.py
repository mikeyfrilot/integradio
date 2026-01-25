"""
UI Behavior Modeler - Semantic intelligence layer for Integradio.

Observes user flows through the EventMesh, encodes them into vectors,
clusters behaviors to discover patterns, and advises on next actions,
missing tests, and UX improvements.

Usage:
    from behavior_modeler import BehaviorModeler, BehaviorModelerConfig

    config = BehaviorModelerConfig(
        db_path=Path("db/behavior.db"),
        ollama_url="http://localhost:11434"
    )
    modeler = BehaviorModeler(config)

    # Connect to EventMesh
    await modeler.connect(mesh)

    # Or ingest flows manually
    await modeler.ingest_flow(session_id, events)
"""

from .config import BehaviorModelerConfig
from .models import (
    Session,
    FlowEvent,
    BehaviorCluster,
    FlowPrediction,
    TestGap,
)

__version__ = "0.1.0"
__all__ = [
    "BehaviorModelerConfig",
    "Session",
    "FlowEvent",
    "BehaviorCluster",
    "FlowPrediction",
    "TestGap",
]
