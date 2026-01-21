"""
Next Action Predictor - Predict likely next actions in user flows.

Combines two complementary approaches:
1. Markov Chain: Fast, interpretable transition probabilities
2. Nearest Neighbor: Semantic similarity to known flows

Based on research showing:
- Markov chains achieve 70-95% accuracy for next-location prediction
- Combining with neural/embedding approaches improves coverage
- N-order Markov (n=2) often optimal for user journeys
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import hashlib
import numpy as np

from .config import BehaviorModelerConfig
from .models import Session, FlowEvent, FlowPrediction
from .store import FlowStore
from .encoder import FlowEncoder
from .index import FlowIndex

logger = logging.getLogger(__name__)


@dataclass
class TransitionKey:
    """Key for Markov transition lookup."""
    prefix: tuple[str, ...]  # (component_type, event_type) pairs
    order: int = 2  # N-gram order

    def __hash__(self):
        return hash(self.prefix)

    def __eq__(self, other):
        return self.prefix == other.prefix


@dataclass
class PredictionResult:
    """Complete prediction result with multiple candidates."""
    predictions: list[FlowPrediction]
    method: str  # 'markov', 'nearest_neighbor', 'hybrid'
    confidence: float  # Overall confidence


class MarkovPredictor:
    """
    Markov chain-based next action predictor.

    Implements n-order Markov model where n=2 (bigram) is default,
    as research shows this balances accuracy with generalization.
    """

    def __init__(self, order: int = 2):
        """
        Initialize Markov predictor.

        Args:
            order: N-gram order (1=unigram, 2=bigram, etc.)
        """
        self.order = order
        # transitions[prefix] = {next_state: count}
        self.transitions: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.total_transitions = 0

    def fit(self, sessions: list[Session]) -> "MarkovPredictor":
        """
        Learn transition probabilities from sessions.

        Args:
            sessions: List of sessions to learn from

        Returns:
            self for chaining
        """
        for session in sessions:
            events = session.events
            if len(events) < self.order + 1:
                continue

            for i in range(len(events) - self.order):
                # Build prefix from last N events
                prefix = tuple(
                    f"{e.component_type}:{e.event_type}"
                    for e in events[i:i + self.order]
                )

                # Next state
                next_event = events[i + self.order]
                next_state = f"{next_event.component_type}:{next_event.event_type}"

                self.transitions[prefix][next_state] += 1
                self.total_transitions += 1

        logger.info(f"Markov model trained: {len(self.transitions)} prefixes, {self.total_transitions} transitions")
        return self

    def predict(
        self,
        recent_events: list[FlowEvent],
        top_k: int = 5,
    ) -> list[FlowPrediction]:
        """
        Predict next actions given recent events.

        Args:
            recent_events: Most recent events in current flow
            top_k: Number of predictions to return

        Returns:
            List of FlowPrediction sorted by confidence
        """
        if len(recent_events) < self.order:
            return []

        # Build prefix from recent events
        prefix = tuple(
            f"{e.component_type}:{e.event_type}"
            for e in recent_events[-self.order:]
        )

        if prefix not in self.transitions:
            # Try shorter prefix (backoff)
            if self.order > 1 and len(recent_events) >= 1:
                shorter_prefix = tuple(
                    f"{e.component_type}:{e.event_type}"
                    for e in recent_events[-(self.order - 1):]
                )
                if shorter_prefix in self.transitions:
                    prefix = shorter_prefix

        if prefix not in self.transitions:
            return []

        # Get transition counts
        next_states = self.transitions[prefix]
        total = sum(next_states.values())

        # Build predictions
        predictions = []
        for state, count in sorted(next_states.items(), key=lambda x: -x[1])[:top_k]:
            component, event = state.split(":", 1) if ":" in state else (state, "unknown")
            confidence = count / total

            predictions.append(FlowPrediction(
                predicted_component=component,
                predicted_event=event,
                confidence=confidence,
            ))

        return predictions

    def get_stats(self) -> dict:
        """Get model statistics."""
        return {
            "order": self.order,
            "n_prefixes": len(self.transitions),
            "total_transitions": self.total_transitions,
            "avg_transitions_per_prefix": (
                self.total_transitions / len(self.transitions)
                if self.transitions else 0
            ),
        }


class NearestNeighborPredictor:
    """
    Nearest neighbor-based predictor using flow embeddings.

    Finds similar historical flows and predicts based on what
    happened next in those flows.
    """

    def __init__(
        self,
        store: FlowStore,
        encoder: FlowEncoder,
        index: FlowIndex,
    ):
        """
        Initialize NN predictor.

        Args:
            store: FlowStore for session retrieval
            encoder: FlowEncoder for embedding partial flows
            index: FlowIndex for similarity search
        """
        self.store = store
        self.encoder = encoder
        self.index = index

    def predict(
        self,
        recent_events: list[FlowEvent],
        top_k: int = 5,
        n_neighbors: int = 10,
    ) -> list[FlowPrediction]:
        """
        Predict next actions by finding similar flows.

        Args:
            recent_events: Events in current partial flow
            top_k: Number of predictions to return
            n_neighbors: Number of similar sessions to consider

        Returns:
            List of FlowPrediction sorted by confidence
        """
        if not recent_events:
            return []

        # Encode partial flow
        partial_vector = self.encoder.encode_partial_flow(recent_events)

        # Find similar sessions
        results = self.index.search(
            partial_vector,
            k=n_neighbors,
            include_session=True,
        )

        if not results:
            return []

        # Collect "what happened next" from similar sessions
        next_actions: dict[str, list[float]] = defaultdict(list)
        current_length = len(recent_events)

        for result in results:
            session = result.session
            if session is None:
                session = self.store.get_session(result.session_id)

            if session is None or len(session.events) <= current_length:
                continue

            # Get next event(s) after current position
            next_event = session.events[current_length]
            key = f"{next_event.component_type}:{next_event.event_type}"

            # Weight by similarity
            next_actions[key].append(result.similarity)

        if not next_actions:
            return []

        # Aggregate predictions
        predictions = []
        for key, similarities in next_actions.items():
            # Confidence = average similarity * frequency factor
            avg_sim = np.mean(similarities)
            freq_factor = len(similarities) / n_neighbors
            confidence = avg_sim * (0.5 + 0.5 * freq_factor)

            component, event = key.split(":", 1) if ":" in key else (key, "unknown")

            predictions.append(FlowPrediction(
                predicted_component=component,
                predicted_event=event,
                confidence=float(confidence),
                similar_sessions=[r.session_id for r in results if r.similarity > 0.5][:3],
            ))

        # Sort by confidence
        predictions.sort(key=lambda p: p.confidence, reverse=True)
        return predictions[:top_k]


class HybridPredictor:
    """
    Hybrid predictor combining Markov and Nearest Neighbor approaches.

    Strategy:
    - Use Markov for common patterns (fast, precise)
    - Fall back to NN for rare/unseen patterns (semantic generalization)
    - Combine when both have predictions (weighted ensemble)
    """

    def __init__(
        self,
        store: FlowStore,
        encoder: FlowEncoder,
        index: FlowIndex,
        config: Optional[BehaviorModelerConfig] = None,
    ):
        """
        Initialize hybrid predictor.

        Args:
            store: FlowStore for data access
            encoder: FlowEncoder for embeddings
            index: FlowIndex for similarity search
            config: Configuration options
        """
        self.config = config or BehaviorModelerConfig()
        self.markov = MarkovPredictor(order=2)
        self.nn = NearestNeighborPredictor(store, encoder, index)
        self._trained = False

    def fit(self, sessions: Optional[list[Session]] = None) -> "HybridPredictor":
        """
        Train the predictor on sessions.

        Args:
            sessions: Sessions to train on (loads from store if None)

        Returns:
            self for chaining
        """
        if sessions is None:
            sessions = list(self.nn.store.iter_sessions(include_events=True))

        self.markov.fit(sessions)
        self._trained = True
        return self

    def predict(
        self,
        recent_events: list[FlowEvent],
        top_k: int = 5,
    ) -> PredictionResult:
        """
        Predict next actions using hybrid approach.

        Args:
            recent_events: Events in current partial flow
            top_k: Number of predictions to return

        Returns:
            PredictionResult with combined predictions
        """
        if len(recent_events) < self.config.prediction_min_prefix:
            return PredictionResult(
                predictions=[],
                method="insufficient_context",
                confidence=0.0,
            )

        # Get predictions from both methods
        markov_preds = self.markov.predict(recent_events, top_k=top_k)
        nn_preds = self.nn.predict(recent_events, top_k=top_k)

        # Determine strategy based on results
        if markov_preds and not nn_preds:
            return PredictionResult(
                predictions=markov_preds,
                method="markov",
                confidence=markov_preds[0].confidence if markov_preds else 0.0,
            )

        if nn_preds and not markov_preds:
            return PredictionResult(
                predictions=nn_preds,
                method="nearest_neighbor",
                confidence=nn_preds[0].confidence if nn_preds else 0.0,
            )

        if not markov_preds and not nn_preds:
            return PredictionResult(
                predictions=[],
                method="no_predictions",
                confidence=0.0,
            )

        # Combine predictions (weighted ensemble)
        combined = self._combine_predictions(markov_preds, nn_preds, top_k)

        return PredictionResult(
            predictions=combined,
            method="hybrid",
            confidence=combined[0].confidence if combined else 0.0,
        )

    def _combine_predictions(
        self,
        markov_preds: list[FlowPrediction],
        nn_preds: list[FlowPrediction],
        top_k: int,
    ) -> list[FlowPrediction]:
        """
        Combine predictions from both methods.

        Weighting: Markov 0.6, NN 0.4 (Markov more precise for known patterns)
        """
        MARKOV_WEIGHT = 0.6
        NN_WEIGHT = 0.4

        # Build lookup by (component, event)
        combined: dict[tuple[str, str], FlowPrediction] = {}

        for pred in markov_preds:
            key = (pred.predicted_component, pred.predicted_event)
            combined[key] = FlowPrediction(
                predicted_component=pred.predicted_component,
                predicted_event=pred.predicted_event,
                predicted_intent=pred.predicted_intent,
                confidence=pred.confidence * MARKOV_WEIGHT,
                similar_sessions=pred.similar_sessions,
            )

        for pred in nn_preds:
            key = (pred.predicted_component, pred.predicted_event)
            if key in combined:
                # Add NN contribution
                combined[key].confidence += pred.confidence * NN_WEIGHT
                # Merge similar sessions
                existing = set(combined[key].similar_sessions)
                combined[key].similar_sessions = list(
                    existing | set(pred.similar_sessions)
                )[:5]
            else:
                combined[key] = FlowPrediction(
                    predicted_component=pred.predicted_component,
                    predicted_event=pred.predicted_event,
                    predicted_intent=pred.predicted_intent,
                    confidence=pred.confidence * NN_WEIGHT,
                    similar_sessions=pred.similar_sessions,
                )

        # Sort and return top_k
        results = sorted(combined.values(), key=lambda p: p.confidence, reverse=True)
        return results[:top_k]

    def get_stats(self) -> dict:
        """Get predictor statistics."""
        return {
            "trained": self._trained,
            "markov": self.markov.get_stats(),
        }


def create_predictor(
    store: FlowStore,
    encoder: FlowEncoder,
    index: FlowIndex,
    config: Optional[BehaviorModelerConfig] = None,
) -> HybridPredictor:
    """
    Create and train a hybrid predictor.

    Args:
        store: FlowStore with sessions
        encoder: FlowEncoder for embeddings
        index: FlowIndex for similarity search
        config: Optional configuration

    Returns:
        Trained HybridPredictor
    """
    predictor = HybridPredictor(store, encoder, index, config)
    predictor.fit()
    return predictor
