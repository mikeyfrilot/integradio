"""Tests for predictors (Markov, NN, Hybrid)."""

import pytest
from datetime import datetime, timezone

from behavior_modeler.predictor import (
    MarkovPredictor,
    NearestNeighborPredictor,
    HybridPredictor,
    PredictionResult,
    create_predictor,
)
from behavior_modeler.models import FlowEvent, FlowPrediction
from behavior_modeler.index import FlowIndex


class TestMarkovPredictor:
    """Tests for MarkovPredictor."""

    def test_markov_initialization(self):
        """Test Markov predictor initializes correctly."""
        predictor = MarkovPredictor(order=2)
        assert predictor.order == 2
        assert len(predictor.transitions) == 0

    def test_markov_fit(self, sample_sessions):
        """Test fitting Markov model."""
        predictor = MarkovPredictor(order=2)
        predictor.fit(sample_sessions)

        assert len(predictor.transitions) > 0
        assert predictor.total_transitions > 0

    def test_markov_predict(self, sample_sessions):
        """Test Markov prediction."""
        predictor = MarkovPredictor(order=2)
        predictor.fit(sample_sessions)

        # Get some events from a session
        session = sample_sessions[0]
        if len(session.events) >= 3:
            recent = session.events[:2]
            predictions = predictor.predict(recent, top_k=5)

            assert isinstance(predictions, list)
            for pred in predictions:
                assert isinstance(pred, FlowPrediction)
                assert 0 <= pred.confidence <= 1

    def test_markov_predict_insufficient_events(self, sample_sessions):
        """Test prediction with too few events."""
        predictor = MarkovPredictor(order=2)
        predictor.fit(sample_sessions)

        # Only 1 event, but order=2
        recent = [sample_sessions[0].events[0]]
        predictions = predictor.predict(recent)

        assert predictions == []

    def test_markov_predict_unknown_prefix(self, sample_sessions):
        """Test prediction with unknown prefix."""
        predictor = MarkovPredictor(order=2)
        predictor.fit(sample_sessions)

        # Create fake events that don't match training data
        fake_events = [
            FlowEvent(
                event_id="f1",
                timestamp=datetime.now(timezone.utc),
                component_type="FakeComponent1",
                event_type="fake_action",
            ),
            FlowEvent(
                event_id="f2",
                timestamp=datetime.now(timezone.utc),
                component_type="FakeComponent2",
                event_type="fake_action",
            ),
        ]
        predictions = predictor.predict(fake_events)

        # Should return empty (or backoff to shorter prefix)
        assert isinstance(predictions, list)

    def test_markov_stats(self, sample_sessions):
        """Test Markov model statistics."""
        predictor = MarkovPredictor(order=2)
        predictor.fit(sample_sessions)

        stats = predictor.get_stats()
        assert stats["order"] == 2
        assert stats["n_prefixes"] > 0
        assert stats["total_transitions"] > 0

    def test_markov_different_orders(self, sample_sessions):
        """Test different Markov orders."""
        for order in [1, 2, 3]:
            predictor = MarkovPredictor(order=order)
            predictor.fit(sample_sessions)

            assert predictor.order == order
            # Higher order = potentially fewer prefixes
            if len(sample_sessions[0].events) > order:
                assert predictor.total_transitions > 0


class TestNearestNeighborPredictor:
    """Tests for NearestNeighborPredictor."""

    @pytest.fixture
    def nn_predictor(self, populated_store, encoder, config):
        """Create NN predictor with built index."""
        index = FlowIndex(populated_store, encoder, config)
        index.build()
        return NearestNeighborPredictor(populated_store, encoder, index)

    def test_nn_initialization(self, nn_predictor):
        """Test NN predictor initializes correctly."""
        assert nn_predictor.store is not None
        assert nn_predictor.encoder is not None
        assert nn_predictor.index is not None

    def test_nn_predict(self, nn_predictor, sample_sessions):
        """Test NN prediction."""
        session = sample_sessions[0]
        if len(session.events) >= 2:
            recent = session.events[:2]
            predictions = nn_predictor.predict(recent, top_k=5)

            assert isinstance(predictions, list)
            for pred in predictions:
                assert isinstance(pred, FlowPrediction)

    def test_nn_predict_empty(self, nn_predictor):
        """Test NN prediction with empty events."""
        predictions = nn_predictor.predict([], top_k=5)
        assert predictions == []


class TestHybridPredictor:
    """Tests for HybridPredictor."""

    @pytest.fixture
    def hybrid_predictor(self, populated_store, encoder, config):
        """Create and train hybrid predictor."""
        index = FlowIndex(populated_store, encoder, config)
        index.build()
        predictor = HybridPredictor(populated_store, encoder, index, config)
        predictor.fit()
        return predictor

    def test_hybrid_initialization(self, populated_store, encoder, config):
        """Test hybrid predictor initializes correctly."""
        index = FlowIndex(populated_store, encoder, config)
        predictor = HybridPredictor(populated_store, encoder, index, config)

        assert predictor.markov is not None
        assert predictor.nn is not None
        assert predictor._trained is False

    def test_hybrid_fit(self, hybrid_predictor):
        """Test hybrid predictor training."""
        assert hybrid_predictor._trained is True
        assert hybrid_predictor.markov.total_transitions > 0

    def test_hybrid_predict(self, hybrid_predictor, sample_sessions):
        """Test hybrid prediction."""
        session = sample_sessions[0]
        if len(session.events) >= 3:
            recent = session.events[:3]
            result = hybrid_predictor.predict(recent, top_k=5)

            assert isinstance(result, PredictionResult)
            assert result.method in ("markov", "nearest_neighbor", "hybrid", "insufficient_context", "no_predictions")
            assert isinstance(result.predictions, list)

    def test_hybrid_predict_insufficient_context(self, hybrid_predictor, config):
        """Test hybrid prediction with too few events."""
        # Single event is less than min_prefix
        recent = [
            FlowEvent(
                event_id="e1",
                timestamp=datetime.now(timezone.utc),
                component_type="Test",
                event_type="click",
            )
        ]
        result = hybrid_predictor.predict(recent)

        assert result.method == "insufficient_context"
        assert result.predictions == []

    def test_hybrid_stats(self, hybrid_predictor):
        """Test hybrid predictor statistics."""
        stats = hybrid_predictor.get_stats()

        assert "trained" in stats
        assert stats["trained"] is True
        assert "markov" in stats


class TestCreatePredictor:
    """Test convenience function."""

    def test_create_predictor_function(self, populated_store, encoder, config):
        """Test create_predictor convenience function."""
        index = FlowIndex(populated_store, encoder, config)
        index.build()

        predictor = create_predictor(populated_store, encoder, index, config)

        assert isinstance(predictor, HybridPredictor)
        assert predictor._trained is True
