"""Tests for SequentialPatternMiner."""

import pytest

from behavior_modeler.patterns import (
    SequentialPattern,
    SequentialPatternMiner,
    PatternMiningResult,
    mine_patterns,
)


class TestSequentialPattern:
    """Tests for SequentialPattern dataclass."""

    def test_pattern_creation(self):
        """Test creating a pattern."""
        pattern = SequentialPattern(
            pattern_id="abc123",
            sequence=["SearchBox:input", "SearchResults:select"],
            support=0.25,
            confidence=0.8,
            occurrence_count=50,
            avg_duration_ms=3000,
        )

        assert pattern.pattern_id == "abc123"
        assert pattern.length == 2
        assert pattern.support == 0.25

    def test_pattern_to_dict(self):
        """Test pattern serialization."""
        pattern = SequentialPattern(
            pattern_id="abc123",
            sequence=["A:click", "B:view"],
            support=0.5,
            confidence=0.9,
            occurrence_count=100,
            avg_duration_ms=2000,
            sample_session_ids=["s1", "s2"],
        )

        d = pattern.to_dict()
        assert d["pattern_id"] == "abc123"
        assert d["length"] == 2
        assert d["sample_session_ids"] == ["s1", "s2"]


class TestSequentialPatternMiner:
    """Tests for SequentialPatternMiner."""

    def test_miner_initialization(self, store, config):
        """Test miner initializes correctly."""
        miner = SequentialPatternMiner(store, config)
        assert miner.store == store

    def test_mine_empty_store(self, store, config):
        """Test mining with no sessions."""
        miner = SequentialPatternMiner(store, config)
        result = miner.mine_patterns()

        assert isinstance(result, PatternMiningResult)
        assert result.patterns == []
        assert result.n_sessions_analyzed == 0

    def test_mine_patterns_basic(self, populated_store, config):
        """Test basic pattern mining."""
        miner = SequentialPatternMiner(populated_store, config)
        result = miner.mine_patterns(min_support=0.05, max_length=4)

        assert isinstance(result, PatternMiningResult)
        assert result.n_sessions_analyzed == 20  # From sample_sessions

        # With 20 sessions and min_support=0.05, should find some patterns
        # (0.05 * 20 = 1 session minimum)
        assert isinstance(result.patterns, list)

    def test_mine_patterns_structure(self, populated_store, config):
        """Test that mined patterns have correct structure."""
        miner = SequentialPatternMiner(populated_store, config)
        result = miner.mine_patterns(min_support=0.1)

        for pattern in result.patterns:
            assert isinstance(pattern, SequentialPattern)
            assert len(pattern.sequence) >= 2  # min_length default is 2
            assert 0 <= pattern.support <= 1
            assert 0 <= pattern.confidence <= 1
            assert pattern.occurrence_count > 0

    def test_mine_patterns_sorted_by_support(self, populated_store, config):
        """Test that patterns are sorted by support."""
        miner = SequentialPatternMiner(populated_store, config)
        result = miner.mine_patterns(min_support=0.05)

        supports = [p.support for p in result.patterns]
        assert supports == sorted(supports, reverse=True)

    def test_mine_patterns_min_support_filtering(self, populated_store, config):
        """Test that min_support filters correctly."""
        miner = SequentialPatternMiner(populated_store, config)

        # High support threshold
        result_high = miner.mine_patterns(min_support=0.5)
        # Low support threshold
        result_low = miner.mine_patterns(min_support=0.05)

        # Lower threshold should find more (or equal) patterns
        assert len(result_low.patterns) >= len(result_high.patterns)

        # All patterns should meet their respective thresholds
        for p in result_high.patterns:
            assert p.support >= 0.5

    def test_mine_patterns_max_length(self, populated_store, config):
        """Test max_length parameter."""
        miner = SequentialPatternMiner(populated_store, config)
        result = miner.mine_patterns(min_support=0.05, max_length=3)

        for pattern in result.patterns:
            assert len(pattern.sequence) <= 3

    def test_mine_patterns_min_length(self, populated_store, config):
        """Test min_length parameter."""
        miner = SequentialPatternMiner(populated_store, config)
        result = miner.mine_patterns(min_support=0.05, min_length=3, max_length=5)

        for pattern in result.patterns:
            assert len(pattern.sequence) >= 3

    def test_sequence_contains_basic(self, store, config):
        """Test _sequence_contains helper."""
        miner = SequentialPatternMiner(store, config)

        sequence = ["A:x", "B:y", "C:z", "D:w"]
        pattern = ("A:x", "C:z")

        assert miner._sequence_contains(sequence, pattern, max_gap=2)

    def test_sequence_contains_gap_constraint(self, store, config):
        """Test gap constraint in _sequence_contains."""
        miner = SequentialPatternMiner(store, config)

        sequence = ["A:x", "B:y", "C:z", "D:w", "E:v"]
        pattern = ("A:x", "E:v")

        # With gap=1, A->E is too far (3 elements between)
        assert not miner._sequence_contains(sequence, pattern, max_gap=1)

        # With gap=3, should work
        assert miner._sequence_contains(sequence, pattern, max_gap=3)


class TestPatternGaps:
    """Tests for finding pattern gaps (untested flows)."""

    def test_find_pattern_gaps(self, populated_store, config):
        """Test finding pattern gaps."""
        miner = SequentialPatternMiner(populated_store, config)
        gaps = miner.find_pattern_gaps(min_support=0.1)

        assert isinstance(gaps, list)
        for gap in gaps:
            assert "pattern" in gap
            assert "support" in gap
            assert "suggested_test_name" in gap
            assert "priority" in gap

    def test_find_pattern_gaps_with_existing_tests(self, populated_store, config):
        """Test gap detection with some patterns already tested."""
        miner = SequentialPatternMiner(populated_store, config)

        # Mine patterns first to get some
        result = miner.mine_patterns(min_support=0.1)
        if result.patterns:
            # Pretend first pattern is already tested
            tested = [result.patterns[0].sequence]
            gaps = miner.find_pattern_gaps(test_patterns=tested, min_support=0.1)

            # Should have one fewer gap
            assert len(gaps) == len(result.patterns) - 1


class TestConvenienceFunction:
    """Test convenience functions."""

    def test_mine_patterns_function(self, populated_store, config):
        """Test mine_patterns convenience function."""
        result = mine_patterns(populated_store, min_support=0.1, config=config)

        assert isinstance(result, PatternMiningResult)
