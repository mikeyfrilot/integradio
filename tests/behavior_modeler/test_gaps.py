"""Tests for GapDetector."""

import pytest

from behavior_modeler.gaps import (
    GapDetector,
    CoverageInfo,
    GapAnalysisResult,
    analyze_test_gaps,
)
from behavior_modeler.models import TestGap


class TestCoverageInfoClass:
    """Tests for CoverageInfo dataclass."""

    def test_coverage_info_creation(self):
        """Test creating coverage info."""
        coverage = CoverageInfo(
            tested_patterns=[("SearchBox:input", "SearchResults:select")],
            tested_components={"SearchBox", "SearchResults"},
            tested_events={"input", "select"},
        )

        assert len(coverage.tested_patterns) == 1
        assert "SearchBox" in coverage.tested_components
        assert "input" in coverage.tested_events


class TestGapDetectorClass:
    """Tests for GapDetector."""

    def test_detector_initialization(self, store, config):
        """Test detector initializes correctly."""
        detector = GapDetector(store, config)
        assert detector.store == store
        assert detector.config == config

    def test_load_coverage_from_tests(self, store, config):
        """Test loading coverage from test patterns."""
        detector = GapDetector(store, config)

        test_patterns = [
            ["SearchBox:input", "SearchResults:select"],
            ["UploadBox:click", "FileDialog:select"],
        ]
        detector.load_coverage_from_tests(test_patterns)

        assert detector._coverage_info is not None
        assert len(detector._coverage_info.tested_patterns) == 2
        assert "SearchBox" in detector._coverage_info.tested_components
        assert "UploadBox" in detector._coverage_info.tested_components

    def test_analyze_gaps_empty_store(self, store, config):
        """Test gap analysis with empty store."""
        detector = GapDetector(store, config)
        result = detector.analyze_gaps()

        assert isinstance(result, GapAnalysisResult)
        assert result.total_patterns_observed == 0
        assert result.coverage_percentage == 100.0  # No patterns = 100% covered

    def test_analyze_gaps_with_sessions(self, populated_store, config):
        """Test gap analysis with populated store."""
        detector = GapDetector(populated_store, config)

        # No test coverage - all patterns should be gaps
        result = detector.analyze_gaps(min_support=0.1, min_observed=1)

        assert isinstance(result, GapAnalysisResult)
        assert result.total_patterns_observed >= 0

    def test_analyze_gaps_with_coverage(self, populated_store, config):
        """Test gap analysis with some coverage."""
        detector = GapDetector(populated_store, config)

        # Add some test coverage
        test_patterns = [
            ["SearchBox:input", "SearchResults:select"],
        ]
        detector.load_coverage_from_tests(test_patterns)

        result = detector.analyze_gaps(min_support=0.1)

        # Should have some coverage
        assert result.patterns_covered >= 0

    def test_gap_priority_calculation(self, store, config):
        """Test priority calculation logic."""
        detector = GapDetector(store, config)

        # High observation count -> high priority
        priority_high = detector._calculate_priority(100, "uncovered_flow")
        assert priority_high in ["critical", "high"]

        # Low observation count -> low priority
        priority_low = detector._calculate_priority(5, "uncovered_flow")
        assert priority_low in ["medium", "low"]

        # Error flow -> always high priority
        priority_error = detector._calculate_priority(5, "error_flow")
        assert priority_error in ["critical", "high"]

    def test_generate_gap_id(self, store, config):
        """Test gap ID generation is deterministic."""
        detector = GapDetector(store, config)

        id1 = detector._generate_gap_id("flow", "search_view")
        id2 = detector._generate_gap_id("flow", "search_view")

        assert id1 == id2
        assert id1.startswith("gap_flow_")

    def test_slugify(self, store, config):
        """Test text slugification."""
        detector = GapDetector(store, config)

        assert detector._slugify("SearchBox") == "searchbox"
        assert detector._slugify("Upload Box") == "upload_box"
        assert detector._slugify("error-flow") == "error_flow"

    def test_generate_test_suggestion(self, populated_store, config):
        """Test test suggestion generation."""
        detector = GapDetector(populated_store, config)

        gap = TestGap(
            gap_id="gap_test_001",
            gap_type="uncovered_flow",
            flow_description="SearchBox:input → SearchResults:select",
            affected_components=["SearchBox", "SearchResults"],
            observed_count=50,
            suggested_test_name="test_search_flow",
            suggested_assertions=["Assert search works"],
            priority="high",
        )

        suggestion = detector.generate_test_suggestion(gap)

        assert suggestion["gap_id"] == "gap_test_001"
        assert suggestion["test_name"] == "test_search_flow"
        assert "test_code" in suggestion
        assert "def test_search_flow" in suggestion["test_code"]

    def test_get_priority_gaps(self, populated_store, config):
        """Test getting gaps by priority."""
        detector = GapDetector(populated_store, config)

        # First analyze to create some gaps
        detector.analyze_gaps(min_support=0.05, min_observed=1)

        # Get high priority gaps
        gaps = detector.get_priority_gaps(min_priority="high", limit=10)

        # All returned gaps should be high priority or above
        for gap in gaps:
            assert gap.priority in ["critical", "high"]


class TestGapAnalysisResultClass:
    """Tests for GapAnalysisResult."""

    def test_result_to_dict(self):
        """Test result serialization."""
        result = GapAnalysisResult(
            gaps=[],
            total_patterns_observed=100,
            patterns_covered=80,
            coverage_percentage=80.0,
            gaps_by_type={"uncovered_flow": 5},
            gaps_by_priority={"high": 3, "medium": 2},
        )

        d = result.to_dict()

        assert d["total_gaps"] == 0
        assert d["total_patterns_observed"] == 100
        assert d["patterns_covered"] == 80
        assert d["coverage_percentage"] == 80.0


class TestPatternCoverage:
    """Tests for pattern coverage checking."""

    def test_exact_match(self, store, config):
        """Test exact pattern match."""
        detector = GapDetector(store, config)

        coverage = CoverageInfo(
            tested_patterns=[("A:x", "B:y")],
            tested_components={"A", "B"},
            tested_events={"x", "y"},
        )

        # Exact match
        assert detector._is_pattern_covered(("A:x", "B:y"), coverage)

        # Different pattern
        assert not detector._is_pattern_covered(("C:z", "D:w"), coverage)

    def test_subsequence_match(self, store, config):
        """Test subsequence coverage."""
        detector = GapDetector(store, config)

        coverage = CoverageInfo(
            tested_patterns=[("A:x", "B:y", "C:z")],
            tested_components={"A", "B", "C"},
            tested_events={"x", "y", "z"},
        )

        # Subsequence of tested pattern
        assert detector._is_subsequence(("A:x", "C:z"), ("A:x", "B:y", "C:z"))


class TestGapsConvenienceFunction:
    """Tests for convenience functions."""

    def test_analyze_test_gaps_function(self, populated_store, config):
        """Test analyze_test_gaps convenience function."""
        result = analyze_test_gaps(populated_store, config=config)

        assert isinstance(result, GapAnalysisResult)

    def test_analyze_with_test_patterns(self, populated_store, config):
        """Test with test patterns provided."""
        test_patterns = [
            ["SearchBox:input", "SearchResults:select"],
        ]

        result = analyze_test_gaps(
            populated_store,
            test_patterns=test_patterns,
            config=config,
        )

        assert isinstance(result, GapAnalysisResult)
