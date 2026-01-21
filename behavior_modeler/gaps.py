"""
Test gap detection and Code Covered integration.

Compares observed user behavior patterns against existing test coverage
to identify untested flows and generate test suggestions.
"""

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .config import BehaviorModelerConfig
from .models import Session, FlowEvent, TestGap
from .store import FlowStore
from .patterns import SequentialPatternMiner, PatternMiningResult


@dataclass
class CoverageInfo:
    """Information about existing test coverage."""

    # Patterns/flows that are explicitly tested
    tested_patterns: list[tuple[str, ...]] = field(default_factory=list)

    # Components that have tests
    tested_components: set[str] = field(default_factory=set)

    # Event types that are tested
    tested_events: set[str] = field(default_factory=set)

    # Test file locations (for reference)
    test_files: list[str] = field(default_factory=list)


@dataclass
class GapAnalysisResult:
    """Results from gap analysis."""

    gaps: list[TestGap]
    total_patterns_observed: int
    patterns_covered: int
    coverage_percentage: float

    # Breakdown by type
    gaps_by_type: dict[str, int] = field(default_factory=dict)
    gaps_by_priority: dict[str, int] = field(default_factory=dict)

    # Component-level analysis
    untested_components: list[str] = field(default_factory=list)
    undertested_components: list[tuple[str, int]] = field(default_factory=list)  # (component, observed_count)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_gaps": len(self.gaps),
            "total_patterns_observed": self.total_patterns_observed,
            "patterns_covered": self.patterns_covered,
            "coverage_percentage": round(self.coverage_percentage, 2),
            "gaps_by_type": self.gaps_by_type,
            "gaps_by_priority": self.gaps_by_priority,
            "untested_components": self.untested_components,
            "undertested_components": [
                {"component": c, "observed_count": count}
                for c, count in self.undertested_components
            ],
            "gaps": [g.to_dict() for g in self.gaps],
        }


class GapDetector:
    """
    Detects gaps between observed user behavior and test coverage.

    Uses pattern mining to find frequently occurring user flows,
    then compares against known test patterns to identify gaps.
    """

    def __init__(
        self,
        store: FlowStore,
        config: Optional[BehaviorModelerConfig] = None,
    ):
        """
        Initialize the detector.

        Args:
            store: Flow store with observed sessions
            config: Configuration
        """
        self.store = store
        self.config = config or BehaviorModelerConfig()
        self._pattern_miner = SequentialPatternMiner(store, config)
        self._coverage_info: Optional[CoverageInfo] = None

    def set_coverage_info(self, coverage: CoverageInfo) -> None:
        """
        Set test coverage information.

        Args:
            coverage: Information about existing test coverage
        """
        self._coverage_info = coverage

    def load_coverage_from_tests(self, test_patterns: list[list[str]]) -> None:
        """
        Load coverage information from test patterns.

        Args:
            test_patterns: List of tested flow patterns
                          (each pattern is a list of "Component:event" strings)
        """
        coverage = CoverageInfo()

        for pattern in test_patterns:
            coverage.tested_patterns.append(tuple(pattern))

            for step in pattern:
                if ":" in step:
                    component, event = step.split(":", 1)
                    coverage.tested_components.add(component)
                    coverage.tested_events.add(event)
                else:
                    coverage.tested_components.add(step)

        self._coverage_info = coverage

    def analyze_gaps(
        self,
        min_support: float = 0.05,
        min_observed: int = 3,
        include_component_gaps: bool = True,
    ) -> GapAnalysisResult:
        """
        Analyze gaps between observed behavior and test coverage.

        Args:
            min_support: Minimum support threshold for patterns
            min_observed: Minimum observation count to report as gap
            include_component_gaps: Also detect untested components

        Returns:
            Gap analysis results
        """
        gaps: list[TestGap] = []

        # Mine patterns from observed behavior
        pattern_result = self._pattern_miner.mine_patterns(
            min_support=min_support,
            min_length=2,
            max_length=6,
        )

        coverage = self._coverage_info or CoverageInfo()

        # Track statistics
        patterns_covered = 0
        observed_components: Counter = Counter()

        # Analyze each pattern
        for pattern in pattern_result.patterns:
            pattern_tuple = tuple(pattern.sequence)

            # Check if this pattern is covered by tests
            is_covered = self._is_pattern_covered(pattern_tuple, coverage)

            if is_covered:
                patterns_covered += 1
                continue

            # This is a gap
            if pattern.occurrence_count >= min_observed:
                gap = self._create_gap_from_pattern(pattern)
                gaps.append(gap)

            # Track components
            for step in pattern.sequence:
                if ":" in step:
                    component = step.split(":")[0]
                    observed_components[component] += pattern.occurrence_count

        # Detect component-level gaps
        untested_components: list[str] = []
        undertested_components: list[tuple[str, int]] = []

        if include_component_gaps:
            # Get all components from sessions
            all_components = self._get_all_observed_components()

            for component, count in all_components.most_common():
                if component not in coverage.tested_components:
                    untested_components.append(component)

                    if count >= min_observed:
                        # Create component-level gap
                        gap = TestGap(
                            gap_id=self._generate_gap_id("component", component),
                            gap_type="uncovered_component",
                            flow_description=f"Component '{component}' is used but not tested",
                            affected_components=[component],
                            observed_count=count,
                            suggested_test_name=f"test_{self._slugify(component)}_interactions",
                            suggested_assertions=[
                                f"Test that {component} renders correctly",
                                f"Test that {component} handles user interactions",
                            ],
                            priority=self._calculate_priority(count, "component"),
                            priority_reason=f"Observed {count} times in user sessions",
                        )
                        gaps.append(gap)

                # Track undertested (tested but maybe not thoroughly)
                elif count > 50 and component in coverage.tested_components:
                    undertested_components.append((component, count))

        # Calculate coverage percentage
        total_patterns = len(pattern_result.patterns)
        coverage_pct = (patterns_covered / total_patterns * 100) if total_patterns > 0 else 100.0

        # Aggregate by type and priority
        gaps_by_type: dict[str, int] = defaultdict(int)
        gaps_by_priority: dict[str, int] = defaultdict(int)

        for gap in gaps:
            gaps_by_type[gap.gap_type] += 1
            gaps_by_priority[gap.priority] += 1

        # Save gaps to store
        for gap in gaps:
            self.store.save_test_gap(gap)

        return GapAnalysisResult(
            gaps=gaps,
            total_patterns_observed=total_patterns,
            patterns_covered=patterns_covered,
            coverage_percentage=coverage_pct,
            gaps_by_type=dict(gaps_by_type),
            gaps_by_priority=dict(gaps_by_priority),
            untested_components=untested_components,
            undertested_components=undertested_components[:10],  # Top 10
        )

    def get_priority_gaps(
        self,
        min_priority: str = "medium",
        limit: int = 20,
    ) -> list[TestGap]:
        """
        Get gaps by priority.

        Args:
            min_priority: Minimum priority level
            limit: Maximum number to return

        Returns:
            List of high-priority gaps
        """
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        min_level = priority_order.get(min_priority, 2)

        # Get gaps from store
        gaps = self.store.get_test_gaps(status="open")

        # Filter by priority
        filtered = [
            g for g in gaps
            if priority_order.get(g.priority, 3) <= min_level
        ]

        # Sort by priority then by observation count
        filtered.sort(
            key=lambda g: (priority_order.get(g.priority, 3), -g.observed_count)
        )

        return filtered[:limit]

    def generate_test_suggestion(self, gap: TestGap) -> dict:
        """
        Generate a test suggestion for a specific gap.

        Args:
            gap: The test gap to generate a suggestion for

        Returns:
            Test suggestion with code template
        """
        # Get sample session if available
        sample_session = None
        if gap.sample_session_id:
            sample_session = self.store.get_session(gap.sample_session_id)

        # Generate test code template
        test_code = self._generate_test_template(gap, sample_session)

        return {
            "gap_id": gap.gap_id,
            "test_name": gap.suggested_test_name,
            "test_file": f"test_{self._slugify(gap.gap_type)}.py",
            "test_code": test_code,
            "assertions": gap.suggested_assertions,
            "affected_components": gap.affected_components,
            "priority": gap.priority,
            "sample_session_id": gap.sample_session_id,
        }

    def _is_pattern_covered(
        self,
        pattern: tuple[str, ...],
        coverage: CoverageInfo,
    ) -> bool:
        """Check if a pattern is covered by existing tests."""
        # Exact match
        if pattern in coverage.tested_patterns:
            return True

        # Subsequence match (pattern is contained in a tested pattern)
        for tested in coverage.tested_patterns:
            if self._is_subsequence(pattern, tested):
                return True

        # All components tested individually (weak coverage)
        components = set()
        for step in pattern:
            if ":" in step:
                components.add(step.split(":")[0])
            else:
                components.add(step)

        if components and components.issubset(coverage.tested_components):
            # Has component coverage, but not flow coverage
            return False

        return False

    def _is_subsequence(self, sub: tuple, full: tuple) -> bool:
        """Check if sub is a subsequence of full."""
        if len(sub) > len(full):
            return False

        sub_idx = 0
        for item in full:
            if sub_idx < len(sub) and item == sub[sub_idx]:
                sub_idx += 1

        return sub_idx == len(sub)

    def _create_gap_from_pattern(self, pattern) -> TestGap:
        """Create a TestGap from a mined pattern."""
        # Extract components and events
        components = []
        events = []

        for step in pattern.sequence:
            if ":" in step:
                comp, evt = step.split(":", 1)
                components.append(comp)
                events.append(evt)
            else:
                components.append(step)

        # Create human-readable description
        flow_desc = " → ".join(pattern.sequence)

        # Generate test name
        test_name = "test_" + "_to_".join(
            self._slugify(c) for c in components[:3]
        ) + "_flow"

        # Generate assertions
        assertions = []
        for i, step in enumerate(pattern.sequence):
            if ":" in step:
                comp, evt = step.split(":", 1)
                if evt == "click":
                    assertions.append(f"Assert {comp} is clickable")
                elif evt == "input":
                    assertions.append(f"Assert {comp} accepts input")
                elif evt == "view":
                    assertions.append(f"Assert {comp} is visible")
                elif evt == "select":
                    assertions.append(f"Assert {comp} selection works")

        if not assertions:
            assertions = [f"Assert flow {flow_desc} completes successfully"]

        # Determine gap type
        if any("error" in s.lower() for s in pattern.sequence):
            gap_type = "error_flow"
        elif pattern.support < 0.1:
            gap_type = "rare_path"
        else:
            gap_type = "uncovered_flow"

        return TestGap(
            gap_id=self._generate_gap_id("flow", "_".join(components)),
            gap_type=gap_type,
            flow_description=flow_desc,
            affected_components=list(set(components)),
            observed_count=pattern.occurrence_count,
            sample_session_id=pattern.sample_session_ids[0] if pattern.sample_session_ids else None,
            suggested_test_name=test_name,
            suggested_assertions=assertions,
            priority=self._calculate_priority(pattern.occurrence_count, gap_type),
            priority_reason=f"Support: {pattern.support:.1%}, observed {pattern.occurrence_count} times",
        )

    def _get_all_observed_components(self) -> Counter:
        """Get all components from observed sessions."""
        components: Counter = Counter()

        for session in self.store.iter_sessions(include_events=True):
            for event in session.events:
                if event.component_type:
                    components[event.component_type] += 1

        return components

    def _calculate_priority(self, observed_count: int, gap_type: str) -> str:
        """Calculate priority based on observation count and gap type."""
        if gap_type == "error_flow":
            # Error flows are always high priority
            return "critical" if observed_count > 20 else "high"

        if observed_count > 100:
            return "critical"
        elif observed_count > 50:
            return "high"
        elif observed_count > 10:
            return "medium"
        else:
            return "low"

    def _generate_gap_id(self, prefix: str, identifier: str) -> str:
        """Generate a unique gap ID."""
        content = f"{prefix}:{identifier}"
        hash_val = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"gap_{prefix}_{hash_val}"

    def _slugify(self, text: str) -> str:
        """Convert text to slug format."""
        return text.lower().replace(" ", "_").replace("-", "_").replace(".", "_")

    def _generate_test_template(
        self,
        gap: TestGap,
        sample_session: Optional[Session],
    ) -> str:
        """Generate a test code template."""
        test_name = gap.suggested_test_name
        components = gap.affected_components

        # Build test steps from sample session or gap description
        steps = []
        if sample_session and sample_session.events:
            for event in sample_session.events[:10]:  # Limit to first 10 events
                step = f"# Step: {event.component_type}.{event.event_type}"
                if event.event_data:
                    step += f" with data: {json.dumps(event.event_data)}"
                steps.append(step)
        else:
            for comp in components:
                steps.append(f"# Interact with {comp}")

        steps_code = "\n    ".join(steps) if steps else "# Add test steps here"

        assertions_code = "\n    ".join(
            f"# {assertion}" for assertion in gap.suggested_assertions
        )

        template = f'''"""
Test for: {gap.flow_description}
Priority: {gap.priority}
Observed: {gap.observed_count} times
"""

import pytest


def {test_name}(client):
    """
    Test the {gap.gap_type} flow.

    Flow: {gap.flow_description}
    """
    {steps_code}

    # Assertions
    {assertions_code}

    # TODO: Add actual implementation
    raise NotImplementedError("Test needs implementation")
'''

        return template


def analyze_test_gaps(
    store: FlowStore,
    test_patterns: Optional[list[list[str]]] = None,
    config: Optional[BehaviorModelerConfig] = None,
) -> GapAnalysisResult:
    """
    Convenience function to analyze test coverage gaps.

    Args:
        store: Flow store with observed sessions
        test_patterns: Optional list of already-tested patterns
        config: Configuration

    Returns:
        Gap analysis results
    """
    detector = GapDetector(store, config)

    if test_patterns:
        detector.load_coverage_from_tests(test_patterns)

    return detector.analyze_gaps()
