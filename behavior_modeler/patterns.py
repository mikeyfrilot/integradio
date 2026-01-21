"""
Sequential Pattern Mining - Discover common flow patterns.

Implements simplified GSP (Generalized Sequential Pattern) algorithm
for finding frequent subsequences in user flows.

Based on clickstream analysis best practices:
- Mine patterns at multiple granularities (events, segments, sequences)
- Support gap constraints for flexible matching
- Calculate support/confidence metrics for pattern quality
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Iterator
import hashlib

from .config import BehaviorModelerConfig
from .models import Session, FlowEvent
from .store import FlowStore

logger = logging.getLogger(__name__)


@dataclass
class SequentialPattern:
    """A discovered sequential pattern in user flows."""
    pattern_id: str
    sequence: list[str]  # e.g., ["SearchBox:input", "SearchResults:select", "CodePanel:view"]
    support: float       # Fraction of sessions containing this pattern
    confidence: float    # How often pattern leads to completion
    occurrence_count: int
    avg_duration_ms: float  # Average time to complete pattern
    sample_session_ids: list[str] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.sequence)

    def to_dict(self) -> dict:
        return {
            "pattern_id": self.pattern_id,
            "sequence": self.sequence,
            "support": self.support,
            "confidence": self.confidence,
            "occurrence_count": self.occurrence_count,
            "avg_duration_ms": self.avg_duration_ms,
            "length": self.length,
            "sample_session_ids": self.sample_session_ids,
        }


@dataclass
class PatternMiningResult:
    """Result from pattern mining operation."""
    patterns: list[SequentialPattern]
    n_sessions_analyzed: int
    min_support_used: float
    max_pattern_length: int


class SequentialPatternMiner:
    """
    Mine frequent sequential patterns from user flows.

    Implements a simplified GSP-like algorithm:
    1. Generate candidate sequences of length k
    2. Count support in session database
    3. Prune candidates below min_support
    4. Generate candidates of length k+1
    5. Repeat until no candidates remain
    """

    def __init__(
        self,
        store: FlowStore,
        config: Optional[BehaviorModelerConfig] = None,
    ):
        """
        Initialize pattern miner.

        Args:
            store: FlowStore containing sessions
            config: Configuration options
        """
        self.store = store
        self.config = config or BehaviorModelerConfig()

    def mine_patterns(
        self,
        min_support: float = 0.05,
        max_length: int = 5,
        min_length: int = 2,
        max_gap: int = 2,
    ) -> PatternMiningResult:
        """
        Mine sequential patterns from all sessions.

        Args:
            min_support: Minimum support threshold (0.0 to 1.0)
            max_length: Maximum pattern length
            min_length: Minimum pattern length
            max_gap: Maximum gap between pattern elements

        Returns:
            PatternMiningResult with discovered patterns
        """
        # Load and convert sessions to sequences
        sessions = list(self.store.iter_sessions(include_events=True))
        sequences = [self._session_to_sequence(s) for s in sessions]
        n_sessions = len(sequences)

        if n_sessions == 0:
            return PatternMiningResult(
                patterns=[],
                n_sessions_analyzed=0,
                min_support_used=min_support,
                max_pattern_length=max_length,
            )

        min_count = max(1, int(n_sessions * min_support))
        logger.info(f"Mining patterns: {n_sessions} sessions, min_count={min_count}")

        # Generate length-1 patterns
        item_counts = defaultdict(int)
        for seq in sequences:
            seen = set()
            for item in seq:
                if item not in seen:
                    item_counts[item] += 1
                    seen.add(item)

        # Filter by support
        frequent_items = {
            item for item, count in item_counts.items()
            if count >= min_count
        }

        if not frequent_items:
            return PatternMiningResult(
                patterns=[],
                n_sessions_analyzed=n_sessions,
                min_support_used=min_support,
                max_pattern_length=max_length,
            )

        # Iteratively find longer patterns
        all_patterns = []
        current_patterns = {(item,): item_counts[item] for item in frequent_items}

        for length in range(2, max_length + 1):
            # Generate candidates
            candidates = self._generate_candidates(current_patterns.keys(), length)

            if not candidates:
                break

            # Count support
            candidate_counts = defaultdict(int)
            candidate_sessions = defaultdict(list)
            candidate_durations = defaultdict(list)

            for session, seq in zip(sessions, sequences):
                for candidate in candidates:
                    if self._sequence_contains(seq, candidate, max_gap):
                        candidate_counts[candidate] += 1
                        candidate_sessions[candidate].append(session.session_id)
                        # Estimate duration for this pattern
                        duration = self._estimate_pattern_duration(session, candidate)
                        if duration:
                            candidate_durations[candidate].append(duration)

            # Filter by support
            current_patterns = {
                pattern: count
                for pattern, count in candidate_counts.items()
                if count >= min_count
            }

            # Build pattern objects for patterns meeting min_length
            if length >= min_length:
                for pattern, count in current_patterns.items():
                    support = count / n_sessions
                    confidence = self._calculate_confidence(sessions, pattern)
                    durations = candidate_durations[pattern]

                    all_patterns.append(SequentialPattern(
                        pattern_id=self._generate_pattern_id(pattern),
                        sequence=list(pattern),
                        support=support,
                        confidence=confidence,
                        occurrence_count=count,
                        avg_duration_ms=sum(durations) / len(durations) if durations else 0,
                        sample_session_ids=candidate_sessions[pattern][:5],
                    ))

            logger.debug(f"Length {length}: {len(current_patterns)} frequent patterns")

        # Sort by support
        all_patterns.sort(key=lambda p: (-p.support, -p.length))

        logger.info(f"Found {len(all_patterns)} patterns")

        return PatternMiningResult(
            patterns=all_patterns,
            n_sessions_analyzed=n_sessions,
            min_support_used=min_support,
            max_pattern_length=max_length,
        )

    def _session_to_sequence(self, session: Session) -> list[str]:
        """Convert session events to sequence of items."""
        return [
            f"{e.component_type}:{e.event_type}"
            for e in session.events
            if e.component_type  # Skip events without component
        ]

    def _generate_candidates(
        self,
        patterns: Iterator[tuple[str, ...]],
        target_length: int,
    ) -> set[tuple[str, ...]]:
        """
        Generate candidate patterns of target length.

        Uses apriori-like joining: patterns sharing k-1 prefix/suffix.
        """
        patterns = list(patterns)
        candidates = set()

        for p1 in patterns:
            for p2 in patterns:
                # Join if suffix of p1 matches prefix of p2
                if p1[1:] == p2[:-1]:
                    candidate = p1 + (p2[-1],)
                    if len(candidate) == target_length:
                        candidates.add(candidate)

        return candidates

    def _sequence_contains(
        self,
        sequence: list[str],
        pattern: tuple[str, ...],
        max_gap: int,
    ) -> bool:
        """
        Check if sequence contains pattern with gap constraint.

        Args:
            sequence: Full sequence to search
            pattern: Pattern to find
            max_gap: Maximum gap between consecutive pattern elements

        Returns:
            True if pattern found within gap constraints
        """
        if not pattern:
            return True

        pattern_idx = 0
        last_match = -1

        for i, item in enumerate(sequence):
            if item == pattern[pattern_idx]:
                # Check gap constraint
                if last_match >= 0 and (i - last_match - 1) > max_gap:
                    # Gap too large, restart
                    pattern_idx = 0
                    if item == pattern[0]:
                        pattern_idx = 1
                        last_match = i
                    continue

                pattern_idx += 1
                last_match = i

                if pattern_idx == len(pattern):
                    return True

        return False

    def _calculate_confidence(
        self,
        sessions: list[Session],
        pattern: tuple[str, ...],
    ) -> float:
        """
        Calculate confidence: how often pattern leads to completion.

        Confidence = sessions_with_pattern_that_complete / sessions_with_pattern
        """
        contains_pattern = 0
        completes_with_pattern = 0

        for session in sessions:
            seq = self._session_to_sequence(session)
            if self._sequence_contains(seq, pattern, max_gap=2):
                contains_pattern += 1
                if session.is_complete:
                    completes_with_pattern += 1

        if contains_pattern == 0:
            return 0.0

        return completes_with_pattern / contains_pattern

    def _estimate_pattern_duration(
        self,
        session: Session,
        pattern: tuple[str, ...],
    ) -> Optional[int]:
        """Estimate duration to complete pattern within session."""
        seq = self._session_to_sequence(session)
        events = session.events

        # Find pattern occurrence
        pattern_idx = 0
        start_time = None
        end_time = None

        for i, item in enumerate(seq):
            if item == pattern[pattern_idx]:
                if pattern_idx == 0:
                    start_time = events[i].timestamp
                pattern_idx += 1
                if pattern_idx == len(pattern):
                    end_time = events[i].timestamp
                    break

        if start_time and end_time:
            return int((end_time - start_time).total_seconds() * 1000)
        return None

    def _generate_pattern_id(self, pattern: tuple[str, ...]) -> str:
        """Generate unique ID for pattern."""
        content = "->".join(pattern)
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def find_pattern_gaps(
        self,
        test_patterns: Optional[list[list[str]]] = None,
        min_support: float = 0.05,
    ) -> list[dict]:
        """
        Find patterns that exist in production but may not be tested.

        Args:
            test_patterns: Known test patterns (if None, assumes none tested)
            min_support: Minimum support for patterns to consider

        Returns:
            List of pattern gaps with suggested test names
        """
        # Mine current patterns
        result = self.mine_patterns(min_support=min_support)

        test_patterns = test_patterns or []
        test_set = {tuple(p) for p in test_patterns}

        gaps = []
        for pattern in result.patterns:
            if tuple(pattern.sequence) not in test_set:
                # Generate suggested test name
                components = [s.split(":")[0] for s in pattern.sequence]
                test_name = "test_" + "_to_".join(components[:3]).lower() + "_flow"

                gaps.append({
                    "pattern": pattern.sequence,
                    "support": pattern.support,
                    "confidence": pattern.confidence,
                    "occurrence_count": pattern.occurrence_count,
                    "suggested_test_name": test_name,
                    "priority": "high" if pattern.support > 0.1 else "medium",
                })

        return gaps


def mine_patterns(
    store: FlowStore,
    min_support: float = 0.05,
    config: Optional[BehaviorModelerConfig] = None,
) -> PatternMiningResult:
    """
    Convenience function to mine patterns from store.

    Args:
        store: FlowStore with sessions
        min_support: Minimum support threshold
        config: Optional configuration

    Returns:
        PatternMiningResult
    """
    miner = SequentialPatternMiner(store, config)
    return miner.mine_patterns(min_support=min_support)
