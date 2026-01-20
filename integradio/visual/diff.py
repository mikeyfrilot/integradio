"""
Spec Diffing - Compare visual specifications for changes.

Provides:
- Token-level diff (color, spacing, etc.)
- Component-level diff (added, removed, modified)
- Semantic versioning suggestions
- Change reports and changelogs

Based on:
- Semantic Versioning (semver.org)
- DTCG token format
- Design system versioning best practices
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
from enum import Enum
import json
from datetime import datetime


# =============================================================================
# Change Types
# =============================================================================

class ChangeType(str, Enum):
    """Types of changes between specs."""
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"


class ChangeLevel(str, Enum):
    """
    Semantic version impact level.

    Based on semver.org:
    - MAJOR: Breaking changes (removals, incompatible modifications)
    - MINOR: New features (additions, backwards-compatible additions)
    - PATCH: Bug fixes (non-breaking modifications)
    """
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"
    NONE = "none"


class ChangeCategory(str, Enum):
    """Categories of changes."""
    TOKEN = "token"
    COMPONENT = "component"
    LAYOUT = "layout"
    ANIMATION = "animation"
    TYPOGRAPHY = "typography"
    COLOR = "color"
    SPACING = "spacing"
    OTHER = "other"


# =============================================================================
# Change Records
# =============================================================================

@dataclass
class Change:
    """A single change between specs."""
    path: str  # JSONPath-like path to changed value
    change_type: ChangeType
    category: ChangeCategory
    old_value: Any = None
    new_value: Any = None
    description: str = ""

    @property
    def level(self) -> ChangeLevel:
        """Determine semantic version impact."""
        if self.change_type == ChangeType.REMOVED:
            return ChangeLevel.MAJOR
        elif self.change_type == ChangeType.ADDED:
            return ChangeLevel.MINOR
        elif self.change_type == ChangeType.MODIFIED:
            # Modifications can be breaking or not
            if self._is_breaking_modification():
                return ChangeLevel.MAJOR
            return ChangeLevel.PATCH
        return ChangeLevel.NONE

    def _is_breaking_modification(self) -> bool:
        """Determine if a modification is breaking."""
        # Type changes are breaking
        if type(self.old_value) != type(self.new_value):
            return True

        # Structural changes in dicts/lists are breaking
        if isinstance(self.old_value, dict) and isinstance(self.new_value, dict):
            old_keys = set(self.old_value.keys())
            new_keys = set(self.new_value.keys())
            # Removed keys are breaking
            if old_keys - new_keys:
                return True

        return False

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "type": self.change_type.value,
            "category": self.category.value,
            "level": self.level.value,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "description": self.description,
        }


@dataclass
class DiffReport:
    """Complete diff report between two specs."""
    old_version: str
    new_version: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    changes: list[Change] = field(default_factory=list)

    @property
    def total_changes(self) -> int:
        return len(self.changes)

    @property
    def added_count(self) -> int:
        return len([c for c in self.changes if c.change_type == ChangeType.ADDED])

    @property
    def removed_count(self) -> int:
        return len([c for c in self.changes if c.change_type == ChangeType.REMOVED])

    @property
    def modified_count(self) -> int:
        return len([c for c in self.changes if c.change_type == ChangeType.MODIFIED])

    @property
    def suggested_version_bump(self) -> ChangeLevel:
        """Suggest version bump based on changes."""
        if not self.changes:
            return ChangeLevel.NONE

        levels = [c.level for c in self.changes]
        if ChangeLevel.MAJOR in levels:
            return ChangeLevel.MAJOR
        elif ChangeLevel.MINOR in levels:
            return ChangeLevel.MINOR
        elif ChangeLevel.PATCH in levels:
            return ChangeLevel.PATCH
        return ChangeLevel.NONE

    def get_changes_by_category(self) -> dict[str, list[Change]]:
        """Group changes by category."""
        by_category: dict[str, list[Change]] = {}
        for change in self.changes:
            cat = change.category.value
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(change)
        return by_category

    def get_changes_by_level(self) -> dict[str, list[Change]]:
        """Group changes by impact level."""
        by_level: dict[str, list[Change]] = {}
        for change in self.changes:
            level = change.level.value
            if level not in by_level:
                by_level[level] = []
            by_level[level].append(change)
        return by_level

    def to_dict(self) -> dict:
        return {
            "old_version": self.old_version,
            "new_version": self.new_version,
            "timestamp": self.timestamp,
            "summary": {
                "total": self.total_changes,
                "added": self.added_count,
                "removed": self.removed_count,
                "modified": self.modified_count,
                "suggested_bump": self.suggested_version_bump.value,
            },
            "changes": [c.to_dict() for c in self.changes],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# =============================================================================
# Spec Differ
# =============================================================================

class SpecDiffer:
    """
    Compare two visual specifications.

    Handles:
    - Token-level comparison
    - Component-level comparison
    - Nested structure diffing
    - Change categorization
    """

    def __init__(self):
        self.changes: list[Change] = []

    def diff(
        self,
        old_spec: dict,
        new_spec: dict,
        old_version: str = "0.0.0",
        new_version: str = "0.0.1",
    ) -> DiffReport:
        """
        Compare two specs and generate diff report.

        Args:
            old_spec: Previous spec dict
            new_spec: New spec dict
            old_version: Version string for old spec
            new_version: Version string for new spec

        Returns:
            DiffReport with all changes
        """
        self.changes = []
        self._diff_recursive(old_spec, new_spec, "")

        return DiffReport(
            old_version=old_version,
            new_version=new_version,
            changes=self.changes,
        )

    def _diff_recursive(
        self,
        old: Any,
        new: Any,
        path: str,
    ) -> None:
        """Recursively diff two values."""
        # Handle None cases
        if old is None and new is not None:
            self._record_change(path, ChangeType.ADDED, None, new)
            return
        if old is not None and new is None:
            self._record_change(path, ChangeType.REMOVED, old, None)
            return
        if old is None and new is None:
            return

        # Type mismatch
        if type(old) != type(new):
            self._record_change(path, ChangeType.MODIFIED, old, new)
            return

        # Dict comparison
        if isinstance(old, dict):
            all_keys = set(old.keys()) | set(new.keys())
            for key in all_keys:
                key_path = f"{path}.{key}" if path else key
                if key not in old:
                    self._record_change(key_path, ChangeType.ADDED, None, new[key])
                elif key not in new:
                    self._record_change(key_path, ChangeType.REMOVED, old[key], None)
                else:
                    self._diff_recursive(old[key], new[key], key_path)
            return

        # List comparison
        if isinstance(old, list):
            # For lists, do simple length/content comparison
            if len(old) != len(new):
                self._record_change(path, ChangeType.MODIFIED, old, new)
            else:
                for i, (old_item, new_item) in enumerate(zip(old, new)):
                    self._diff_recursive(old_item, new_item, f"{path}[{i}]")
            return

        # Primitive comparison
        if old != new:
            self._record_change(path, ChangeType.MODIFIED, old, new)

    def _record_change(
        self,
        path: str,
        change_type: ChangeType,
        old_value: Any,
        new_value: Any,
    ) -> None:
        """Record a change."""
        category = self._categorize_path(path)
        description = self._describe_change(path, change_type, old_value, new_value)

        self.changes.append(Change(
            path=path,
            change_type=change_type,
            category=category,
            old_value=old_value,
            new_value=new_value,
            description=description,
        ))

    def _categorize_path(self, path: str) -> ChangeCategory:
        """Determine category from path."""
        path_lower = path.lower()

        if "token" in path_lower:
            # Further categorize tokens
            if "color" in path_lower:
                return ChangeCategory.COLOR
            if "spacing" in path_lower or "margin" in path_lower or "padding" in path_lower:
                return ChangeCategory.SPACING
            if "font" in path_lower or "text" in path_lower or "typography" in path_lower:
                return ChangeCategory.TYPOGRAPHY
            if "animation" in path_lower or "transition" in path_lower:
                return ChangeCategory.ANIMATION
            return ChangeCategory.TOKEN

        if "component" in path_lower:
            return ChangeCategory.COMPONENT
        if "layout" in path_lower:
            return ChangeCategory.LAYOUT
        if "animation" in path_lower:
            return ChangeCategory.ANIMATION
        if "color" in path_lower:
            return ChangeCategory.COLOR
        if "spacing" in path_lower:
            return ChangeCategory.SPACING

        return ChangeCategory.OTHER

    def _describe_change(
        self,
        path: str,
        change_type: ChangeType,
        old_value: Any,
        new_value: Any,
    ) -> str:
        """Generate human-readable description."""
        name = path.split(".")[-1] if "." in path else path

        if change_type == ChangeType.ADDED:
            return f"Added {name}"
        elif change_type == ChangeType.REMOVED:
            return f"Removed {name}"
        elif change_type == ChangeType.MODIFIED:
            if isinstance(old_value, (int, float, str, bool)):
                return f"Changed {name} from '{old_value}' to '{new_value}'"
            return f"Modified {name}"

        return f"{name} unchanged"


# =============================================================================
# Changelog Generator
# =============================================================================

def generate_changelog(report: DiffReport) -> str:
    """
    Generate a markdown changelog from diff report.

    Args:
        report: DiffReport to convert

    Returns:
        Markdown changelog string
    """
    lines = [
        f"# Changelog: {report.old_version} → {report.new_version}",
        "",
        f"Generated: {report.timestamp}",
        "",
        "## Summary",
        "",
        f"- **Total changes**: {report.total_changes}",
        f"- **Added**: {report.added_count}",
        f"- **Removed**: {report.removed_count}",
        f"- **Modified**: {report.modified_count}",
        f"- **Suggested bump**: {report.suggested_version_bump.value.upper()}",
        "",
    ]

    # Group by level for organized output
    by_level = report.get_changes_by_level()

    if "major" in by_level:
        lines.extend([
            "## ⚠️ Breaking Changes",
            "",
        ])
        for change in by_level["major"]:
            lines.append(f"- {change.description}")
        lines.append("")

    if "minor" in by_level:
        lines.extend([
            "## ✨ New Features",
            "",
        ])
        for change in by_level["minor"]:
            lines.append(f"- {change.description}")
        lines.append("")

    if "patch" in by_level:
        lines.extend([
            "## 🔧 Fixes & Updates",
            "",
        ])
        for change in by_level["patch"]:
            lines.append(f"- {change.description}")
        lines.append("")

    return "\n".join(lines)


def generate_json_changelog(report: DiffReport) -> str:
    """
    Generate a JSON changelog from diff report.

    Args:
        report: DiffReport to convert

    Returns:
        JSON changelog string
    """
    changelog = {
        "version": report.new_version,
        "previous_version": report.old_version,
        "date": report.timestamp,
        "suggested_bump": report.suggested_version_bump.value,
        "breaking_changes": [
            c.to_dict() for c in report.changes
            if c.level == ChangeLevel.MAJOR
        ],
        "features": [
            c.to_dict() for c in report.changes
            if c.level == ChangeLevel.MINOR
        ],
        "fixes": [
            c.to_dict() for c in report.changes
            if c.level == ChangeLevel.PATCH
        ],
    }
    return json.dumps(changelog, indent=2)


# =============================================================================
# Version Utilities
# =============================================================================

def parse_version(version: str) -> tuple[int, int, int]:
    """Parse semver string to tuple."""
    parts = version.lstrip("v").split(".")
    major = int(parts[0]) if len(parts) > 0 else 0
    minor = int(parts[1]) if len(parts) > 1 else 0
    patch = int(parts[2]) if len(parts) > 2 else 0
    return (major, minor, patch)


def bump_version(version: str, level: ChangeLevel) -> str:
    """
    Bump version according to level.

    Args:
        version: Current version (e.g., "1.2.3")
        level: What to bump

    Returns:
        New version string
    """
    major, minor, patch = parse_version(version)

    if level == ChangeLevel.MAJOR:
        return f"{major + 1}.0.0"
    elif level == ChangeLevel.MINOR:
        return f"{major}.{minor + 1}.0"
    elif level == ChangeLevel.PATCH:
        return f"{major}.{minor}.{patch + 1}"

    return version


def suggest_version(old_version: str, report: DiffReport) -> str:
    """
    Suggest new version based on changes.

    Args:
        old_version: Current version
        report: Diff report

    Returns:
        Suggested new version
    """
    return bump_version(old_version, report.suggested_version_bump)


# =============================================================================
# Convenience Functions
# =============================================================================

def diff_specs(
    old_spec: dict,
    new_spec: dict,
    old_version: str = "0.0.0",
    new_version: str | None = None,
) -> DiffReport:
    """
    Compare two specs and return diff report.

    Args:
        old_spec: Previous spec
        new_spec: New spec
        old_version: Previous version string
        new_version: New version (auto-suggested if None)

    Returns:
        DiffReport
    """
    differ = SpecDiffer()
    report = differ.diff(old_spec, new_spec, old_version, new_version or "0.0.0")

    # Update new_version if auto-suggested
    if new_version is None:
        report.new_version = suggest_version(old_version, report)

    return report


def diff_visual_specs(
    old_spec: "VisualSpec",
    new_spec: "VisualSpec",
) -> DiffReport:
    """
    Compare two VisualSpec objects.

    Args:
        old_spec: Previous VisualSpec
        new_spec: New VisualSpec

    Returns:
        DiffReport
    """
    old_dict = old_spec.to_dict() if hasattr(old_spec, "to_dict") else {}
    new_dict = new_spec.to_dict() if hasattr(new_spec, "to_dict") else {}

    return diff_specs(old_dict, new_dict)


def diff_ui_specs(
    old_spec: "UISpec",
    new_spec: "UISpec",
    old_version: str = "0.0.0",
) -> DiffReport:
    """
    Compare two UISpec objects.

    Args:
        old_spec: Previous UISpec
        new_spec: New UISpec
        old_version: Previous version

    Returns:
        DiffReport
    """
    old_dict = old_spec.to_dict() if hasattr(old_spec, "to_dict") else {}
    new_dict = new_spec.to_dict() if hasattr(new_spec, "to_dict") else {}

    return diff_specs(old_dict, new_dict, old_version)
