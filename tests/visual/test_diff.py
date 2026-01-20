"""
Tests for Spec Diffing module.

Tests:
- ChangeType, ChangeLevel, ChangeCategory enums
- Change dataclass
- DiffReport dataclass
- SpecDiffer diffing logic
- Changelog generation
- Version utilities
- Convenience functions
"""

import pytest
import json
from integradio.visual.diff import (
    ChangeType,
    ChangeLevel,
    ChangeCategory,
    Change,
    DiffReport,
    SpecDiffer,
    generate_changelog,
    generate_json_changelog,
    parse_version,
    bump_version,
    suggest_version,
    diff_specs,
    diff_visual_specs,
    diff_ui_specs,
)


class TestChangeType:
    """Tests for ChangeType enum."""

    def test_change_types(self):
        """Test all change type values."""
        assert ChangeType.ADDED.value == "added"
        assert ChangeType.REMOVED.value == "removed"
        assert ChangeType.MODIFIED.value == "modified"
        assert ChangeType.UNCHANGED.value == "unchanged"

    def test_from_string(self):
        """Test creating from string."""
        assert ChangeType("added") == ChangeType.ADDED
        assert ChangeType("removed") == ChangeType.REMOVED


class TestChangeLevel:
    """Tests for ChangeLevel enum."""

    def test_change_levels(self):
        """Test all change level values."""
        assert ChangeLevel.MAJOR.value == "major"
        assert ChangeLevel.MINOR.value == "minor"
        assert ChangeLevel.PATCH.value == "patch"
        assert ChangeLevel.NONE.value == "none"


class TestChangeCategory:
    """Tests for ChangeCategory enum."""

    def test_categories(self):
        """Test all category values."""
        assert ChangeCategory.TOKEN.value == "token"
        assert ChangeCategory.COMPONENT.value == "component"
        assert ChangeCategory.LAYOUT.value == "layout"
        assert ChangeCategory.ANIMATION.value == "animation"
        assert ChangeCategory.COLOR.value == "color"
        assert ChangeCategory.SPACING.value == "spacing"


class TestChange:
    """Tests for Change dataclass."""

    def test_create_change(self):
        """Test creating a change."""
        change = Change(
            path="tokens.color.primary",
            change_type=ChangeType.MODIFIED,
            category=ChangeCategory.COLOR,
            old_value="#ff0000",
            new_value="#0000ff",
        )

        assert change.path == "tokens.color.primary"
        assert change.change_type == ChangeType.MODIFIED
        assert change.category == ChangeCategory.COLOR

    def test_added_change_level(self):
        """Test that additions are MINOR level."""
        change = Change(
            path="new.feature",
            change_type=ChangeType.ADDED,
            category=ChangeCategory.OTHER,
            new_value="value",
        )

        assert change.level == ChangeLevel.MINOR

    def test_removed_change_level(self):
        """Test that removals are MAJOR level."""
        change = Change(
            path="old.feature",
            change_type=ChangeType.REMOVED,
            category=ChangeCategory.OTHER,
            old_value="value",
        )

        assert change.level == ChangeLevel.MAJOR

    def test_modified_change_level_patch(self):
        """Test that non-breaking modifications are PATCH level."""
        change = Change(
            path="value",
            change_type=ChangeType.MODIFIED,
            category=ChangeCategory.OTHER,
            old_value="old",
            new_value="new",
        )

        assert change.level == ChangeLevel.PATCH

    def test_modified_change_level_major_type_change(self):
        """Test that type changes are MAJOR level."""
        change = Change(
            path="value",
            change_type=ChangeType.MODIFIED,
            category=ChangeCategory.OTHER,
            old_value="string",
            new_value=123,
        )

        assert change.level == ChangeLevel.MAJOR

    def test_unchanged_level(self):
        """Test that unchanged is NONE level."""
        change = Change(
            path="value",
            change_type=ChangeType.UNCHANGED,
            category=ChangeCategory.OTHER,
        )

        assert change.level == ChangeLevel.NONE

    def test_change_to_dict(self):
        """Test change serialization."""
        change = Change(
            path="test.path",
            change_type=ChangeType.ADDED,
            category=ChangeCategory.TOKEN,
            new_value="new",
            description="Added test",
        )

        data = change.to_dict()

        assert data["path"] == "test.path"
        assert data["type"] == "added"
        assert data["category"] == "token"
        assert data["level"] == "minor"
        assert data["new_value"] == "new"


class TestDiffReport:
    """Tests for DiffReport dataclass."""

    def test_create_report(self):
        """Test creating a report."""
        report = DiffReport(
            old_version="1.0.0",
            new_version="1.1.0",
        )

        assert report.old_version == "1.0.0"
        assert report.new_version == "1.1.0"
        assert report.total_changes == 0

    def test_report_with_changes(self):
        """Test report with changes."""
        changes = [
            Change("a", ChangeType.ADDED, ChangeCategory.OTHER),
            Change("b", ChangeType.ADDED, ChangeCategory.OTHER),
            Change("c", ChangeType.REMOVED, ChangeCategory.OTHER),
            Change("d", ChangeType.MODIFIED, ChangeCategory.OTHER),
        ]

        report = DiffReport(
            old_version="1.0.0",
            new_version="2.0.0",
            changes=changes,
        )

        assert report.total_changes == 4
        assert report.added_count == 2
        assert report.removed_count == 1
        assert report.modified_count == 1

    def test_suggested_version_bump_major(self):
        """Test major version bump suggestion."""
        changes = [
            Change("removed", ChangeType.REMOVED, ChangeCategory.OTHER),
        ]

        report = DiffReport(
            old_version="1.0.0",
            new_version="1.0.0",
            changes=changes,
        )

        assert report.suggested_version_bump == ChangeLevel.MAJOR

    def test_suggested_version_bump_minor(self):
        """Test minor version bump suggestion."""
        changes = [
            Change("added", ChangeType.ADDED, ChangeCategory.OTHER),
        ]

        report = DiffReport(
            old_version="1.0.0",
            new_version="1.0.0",
            changes=changes,
        )

        assert report.suggested_version_bump == ChangeLevel.MINOR

    def test_suggested_version_bump_patch(self):
        """Test patch version bump suggestion."""
        changes = [
            Change("mod", ChangeType.MODIFIED, ChangeCategory.OTHER, "old", "new"),
        ]

        report = DiffReport(
            old_version="1.0.0",
            new_version="1.0.0",
            changes=changes,
        )

        assert report.suggested_version_bump == ChangeLevel.PATCH

    def test_suggested_version_bump_none(self):
        """Test no bump suggestion for no changes."""
        report = DiffReport(
            old_version="1.0.0",
            new_version="1.0.0",
            changes=[],
        )

        assert report.suggested_version_bump == ChangeLevel.NONE

    def test_get_changes_by_category(self):
        """Test grouping changes by category."""
        changes = [
            Change("a", ChangeType.ADDED, ChangeCategory.COLOR),
            Change("b", ChangeType.ADDED, ChangeCategory.COLOR),
            Change("c", ChangeType.ADDED, ChangeCategory.SPACING),
        ]

        report = DiffReport("1.0.0", "1.1.0", changes=changes)
        by_category = report.get_changes_by_category()

        assert len(by_category["color"]) == 2
        assert len(by_category["spacing"]) == 1

    def test_get_changes_by_level(self):
        """Test grouping changes by level."""
        changes = [
            Change("a", ChangeType.ADDED, ChangeCategory.OTHER),
            Change("b", ChangeType.REMOVED, ChangeCategory.OTHER),
            Change("c", ChangeType.MODIFIED, ChangeCategory.OTHER, "x", "y"),
        ]

        report = DiffReport("1.0.0", "2.0.0", changes=changes)
        by_level = report.get_changes_by_level()

        assert len(by_level["minor"]) == 1
        assert len(by_level["major"]) == 1
        assert len(by_level["patch"]) == 1

    def test_report_to_dict(self):
        """Test report serialization."""
        changes = [
            Change("test", ChangeType.ADDED, ChangeCategory.TOKEN),
        ]

        report = DiffReport("1.0.0", "1.1.0", changes=changes)
        data = report.to_dict()

        assert data["old_version"] == "1.0.0"
        assert data["new_version"] == "1.1.0"
        assert data["summary"]["total"] == 1
        assert len(data["changes"]) == 1

    def test_report_to_json(self):
        """Test JSON serialization."""
        report = DiffReport("1.0.0", "1.1.0")
        json_str = report.to_json()

        parsed = json.loads(json_str)
        assert parsed["old_version"] == "1.0.0"


class TestSpecDiffer:
    """Tests for SpecDiffer class."""

    def test_diff_identical(self):
        """Test diffing identical specs."""
        spec = {"key": "value"}
        differ = SpecDiffer()
        report = differ.diff(spec, spec)

        assert report.total_changes == 0

    def test_diff_added_key(self):
        """Test detecting added key."""
        old = {"a": 1}
        new = {"a": 1, "b": 2}

        differ = SpecDiffer()
        report = differ.diff(old, new)

        assert report.added_count == 1
        assert report.changes[0].path == "b"

    def test_diff_removed_key(self):
        """Test detecting removed key."""
        old = {"a": 1, "b": 2}
        new = {"a": 1}

        differ = SpecDiffer()
        report = differ.diff(old, new)

        assert report.removed_count == 1
        assert report.changes[0].path == "b"

    def test_diff_modified_value(self):
        """Test detecting modified value."""
        old = {"a": 1}
        new = {"a": 2}

        differ = SpecDiffer()
        report = differ.diff(old, new)

        assert report.modified_count == 1
        assert report.changes[0].old_value == 1
        assert report.changes[0].new_value == 2

    def test_diff_nested(self):
        """Test diffing nested structures."""
        old = {"parent": {"child": "old"}}
        new = {"parent": {"child": "new"}}

        differ = SpecDiffer()
        report = differ.diff(old, new)

        assert report.modified_count == 1
        assert "parent.child" in report.changes[0].path

    def test_diff_list_length_change(self):
        """Test detecting list length change."""
        old = {"items": [1, 2]}
        new = {"items": [1, 2, 3]}

        differ = SpecDiffer()
        report = differ.diff(old, new)

        assert report.modified_count == 1

    def test_diff_list_item_change(self):
        """Test detecting list item change."""
        old = {"items": [1, 2, 3]}
        new = {"items": [1, 9, 3]}

        differ = SpecDiffer()
        report = differ.diff(old, new)

        assert report.modified_count == 1
        assert "[1]" in report.changes[0].path

    def test_diff_type_change(self):
        """Test detecting type change."""
        old = {"value": "string"}
        new = {"value": 123}

        differ = SpecDiffer()
        report = differ.diff(old, new)

        assert report.modified_count == 1
        assert report.changes[0].level == ChangeLevel.MAJOR

    def test_diff_versions(self):
        """Test version strings are preserved."""
        differ = SpecDiffer()
        report = differ.diff({}, {}, "1.0.0", "1.1.0")

        assert report.old_version == "1.0.0"
        assert report.new_version == "1.1.0"

    def test_categorize_token_color(self):
        """Test color token categorization."""
        old = {}
        new = {"tokens": {"color": {"primary": "#ff0000"}}}

        differ = SpecDiffer()
        report = differ.diff(old, new)

        # Should have color-related changes
        categories = [c.category for c in report.changes]
        assert ChangeCategory.COLOR in categories or ChangeCategory.TOKEN in categories

    def test_categorize_spacing(self):
        """Test spacing categorization."""
        old = {}
        new = {"spacing": {"margin": "16px"}}

        differ = SpecDiffer()
        report = differ.diff(old, new)

        categories = [c.category for c in report.changes]
        assert any(c in categories for c in [ChangeCategory.SPACING, ChangeCategory.OTHER])


class TestChangelogGeneration:
    """Tests for changelog generation."""

    def test_generate_changelog_empty(self):
        """Test changelog for empty report."""
        report = DiffReport("1.0.0", "1.0.0")
        changelog = generate_changelog(report)

        assert "1.0.0" in changelog
        assert "Total changes" in changelog and "0" in changelog

    def test_generate_changelog_with_changes(self):
        """Test changelog with various changes."""
        changes = [
            Change("new_feature", ChangeType.ADDED, ChangeCategory.OTHER, description="Added new feature"),
            Change("old_api", ChangeType.REMOVED, ChangeCategory.OTHER, description="Removed old API"),
            Change("fix", ChangeType.MODIFIED, ChangeCategory.OTHER, "old", "new", "Fixed bug"),
        ]

        report = DiffReport("1.0.0", "2.0.0", changes=changes)
        changelog = generate_changelog(report)

        assert "1.0.0" in changelog
        assert "2.0.0" in changelog
        assert "Breaking Changes" in changelog
        assert "New Features" in changelog
        assert "Fixes" in changelog

    def test_generate_changelog_markdown_format(self):
        """Test changelog is valid markdown."""
        changes = [Change("test", ChangeType.ADDED, ChangeCategory.OTHER)]
        report = DiffReport("1.0.0", "1.1.0", changes=changes)
        changelog = generate_changelog(report)

        # Should have markdown headers
        assert "# " in changelog or "## " in changelog

    def test_generate_json_changelog(self):
        """Test JSON changelog generation."""
        changes = [
            Change("feature", ChangeType.ADDED, ChangeCategory.OTHER),
            Change("removed", ChangeType.REMOVED, ChangeCategory.OTHER),
        ]

        report = DiffReport("1.0.0", "2.0.0", changes=changes)
        json_changelog = generate_json_changelog(report)

        parsed = json.loads(json_changelog)

        assert parsed["version"] == "2.0.0"
        assert parsed["previous_version"] == "1.0.0"
        assert len(parsed["features"]) == 1
        assert len(parsed["breaking_changes"]) == 1


class TestVersionUtilities:
    """Tests for version utility functions."""

    def test_parse_version_full(self):
        """Test parsing full version string."""
        major, minor, patch = parse_version("1.2.3")

        assert major == 1
        assert minor == 2
        assert patch == 3

    def test_parse_version_with_v(self):
        """Test parsing version with v prefix."""
        major, minor, patch = parse_version("v2.1.0")

        assert major == 2
        assert minor == 1
        assert patch == 0

    def test_parse_version_partial(self):
        """Test parsing partial version."""
        major, minor, patch = parse_version("1")

        assert major == 1
        assert minor == 0
        assert patch == 0

    def test_bump_version_major(self):
        """Test major version bump."""
        new_version = bump_version("1.2.3", ChangeLevel.MAJOR)
        assert new_version == "2.0.0"

    def test_bump_version_minor(self):
        """Test minor version bump."""
        new_version = bump_version("1.2.3", ChangeLevel.MINOR)
        assert new_version == "1.3.0"

    def test_bump_version_patch(self):
        """Test patch version bump."""
        new_version = bump_version("1.2.3", ChangeLevel.PATCH)
        assert new_version == "1.2.4"

    def test_bump_version_none(self):
        """Test no bump."""
        new_version = bump_version("1.2.3", ChangeLevel.NONE)
        assert new_version == "1.2.3"

    def test_suggest_version(self):
        """Test version suggestion based on report."""
        changes = [Change("new", ChangeType.ADDED, ChangeCategory.OTHER)]
        report = DiffReport("1.0.0", "1.0.0", changes=changes)

        suggested = suggest_version("1.0.0", report)
        assert suggested == "1.1.0"


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_diff_specs(self):
        """Test diff_specs function."""
        old = {"a": 1}
        new = {"a": 2}

        report = diff_specs(old, new, "1.0.0", "1.0.1")

        assert isinstance(report, DiffReport)
        assert report.modified_count == 1

    def test_diff_specs_auto_version(self):
        """Test diff_specs with auto version suggestion."""
        old = {"a": 1}
        new = {"a": 1, "b": 2}

        report = diff_specs(old, new, "1.0.0")

        # Should suggest minor bump
        assert report.new_version == "1.1.0"

    def test_diff_visual_specs(self):
        """Test diff_visual_specs function."""
        # Mock VisualSpec-like objects
        class MockSpec:
            def to_dict(self):
                return {"id": "test"}

        old = MockSpec()
        new = MockSpec()

        report = diff_visual_specs(old, new)

        assert isinstance(report, DiffReport)

    def test_diff_ui_specs(self):
        """Test diff_ui_specs function."""
        class MockUISpec:
            def to_dict(self):
                return {"pages": {}}

        old = MockUISpec()
        new = MockUISpec()

        report = diff_ui_specs(old, new, "1.0.0")

        assert isinstance(report, DiffReport)


class TestEdgeCases:
    """Edge case tests."""

    def test_diff_none_values(self):
        """Test diffing with None values."""
        old = {"key": None}
        new = {"key": "value"}

        differ = SpecDiffer()
        report = differ.diff(old, new)

        assert report.total_changes > 0

    def test_diff_both_none(self):
        """Test diffing when both are None."""
        differ = SpecDiffer()
        report = differ.diff(None, None)

        assert report.total_changes == 0

    def test_diff_empty_dicts(self):
        """Test diffing empty dicts."""
        differ = SpecDiffer()
        report = differ.diff({}, {})

        assert report.total_changes == 0

    def test_diff_deep_nesting(self):
        """Test diffing deeply nested structures."""
        old = {"a": {"b": {"c": {"d": {"e": "old"}}}}}
        new = {"a": {"b": {"c": {"d": {"e": "new"}}}}}

        differ = SpecDiffer()
        report = differ.diff(old, new)

        assert report.modified_count == 1
        assert "a.b.c.d.e" in report.changes[0].path

    def test_diff_large_spec(self):
        """Test diffing large spec."""
        old = {f"key{i}": i for i in range(100)}
        new = {f"key{i}": i + 1 for i in range(100)}

        differ = SpecDiffer()
        report = differ.diff(old, new)

        assert report.modified_count == 100

    def test_changelog_special_characters(self):
        """Test changelog with special characters."""
        changes = [
            Change(
                "path",
                ChangeType.ADDED,
                ChangeCategory.OTHER,
                description="Added <html> & \"special\" chars",
            )
        ]

        report = DiffReport("1.0.0", "1.1.0", changes=changes)
        changelog = generate_changelog(report)

        # Should contain the text (may be escaped)
        assert "special" in changelog

    def test_json_changelog_serializable(self):
        """Test JSON changelog is fully serializable."""
        changes = [
            Change("path", ChangeType.MODIFIED, ChangeCategory.OTHER,
                   old_value={"nested": [1, 2, 3]},
                   new_value={"nested": [1, 2, 4]}),
        ]

        report = DiffReport("1.0.0", "1.0.1", changes=changes)
        json_str = generate_json_changelog(report)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
