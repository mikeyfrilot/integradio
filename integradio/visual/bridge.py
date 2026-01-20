"""
Test Bridge - Extract visual assertions from test files.

This module parses pytest test files to:
1. Find component references (what components exist)
2. Extract assertions about component behavior
3. Link test coverage to visual specs
4. Generate visual specs from test patterns

This creates a bidirectional link: tests validate behavior,
visual specs validate appearance, both describe the same components.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .spec import VisualSpec, PageSpec, UISpec


# =============================================================================
# AST Extraction Types
# =============================================================================

@dataclass
class ComponentReference:
    """A reference to a component found in test code."""
    component_id: str
    component_type: str | None = None
    file_path: str = ""
    line_number: int = 0
    context: str = ""  # The test function/class name


@dataclass
class TestAssertion:
    """An assertion about a component from a test."""
    component_id: str
    assertion_type: str  # "visibility", "value", "enabled", "label", etc.
    expected_value: Any = None
    test_name: str = ""
    file_path: str = ""
    line_number: int = 0


@dataclass
class DataFlowEdge:
    """A data flow relationship found in tests."""
    source_id: str
    target_id: str
    event_type: str  # "click", "change", "submit", etc.
    test_name: str = ""


@dataclass
class TestExtraction:
    """Complete extraction from a test file."""
    file_path: str
    components: list[ComponentReference] = field(default_factory=list)
    assertions: list[TestAssertion] = field(default_factory=list)
    data_flows: list[DataFlowEdge] = field(default_factory=list)
    test_functions: list[str] = field(default_factory=list)


# =============================================================================
# Test File Parser
# =============================================================================

class TestFileParser:
    """
    Parse pytest files to extract component information.

    Handles common patterns:
    - gr.Textbox(elem_id="search-input")
    - semantic(gr.Button("Search"), intent="trigger search")
    - assert component.value == expected
    - button.click(fn=handler, inputs=inp, outputs=out)
    """

    # Patterns for finding components
    GRADIO_COMPONENT_PATTERN = re.compile(
        r'gr\.(\w+)\s*\([^)]*elem_id\s*=\s*["\']([^"\']+)["\']'
    )
    SEMANTIC_PATTERN = re.compile(
        r'semantic\s*\(\s*gr\.(\w+)[^)]*,\s*intent\s*=\s*["\']([^"\']+)["\']'
    )
    ELEM_ID_PATTERN = re.compile(
        r'elem_id\s*=\s*["\']([^"\']+)["\']'
    )
    LABEL_PATTERN = re.compile(
        r'label\s*=\s*["\']([^"\']+)["\']'
    )

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self.extraction = TestExtraction(file_path=str(file_path))

    def parse(self) -> TestExtraction:
        """Parse the test file and extract information."""
        content = self.file_path.read_text(encoding="utf-8")

        # Parse AST
        try:
            tree = ast.parse(content)
            self._walk_ast(tree)
        except SyntaxError:
            # Fall back to regex-only parsing
            pass

        # Regex-based extraction (catches things AST misses)
        self._extract_with_regex(content)

        return self.extraction

    def _walk_ast(self, tree: ast.AST) -> None:
        """Walk the AST to find test functions and assertions."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith("test_"):
                    self.extraction.test_functions.append(node.name)
                    self._extract_from_function(node)

            elif isinstance(node, ast.ClassDef):
                if node.name.startswith("Test"):
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name.startswith("test_"):
                            self.extraction.test_functions.append(f"{node.name}.{item.name}")
                            self._extract_from_function(item, class_name=node.name)

    def _extract_from_function(
        self,
        func: ast.FunctionDef,
        class_name: str = "",
    ) -> None:
        """Extract component references and assertions from a function."""
        context = f"{class_name}.{func.name}" if class_name else func.name

        for node in ast.walk(func):
            # Find assertions
            if isinstance(node, ast.Assert):
                self._extract_assertion(node, context)

            # Find method calls (component.click(), etc.)
            if isinstance(node, ast.Call):
                self._extract_call(node, context)

    def _extract_assertion(self, node: ast.Assert, context: str) -> None:
        """Extract assertion information."""
        # Handle: assert component.value == expected
        if isinstance(node.test, ast.Compare):
            left = node.test.left

            # Check for attribute access (component.value)
            if isinstance(left, ast.Attribute):
                attr_name = left.attr
                if isinstance(left.value, ast.Name):
                    var_name = left.value.id

                    # Try to get expected value
                    expected = None
                    if node.test.comparators:
                        expected = self._get_literal_value(node.test.comparators[0])

                    self.extraction.assertions.append(TestAssertion(
                        component_id=var_name,  # Variable name, will map later
                        assertion_type=attr_name,
                        expected_value=expected,
                        test_name=context,
                        file_path=str(self.file_path),
                        line_number=node.lineno,
                    ))

    def _extract_call(self, node: ast.Call, context: str) -> None:
        """Extract method calls for data flow."""
        # Handle: button.click(fn=handler, inputs=inp, outputs=out)
        if isinstance(node.func, ast.Attribute):
            method_name = node.func.attr
            if method_name in ("click", "change", "submit", "input", "select"):
                if isinstance(node.func.value, ast.Name):
                    source_var = node.func.value.id

                    # Find outputs keyword
                    for kw in node.keywords:
                        if kw.arg == "outputs":
                            targets = self._get_output_names(kw.value)
                            for target in targets:
                                self.extraction.data_flows.append(DataFlowEdge(
                                    source_id=source_var,
                                    target_id=target,
                                    event_type=method_name,
                                    test_name=context,
                                ))

    def _get_literal_value(self, node: ast.AST) -> Any:
        """Extract literal value from AST node."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Str):  # Python 3.7 compat
            return node.s
        if isinstance(node, ast.Num):  # Python 3.7 compat
            return node.n
        if isinstance(node, ast.NameConstant):  # True/False/None
            return node.value
        return None

    def _get_output_names(self, node: ast.AST) -> list[str]:
        """Extract output variable names from outputs kwarg."""
        names = []
        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.List):
            for elt in node.elts:
                if isinstance(elt, ast.Name):
                    names.append(elt.id)
        elif isinstance(node, ast.Tuple):
            for elt in node.elts:
                if isinstance(elt, ast.Name):
                    names.append(elt.id)
        return names

    def _extract_with_regex(self, content: str) -> None:
        """Extract component information using regex patterns."""
        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            # Find Gradio components with elem_id
            for match in self.GRADIO_COMPONENT_PATTERN.finditer(line):
                comp_type, elem_id = match.groups()
                self.extraction.components.append(ComponentReference(
                    component_id=elem_id,
                    component_type=comp_type,
                    file_path=str(self.file_path),
                    line_number=i,
                ))

            # Find semantic() wrappers
            for match in self.SEMANTIC_PATTERN.finditer(line):
                comp_type, intent = match.groups()
                # Use intent as component_id if no elem_id
                comp_id = intent.replace(" ", "-").lower()
                self.extraction.components.append(ComponentReference(
                    component_id=comp_id,
                    component_type=comp_type,
                    file_path=str(self.file_path),
                    line_number=i,
                    context=intent,
                ))


# =============================================================================
# Test Suite Scanner
# =============================================================================

class TestSuiteScanner:
    """
    Scan an entire test directory for component references.

    Builds a complete picture of what components are tested.
    """

    def __init__(self, test_dir: str | Path):
        self.test_dir = Path(test_dir)
        self.extractions: list[TestExtraction] = []

    def scan(self) -> list[TestExtraction]:
        """Scan all test files."""
        test_files = list(self.test_dir.glob("**/test_*.py"))
        test_files.extend(self.test_dir.glob("**/*_test.py"))

        for test_file in test_files:
            parser = TestFileParser(test_file)
            self.extractions.append(parser.parse())

        return self.extractions

    def get_all_components(self) -> list[ComponentReference]:
        """Get all component references across all test files."""
        components = []
        for extraction in self.extractions:
            components.extend(extraction.components)
        return components

    def get_all_assertions(self) -> list[TestAssertion]:
        """Get all assertions across all test files."""
        assertions = []
        for extraction in self.extractions:
            assertions.extend(extraction.assertions)
        return assertions

    def get_coverage_map(self) -> dict[str, list[str]]:
        """
        Get mapping of component_id -> list of test names.

        Shows which components have test coverage.
        """
        coverage: dict[str, list[str]] = {}

        for extraction in self.extractions:
            for assertion in extraction.assertions:
                if assertion.component_id not in coverage:
                    coverage[assertion.component_id] = []
                if assertion.test_name not in coverage[assertion.component_id]:
                    coverage[assertion.component_id].append(assertion.test_name)

        return coverage


# =============================================================================
# Spec Generator from Tests
# =============================================================================

class SpecGenerator:
    """
    Generate visual specs from test extractions.

    Creates placeholder VisualSpecs for all components found in tests,
    which can then be filled in with actual visual properties.
    """

    def __init__(self, extractions: list[TestExtraction]):
        self.extractions = extractions

    def generate_page_spec(
        self,
        page_name: str,
        route: str,
    ) -> PageSpec:
        """Generate a PageSpec with placeholder components."""
        page = PageSpec(name=page_name, route=route)

        # Deduplicate components
        seen: set[str] = set()
        for extraction in self.extractions:
            for comp in extraction.components:
                if comp.component_id not in seen:
                    seen.add(comp.component_id)

                    spec = VisualSpec(
                        component_id=comp.component_id,
                        component_type=comp.component_type or "Unknown",
                        test_file=comp.file_path,
                        test_line=comp.line_number,
                    )
                    page.add_component(spec)

        return page

    def generate_ui_spec(self, name: str = "From Tests") -> UISpec:
        """Generate a complete UISpec from test extractions."""
        spec = UISpec(name=name)

        # Group components by test file (each file = a page)
        for extraction in self.extractions:
            if extraction.components:
                # Use test file name as page name
                page_name = Path(extraction.file_path).stem
                page = PageSpec(
                    name=page_name,
                    route=f"/{page_name}",
                )

                for comp in extraction.components:
                    visual = VisualSpec(
                        component_id=comp.component_id,
                        component_type=comp.component_type or "Unknown",
                        test_file=extraction.file_path,
                        test_line=comp.line_number,
                    )
                    page.add_component(visual)

                spec.add_page(page)

        return spec


# =============================================================================
# Test-Spec Linker
# =============================================================================

@dataclass
class LinkReport:
    """Report of test-to-spec linking."""
    total_test_components: int = 0
    total_spec_components: int = 0
    linked: int = 0
    unlinked_in_tests: list[str] = field(default_factory=list)
    unlinked_in_spec: list[str] = field(default_factory=list)

    @property
    def coverage_percentage(self) -> float:
        """Percentage of test components that have visual specs."""
        if self.total_test_components == 0:
            return 100.0
        return (self.linked / self.total_test_components) * 100


class TestSpecLinker:
    """
    Link test extractions to visual specifications.

    Identifies:
    - Components in tests without visual specs
    - Components in specs without test coverage
    - Complete coverage (both behavioral and visual)
    """

    def __init__(self, spec: UISpec, extractions: list[TestExtraction]):
        self.spec = spec
        self.extractions = extractions

    def link(self) -> LinkReport:
        """Perform linking and return report."""
        report = LinkReport()

        # Get all component IDs from tests
        test_ids: set[str] = set()
        for extraction in self.extractions:
            for comp in extraction.components:
                test_ids.add(comp.component_id)

        # Get all component IDs from specs
        spec_ids: set[str] = set()
        for page in self.spec.pages.values():
            for comp_id in page.components:
                spec_ids.add(comp_id)

        report.total_test_components = len(test_ids)
        report.total_spec_components = len(spec_ids)

        # Find linked components
        linked = test_ids & spec_ids
        report.linked = len(linked)

        # Find unlinked
        report.unlinked_in_tests = list(test_ids - spec_ids)
        report.unlinked_in_spec = list(spec_ids - test_ids)

        return report

    def add_missing_specs(self) -> int:
        """Add placeholder VisualSpecs for components only in tests."""
        report = self.link()
        added = 0

        # Find or create a default page
        if not self.spec.pages:
            default_page = PageSpec(name="Default", route="/")
            self.spec.add_page(default_page)

        default_page = list(self.spec.pages.values())[0]

        for comp_id in report.unlinked_in_tests:
            # Find component info from extractions
            comp_type = "Unknown"
            test_file = ""
            line = 0

            for extraction in self.extractions:
                for comp in extraction.components:
                    if comp.component_id == comp_id:
                        comp_type = comp.component_type or "Unknown"
                        test_file = comp.file_path
                        line = comp.line_number
                        break

            spec = VisualSpec(
                component_id=comp_id,
                component_type=comp_type,
                test_file=test_file,
                test_line=line,
            )
            default_page.add_component(spec)
            added += 1

        return added


# =============================================================================
# Convenience Functions
# =============================================================================

def parse_test_file(path: str | Path) -> TestExtraction:
    """Parse a single test file."""
    parser = TestFileParser(path)
    return parser.parse()


def scan_tests(test_dir: str | Path) -> list[TestExtraction]:
    """Scan a test directory."""
    scanner = TestSuiteScanner(test_dir)
    return scanner.scan()


def generate_spec_from_tests(test_dir: str | Path, name: str = "From Tests") -> UISpec:
    """Generate a UISpec from test files."""
    extractions = scan_tests(test_dir)
    generator = SpecGenerator(extractions)
    return generator.generate_ui_spec(name)


def link_tests_to_spec(
    spec: UISpec,
    test_dir: str | Path,
) -> LinkReport:
    """Link tests to a spec and return coverage report."""
    extractions = scan_tests(test_dir)
    linker = TestSpecLinker(spec, extractions)
    return linker.link()


def auto_fill_spec_from_tests(
    spec: UISpec,
    test_dir: str | Path,
) -> int:
    """Add missing visual specs from test components."""
    extractions = scan_tests(test_dir)
    linker = TestSpecLinker(spec, extractions)
    return linker.add_missing_specs()
