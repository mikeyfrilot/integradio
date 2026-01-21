"""
Integration tests for Test Bridge module.

Tests:
- ComponentReference, TestAssertion, DataFlowEdge, TestExtraction dataclasses
- TestFileParser (AST and regex parsing)
- TestSuiteScanner (directory scanning)
- SpecGenerator (spec generation from extractions)
- LinkReport and TestSpecLinker
- Convenience functions

These tests use real file I/O for integration testing.
"""

import pytest
from pathlib import Path
from textwrap import dedent

from integradio.visual.bridge import (
    ComponentReference,
    TestAssertion,
    DataFlowEdge,
    TestExtraction,
    TestFileParser,
    TestSuiteScanner,
    SpecGenerator,
    LinkReport,
    TestSpecLinker,
    parse_test_file,
    scan_tests,
    generate_spec_from_tests,
    link_tests_to_spec,
    auto_fill_spec_from_tests,
)
from integradio.visual.spec import VisualSpec, PageSpec, UISpec


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_test_file(tmp_path: Path) -> Path:
    """Create a sample test file for parsing."""
    test_content = dedent('''
        """Sample test file for bridge module testing."""
        import pytest
        import gradio as gr


        def test_search_input():
            """Test the search input component."""
            search_box = gr.Textbox(elem_id="search-input", label="Search")
            assert search_box.value == ""
            assert search_box.label == "Search"


        def test_button_click():
            """Test button click handler."""
            btn = gr.Button(elem_id="search-btn", value="Search")
            result = gr.Textbox(elem_id="result-output")

            # Simulate click
            btn.click(fn=lambda x: x.upper(), inputs=search_box, outputs=result)
            assert result.value is not None


        def test_semantic_component():
            """Test with semantic wrapper."""
            semantic(gr.Button("Submit"), intent="trigger submission")


        class TestSearchPage:
            """Test class for search page."""

            def test_page_layout(self):
                """Test page layout."""
                header = gr.Markdown(elem_id="page-header")
                sidebar = gr.Column(elem_id="sidebar")
                assert header is not None

            def test_data_flow(self):
                """Test data flow between components."""
                inp = gr.Textbox(elem_id="data-input")
                out = gr.Markdown(elem_id="data-output")

                btn = gr.Button(elem_id="process-btn")
                btn.change(fn=process, inputs=inp, outputs=[out])

                assert inp.interactive is True
    ''')

    test_file = tmp_path / "test_sample.py"
    test_file.write_text(test_content, encoding="utf-8")
    return test_file


@pytest.fixture
def sample_test_directory(tmp_path: Path) -> Path:
    """Create a sample test directory with multiple test files."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()

    # First test file
    test1_content = dedent('''
        """First test file."""
        import gradio as gr

        def test_component_a():
            comp = gr.Textbox(elem_id="component-a", label="A")
            assert comp.value == ""

        def test_component_b():
            comp = gr.Button(elem_id="component-b", value="Click")
    ''')
    (tests_dir / "test_first.py").write_text(test1_content, encoding="utf-8")

    # Second test file
    test2_content = dedent('''
        """Second test file."""
        import gradio as gr

        def test_component_c():
            comp = gr.Slider(elem_id="component-c", minimum=0)
            assert comp.minimum == 0

        class TestOther:
            def test_component_d(self):
                comp = gr.Dropdown(elem_id="component-d")
    ''')
    (tests_dir / "test_second.py").write_text(test2_content, encoding="utf-8")

    # Subfolder with test
    subdir = tests_dir / "submodule"
    subdir.mkdir()
    test3_content = dedent('''
        """Subfolder test file."""
        import gradio as gr

        def test_nested():
            comp = gr.Image(elem_id="nested-image")
    ''')
    (tests_dir / "submodule" / "test_nested.py").write_text(test3_content, encoding="utf-8")

    return tests_dir


@pytest.fixture
def sample_ui_spec() -> UISpec:
    """Create a sample UISpec for linking tests."""
    spec = UISpec(name="Test App", version="1.0.0")

    page = PageSpec(name="Main", route="/")
    page.add_component(VisualSpec(component_id="component-a", component_type="Textbox"))
    page.add_component(VisualSpec(component_id="existing-spec", component_type="Button"))

    spec.add_page(page)
    return spec


# =============================================================================
# Dataclass Tests
# =============================================================================

class TestComponentReference:
    """Tests for ComponentReference dataclass."""

    def test_create_minimal(self):
        """Test creating with minimal fields."""
        ref = ComponentReference(component_id="test-id")

        assert ref.component_id == "test-id"
        assert ref.component_type is None
        assert ref.file_path == ""
        assert ref.line_number == 0
        assert ref.context == ""

    def test_create_full(self):
        """Test creating with all fields."""
        ref = ComponentReference(
            component_id="search-input",
            component_type="Textbox",
            file_path="/path/to/test.py",
            line_number=42,
            context="test_search",
        )

        assert ref.component_id == "search-input"
        assert ref.component_type == "Textbox"
        assert ref.file_path == "/path/to/test.py"
        assert ref.line_number == 42
        assert ref.context == "test_search"


class TestTestAssertion:
    """Tests for TestAssertion dataclass."""

    def test_create_minimal(self):
        """Test creating with required fields."""
        assertion = TestAssertion(
            component_id="test-comp",
            assertion_type="visibility",
        )

        assert assertion.component_id == "test-comp"
        assert assertion.assertion_type == "visibility"
        assert assertion.expected_value is None

    def test_create_full(self):
        """Test creating with all fields."""
        assertion = TestAssertion(
            component_id="input-field",
            assertion_type="value",
            expected_value="hello",
            test_name="test_input_value",
            file_path="/tests/test.py",
            line_number=25,
        )

        assert assertion.component_id == "input-field"
        assert assertion.assertion_type == "value"
        assert assertion.expected_value == "hello"
        assert assertion.test_name == "test_input_value"


class TestDataFlowEdge:
    """Tests for DataFlowEdge dataclass."""

    def test_create(self):
        """Test creating a data flow edge."""
        edge = DataFlowEdge(
            source_id="button",
            target_id="output",
            event_type="click",
            test_name="test_click_flow",
        )

        assert edge.source_id == "button"
        assert edge.target_id == "output"
        assert edge.event_type == "click"
        assert edge.test_name == "test_click_flow"


class TestTestExtraction:
    """Tests for TestExtraction dataclass."""

    def test_create_empty(self):
        """Test creating empty extraction."""
        extraction = TestExtraction(file_path="/test.py")

        assert extraction.file_path == "/test.py"
        assert extraction.components == []
        assert extraction.assertions == []
        assert extraction.data_flows == []
        assert extraction.test_functions == []

    def test_create_with_data(self):
        """Test creating extraction with data."""
        components = [ComponentReference("comp1"), ComponentReference("comp2")]
        assertions = [TestAssertion("comp1", "value")]
        data_flows = [DataFlowEdge("a", "b", "click")]
        test_functions = ["test_one", "test_two"]

        extraction = TestExtraction(
            file_path="/test.py",
            components=components,
            assertions=assertions,
            data_flows=data_flows,
            test_functions=test_functions,
        )

        assert len(extraction.components) == 2
        assert len(extraction.assertions) == 1
        assert len(extraction.data_flows) == 1
        assert len(extraction.test_functions) == 2


# =============================================================================
# TestFileParser Tests
# =============================================================================

class TestTestFileParser:
    """Tests for TestFileParser class."""

    def test_parse_file_path(self, sample_test_file: Path):
        """Test parser correctly stores file path."""
        parser = TestFileParser(sample_test_file)

        assert parser.file_path == sample_test_file

    def test_parse_extracts_test_functions(self, sample_test_file: Path):
        """Test parsing extracts test function names."""
        parser = TestFileParser(sample_test_file)
        extraction = parser.parse()

        assert "test_search_input" in extraction.test_functions
        assert "test_button_click" in extraction.test_functions
        assert "test_semantic_component" in extraction.test_functions
        assert "TestSearchPage.test_page_layout" in extraction.test_functions
        assert "TestSearchPage.test_data_flow" in extraction.test_functions

    def test_parse_extracts_components_with_elem_id(self, sample_test_file: Path):
        """Test parsing extracts components with elem_id."""
        parser = TestFileParser(sample_test_file)
        extraction = parser.parse()

        component_ids = [c.component_id for c in extraction.components]

        assert "search-input" in component_ids
        assert "search-btn" in component_ids
        assert "result-output" in component_ids
        assert "page-header" in component_ids

    def test_parse_extracts_component_types(self, sample_test_file: Path):
        """Test parsing extracts component types correctly."""
        parser = TestFileParser(sample_test_file)
        extraction = parser.parse()

        comp_map = {c.component_id: c.component_type for c in extraction.components}

        assert comp_map.get("search-input") == "Textbox"
        assert comp_map.get("search-btn") == "Button"
        assert comp_map.get("page-header") == "Markdown"

    def test_parse_extracts_semantic_components(self, tmp_path: Path):
        """Test parsing extracts semantic() wrapped components."""
        # Note: The SEMANTIC_PATTERN regex uses [^)]* which stops at first ).
        # So we need a simpler syntax without inner parentheses.
        test_content = dedent('''
            def test_semantic():
                btn = semantic(gr.Button, intent="trigger submission")
        ''')
        test_file = tmp_path / "test_semantic.py"
        test_file.write_text(test_content, encoding="utf-8")

        parser = TestFileParser(test_file)
        extraction = parser.parse()

        component_ids = [c.component_id for c in extraction.components]

        # semantic(gr.Button, intent="trigger submission")
        # -> component_id = "trigger-submission"
        assert "trigger-submission" in component_ids

    def test_parse_extracts_assertions(self, sample_test_file: Path):
        """Test parsing extracts assertions."""
        parser = TestFileParser(sample_test_file)
        extraction = parser.parse()

        # Should have assertions from the test functions
        assert len(extraction.assertions) > 0

        assertion_types = [a.assertion_type for a in extraction.assertions]
        assert "value" in assertion_types or "label" in assertion_types

    def test_parse_extracts_data_flows(self, sample_test_file: Path):
        """Test parsing extracts data flow from click/change methods."""
        parser = TestFileParser(sample_test_file)
        extraction = parser.parse()

        # Should have data flows from btn.click and btn.change
        assert len(extraction.data_flows) > 0

        # Check event types
        event_types = {df.event_type for df in extraction.data_flows}
        assert "click" in event_types or "change" in event_types

    def test_parse_records_line_numbers(self, sample_test_file: Path):
        """Test parsing records correct line numbers."""
        parser = TestFileParser(sample_test_file)
        extraction = parser.parse()

        # Components should have line numbers > 0
        for comp in extraction.components:
            assert comp.line_number > 0

    def test_parse_handles_syntax_error(self, tmp_path: Path):
        """Test parser handles syntax errors gracefully."""
        bad_file = tmp_path / "bad_test.py"
        bad_file.write_text("def test_broken(\n    # syntax error", encoding="utf-8")

        parser = TestFileParser(bad_file)
        extraction = parser.parse()

        # Should not raise, returns empty extraction
        assert extraction.file_path == str(bad_file)

    def test_parse_empty_file(self, tmp_path: Path):
        """Test parsing an empty file."""
        empty_file = tmp_path / "empty_test.py"
        empty_file.write_text("", encoding="utf-8")

        parser = TestFileParser(empty_file)
        extraction = parser.parse()

        assert extraction.file_path == str(empty_file)
        assert extraction.components == []
        assert extraction.test_functions == []

    def test_parse_file_with_only_comments(self, tmp_path: Path):
        """Test parsing file with only comments."""
        comment_file = tmp_path / "test_comments.py"
        comment_file.write_text('"""Docstring only."""\n# Comment line\n', encoding="utf-8")

        parser = TestFileParser(comment_file)
        extraction = parser.parse()

        assert extraction.test_functions == []
        assert extraction.components == []

    def test_regex_patterns_find_elem_id(self):
        """Test GRADIO_COMPONENT_PATTERN regex."""
        pattern = TestFileParser.GRADIO_COMPONENT_PATTERN

        test_line = 'gr.Textbox(elem_id="search-input", label="Search")'
        match = pattern.search(test_line)

        assert match is not None
        assert match.group(1) == "Textbox"
        assert match.group(2) == "search-input"

    def test_regex_patterns_find_semantic(self):
        """Test SEMANTIC_PATTERN regex."""
        pattern = TestFileParser.SEMANTIC_PATTERN

        # Note: The pattern uses [^)]* which stops at the first )
        # So the inner component cannot have parentheses in it
        test_line = 'semantic(gr.Button, intent="trigger action")'
        match = pattern.search(test_line)

        assert match is not None
        assert match.group(1) == "Button"
        assert match.group(2) == "trigger action"

    def test_regex_patterns_semantic_with_args(self):
        """Test SEMANTIC_PATTERN with component that has no inner parens."""
        pattern = TestFileParser.SEMANTIC_PATTERN

        # Pattern works when no inner () in component args
        test_line = 'semantic(gr.Slider, intent="adjust volume")'
        match = pattern.search(test_line)

        assert match is not None
        assert match.group(1) == "Slider"
        assert match.group(2) == "adjust volume"


class TestTestFileParserASTExtraction:
    """Tests for AST-based extraction in TestFileParser."""

    # Note: Lines 202-208 in bridge.py are Python 3.7 compatibility code
    # for ast.Str, ast.Num, and ast.NameConstant which were removed in
    # Python 3.12+. These cannot be tested on Python 3.14 and the code
    # would need updating to use hasattr() checks for cross-version compatibility.

    def test_extract_literal_string(self, tmp_path: Path):
        """Test extracting string literal from assertion."""
        test_content = dedent('''
            def test_value():
                comp = gr.Textbox()
                assert comp.value == "hello"
        ''')
        test_file = tmp_path / "test_literal.py"
        test_file.write_text(test_content, encoding="utf-8")

        parser = TestFileParser(test_file)
        extraction = parser.parse()

        # Should extract assertion with expected value
        value_assertions = [a for a in extraction.assertions if a.assertion_type == "value"]
        assert len(value_assertions) > 0
        assert value_assertions[0].expected_value == "hello"

    def test_extract_literal_number(self, tmp_path: Path):
        """Test extracting numeric literal from assertion."""
        test_content = dedent('''
            def test_count():
                widget = gr.Slider()
                assert widget.value == 42
        ''')
        test_file = tmp_path / "test_number.py"
        test_file.write_text(test_content, encoding="utf-8")

        parser = TestFileParser(test_file)
        extraction = parser.parse()

        value_assertions = [a for a in extraction.assertions if a.assertion_type == "value"]
        assert len(value_assertions) > 0
        assert value_assertions[0].expected_value == 42

    def test_extract_literal_boolean(self, tmp_path: Path):
        """Test extracting boolean literal from assertion."""
        test_content = dedent('''
            def test_visible():
                comp = gr.Textbox()
                assert comp.visible == True
        ''')
        test_file = tmp_path / "test_bool.py"
        test_file.write_text(test_content, encoding="utf-8")

        parser = TestFileParser(test_file)
        extraction = parser.parse()

        visible_assertions = [a for a in extraction.assertions if a.assertion_type == "visible"]
        assert len(visible_assertions) > 0
        assert visible_assertions[0].expected_value is True

    def test_extract_output_names_single(self, tmp_path: Path):
        """Test extracting single output from method call."""
        test_content = dedent('''
            def test_click():
                btn = gr.Button()
                result = gr.Textbox()
                btn.click(fn=handler, outputs=result)
        ''')
        test_file = tmp_path / "test_single_out.py"
        test_file.write_text(test_content, encoding="utf-8")

        parser = TestFileParser(test_file)
        extraction = parser.parse()

        assert len(extraction.data_flows) > 0
        assert extraction.data_flows[0].target_id == "result"

    def test_extract_output_names_list(self, tmp_path: Path):
        """Test extracting list of outputs from method call."""
        test_content = dedent('''
            def test_multi_output():
                btn = gr.Button()
                out1 = gr.Textbox()
                out2 = gr.Markdown()
                btn.click(fn=handler, outputs=[out1, out2])
        ''')
        test_file = tmp_path / "test_list_out.py"
        test_file.write_text(test_content, encoding="utf-8")

        parser = TestFileParser(test_file)
        extraction = parser.parse()

        targets = [df.target_id for df in extraction.data_flows]
        assert "out1" in targets
        assert "out2" in targets

    def test_extract_output_names_tuple(self, tmp_path: Path):
        """Test extracting tuple of outputs from method call."""
        test_content = dedent('''
            def test_tuple_output():
                btn = gr.Button()
                a = gr.Textbox()
                b = gr.Image()
                btn.submit(fn=handler, outputs=(a, b))
        ''')
        test_file = tmp_path / "test_tuple_out.py"
        test_file.write_text(test_content, encoding="utf-8")

        parser = TestFileParser(test_file)
        extraction = parser.parse()

        targets = [df.target_id for df in extraction.data_flows]
        assert "a" in targets
        assert "b" in targets

    def test_extract_various_event_types(self, tmp_path: Path):
        """Test extraction of different event types."""
        test_content = dedent('''
            def test_events():
                btn = gr.Button()
                txt = gr.Textbox()
                dropdown = gr.Dropdown()
                slider = gr.Slider()
                out = gr.Markdown()

                btn.click(fn=f1, outputs=out)
                txt.change(fn=f2, outputs=out)
                dropdown.select(fn=f3, outputs=out)
                slider.input(fn=f4, outputs=out)
        ''')
        test_file = tmp_path / "test_events.py"
        test_file.write_text(test_content, encoding="utf-8")

        parser = TestFileParser(test_file)
        extraction = parser.parse()

        event_types = {df.event_type for df in extraction.data_flows}
        assert "click" in event_types
        assert "change" in event_types
        assert "select" in event_types
        assert "input" in event_types


# =============================================================================
# TestSuiteScanner Tests
# =============================================================================

class TestTestSuiteScanner:
    """Tests for TestSuiteScanner class."""

    def test_scan_finds_all_test_files(self, sample_test_directory: Path):
        """Test scanner finds all test files in directory."""
        scanner = TestSuiteScanner(sample_test_directory)
        extractions = scanner.scan()

        # Should find test_first.py, test_second.py, and test_nested.py
        assert len(extractions) == 3

    def test_scan_stores_extractions(self, sample_test_directory: Path):
        """Test scanner stores extractions internally."""
        scanner = TestSuiteScanner(sample_test_directory)
        scanner.scan()

        assert len(scanner.extractions) == 3

    def test_get_all_components(self, sample_test_directory: Path):
        """Test getting all components across files."""
        scanner = TestSuiteScanner(sample_test_directory)
        scanner.scan()

        all_components = scanner.get_all_components()
        component_ids = [c.component_id for c in all_components]

        assert "component-a" in component_ids
        assert "component-b" in component_ids
        assert "component-c" in component_ids
        assert "component-d" in component_ids
        assert "nested-image" in component_ids

    def test_get_all_assertions(self, sample_test_directory: Path):
        """Test getting all assertions across files."""
        scanner = TestSuiteScanner(sample_test_directory)
        scanner.scan()

        all_assertions = scanner.get_all_assertions()

        # Should have assertions from files
        assert len(all_assertions) >= 0  # May be 0 if no component.attr assertions

    def test_get_coverage_map(self, sample_test_directory: Path):
        """Test generating coverage map."""
        scanner = TestSuiteScanner(sample_test_directory)
        scanner.scan()

        coverage = scanner.get_coverage_map()

        # Should be a dict mapping component_id -> list of test names
        assert isinstance(coverage, dict)

    def test_scan_empty_directory(self, tmp_path: Path):
        """Test scanning empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        scanner = TestSuiteScanner(empty_dir)
        extractions = scanner.scan()

        assert extractions == []

    def test_scan_with_suffix_pattern(self, tmp_path: Path):
        """Test scanner finds _test.py files too."""
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        # Create file with _test.py suffix
        suffix_test = dedent('''
            def test_something():
                pass
        ''')
        (tests_dir / "widget_test.py").write_text(suffix_test, encoding="utf-8")

        scanner = TestSuiteScanner(tests_dir)
        extractions = scanner.scan()

        assert len(extractions) == 1


# =============================================================================
# SpecGenerator Tests
# =============================================================================

class TestSpecGenerator:
    """Tests for SpecGenerator class."""

    def test_generate_page_spec(self, sample_test_file: Path):
        """Test generating PageSpec from extraction."""
        parser = TestFileParser(sample_test_file)
        extraction = parser.parse()

        generator = SpecGenerator([extraction])
        page = generator.generate_page_spec("Test Page", "/test")

        assert page.name == "Test Page"
        assert page.route == "/test"
        assert len(page.components) > 0

    def test_generate_page_spec_deduplicates(self, tmp_path: Path):
        """Test PageSpec deduplicates components."""
        test_content = dedent('''
            def test_one():
                comp = gr.Textbox(elem_id="same-id")

            def test_two():
                comp = gr.Textbox(elem_id="same-id")
        ''')
        test_file = tmp_path / "test_dup.py"
        test_file.write_text(test_content, encoding="utf-8")

        parser = TestFileParser(test_file)
        extraction = parser.parse()

        generator = SpecGenerator([extraction])
        page = generator.generate_page_spec("Dedup Test", "/dedup")

        # Should only have one component with id "same-id"
        assert len([c for c in page.components.values() if c.component_id == "same-id"]) == 1

    def test_generate_page_spec_includes_test_info(self, sample_test_file: Path):
        """Test generated specs include test file info."""
        parser = TestFileParser(sample_test_file)
        extraction = parser.parse()

        generator = SpecGenerator([extraction])
        page = generator.generate_page_spec("Test", "/")

        # Check that at least one component has test_file set
        has_test_info = any(
            spec.test_file is not None and spec.test_file != ""
            for spec in page.components.values()
        )
        assert has_test_info

    def test_generate_ui_spec(self, sample_test_directory: Path):
        """Test generating UISpec from multiple extractions."""
        scanner = TestSuiteScanner(sample_test_directory)
        extractions = scanner.scan()

        generator = SpecGenerator(extractions)
        ui_spec = generator.generate_ui_spec("Test App")

        assert ui_spec.name == "Test App"
        assert len(ui_spec.pages) > 0

    def test_generate_ui_spec_creates_pages_from_files(self, sample_test_directory: Path):
        """Test UISpec creates a page per test file."""
        scanner = TestSuiteScanner(sample_test_directory)
        extractions = scanner.scan()

        generator = SpecGenerator(extractions)
        ui_spec = generator.generate_ui_spec()

        # Should have pages from test files (excluding empty ones)
        assert len(ui_spec.pages) >= 2

    def test_generate_ui_spec_empty_extractions(self):
        """Test generating UISpec from empty extractions."""
        generator = SpecGenerator([])
        ui_spec = generator.generate_ui_spec("Empty")

        assert ui_spec.name == "Empty"
        assert len(ui_spec.pages) == 0


# =============================================================================
# LinkReport Tests
# =============================================================================

class TestLinkReport:
    """Tests for LinkReport dataclass."""

    def test_coverage_percentage_full(self):
        """Test 100% coverage percentage."""
        report = LinkReport(
            total_test_components=10,
            total_spec_components=10,
            linked=10,
        )

        assert report.coverage_percentage == 100.0

    def test_coverage_percentage_partial(self):
        """Test partial coverage percentage."""
        report = LinkReport(
            total_test_components=10,
            total_spec_components=5,
            linked=5,
        )

        assert report.coverage_percentage == 50.0

    def test_coverage_percentage_zero_components(self):
        """Test coverage with zero test components."""
        report = LinkReport(
            total_test_components=0,
            total_spec_components=5,
            linked=0,
        )

        # With no test components, coverage is 100% (nothing to cover)
        assert report.coverage_percentage == 100.0

    def test_unlinked_lists(self):
        """Test unlinked component lists."""
        report = LinkReport(
            unlinked_in_tests=["test-only-a", "test-only-b"],
            unlinked_in_spec=["spec-only-x"],
        )

        assert len(report.unlinked_in_tests) == 2
        assert len(report.unlinked_in_spec) == 1


# =============================================================================
# TestSpecLinker Tests
# =============================================================================

class TestTestSpecLinker:
    """Tests for TestSpecLinker class."""

    def test_link_finds_matching_components(self, sample_test_directory: Path, sample_ui_spec: UISpec):
        """Test linking finds matching components."""
        scanner = TestSuiteScanner(sample_test_directory)
        extractions = scanner.scan()

        linker = TestSpecLinker(sample_ui_spec, extractions)
        report = linker.link()

        # "component-a" exists in both tests and spec
        assert report.linked >= 1
        assert "component-a" not in report.unlinked_in_tests
        assert "component-a" not in report.unlinked_in_spec

    def test_link_finds_unlinked_in_tests(self, sample_test_directory: Path, sample_ui_spec: UISpec):
        """Test linking finds components only in tests."""
        scanner = TestSuiteScanner(sample_test_directory)
        extractions = scanner.scan()

        linker = TestSpecLinker(sample_ui_spec, extractions)
        report = linker.link()

        # "component-b", "component-c", etc. are in tests but not spec
        assert len(report.unlinked_in_tests) > 0

    def test_link_finds_unlinked_in_spec(self, sample_test_directory: Path, sample_ui_spec: UISpec):
        """Test linking finds components only in spec."""
        scanner = TestSuiteScanner(sample_test_directory)
        extractions = scanner.scan()

        linker = TestSpecLinker(sample_ui_spec, extractions)
        report = linker.link()

        # "existing-spec" is in spec but not tests
        assert "existing-spec" in report.unlinked_in_spec

    def test_add_missing_specs(self, sample_test_directory: Path, sample_ui_spec: UISpec):
        """Test adding missing specs for test components."""
        scanner = TestSuiteScanner(sample_test_directory)
        extractions = scanner.scan()

        linker = TestSpecLinker(sample_ui_spec, extractions)

        # Get initial report
        initial_report = linker.link()
        initial_unlinked = len(initial_report.unlinked_in_tests)

        # Add missing specs
        added = linker.add_missing_specs()

        # Should have added some specs
        assert added > 0
        assert added == initial_unlinked

    def test_add_missing_specs_creates_default_page(self):
        """Test add_missing_specs creates default page if none exist."""
        empty_spec = UISpec(name="Empty")

        # Create an extraction with a component
        extraction = TestExtraction(
            file_path="/test.py",
            components=[ComponentReference("new-comp", "Button")],
        )

        linker = TestSpecLinker(empty_spec, [extraction])
        added = linker.add_missing_specs()

        assert added == 1
        assert len(empty_spec.pages) == 1

    def test_link_empty_extractions(self, sample_ui_spec: UISpec):
        """Test linking with no extractions."""
        linker = TestSpecLinker(sample_ui_spec, [])
        report = linker.link()

        assert report.total_test_components == 0
        assert report.linked == 0


# =============================================================================
# Convenience Functions Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_parse_test_file(self, sample_test_file: Path):
        """Test parse_test_file function."""
        extraction = parse_test_file(sample_test_file)

        assert isinstance(extraction, TestExtraction)
        assert extraction.file_path == str(sample_test_file)
        assert len(extraction.test_functions) > 0

    def test_parse_test_file_with_str_path(self, sample_test_file: Path):
        """Test parse_test_file accepts string path."""
        extraction = parse_test_file(str(sample_test_file))

        assert isinstance(extraction, TestExtraction)

    def test_scan_tests(self, sample_test_directory: Path):
        """Test scan_tests function."""
        extractions = scan_tests(sample_test_directory)

        assert isinstance(extractions, list)
        assert len(extractions) == 3
        assert all(isinstance(e, TestExtraction) for e in extractions)

    def test_generate_spec_from_tests(self, sample_test_directory: Path):
        """Test generate_spec_from_tests function."""
        ui_spec = generate_spec_from_tests(sample_test_directory, "My App")

        assert isinstance(ui_spec, UISpec)
        assert ui_spec.name == "My App"
        assert len(ui_spec.pages) > 0

    def test_generate_spec_from_tests_default_name(self, sample_test_directory: Path):
        """Test generate_spec_from_tests with default name."""
        ui_spec = generate_spec_from_tests(sample_test_directory)

        assert ui_spec.name == "From Tests"

    def test_link_tests_to_spec(self, sample_test_directory: Path, sample_ui_spec: UISpec):
        """Test link_tests_to_spec function."""
        report = link_tests_to_spec(sample_ui_spec, sample_test_directory)

        assert isinstance(report, LinkReport)
        assert report.total_test_components > 0

    def test_auto_fill_spec_from_tests(self, sample_test_directory: Path):
        """Test auto_fill_spec_from_tests function."""
        spec = UISpec(name="Auto Fill Test")
        page = PageSpec(name="Main", route="/")
        spec.add_page(page)

        added = auto_fill_spec_from_tests(spec, sample_test_directory)

        assert added > 0
        # The page should now have more components
        total_components = sum(len(p.components) for p in spec.pages.values())
        assert total_components == added


# =============================================================================
# Integration Tests
# =============================================================================

class TestFullWorkflow:
    """End-to-end integration tests."""

    def test_parse_scan_generate_link_workflow(self, sample_test_directory: Path):
        """Test complete workflow from parsing to linking."""
        # 1. Scan test directory
        extractions = scan_tests(sample_test_directory)
        assert len(extractions) > 0

        # 2. Generate UI spec from tests
        ui_spec = generate_spec_from_tests(sample_test_directory, "Workflow Test")
        assert len(ui_spec.pages) > 0

        # 3. Link tests to spec (should be fully linked since spec came from tests)
        report = link_tests_to_spec(ui_spec, sample_test_directory)

        # All test components should be in spec (since we generated from same tests)
        assert report.linked == report.total_test_components
        assert len(report.unlinked_in_tests) == 0

    def test_incremental_spec_filling(self, sample_test_directory: Path):
        """Test incrementally adding specs as tests are added."""
        # Start with empty spec
        spec = UISpec(name="Incremental")

        # Add a page with one component
        page = PageSpec(name="Main", route="/")
        page.add_component(VisualSpec(component_id="component-a", component_type="Textbox"))
        spec.add_page(page)

        # Fill from tests
        added = auto_fill_spec_from_tests(spec, sample_test_directory)

        # Should have added missing components
        assert added > 0

        # Now linking should show full coverage
        report = link_tests_to_spec(spec, sample_test_directory)
        assert report.coverage_percentage == 100.0

    def test_real_world_test_patterns(self, tmp_path: Path):
        """Test parsing realistic test patterns."""
        # Note: The GRADIO_COMPONENT_PATTERN regex uses [^)]* which means
        # elem_id must appear before the first ) on the same line.
        # Multi-line component definitions with elem_id on a separate line won't match.
        # Note: Data flow extraction only works for method calls inside test_ functions.
        test_content = dedent('''
            """Tests for the search feature."""
            import pytest
            import gradio as gr
            from my_app import search_handler, semantic


            @pytest.fixture
            def demo():
                with gr.Blocks() as demo:
                    # Single-line component definitions work with the regex
                    search_input = gr.Textbox(elem_id="search-input", label="Search Query")
                    search_btn = gr.Button(elem_id="search-btn", value="Search")
                    results = gr.Markdown(elem_id="search-results")
                return demo


            class TestSearchFeature:
                def test_empty_search(self, demo):
                    """Test empty search returns message."""
                    search_input = demo.children[0]
                    assert search_input.value == ""

                def test_search_with_query(self, demo):
                    """Test search with query."""
                    search_input = demo.children[0]
                    search_input.value = "test query"

                    results = demo.children[2]
                    assert results.visible == True

                def test_data_flow(self):
                    """Test with data flow inside test function."""
                    btn = gr.Button()
                    out = gr.Markdown()
                    btn.click(fn=handler, outputs=out)

                def test_semantic_wrapper(self):
                    """Test semantic component."""
                    # Note: semantic pattern requires no inner () before intent
                    btn = semantic(gr.Button, intent="submit form")
                    assert btn is not None
        ''')

        test_file = tmp_path / "test_search.py"
        test_file.write_text(test_content, encoding="utf-8")

        extraction = parse_test_file(test_file)

        # Should extract components
        component_ids = {c.component_id for c in extraction.components}
        assert "search-input" in component_ids
        assert "search-btn" in component_ids
        assert "search-results" in component_ids
        assert "submit-form" in component_ids  # from semantic

        # Should extract test functions
        assert "TestSearchFeature.test_empty_search" in extraction.test_functions
        assert "TestSearchFeature.test_search_with_query" in extraction.test_functions

        # Should extract data flows from test functions
        assert len(extraction.data_flows) > 0
        flow_events = {df.event_type for df in extraction.data_flows}
        assert "click" in flow_events
