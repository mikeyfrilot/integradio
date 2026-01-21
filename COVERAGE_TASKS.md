# Coverage Tasks for 3 Parallel Agents

**Current: 71% (2072 tests) → Target: 95%+**

Each agent uses the same dev-brain. They work on different files so no conflicts.

---

## Copy-Paste Prompts for Each Agent

### AGENT 1 PROMPT:
```
Generate integration tests for the integradio events module to achieve 100% coverage.

Working directory: F:/AI/integradio-dev

Files to cover (in priority order):
1. integradio/events/websocket.py (16% → 100%) - 339 lines uncovered, CRITICAL
2. integradio/events/security.py (78% → 100%) - 50 lines
3. integradio/events/mesh.py (91% → 100%) - 14 lines
4. integradio/events/event.py (90% → 100%) - 11 lines
5. integradio/events/handlers.py (98% → 100%) - 3 lines
6. integradio/embedder.py (80% → 100%) - 43 lines

Rules:
1. Read each source file first to understand what needs testing
2. Create tests/test_websocket.py (new file)
3. Add to tests/test_events.py and tests/test_embedder.py (existing)
4. Run: python -m pytest tests/[file].py --cov=integradio.[module] --cov-report=term-missing
5. Write integration tests (minimal mocking, test real behavior)
6. Verify all tests pass before moving to next file
7. Check system with: powershell -ExecutionPolicy Bypass -File F:/AI/integradio-dev/monitor.ps1

Target: ~150-200 new tests
```

---

### AGENT 2 PROMPT:
```
Generate integration tests for the integradio inspector and agent modules to achieve 100% coverage.

Working directory: F:/AI/integradio-dev

Files to cover (in priority order):
1. integradio/inspector/panel.py (38% → 100%) - 61 lines uncovered, CRITICAL
2. integradio/agent/tools.py (65% → 100%) - 83 lines
3. integradio/inspector/core.py (63% → 100%) - 51 lines
4. integradio/inspector/search.py (64% → 100%) - 46 lines
5. integradio/agent/langchain.py (49% → 100%) - 42 lines
6. integradio/inspector/dataflow.py (76% → 100%) - 39 lines
7. integradio/inspector/tree.py (80% → 100%) - 30 lines
8. integradio/agent/mcp.py (90% → 100%) - 15 lines

Rules:
1. Read each source file first to understand what needs testing
2. Create tests/inspector/ directory with __init__.py
3. Create tests/agent/ directory with __init__.py
4. Create test files: test_panel.py, test_core.py, test_search.py, test_dataflow.py, test_tree.py, test_tools.py, test_langchain.py, test_mcp.py
5. Run: python -m pytest tests/[dir]/[file].py --cov=integradio.[module] --cov-report=term-missing
6. Write integration tests (minimal mocking, test real behavior)
7. Verify all tests pass before moving to next file
8. Check system with: powershell -ExecutionPolicy Bypass -File F:/AI/integradio-dev/monitor.ps1

Target: ~180-220 new tests
```

---

### AGENT 3 PROMPT:
```
Generate integration tests for the integradio pages and visual modules to achieve 100% coverage.

Working directory: F:/AI/integradio-dev

Files to cover (in priority order):
1. integradio/visual/spec.py (77% → 100%) - 100 lines uncovered, CRITICAL
2. integradio/visual/screenshot.py (85% → 100%) - 63 lines
3. integradio/pages/upload.py (68% → 100%) - 61 lines
4. integradio/visual/tokens.py (87% → 100%) - 38 lines
5. integradio/pages/chat.py (71% → 100%) - 30 lines
6. integradio/pages/utils.py (39% → 100%) - 22 lines
7. integradio/pages/datatable.py (81% → 100%) - 22 lines
8. integradio/pages/form.py (87% → 100%) - 19 lines
9. integradio/visual/library.py (95% → 100%) - 16 lines
10. integradio/pages/gallery.py (83% → 100%) - 15 lines
11. integradio/visual/diff.py (94% → 100%) - 13 lines
12. integradio/viz.py (87% → 100%) - 12 lines

Also finish these near-complete:
- integradio/visual/viewer.py (90% → 100%) - 25 lines
- integradio/pages/dashboard.py (91% → 100%) - 9 lines
- integradio/visual/bridge.py (97% → 100%) - 7 lines
- integradio/blocks.py (96% → 100%) - 5 lines

Rules:
1. Read each source file first to understand what needs testing
2. Create tests/visual/test_spec.py, test_screenshot.py, test_tokens.py (new files)
3. Add to tests/test_pages.py and tests/test_viz.py (existing)
4. Add to tests/visual/test_viewer.py, test_bridge.py, test_diff.py, test_library.py (existing)
5. Run: python -m pytest tests/[file].py --cov=integradio.[module] --cov-report=term-missing
6. Write integration tests (minimal mocking, test real behavior)
7. Verify all tests pass before moving to next file
8. Check system with: powershell -ExecutionPolicy Bypass -File F:/AI/integradio-dev/monitor.ps1

Target: ~200-250 new tests
```

---

## Quick Reference

### Run tests for a specific module:
```bash
cd F:/AI/integradio-dev
python -m pytest tests/[test_file].py --cov=integradio.[module] --cov-report=term-missing -v
```

### Run all tests with coverage:
```bash
python -m pytest --cov=integradio --cov-report=term -q
```

### Check system resources:
```bash
powershell -ExecutionPolicy Bypass -File F:/AI/integradio-dev/monitor.ps1
```

### Python 3.14 Notes:
- Coverage with numpy can have import issues - run tests in subprocess if needed
- Use `python -c "import subprocess; ..."` pattern if direct pytest fails

---

## Expected Results

| Metric | Current | After All 3 Agents |
|--------|---------|-------------------|
| Tests | 2072 | ~2600-2700 |
| Coverage | 71% | 95%+ |

Each agent should complete in ~30-45 minutes.
