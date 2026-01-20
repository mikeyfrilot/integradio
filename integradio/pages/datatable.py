"""
Data Table Page - Interactive data grid with sorting/filtering.

Features:
- Sortable columns
- Column filtering
- Pagination
- Row selection
- Export functionality
- Inline editing (optional)
"""

from typing import Optional, Callable, Any
from dataclasses import dataclass, field

import gradio as gr

from ..components import semantic
from ..blocks import SemanticBlocks


@dataclass
class ColumnDef:
    """Definition of a table column."""
    key: str
    label: str
    sortable: bool = True
    filterable: bool = True
    width: Optional[str] = None
    type: str = "text"  # "text", "number", "date", "boolean", "link"


@dataclass
class DataTableConfig:
    """Configuration for data table."""
    title: str = "Data Table"
    columns: list[ColumnDef] = field(default_factory=lambda: [
        ColumnDef("id", "ID", type="number"),
        ColumnDef("name", "Name"),
        ColumnDef("email", "Email"),
        ColumnDef("status", "Status"),
        ColumnDef("created", "Created", type="date"),
    ])
    page_size: int = 10
    page_sizes: list[int] = field(default_factory=lambda: [10, 25, 50, 100])
    show_search: bool = True
    show_filters: bool = True
    show_export: bool = True
    allow_selection: bool = True
    allow_edit: bool = False
    show_row_numbers: bool = True


# Sample data for demonstration
SAMPLE_DATA = [
    {"id": 1, "name": "Alice Johnson", "email": "alice@example.com", "status": "Active", "created": "2024-01-15"},
    {"id": 2, "name": "Bob Smith", "email": "bob@example.com", "status": "Active", "created": "2024-01-20"},
    {"id": 3, "name": "Carol Williams", "email": "carol@example.com", "status": "Inactive", "created": "2024-02-01"},
    {"id": 4, "name": "David Brown", "email": "david@example.com", "status": "Pending", "created": "2024-02-10"},
    {"id": 5, "name": "Eva Martinez", "email": "eva@example.com", "status": "Active", "created": "2024-02-15"},
]


def create_data_table(
    config: Optional[DataTableConfig] = None,
    initial_data: Optional[list[dict]] = None,
    on_row_select: Optional[Callable] = None,
    on_edit: Optional[Callable] = None,
    on_export: Optional[Callable] = None,
) -> dict[str, Any]:
    """
    Create a data table with semantic-tracked components.

    Args:
        config: Table configuration
        initial_data: Initial data to display
        on_row_select: Row selection handler
        on_edit: Edit handler
        on_export: Export handler

    Returns:
        Dict of component references
    """
    config = config or DataTableConfig()
    data = initial_data or SAMPLE_DATA
    components = {}

    # Header
    components["title"] = semantic(
        gr.Markdown(f"# 📊 {config.title}"),
        intent="displays data table page title",
        tags=["header"],
    )

    # Toolbar
    with gr.Row():
        # Search
        if config.show_search:
            with gr.Column(scale=3):
                components["search"] = semantic(
                    gr.Textbox(
                        placeholder="Search all columns...",
                        label="",
                        show_label=False,
                        elem_id="table-search",
                    ),
                    intent="searches across all table data",
                    tags=["filter", "search"],
                )

        # Page size
        with gr.Column(scale=1):
            components["page_size"] = semantic(
                gr.Dropdown(
                    choices=[str(s) for s in config.page_sizes],
                    value=str(config.page_size),
                    label="Rows per page",
                ),
                intent="controls number of rows displayed per page",
                tags=["pagination", "config"],
            )

        # Export button
        if config.show_export:
            with gr.Column(scale=1):
                components["export_btn"] = semantic(
                    gr.Button("📥 Export", variant="secondary"),
                    intent="exports table data to file",
                    tags=["action", "export"],
                )

        # Add new row button
        if config.allow_edit:
            with gr.Column(scale=1):
                components["add_btn"] = semantic(
                    gr.Button("➕ Add Row", variant="primary"),
                    intent="adds new row to table",
                    tags=["action", "create"],
                )

    # Column Filters (collapsible)
    if config.show_filters:
        with gr.Accordion("🔍 Column Filters", open=False):
            with gr.Row():
                for col in config.columns[:4]:  # Show first 4 filterable columns
                    if col.filterable:
                        components[f"filter_{col.key}"] = semantic(
                            gr.Textbox(
                                label=col.label,
                                placeholder=f"Filter {col.label}...",
                                elem_id=f"filter-{col.key}",
                            ),
                            intent=f"filters table by {col.label} column",
                            tags=["filter", "column", col.key],
                        )

            components["apply_filters"] = semantic(
                gr.Button("Apply Filters", variant="secondary", size="sm"),
                intent="applies column filters to table",
                tags=["action", "filter"],
            )

            components["clear_filters"] = semantic(
                gr.Button("Clear Filters", variant="secondary", size="sm"),
                intent="clears all column filters",
                tags=["action", "filter", "reset"],
            )

    # Stats row
    components["stats"] = semantic(
        gr.Markdown(f"Showing **{len(data)}** records"),
        intent="displays count of visible table records",
        tags=["status", "count"],
    )

    # Main Data Table
    # Convert data to list of lists for DataFrame
    headers = [col.label for col in config.columns]
    if config.show_row_numbers:
        headers = ["#"] + headers

    table_data = []
    for i, row in enumerate(data):
        row_data = [row.get(col.key, "") for col in config.columns]
        if config.show_row_numbers:
            row_data = [i + 1] + row_data
        table_data.append(row_data)

    components["table"] = semantic(
        gr.Dataframe(
            value=table_data,
            headers=headers,
            interactive=config.allow_edit,
            wrap=True,
            elem_id="main-table",
        ),
        intent="displays interactive data grid",
        tags=["table", "data", "primary"],
    )

    # Pagination
    with gr.Row():
        components["first_btn"] = semantic(
            gr.Button("⏮️ First", size="sm", variant="secondary"),
            intent="navigates to first page of results",
            tags=["pagination", "navigation"],
        )

        components["prev_btn"] = semantic(
            gr.Button("◀️ Previous", size="sm", variant="secondary"),
            intent="navigates to previous page of results",
            tags=["pagination", "navigation"],
        )

        components["page_info"] = semantic(
            gr.Markdown("Page **1** of **1**"),
            intent="displays current page information",
            tags=["pagination", "status"],
        )

        components["next_btn"] = semantic(
            gr.Button("Next ▶️", size="sm", variant="secondary"),
            intent="navigates to next page of results",
            tags=["pagination", "navigation"],
        )

        components["last_btn"] = semantic(
            gr.Button("Last ⏭️", size="sm", variant="secondary"),
            intent="navigates to last page of results",
            tags=["pagination", "navigation"],
        )

    # Selected row details (if selection enabled)
    if config.allow_selection:
        gr.Markdown("---")
        components["selection_title"] = semantic(
            gr.Markdown("### 📋 Selected Row"),
            intent="introduces selected row details section",
            tags=["header", "section"],
        )

        components["selected_row"] = semantic(
            gr.JSON(
                label="Row Data",
                elem_id="selected-row-data",
            ),
            intent="displays details of selected table row",
            tags=["display", "selection", "details"],
        )

        with gr.Row():
            components["edit_selected"] = semantic(
                gr.Button("✏️ Edit", variant="secondary", size="sm"),
                intent="opens editor for selected row",
                tags=["action", "edit"],
            )

            components["delete_selected"] = semantic(
                gr.Button("🗑️ Delete", variant="stop", size="sm"),
                intent="deletes selected row from table",
                tags=["action", "delete", "destructive"],
            )

    # Wire up row selection
    def handle_select(evt: gr.SelectData, table_data):
        if evt is None:
            return {}
        row_idx = evt.index[0]
        if row_idx < len(data):
            return data[row_idx]
        return {}

    if config.allow_selection:
        components["table"].select(
            fn=handle_select if not on_row_select else on_row_select,
            inputs=[components["table"]],
            outputs=[components["selected_row"]],
        )

    # Wire up export
    if config.show_export and on_export:
        components["export_btn"].click(
            fn=on_export,
            inputs=[components["table"]],
        )

    return components


class DataTablePage:
    """
    Complete data table page with SemanticBlocks integration.

    Usage:
        page = DataTablePage(
            title="Users",
            columns=[ColumnDef(...), ...],
            initial_data=my_data,
        )
        page.launch()
    """

    def __init__(
        self,
        title: str = "Data Table",
        columns: Optional[list[ColumnDef]] = None,
        initial_data: Optional[list[dict]] = None,
        on_row_select: Optional[Callable] = None,
        on_export: Optional[Callable] = None,
        **config_kwargs,
    ):
        self.config = DataTableConfig(title=title, **config_kwargs)
        if columns:
            self.config.columns = columns
        self.initial_data = initial_data
        self.on_row_select = on_row_select
        self.on_export = on_export
        self.components: dict[str, Any] = {}
        self.blocks: Optional[SemanticBlocks] = None

    def build(self) -> SemanticBlocks:
        """Build the data table page."""
        self.blocks = SemanticBlocks(
            title=self.config.title,
            theme=gr.themes.Soft(),
        )

        with self.blocks:
            self.components = create_data_table(
                config=self.config,
                initial_data=self.initial_data,
                on_row_select=self.on_row_select,
                on_export=self.on_export,
            )

        return self.blocks

    def launch(self, **kwargs) -> None:
        """Build and launch the data table page."""
        if self.blocks is None:
            self.build()
        self.blocks.launch(**kwargs)

    @staticmethod
    def render(
        config: Optional[DataTableConfig] = None,
        initial_data: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        """Render data table into existing Blocks context."""
        return create_data_table(config=config, initial_data=initial_data)
