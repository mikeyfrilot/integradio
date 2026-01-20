"""
Analytics Page - Data visualization dashboard with charts.

Features:
- Multiple chart types (line, bar, pie, area)
- Date range picker
- Metric selectors
- Real-time updates
- Export reports
"""

from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import random

import gradio as gr

from ..components import semantic
from ..blocks import SemanticBlocks


@dataclass
class MetricDefinition:
    """Definition of an analytics metric."""
    key: str
    label: str
    icon: str = "📊"
    format: str = "number"  # "number", "currency", "percent", "duration"
    color: str = "#6366f1"


@dataclass
class AnalyticsConfig:
    """Configuration for analytics dashboard."""
    title: str = "Analytics"
    subtitle: str = "Track your performance metrics"
    date_ranges: list[str] = field(default_factory=lambda: [
        "Today",
        "Yesterday",
        "Last 7 Days",
        "Last 30 Days",
        "This Month",
        "Last Month",
        "Custom",
    ])
    metrics: list[MetricDefinition] = field(default_factory=lambda: [
        MetricDefinition("views", "Page Views", "👁️"),
        MetricDefinition("visitors", "Unique Visitors", "👥"),
        MetricDefinition("sessions", "Sessions", "📱"),
        MetricDefinition("bounce_rate", "Bounce Rate", "↩️", format="percent"),
        MetricDefinition("avg_duration", "Avg. Session Duration", "⏱️", format="duration"),
        MetricDefinition("conversions", "Conversions", "🎯"),
    ])
    chart_types: list[str] = field(default_factory=lambda: [
        "Line Chart",
        "Bar Chart",
        "Area Chart",
    ])
    show_realtime: bool = True
    refresh_interval: int = 30  # seconds


def generate_sample_data(days: int = 7) -> list[dict]:
    """Generate sample analytics data."""
    data = []
    base_date = datetime.now() - timedelta(days=days)

    for i in range(days):
        date = base_date + timedelta(days=i)
        data.append({
            "date": date.strftime("%Y-%m-%d"),
            "views": random.randint(1000, 5000),
            "visitors": random.randint(500, 2000),
            "sessions": random.randint(600, 2500),
            "bounce_rate": random.uniform(0.3, 0.6),
            "avg_duration": random.randint(60, 300),
            "conversions": random.randint(20, 100),
        })

    return data


def create_analytics_dashboard(
    config: Optional[AnalyticsConfig] = None,
    on_date_change: Optional[Callable] = None,
    on_export: Optional[Callable] = None,
    initial_data: Optional[list[dict]] = None,
) -> dict[str, Any]:
    """
    Create an analytics dashboard with semantic-tracked components.

    Args:
        config: Analytics configuration
        on_date_change: Date range change handler
        on_export: Export handler
        initial_data: Initial data to display

    Returns:
        Dict of component references
    """
    config = config or AnalyticsConfig()
    data = initial_data or generate_sample_data(30)
    components = {}

    # Header
    with gr.Row():
        with gr.Column(scale=3):
            components["title"] = semantic(
                gr.Markdown(f"# 📈 {config.title}"),
                intent="displays analytics dashboard title",
                tags=["header"],
            )
            components["subtitle"] = semantic(
                gr.Markdown(f"*{config.subtitle}*"),
                intent="displays analytics dashboard subtitle",
                tags=["header", "subtitle"],
            )

        with gr.Column(scale=1):
            components["date_range"] = semantic(
                gr.Dropdown(
                    choices=config.date_ranges,
                    value="Last 7 Days",
                    label="Date Range",
                    elem_id="date-range",
                ),
                intent="selects time period for analytics data",
                tags=["filter", "date", "primary"],
            )

        with gr.Column(scale=1):
            components["refresh_btn"] = semantic(
                gr.Button("🔄 Refresh", variant="secondary"),
                intent="refreshes analytics data",
                tags=["action", "refresh"],
            )

            components["export_btn"] = semantic(
                gr.Button("📥 Export Report", variant="secondary"),
                intent="exports analytics report as PDF/CSV",
                tags=["action", "export"],
            )

    gr.Markdown("---")

    # Metric Cards (KPIs)
    components["metrics_title"] = semantic(
        gr.Markdown("### 📊 Key Metrics"),
        intent="introduces key metrics section",
        tags=["header", "section"],
    )

    with gr.Row():
        for metric in config.metrics[:6]:  # Show first 6 metrics
            with gr.Column():
                # Calculate sample values
                if metric.format == "percent":
                    value = f"{random.uniform(0.3, 0.6):.1%}"
                elif metric.format == "duration":
                    mins = random.randint(2, 8)
                    secs = random.randint(0, 59)
                    value = f"{mins}:{secs:02d}"
                elif metric.format == "currency":
                    value = f"${random.randint(1000, 10000):,}"
                else:
                    value = f"{random.randint(1000, 50000):,}"

                change = random.uniform(-0.2, 0.3)
                change_str = f"+{change:.1%}" if change > 0 else f"{change:.1%}"
                trend = "📈" if change > 0 else "📉"

                components[f"metric_{metric.key}"] = semantic(
                    gr.Markdown(
                        f"### {metric.icon} {metric.label}\n\n"
                        f"# {value}\n\n"
                        f"{trend} {change_str} vs previous period"
                    ),
                    intent=f"displays {metric.label} metric with trend",
                    tags=["metric", "kpi", metric.key],
                )

    gr.Markdown("---")

    # Charts Section
    with gr.Row():
        # Main Chart
        with gr.Column(scale=2):
            components["chart_title"] = semantic(
                gr.Markdown("### 📈 Trend Analysis"),
                intent="introduces trend chart section",
                tags=["header", "section"],
            )

            with gr.Row():
                components["chart_metric"] = semantic(
                    gr.Dropdown(
                        choices=[m.label for m in config.metrics],
                        value=config.metrics[0].label,
                        label="Metric",
                    ),
                    intent="selects metric to display in chart",
                    tags=["filter", "metric", "chart"],
                )

                components["chart_type"] = semantic(
                    gr.Dropdown(
                        choices=config.chart_types,
                        value="Line Chart",
                        label="Chart Type",
                    ),
                    intent="selects visualization type for chart",
                    tags=["filter", "chart-type"],
                )

            # Line plot
            import pandas as pd
            df = pd.DataFrame(data)

            components["main_chart"] = semantic(
                gr.LinePlot(
                    value=df,
                    x="date",
                    y="views",
                    title="Page Views Over Time",
                ),
                intent="displays primary trend visualization",
                tags=["chart", "visualization", "primary"],
            )

        # Secondary Charts
        with gr.Column(scale=1):
            components["breakdown_title"] = semantic(
                gr.Markdown("### 🥧 Breakdown"),
                intent="introduces breakdown chart section",
                tags=["header", "section"],
            )

            # Traffic sources (simulated)
            sources_data = pd.DataFrame({
                "source": ["Direct", "Organic", "Social", "Referral", "Email"],
                "visitors": [random.randint(500, 2000) for _ in range(5)],
            })

            components["breakdown_chart"] = semantic(
                gr.BarPlot(
                    value=sources_data,
                    x="source",
                    y="visitors",
                    title="Traffic Sources",
                ),
                intent="displays traffic source breakdown",
                tags=["chart", "visualization", "breakdown"],
            )

    gr.Markdown("---")

    # Real-time Section (if enabled)
    if config.show_realtime:
        components["realtime_title"] = semantic(
            gr.Markdown("### ⚡ Real-time"),
            intent="introduces real-time metrics section",
            tags=["header", "section", "realtime"],
        )

        with gr.Row():
            components["active_users"] = semantic(
                gr.Markdown(f"## 🟢 {random.randint(50, 200)} Active Users"),
                intent="displays current active users count",
                tags=["metric", "realtime", "live"],
            )

            components["events_per_min"] = semantic(
                gr.Markdown(f"## ⚡ {random.randint(100, 500)} Events/min"),
                intent="displays current events rate",
                tags=["metric", "realtime", "live"],
            )

            components["realtime_indicator"] = semantic(
                gr.Markdown(f"*Last updated: {datetime.now().strftime('%H:%M:%S')}*"),
                intent="shows when data was last refreshed",
                tags=["status", "timestamp"],
            )

    gr.Markdown("---")

    # Detailed Data Table
    with gr.Accordion("📋 Detailed Data", open=False):
        components["data_table"] = semantic(
            gr.Dataframe(
                value=[[
                    d["date"],
                    f"{d['views']:,}",
                    f"{d['visitors']:,}",
                    f"{d['sessions']:,}",
                    f"{d['bounce_rate']:.1%}",
                    f"{d['conversions']:,}",
                ] for d in data[-10:]],
                headers=["Date", "Views", "Visitors", "Sessions", "Bounce Rate", "Conversions"],
                interactive=False,
            ),
            intent="displays detailed analytics data in table format",
            tags=["table", "data", "detailed"],
        )

        components["download_csv"] = semantic(
            gr.Button("📥 Download CSV", variant="secondary", size="sm"),
            intent="downloads analytics data as CSV file",
            tags=["action", "export", "csv"],
        )

    # Comparison Section
    with gr.Accordion("📊 Period Comparison", open=False):
        components["comparison_title"] = semantic(
            gr.Markdown("Compare metrics between two time periods"),
            intent="introduces period comparison feature",
            tags=["header", "comparison"],
        )

        with gr.Row():
            components["compare_period1"] = semantic(
                gr.Dropdown(
                    choices=config.date_ranges[:-1],
                    value="Last 7 Days",
                    label="Period 1",
                ),
                intent="selects first period for comparison",
                tags=["filter", "comparison"],
            )

            components["compare_period2"] = semantic(
                gr.Dropdown(
                    choices=config.date_ranges[:-1],
                    value="Last 30 Days",
                    label="Period 2",
                ),
                intent="selects second period for comparison",
                tags=["filter", "comparison"],
            )

            components["compare_btn"] = semantic(
                gr.Button("Compare", variant="primary"),
                intent="generates comparison between selected periods",
                tags=["action", "compare"],
            )

        components["comparison_result"] = semantic(
            gr.Markdown("*Select periods and click Compare*"),
            intent="displays period comparison results",
            tags=["output", "comparison"],
        )

    # Wire up date range change
    if on_date_change:
        components["date_range"].change(
            fn=on_date_change,
            inputs=[components["date_range"]],
            outputs=[components["main_chart"], components["data_table"]],
        )

    return components


class AnalyticsPage:
    """
    Complete analytics dashboard with SemanticBlocks integration.

    Usage:
        page = AnalyticsPage(
            title="Website Analytics",
            metrics=[MetricDefinition(...), ...],
        )
        page.launch()
    """

    def __init__(
        self,
        title: str = "Analytics",
        on_date_change: Optional[Callable] = None,
        on_export: Optional[Callable] = None,
        initial_data: Optional[list[dict]] = None,
        **config_kwargs,
    ):
        self.config = AnalyticsConfig(title=title, **config_kwargs)
        self.on_date_change = on_date_change
        self.on_export = on_export
        self.initial_data = initial_data
        self.components: dict[str, Any] = {}
        self.blocks: Optional[SemanticBlocks] = None

    def build(self) -> SemanticBlocks:
        """Build the analytics dashboard."""
        self.blocks = SemanticBlocks(
            title=self.config.title,
            theme=gr.themes.Soft(),
        )

        with self.blocks:
            self.components = create_analytics_dashboard(
                config=self.config,
                on_date_change=self.on_date_change,
                on_export=self.on_export,
                initial_data=self.initial_data,
            )

        return self.blocks

    def launch(self, **kwargs) -> None:
        """Build and launch the analytics dashboard."""
        if self.blocks is None:
            self.build()
        self.blocks.launch(**kwargs)

    @staticmethod
    def render(config: Optional[AnalyticsConfig] = None) -> dict[str, Any]:
        """Render analytics dashboard into existing Blocks context."""
        return create_analytics_dashboard(config=config)
