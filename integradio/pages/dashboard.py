"""
Dashboard Page - Overview dashboard with KPIs and widgets.

Features:
- KPI cards with trends
- Activity feed
- Quick actions
- Status indicators
- Refreshable widgets
"""

from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime

import gradio as gr

from ..components import semantic
from ..blocks import SemanticBlocks


@dataclass
class KPICard:
    """A key performance indicator card."""
    title: str
    value: str
    change: str = ""  # e.g., "+12%"
    trend: str = "neutral"  # "up", "down", "neutral"
    icon: str = "📊"


@dataclass
class ActivityItem:
    """An activity feed item."""
    message: str
    timestamp: str
    type: str = "info"  # "info", "success", "warning", "error"
    icon: str = "📌"


@dataclass
class QuickAction:
    """A quick action button."""
    label: str
    icon: str
    action_id: str


@dataclass
class DashboardConfig:
    """Configuration for dashboard."""
    title: str = "Dashboard"
    subtitle: str = "Welcome back!"
    username: str = "User"
    show_date: bool = True
    kpis: list[KPICard] = field(default_factory=lambda: [
        KPICard("Total Users", "12,543", "+12%", "up", "👥"),
        KPICard("Revenue", "$45,231", "+8%", "up", "💰"),
        KPICard("Active Sessions", "1,234", "-3%", "down", "📈"),
        KPICard("Conversion Rate", "3.2%", "+0.5%", "up", "🎯"),
    ])
    activities: list[ActivityItem] = field(default_factory=lambda: [
        ActivityItem("New user registered", "2 min ago", "success", "👤"),
        ActivityItem("Payment received", "15 min ago", "success", "💳"),
        ActivityItem("Server alert resolved", "1 hour ago", "warning", "⚠️"),
        ActivityItem("Report generated", "2 hours ago", "info", "📄"),
    ])
    quick_actions: list[QuickAction] = field(default_factory=lambda: [
        QuickAction("New Post", "📝", "new_post"),
        QuickAction("Upload File", "📤", "upload"),
        QuickAction("Send Message", "✉️", "message"),
        QuickAction("View Reports", "📊", "reports"),
    ])


def create_dashboard(
    config: Optional[DashboardConfig] = None,
    on_refresh: Optional[Callable] = None,
    on_action: Optional[Callable] = None,
) -> dict[str, Any]:
    """
    Create a dashboard with semantic-tracked components.

    Args:
        config: Dashboard configuration
        on_refresh: Refresh handler
        on_action: Quick action handler

    Returns:
        Dict of component references
    """
    config = config or DashboardConfig()
    components = {}

    # Header with greeting and date
    with gr.Row():
        with gr.Column(scale=4):
            components["greeting"] = semantic(
                gr.Markdown(f"# 👋 {config.subtitle}, {config.username}!"),
                intent="displays personalized dashboard greeting",
                tags=["header", "greeting"],
            )

        with gr.Column(scale=1):
            if config.show_date:
                today = datetime.now().strftime("%B %d, %Y")
                components["date"] = semantic(
                    gr.Markdown(f"**{today}**"),
                    intent="displays current date",
                    tags=["header", "date"],
                )

            components["refresh_btn"] = semantic(
                gr.Button("🔄 Refresh", variant="secondary", size="sm"),
                intent="refreshes all dashboard data",
                tags=["action", "refresh"],
            )

    gr.Markdown("---")

    # KPI Cards Row
    components["kpi_title"] = semantic(
        gr.Markdown("### 📊 Key Metrics"),
        intent="introduces key performance indicators section",
        tags=["header", "section"],
    )

    with gr.Row():
        for i, kpi in enumerate(config.kpis):
            with gr.Column():
                # Trend indicator
                trend_emoji = {"up": "📈", "down": "📉", "neutral": "➡️"}[kpi.trend]
                trend_color = {"up": "green", "down": "red", "neutral": "gray"}[kpi.trend]

                kpi_md = f"""
### {kpi.icon} {kpi.title}

# {kpi.value}

{trend_emoji} **{kpi.change}** from last period
"""
                components[f"kpi_{i}"] = semantic(
                    gr.Markdown(kpi_md),
                    intent=f"displays {kpi.title} metric with trend",
                    tags=["kpi", "metric", kpi.trend],
                )

    gr.Markdown("---")

    # Main Content Row
    with gr.Row():
        # Activity Feed (left)
        with gr.Column(scale=2):
            components["activity_title"] = semantic(
                gr.Markdown("### 📋 Recent Activity"),
                intent="introduces activity feed section",
                tags=["header", "section"],
            )

            activity_items = []
            for activity in config.activities:
                type_colors = {
                    "info": "ℹ️",
                    "success": "✅",
                    "warning": "⚠️",
                    "error": "❌",
                }
                icon = type_colors.get(activity.type, activity.icon)
                activity_items.append(f"{icon} **{activity.message}** - *{activity.timestamp}*")

            components["activity_feed"] = semantic(
                gr.Markdown("\n\n".join(activity_items)),
                intent="displays chronological activity feed",
                tags=["feed", "activity", "timeline"],
            )

            components["view_all_activity"] = semantic(
                gr.Button("View All Activity →", variant="secondary", size="sm"),
                intent="navigates to full activity history",
                tags=["navigation", "activity"],
            )

        # Quick Actions (right)
        with gr.Column(scale=1):
            components["actions_title"] = semantic(
                gr.Markdown("### ⚡ Quick Actions"),
                intent="introduces quick actions section",
                tags=["header", "section"],
            )

            for i, action in enumerate(config.quick_actions):
                components[f"action_{action.action_id}"] = semantic(
                    gr.Button(
                        f"{action.icon} {action.label}",
                        variant="secondary",
                        size="sm",
                        elem_id=f"action-{action.action_id}",
                    ),
                    intent=f"quick action to {action.label.lower()}",
                    tags=["action", "quick", action.action_id],
                )

    gr.Markdown("---")

    # Status Section
    with gr.Row():
        # System Status
        with gr.Column():
            components["status_title"] = semantic(
                gr.Markdown("### 🖥️ System Status"),
                intent="introduces system status section",
                tags=["header", "section"],
            )

            status_items = [
                ("API Server", "Operational", "🟢"),
                ("Database", "Operational", "🟢"),
                ("CDN", "Operational", "🟢"),
                ("Background Jobs", "Degraded", "🟡"),
            ]

            status_md = "\n".join(
                f"{icon} **{name}**: {status}" for name, status, icon in status_items
            )

            components["system_status"] = semantic(
                gr.Markdown(status_md),
                intent="displays current system health status",
                tags=["status", "health", "monitoring"],
            )

        # Notifications
        with gr.Column():
            components["notifications_title"] = semantic(
                gr.Markdown("### 🔔 Notifications"),
                intent="introduces notifications section",
                tags=["header", "section"],
            )

            components["notifications"] = semantic(
                gr.Markdown(
                    "📬 You have **3** unread notifications\n\n"
                    "🎉 New feature available: Check out our latest update!\n\n"
                    "📅 Scheduled maintenance: Sunday 2am-4am UTC"
                ),
                intent="displays user notifications and alerts",
                tags=["notifications", "alerts"],
            )

            components["mark_read_btn"] = semantic(
                gr.Button("Mark All as Read", variant="secondary", size="sm"),
                intent="marks all notifications as read",
                tags=["action", "notifications"],
            )

    # Wire up refresh
    if on_refresh:
        kpi_outputs = [components[f"kpi_{i}"] for i in range(len(config.kpis))]
        components["refresh_btn"].click(
            fn=on_refresh,
            outputs=kpi_outputs + [components["activity_feed"]],
        )

    # Wire up actions
    if on_action:
        for action in config.quick_actions:
            components[f"action_{action.action_id}"].click(
                fn=lambda aid=action.action_id: on_action(aid),
            )

    return components


class DashboardPage:
    """
    Complete dashboard page with SemanticBlocks integration.

    Usage:
        page = DashboardPage(
            username="John",
            kpis=[KPICard(...), ...],
        )
        page.launch()
    """

    def __init__(
        self,
        title: str = "Dashboard",
        username: str = "User",
        on_refresh: Optional[Callable] = None,
        on_action: Optional[Callable] = None,
        **config_kwargs,
    ):
        self.config = DashboardConfig(title=title, username=username, **config_kwargs)
        self.on_refresh = on_refresh
        self.on_action = on_action
        self.components: dict[str, Any] = {}
        self.blocks: Optional[SemanticBlocks] = None

    def build(self) -> SemanticBlocks:
        """Build the dashboard."""
        self.blocks = SemanticBlocks(
            title=self.config.title,
            theme=gr.themes.Soft(),
        )

        with self.blocks:
            self.components = create_dashboard(
                config=self.config,
                on_refresh=self.on_refresh,
                on_action=self.on_action,
            )

        return self.blocks

    def launch(self, **kwargs) -> None:
        """Build and launch the dashboard."""
        if self.blocks is None:
            self.build()
        self.blocks.launch(**kwargs)

    @staticmethod
    def render(config: Optional[DashboardConfig] = None) -> dict[str, Any]:
        """Render dashboard into existing Blocks context."""
        return create_dashboard(config=config)
