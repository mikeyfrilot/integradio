"""
Integradio Pages - Pre-built page templates with semantic tracking.

Each page template comes with:
- Semantic intents for all components
- Proper dataflow relationships
- Consistent styling
- Accessibility considerations

Usage:
    from integradio.pages import ChatPage, SettingsPage, GalleryPage

    # Create a chat interface
    chat = ChatPage(
        title="My Assistant",
        system_prompt="You are a helpful assistant",
    )
    chat.launch()

    # Or compose into larger apps
    with SemanticBlocks() as demo:
        with gr.Tab("Chat"):
            ChatPage.render()
        with gr.Tab("Settings"):
            SettingsPage.render()
"""

from .chat import ChatPage, create_chat_interface
from .settings import SettingsPage, create_settings_panel
from .gallery import GalleryPage, create_gallery_grid
from .hero import HeroPage, create_hero_section
from .help import HelpPage, create_help_center
from .dashboard import DashboardPage, create_dashboard
from .form import FormPage, create_form_wizard
from .datatable import DataTablePage, create_data_table
from .upload import UploadPage, create_upload_center
from .analytics import AnalyticsPage, create_analytics_dashboard

__all__ = [
    # Page classes
    "ChatPage",
    "SettingsPage",
    "GalleryPage",
    "HeroPage",
    "HelpPage",
    "DashboardPage",
    "FormPage",
    "DataTablePage",
    "UploadPage",
    "AnalyticsPage",
    # Factory functions
    "create_chat_interface",
    "create_settings_panel",
    "create_gallery_grid",
    "create_hero_section",
    "create_help_center",
    "create_dashboard",
    "create_form_wizard",
    "create_data_table",
    "create_upload_center",
    "create_analytics_dashboard",
]
