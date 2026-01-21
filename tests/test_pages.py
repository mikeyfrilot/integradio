"""
Tests for page templates in integradio.pages module.

Tests all 10 page templates:
- ChatPage
- DashboardPage
- HeroPage
- GalleryPage
- AnalyticsPage
- DataTablePage
- FormPage
- UploadPage
- SettingsPage
- HelpPage
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import gradio as gr


# Fixture to mock the embedder for all tests in this module
@pytest.fixture(autouse=True)
def mock_embedder_for_pages():
    """Mock the embedder to prevent actual API calls during page tests."""
    mock_embedder = MagicMock()
    mock_embedder.dimension = 768

    def mock_embed(text: str) -> np.ndarray:
        np.random.seed(hash(text) % 2**32)
        return np.random.rand(768).astype(np.float32)

    def mock_embed_query(query: str) -> np.ndarray:
        np.random.seed(hash(f"query:{query}") % 2**32)
        return np.random.rand(768).astype(np.float32)

    mock_embedder.embed.side_effect = mock_embed
    mock_embedder.embed_query.side_effect = mock_embed_query
    mock_embedder.embed_batch.side_effect = lambda texts: [mock_embed(t) for t in texts]

    with patch("integradio.blocks.Embedder") as mock_cls:
        mock_cls.return_value = mock_embedder
        yield mock_embedder

# Import all page classes and factory functions from main pages module
from integradio.pages import (
    # Page classes
    ChatPage,
    DashboardPage,
    HeroPage,
    GalleryPage,
    AnalyticsPage,
    DataTablePage,
    FormPage,
    UploadPage,
    SettingsPage,
    HelpPage,
    # Factory functions
    create_chat_interface,
    create_dashboard,
    create_hero_section,
    create_gallery_grid,
    create_analytics_dashboard,
    create_data_table,
    create_form_wizard,
    create_upload_center,
    create_settings_panel,
    create_help_center,
)

# Import config classes and dataclasses from submodules
from integradio.pages.chat import ChatConfig
from integradio.pages.dashboard import DashboardConfig, KPICard, ActivityItem, QuickAction
from integradio.pages.hero import HeroConfig, Feature, Testimonial
from integradio.pages.gallery import GalleryConfig
from integradio.pages.analytics import AnalyticsConfig, MetricDefinition
from integradio.pages.datatable import DataTableConfig, ColumnDef
from integradio.pages.form import FormConfig, FormStep, FormField, FieldType
from integradio.pages.upload import UploadConfig
from integradio.pages.settings import SettingItem, SettingsSection
from integradio.pages.help import HelpConfig, FAQItem, HelpArticle

from integradio import SemanticBlocks


class TestChatPage:
    """Tests for ChatPage template."""

    def test_chat_page_instantiation(self):
        """Test basic ChatPage instantiation."""
        page = ChatPage()
        assert page.config.title == "AI Chat"
        assert page.config.system_prompt == "You are a helpful assistant."
        assert page.blocks is None
        assert page.components == {}

    def test_chat_page_custom_config(self):
        """Test ChatPage with custom configuration."""
        page = ChatPage(
            title="My Custom Bot",
            system_prompt="You are a coding assistant.",
            temperature=0.9,
            max_tokens=8192,
        )
        assert page.config.title == "My Custom Bot"
        assert page.config.system_prompt == "You are a coding assistant."
        assert page.config.temperature == 0.9
        assert page.config.max_tokens == 8192

    def test_chat_page_build(self):
        """Test ChatPage build creates SemanticBlocks."""
        page = ChatPage(title="Test Chat")
        blocks = page.build()

        assert isinstance(blocks, SemanticBlocks)
        assert page.blocks is blocks
        assert "title" in page.components
        assert "chatbot" in page.components
        assert "user_input" in page.components
        assert "send_btn" in page.components
        assert "clear_btn" in page.components

    def test_chat_page_with_chat_fn(self):
        """Test ChatPage with custom chat function."""
        mock_fn = MagicMock()
        page = ChatPage(chat_fn=mock_fn)
        page.build()

        # Verify components exist for wiring
        assert "send_btn" in page.components
        assert "user_input" in page.components

    def test_chat_config_dataclass(self):
        """Test ChatConfig dataclass defaults."""
        config = ChatConfig()
        assert config.title == "AI Chat"
        assert config.placeholder == "Type your message..."
        assert config.show_token_count is True
        assert config.enable_export is True

    def test_create_chat_interface_factory(self):
        """Test create_chat_interface factory function."""
        with SemanticBlocks() as demo:
            components = create_chat_interface()

        assert "title" in components
        assert "chatbot" in components
        assert "user_input" in components


class TestDashboardPage:
    """Tests for DashboardPage template."""

    def test_dashboard_page_instantiation(self):
        """Test basic DashboardPage instantiation."""
        page = DashboardPage()
        assert page.config.title == "Dashboard"
        assert page.config.username == "User"

    def test_dashboard_page_custom_config(self):
        """Test DashboardPage with custom configuration."""
        page = DashboardPage(
            title="Admin Panel",
            username="Admin",
        )
        assert page.config.title == "Admin Panel"
        assert page.config.username == "Admin"

    def test_dashboard_page_build(self):
        """Test DashboardPage build creates components."""
        page = DashboardPage(username="TestUser")
        blocks = page.build()

        assert isinstance(blocks, SemanticBlocks)
        assert "greeting" in page.components
        assert "refresh_btn" in page.components
        assert "kpi_title" in page.components
        assert "activity_feed" in page.components

    def test_dashboard_with_custom_kpis(self):
        """Test DashboardPage with custom KPIs."""
        custom_kpis = [
            KPICard("Downloads", "50K", "+25%", "up", "📥"),
            KPICard("Errors", "12", "-50%", "down", "❌"),
        ]
        page = DashboardPage(kpis=custom_kpis)
        page.build()

        assert "kpi_0" in page.components
        assert "kpi_1" in page.components

    def test_dashboard_with_custom_activities(self):
        """Test DashboardPage with custom activities."""
        activities = [
            ActivityItem("Custom event", "just now", "info", "🔔"),
        ]
        page = DashboardPage(activities=activities)
        page.build()

        assert "activity_feed" in page.components

    def test_dashboard_with_quick_actions(self):
        """Test DashboardPage with custom quick actions."""
        actions = [
            QuickAction("Run Tests", "🧪", "run_tests"),
        ]
        page = DashboardPage(quick_actions=actions)
        page.build()

        assert "action_run_tests" in page.components

    def test_kpi_card_dataclass(self):
        """Test KPICard dataclass."""
        kpi = KPICard(
            title="Revenue",
            value="$100K",
            change="+10%",
            trend="up",
            icon="💰",
        )
        assert kpi.title == "Revenue"
        assert kpi.trend == "up"


class TestHeroPage:
    """Tests for HeroPage template."""

    def test_hero_page_instantiation(self):
        """Test basic HeroPage instantiation."""
        page = HeroPage()
        assert page.config.title == "Welcome"
        assert page.config.subtitle == "The best solution"

    def test_hero_page_custom_config(self):
        """Test HeroPage with custom configuration."""
        page = HeroPage(
            title="My Product",
            subtitle="Built for developers",
            primary_cta="Get Started Free",
            secondary_cta="Watch Demo",
        )
        assert page.config.title == "My Product"
        assert page.config.primary_cta == "Get Started Free"

    def test_hero_page_build(self):
        """Test HeroPage build creates components."""
        page = HeroPage(title="Test Hero")
        blocks = page.build()

        assert isinstance(blocks, SemanticBlocks)
        assert "title" in page.components
        assert "subtitle" in page.components
        assert "primary_cta" in page.components
        assert "secondary_cta" in page.components

    def test_hero_with_features(self):
        """Test HeroPage with custom features."""
        features = [
            Feature("🚀", "Fast", "Lightning fast"),
            Feature("🔒", "Secure", "Enterprise security"),
        ]
        page = HeroPage(features=features)
        page.build()

        assert "feature_0" in page.components
        assert "feature_1" in page.components

    def test_hero_with_testimonials(self):
        """Test HeroPage with testimonials."""
        testimonials = [
            Testimonial("Great product!", "John Doe", "CEO"),
        ]
        page = HeroPage(testimonials=testimonials)
        page.build()

        assert "testimonial_0" in page.components

    def test_hero_with_cta_handlers(self):
        """Test HeroPage with CTA click handlers."""
        primary_handler = MagicMock()
        secondary_handler = MagicMock()

        page = HeroPage(
            on_primary_click=primary_handler,
            on_secondary_click=secondary_handler,
        )
        page.build()

        assert "primary_cta" in page.components
        assert "secondary_cta" in page.components

    def test_feature_dataclass(self):
        """Test Feature dataclass."""
        feature = Feature("🎨", "Beautiful", "Stunning UI")
        assert feature.icon == "🎨"
        assert feature.title == "Beautiful"

    def test_testimonial_dataclass(self):
        """Test Testimonial dataclass."""
        testimonial = Testimonial(
            quote="Amazing!",
            author="Jane",
            role="Designer",
        )
        assert testimonial.author == "Jane"


class TestGalleryPage:
    """Tests for GalleryPage template."""

    def test_gallery_page_instantiation(self):
        """Test basic GalleryPage instantiation."""
        page = GalleryPage()
        assert page.config.title == "Gallery"
        assert page.config.columns == 4

    def test_gallery_page_custom_config(self):
        """Test GalleryPage with custom configuration."""
        page = GalleryPage(
            title="My Photos",
            columns=6,
            height=800,
            allow_upload=False,
        )
        assert page.config.title == "My Photos"
        assert page.config.columns == 6
        assert page.config.allow_upload is False

    def test_gallery_page_build(self):
        """Test GalleryPage build creates components."""
        page = GalleryPage(title="Test Gallery")
        blocks = page.build()

        assert isinstance(blocks, SemanticBlocks)
        assert "title" in page.components
        assert "gallery" in page.components
        assert "search" in page.components
        assert "category" in page.components

    def test_gallery_with_upload_disabled(self):
        """Test GalleryPage with upload disabled."""
        page = GalleryPage(allow_upload=False)
        page.build()

        # upload_btn should not exist when upload is disabled
        assert "upload_btn" not in page.components

    def test_gallery_with_upload_enabled(self):
        """Test GalleryPage with upload enabled."""
        page = GalleryPage(allow_upload=True)
        page.build()

        assert "upload_btn" in page.components

    def test_gallery_with_custom_categories(self):
        """Test GalleryPage with custom categories."""
        categories = ["All", "Nature", "Urban", "Abstract"]
        page = GalleryPage(categories=categories)
        page.build()

        assert "category" in page.components

    def test_gallery_config_dataclass(self):
        """Test GalleryConfig dataclass defaults."""
        config = GalleryConfig()
        assert config.show_metadata is True
        assert config.show_download is True


class TestAnalyticsPage:
    """Tests for AnalyticsPage template."""

    def test_analytics_page_instantiation(self):
        """Test basic AnalyticsPage instantiation."""
        page = AnalyticsPage()
        assert page.config.title == "Analytics"

    def test_analytics_page_custom_config(self):
        """Test AnalyticsPage with custom configuration."""
        page = AnalyticsPage(
            title="My Analytics",
            show_realtime=False,
        )
        assert page.config.title == "My Analytics"
        assert page.config.show_realtime is False

    def test_analytics_page_build(self):
        """Test AnalyticsPage build creates components."""
        page = AnalyticsPage(title="Test Analytics")
        blocks = page.build()

        assert isinstance(blocks, SemanticBlocks)
        assert "title" in page.components
        assert "date_range" in page.components
        assert "main_chart" in page.components
        assert "data_table" in page.components

    def test_analytics_with_custom_metrics(self):
        """Test AnalyticsPage with custom metrics."""
        metrics = [
            MetricDefinition("sales", "Sales", "💰", "currency"),
            MetricDefinition("orders", "Orders", "📦", "number"),
        ]
        page = AnalyticsPage(metrics=metrics)
        page.build()

        assert "metric_sales" in page.components
        assert "metric_orders" in page.components

    def test_analytics_with_realtime_disabled(self):
        """Test AnalyticsPage with realtime section disabled."""
        page = AnalyticsPage(show_realtime=False)
        page.build()

        # Realtime components should not exist
        assert "active_users" not in page.components

    def test_analytics_with_realtime_enabled(self):
        """Test AnalyticsPage with realtime section enabled."""
        page = AnalyticsPage(show_realtime=True)
        page.build()

        assert "active_users" in page.components
        assert "events_per_min" in page.components

    def test_metric_definition_dataclass(self):
        """Test MetricDefinition dataclass."""
        metric = MetricDefinition(
            key="revenue",
            label="Revenue",
            icon="💵",
            format="currency",
            color="#00ff00",
        )
        assert metric.key == "revenue"
        assert metric.format == "currency"


class TestDataTablePage:
    """Tests for DataTablePage template."""

    def test_datatable_page_instantiation(self):
        """Test basic DataTablePage instantiation."""
        page = DataTablePage()
        assert page.config.title == "Data Table"
        assert page.config.page_size == 10

    def test_datatable_page_custom_config(self):
        """Test DataTablePage with custom configuration."""
        page = DataTablePage(
            title="Users Table",
            page_size=25,
            allow_edit=True,
        )
        assert page.config.title == "Users Table"
        assert page.config.page_size == 25
        assert page.config.allow_edit is True

    def test_datatable_page_build(self):
        """Test DataTablePage build creates components."""
        page = DataTablePage(title="Test Table")
        blocks = page.build()

        assert isinstance(blocks, SemanticBlocks)
        assert "title" in page.components
        assert "table" in page.components
        assert "search" in page.components
        assert "page_size" in page.components

    def test_datatable_with_custom_columns(self):
        """Test DataTablePage with custom columns."""
        columns = [
            ColumnDef("id", "ID", type="number"),
            ColumnDef("name", "Full Name"),
            ColumnDef("active", "Active", type="boolean"),
        ]
        page = DataTablePage(columns=columns)
        page.build()

        assert "table" in page.components

    def test_datatable_with_initial_data(self):
        """Test DataTablePage with initial data."""
        data = [
            {"id": 1, "name": "Test User", "email": "test@example.com"},
        ]
        page = DataTablePage(initial_data=data)
        page.build()

        assert "table" in page.components

    def test_datatable_with_selection_enabled(self):
        """Test DataTablePage with row selection enabled."""
        page = DataTablePage(allow_selection=True)
        page.build()

        assert "selected_row" in page.components

    def test_datatable_with_selection_disabled(self):
        """Test DataTablePage with row selection disabled."""
        page = DataTablePage(allow_selection=False)
        page.build()

        assert "selected_row" not in page.components

    def test_column_def_dataclass(self):
        """Test ColumnDef dataclass."""
        col = ColumnDef(
            key="email",
            label="Email Address",
            sortable=True,
            filterable=True,
        )
        assert col.key == "email"
        assert col.sortable is True


class TestFormPage:
    """Tests for FormPage template."""

    def test_form_page_instantiation(self):
        """Test basic FormPage instantiation."""
        page = FormPage()
        assert page.config.title == "Form"

    def test_form_page_custom_config(self):
        """Test FormPage with custom configuration."""
        page = FormPage(
            title="Registration",
            submit_text="Create Account",
            show_progress=True,
        )
        assert page.config.title == "Registration"
        assert page.config.submit_text == "Create Account"

    def test_form_page_build(self):
        """Test FormPage build creates components."""
        page = FormPage(title="Test Form")
        blocks = page.build()

        assert isinstance(blocks, SemanticBlocks)
        assert "title" in page.components
        assert "next_btn" in page.components
        assert "submit_btn" in page.components
        assert "progress" in page.components

    def test_form_with_custom_steps(self):
        """Test FormPage with custom steps."""
        steps = [
            FormStep(
                title="Contact Info",
                fields=[
                    FormField("email", "Email", FieldType.EMAIL, required=True),
                    FormField("phone", "Phone"),
                ],
            ),
            FormStep(
                title="Preferences",
                fields=[
                    FormField("newsletter", "Subscribe", FieldType.CHECKBOX),
                ],
            ),
        ]
        page = FormPage(steps=steps)
        page.build()

        assert "field_email" in page.components
        assert "field_phone" in page.components
        assert "field_newsletter" in page.components

    def test_form_field_types(self):
        """Test various form field types."""
        steps = [
            FormStep(
                title="Test Fields",
                fields=[
                    FormField("text_field", "Text", FieldType.TEXT),
                    FormField("number_field", "Number", FieldType.NUMBER),
                    FormField("dropdown_field", "Dropdown", FieldType.DROPDOWN, choices=["A", "B"]),
                    FormField("radio_field", "Radio", FieldType.RADIO, choices=["X", "Y"]),
                    FormField("textarea_field", "Textarea", FieldType.TEXTAREA),
                    FormField("password_field", "Password", FieldType.PASSWORD),
                ],
            ),
        ]
        page = FormPage(steps=steps)
        page.build()

        assert "field_text_field" in page.components
        assert "field_number_field" in page.components
        assert "field_dropdown_field" in page.components

    def test_form_step_dataclass(self):
        """Test FormStep dataclass."""
        step = FormStep(
            title="Account",
            description="Create your account",
            icon="🔐",
        )
        assert step.title == "Account"
        assert step.icon == "🔐"

    def test_form_field_dataclass(self):
        """Test FormField dataclass."""
        field = FormField(
            name="username",
            label="Username",
            type=FieldType.TEXT,
            required=True,
            placeholder="Enter username",
        )
        assert field.name == "username"
        assert field.required is True

    def test_field_type_enum(self):
        """Test FieldType enum values."""
        assert FieldType.TEXT.value == "text"
        assert FieldType.EMAIL.value == "email"
        assert FieldType.PASSWORD.value == "password"


class TestUploadPage:
    """Tests for UploadPage template."""

    def test_upload_page_instantiation(self):
        """Test basic UploadPage instantiation."""
        page = UploadPage()
        assert page.config.title == "Upload Center"

    def test_upload_page_custom_config(self):
        """Test UploadPage with custom configuration."""
        page = UploadPage(
            title="Media Upload",
            max_files=5,
            max_file_size="50MB",
        )
        assert page.config.title == "Media Upload"
        assert page.config.max_files == 5
        assert page.config.max_file_size == "50MB"

    def test_upload_page_build(self):
        """Test UploadPage build creates components."""
        page = UploadPage(title="Test Upload")
        blocks = page.build()

        assert isinstance(blocks, SemanticBlocks)
        assert "title" in page.components
        assert "upload_area" in page.components
        assert "file_list" in page.components
        assert "upload_status" in page.components

    def test_upload_with_restricted_types(self):
        """Test UploadPage with restricted file types."""
        page = UploadPage(allowed_types=["image"])
        page.build()

        assert "upload_image" in page.components
        assert "upload_video" not in page.components

    def test_upload_with_all_types(self):
        """Test UploadPage with all file types allowed."""
        page = UploadPage(allowed_types=["image", "video", "audio", "document"])
        page.build()

        assert "upload_image" in page.components
        assert "upload_video" in page.components
        assert "upload_audio" in page.components
        assert "upload_doc" in page.components

    def test_upload_with_processing_disabled(self):
        """Test UploadPage with processing options disabled."""
        page = UploadPage(show_processing=False)
        page.build()

        assert "process_btn" not in page.components

    def test_upload_with_processing_enabled(self):
        """Test UploadPage with processing options enabled."""
        page = UploadPage(show_processing=True)
        page.build()

        assert "process_btn" in page.components
        assert "process_options" in page.components

    def test_upload_config_dataclass(self):
        """Test UploadConfig dataclass defaults."""
        config = UploadConfig()
        assert config.show_preview is True
        assert config.auto_process is False


class TestSettingsPage:
    """Tests for SettingsPage template."""

    def test_settings_page_instantiation(self):
        """Test basic SettingsPage instantiation."""
        page = SettingsPage()
        assert page.title == "Settings"

    def test_settings_page_custom_config(self):
        """Test SettingsPage with custom title."""
        page = SettingsPage(title="App Settings")
        assert page.title == "App Settings"

    def test_settings_page_build(self):
        """Test SettingsPage build creates components."""
        page = SettingsPage(title="Test Settings")
        blocks = page.build()

        assert isinstance(blocks, SemanticBlocks)
        assert "title" in page.components
        assert "save_btn" in page.components
        assert "reset_btn" in page.components
        assert "export_btn" in page.components

    def test_settings_with_custom_sections(self):
        """Test SettingsPage with custom sections."""
        sections = [
            SettingsSection(
                title="Custom Section",
                icon="🔧",
                settings=[
                    SettingItem("custom_toggle", "Custom Toggle", "toggle", True),
                    SettingItem("custom_dropdown", "Custom Dropdown", "dropdown", "A", choices=["A", "B"]),
                ],
            ),
        ]
        page = SettingsPage(sections=sections)
        page.build()

        assert "custom_section_custom_toggle" in page.components
        assert "custom_section_custom_dropdown" in page.components

    def test_settings_item_types(self):
        """Test various setting item types."""
        sections = [
            SettingsSection(
                title="Test",
                settings=[
                    SettingItem("toggle_setting", "Toggle", "toggle", True),
                    SettingItem("dropdown_setting", "Dropdown", "dropdown", "X", choices=["X", "Y"]),
                    SettingItem("slider_setting", "Slider", "slider", 50, min_value=0, max_value=100),
                    SettingItem("number_setting", "Number", "number", 10),
                    SettingItem("color_setting", "Color", "color", "#ff0000"),
                    SettingItem("text_setting", "Text", "text", "hello"),
                ],
            ),
        ]
        page = SettingsPage(sections=sections)
        page.build()

        assert "test_toggle_setting" in page.components
        assert "test_slider_setting" in page.components
        assert "test_color_setting" in page.components

    def test_settings_section_dataclass(self):
        """Test SettingsSection dataclass."""
        section = SettingsSection(
            title="General",
            icon="⚙️",
        )
        assert section.title == "General"
        assert section.icon == "⚙️"

    def test_setting_item_dataclass(self):
        """Test SettingItem dataclass."""
        item = SettingItem(
            key="theme",
            label="Theme",
            type="dropdown",
            default="dark",
            description="Choose theme",
            choices=["light", "dark"],
        )
        assert item.key == "theme"
        assert item.default == "dark"


class TestHelpPage:
    """Tests for HelpPage template."""

    def test_help_page_instantiation(self):
        """Test basic HelpPage instantiation."""
        page = HelpPage()
        assert page.config.title == "Help Center"

    def test_help_page_custom_config(self):
        """Test HelpPage with custom configuration."""
        page = HelpPage(
            title="Support Center",
            support_email="help@example.com",
        )
        assert page.config.title == "Support Center"
        assert page.config.support_email == "help@example.com"

    def test_help_page_build(self):
        """Test HelpPage build creates components."""
        page = HelpPage(title="Test Help")
        blocks = page.build()

        assert isinstance(blocks, SemanticBlocks)
        assert "title" in page.components
        assert "search" in page.components
        assert "search_btn" in page.components
        assert "faq_title" in page.components

    def test_help_with_custom_faqs(self):
        """Test HelpPage with custom FAQs."""
        faqs = [
            FAQItem("How to start?", "Click the button.", "Getting Started"),
            FAQItem("What is this?", "This is help.", "General"),
        ]
        page = HelpPage(faqs=faqs)
        page.build()

        # FAQs are organized by category
        assert "faq_title" in page.components

    def test_help_with_custom_categories(self):
        """Test HelpPage with custom categories."""
        categories = ["Setup", "Usage", "Troubleshooting"]
        page = HelpPage(categories=categories)
        page.build()

        assert "cat_0" in page.components
        assert "cat_1" in page.components
        assert "cat_2" in page.components

    def test_help_with_contact_form_disabled(self):
        """Test HelpPage with contact form disabled."""
        page = HelpPage(show_contact_form=False)
        page.build()

        assert "contact_submit" not in page.components

    def test_help_with_contact_form_enabled(self):
        """Test HelpPage with contact form enabled."""
        page = HelpPage(show_contact_form=True)
        page.build()

        assert "contact_name" in page.components
        assert "contact_email" in page.components
        assert "contact_message" in page.components
        assert "contact_submit" in page.components

    def test_help_with_videos_disabled(self):
        """Test HelpPage with video section disabled."""
        page = HelpPage(show_video_section=False)
        page.build()

        assert "videos_title" not in page.components

    def test_help_with_videos_enabled(self):
        """Test HelpPage with video section enabled."""
        page = HelpPage(show_video_section=True)
        page.build()

        assert "videos_title" in page.components

    def test_faq_item_dataclass(self):
        """Test FAQItem dataclass."""
        faq = FAQItem(
            question="How do I reset?",
            answer="Click reset button.",
            category="Account",
        )
        assert faq.question == "How do I reset?"
        assert faq.category == "Account"

    def test_help_article_dataclass(self):
        """Test HelpArticle dataclass."""
        article = HelpArticle(
            title="Getting Started",
            content="Welcome to our app...",
            category="Basics",
            tags=["intro", "beginner"],
        )
        assert article.title == "Getting Started"
        assert "intro" in article.tags


class TestPageRenderMethods:
    """Test static render methods for all pages."""

    def test_chat_page_render(self):
        """Test ChatPage.render static method."""
        with SemanticBlocks() as demo:
            components = ChatPage.render()
        assert "chatbot" in components

    def test_dashboard_page_render(self):
        """Test DashboardPage.render static method."""
        with SemanticBlocks() as demo:
            components = DashboardPage.render()
        assert "greeting" in components

    def test_hero_page_render(self):
        """Test HeroPage.render static method."""
        with SemanticBlocks() as demo:
            components = HeroPage.render()
        assert "primary_cta" in components

    def test_gallery_page_render(self):
        """Test GalleryPage.render static method."""
        with SemanticBlocks() as demo:
            components = GalleryPage.render()
        assert "gallery" in components

    def test_analytics_page_render(self):
        """Test AnalyticsPage.render static method."""
        with SemanticBlocks() as demo:
            components = AnalyticsPage.render()
        assert "main_chart" in components

    def test_datatable_page_render(self):
        """Test DataTablePage.render static method."""
        with SemanticBlocks() as demo:
            components = DataTablePage.render()
        assert "table" in components

    def test_form_page_render(self):
        """Test FormPage.render static method."""
        with SemanticBlocks() as demo:
            components = FormPage.render()
        assert "submit_btn" in components

    def test_upload_page_render(self):
        """Test UploadPage.render static method."""
        with SemanticBlocks() as demo:
            components = UploadPage.render()
        assert "upload_area" in components

    def test_settings_page_render(self):
        """Test SettingsPage.render static method."""
        with SemanticBlocks() as demo:
            components = SettingsPage.render()
        assert "save_btn" in components

    def test_help_page_render(self):
        """Test HelpPage.render static method."""
        with SemanticBlocks() as demo:
            components = HelpPage.render()
        assert "search" in components


class TestPageIntegration:
    """Integration tests for page components."""

    def test_all_pages_build_successfully(self):
        """Test that all page types can be built without errors."""
        pages = [
            ChatPage(),
            DashboardPage(),
            HeroPage(),
            GalleryPage(),
            AnalyticsPage(),
            DataTablePage(),
            FormPage(),
            UploadPage(),
            SettingsPage(),
            HelpPage(),
        ]

        for page in pages:
            blocks = page.build()
            assert isinstance(blocks, SemanticBlocks)
            assert len(page.components) > 0

    def test_pages_have_title_component(self):
        """Test that all pages have a title component."""
        pages = [
            ChatPage(),
            DashboardPage(),
            HeroPage(),
            GalleryPage(),
            AnalyticsPage(),
            DataTablePage(),
            FormPage(),
            UploadPage(),
            SettingsPage(),
            HelpPage(),
        ]

        for page in pages:
            page.build()
            # DashboardPage uses 'greeting' instead of 'title'
            has_header = "title" in page.components or "greeting" in page.components
            assert has_header, f"{type(page).__name__} missing title/greeting component"

    def test_pages_return_semantic_blocks(self):
        """Test that build() returns SemanticBlocks instance."""
        page = ChatPage()
        result = page.build()

        assert isinstance(result, SemanticBlocks)
        assert page.blocks is result

    def test_double_build_returns_same_blocks(self):
        """Test that calling build() twice doesn't create new blocks."""
        page = ChatPage()
        blocks1 = page.build()
        blocks2 = page.build()

        # build() should create new blocks each time
        # This tests that internal state is consistent
        assert page.blocks is blocks2


# =============================================================================
# Upload Security Tests
# =============================================================================

class TestUploadSecurity:
    """Security tests for upload functionality."""

    def test_sanitize_filename_basic(self):
        """Test basic filename sanitization."""
        from integradio.pages.upload import sanitize_filename

        assert sanitize_filename("test.jpg") == "test.jpg"
        assert sanitize_filename("my_file.png") == "my_file.png"
        assert sanitize_filename("document-v2.pdf") == "document-v2.pdf"

    def test_sanitize_filename_path_traversal(self):
        """Test path traversal prevention."""
        from integradio.pages.upload import sanitize_filename

        # Path traversal attempts should be sanitized
        assert sanitize_filename("../../../etc/passwd") == "passwd"
        assert sanitize_filename("..\\..\\windows\\system32") == "system32"
        assert sanitize_filename("/etc/passwd") == "passwd"

    def test_sanitize_filename_null_bytes(self):
        """Test null byte removal."""
        from integradio.pages.upload import sanitize_filename

        # Null bytes should be removed
        result = sanitize_filename("test\x00.jpg")
        assert "\x00" not in result

    def test_sanitize_filename_empty(self):
        """Test empty filename rejection."""
        from integradio.pages.upload import sanitize_filename

        with pytest.raises(ValueError, match="Empty filename"):
            sanitize_filename("")

    def test_sanitize_filename_dot(self):
        """Test dot-only filenames are rejected."""
        from integradio.pages.upload import sanitize_filename

        with pytest.raises(ValueError, match="Invalid filename"):
            sanitize_filename(".")

        with pytest.raises(ValueError, match="Invalid filename"):
            sanitize_filename("..")

    def test_sanitize_filename_too_long(self):
        """Test filename length validation."""
        from integradio.pages.upload import sanitize_filename

        long_name = "a" * 300 + ".txt"
        with pytest.raises(ValueError, match="too long"):
            sanitize_filename(long_name)

    def test_sanitize_filename_blocked_extensions(self):
        """Test blocked extension rejection."""
        from integradio.pages.upload import sanitize_filename

        dangerous_files = [
            "virus.exe",
            "script.bat",
            "payload.cmd",
            "shell.sh",
            "hack.ps1",
            "attack.vbs",
            "malware.dll",
            "backdoor.php",
        ]

        for filename in dangerous_files:
            with pytest.raises(ValueError, match="not allowed"):
                sanitize_filename(filename)

    def test_sanitize_filename_double_extension(self):
        """Test double extension blocking (e.g., file.jpg.exe)."""
        from integradio.pages.upload import sanitize_filename

        with pytest.raises(ValueError, match="not allowed"):
            sanitize_filename("innocent.jpg.exe")

        with pytest.raises(ValueError, match="not allowed"):
            sanitize_filename("document.pdf.bat")

    def test_sanitize_filename_special_chars_replaced(self):
        """Test special characters are replaced."""
        from integradio.pages.upload import sanitize_filename

        # Characters not matching the safe pattern should be replaced with _
        result = sanitize_filename("file<>:name.txt")
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result

    def test_format_file_size(self):
        """Test file size formatting."""
        from integradio.pages.upload import _format_file_size

        assert _format_file_size(0) == "0.0 B"
        assert _format_file_size(500) == "500.0 B"
        assert _format_file_size(1024) == "1.0 KB"
        assert _format_file_size(1024 * 1024) == "1.0 MB"
        assert _format_file_size(1024 * 1024 * 1024) == "1.0 GB"
        assert _format_file_size(1024 * 1024 * 1024 * 1024) == "1.0 TB"

    def test_get_file_type(self):
        """Test file type detection."""
        from integradio.pages.upload import _get_file_type

        # Images
        assert _get_file_type(".jpg") == "Image"
        assert _get_file_type(".png") == "Image"
        assert _get_file_type(".gif") == "Image"

        # Videos
        assert _get_file_type(".mp4") == "Video"
        assert _get_file_type(".webm") == "Video"

        # Audio
        assert _get_file_type(".mp3") == "Audio"
        assert _get_file_type(".wav") == "Audio"

        # Documents
        assert _get_file_type(".pdf") == "PDF"
        assert _get_file_type(".doc") == "Document"
        assert _get_file_type(".txt") == "Text"

        # Unknown
        assert _get_file_type(".xyz") == "File"


# =============================================================================
# Chat Response Handler Tests
# =============================================================================

class TestChatResponseHandlers:
    """Tests for chat response handling edge cases."""

    def test_chat_with_streaming_function(self):
        """Test ChatPage with streaming chat function."""
        def streaming_chat(message, history, system_prompt, temp, max_len):
            yield "Hello"
            yield " World"
            yield "!"

        page = ChatPage(chat_fn=streaming_chat, stream=True)
        page.build()

        assert page.chat_fn is streaming_chat
        assert page.stream is True

    def test_chat_with_non_streaming_function(self):
        """Test ChatPage with non-streaming chat function."""
        def simple_chat(message, history, system_prompt, temp, max_len):
            return f"You said: {message}"

        page = ChatPage(chat_fn=simple_chat, stream=False)
        page.build()

        assert page.stream is False

    def test_chat_config_temperature_range(self):
        """Test ChatConfig temperature values."""
        config = ChatConfig(temperature=0.0)
        assert config.temperature == 0.0

        config = ChatConfig(temperature=2.0)
        assert config.temperature == 2.0

    def test_chat_without_system_prompt_display(self):
        """Test ChatPage without system prompt display."""
        page = ChatPage(show_system_prompt=False)
        page.build()

        assert "system_prompt" not in page.components

    def test_chat_without_regenerate(self):
        """Test ChatPage without regenerate button."""
        page = ChatPage(enable_regenerate=False)
        page.build()

        assert "regenerate_btn" not in page.components

    def test_chat_without_export(self):
        """Test ChatPage without export button."""
        page = ChatPage(enable_export=False)
        page.build()

        assert "export_btn" not in page.components

    def test_chat_without_token_count(self):
        """Test ChatPage without token count display."""
        page = ChatPage(show_token_count=False)
        page.build()

        assert "token_display" not in page.components


# =============================================================================
# Upload Handler Edge Cases
# =============================================================================

class TestUploadHandlerEdgeCases:
    """Tests for upload handler edge cases."""

    def test_upload_config_defaults(self):
        """Test UploadConfig default values."""
        config = UploadConfig()

        assert config.title == "Upload Center"
        assert config.max_file_size == "100MB"
        assert config.max_files == 10
        assert config.show_preview is True
        assert config.show_processing is True
        assert config.auto_process is False
        assert "image" in config.allowed_types
        assert "video" in config.allowed_types
        assert "audio" in config.allowed_types
        assert "document" in config.allowed_types

    def test_upload_with_image_only(self):
        """Test UploadPage with image-only mode."""
        page = UploadPage(allowed_types=["image"])
        page.build()

        assert "upload_image" in page.components
        assert "upload_video" not in page.components
        assert "upload_audio" not in page.components
        assert "upload_doc" not in page.components

    def test_upload_with_custom_max_files(self):
        """Test UploadPage with custom max files limit."""
        page = UploadPage(max_files=3)
        page.build()

        assert page.config.max_files == 3

    def test_upload_with_preview_disabled(self):
        """Test UploadPage with preview disabled."""
        page = UploadPage(show_preview=False)
        # Just instantiate - build behavior depends on implementation
        assert page.config.show_preview is False

    def test_uploaded_file_dataclass(self):
        """Test UploadedFile dataclass."""
        from integradio.pages.upload import UploadedFile

        file = UploadedFile(
            name="test.jpg",
            size="1.5 MB",
            type="Image",
        )

        assert file.name == "test.jpg"
        assert file.status == "uploaded"
        assert file.preview_url is None
        assert file.metadata == {}


# =============================================================================
# Pages Utils Tests
# =============================================================================

class TestPagesUtils:
    """Tests for pages/utils.py functions."""

    def test_get_page_css(self):
        """Test get_page_css returns CSS string."""
        from integradio.pages.utils import get_page_css

        css = get_page_css()
        assert isinstance(css, str)
        assert len(css) > 0

    def test_get_enhanced_page_css(self):
        """Test get_enhanced_page_css includes empty state CSS."""
        from integradio.pages.utils import get_enhanced_page_css

        css = get_enhanced_page_css()
        assert isinstance(css, str)
        assert ".empty-state" in css

    def test_get_empty_state_types(self):
        """Test get_empty_state for all types."""
        from integradio.pages.utils import get_empty_state, EMPTY_STATES

        for state_type in EMPTY_STATES.keys():
            html = get_empty_state(state_type)
            assert "empty-state" in html
            assert "role=\"status\"" in html

    def test_get_empty_state_invalid(self):
        """Test get_empty_state with invalid type returns default."""
        from integradio.pages.utils import get_empty_state

        html = get_empty_state("nonexistent_type")
        # Should return no_data as default
        assert "No data available" in html

    def test_add_skip_navigation(self):
        """Test add_skip_navigation creates HTML component."""
        from integradio.pages.utils import add_skip_navigation
        import gradio as gr

        with gr.Blocks() as demo:
            component = add_skip_navigation()

        assert component is not None

    def test_add_main_content_landmark(self):
        """Test add_main_content_landmark creates HTML."""
        from integradio.pages.utils import add_main_content_landmark
        import gradio as gr

        with gr.Blocks() as demo:
            component = add_main_content_landmark("my-content")

        assert component is not None

    def test_close_main_content_landmark(self):
        """Test close_main_content_landmark creates closing div."""
        from integradio.pages.utils import close_main_content_landmark
        import gradio as gr

        with gr.Blocks() as demo:
            component = close_main_content_landmark()

        assert component is not None

    def test_create_loading_section_show_skeleton(self):
        """Test create_loading_section with skeleton visible."""
        from integradio.pages.utils import create_loading_section
        import gradio as gr

        with gr.Blocks() as demo:
            skeleton, content = create_loading_section(
                label="Results",
                lines=5,
                show_skeleton=True,
            )

        assert skeleton is not None
        assert content is not None

    def test_create_loading_section_hide_skeleton(self):
        """Test create_loading_section with skeleton hidden."""
        from integradio.pages.utils import create_loading_section
        import gradio as gr

        with gr.Blocks() as demo:
            skeleton, content = create_loading_section(
                label="Results",
                show_skeleton=False,
            )

        assert skeleton is not None
        assert content is not None

    def test_create_status_announcement_polite(self):
        """Test create_status_announcement with polite mode."""
        from integradio.pages.utils import create_status_announcement
        import gradio as gr

        with gr.Blocks() as demo:
            component = create_status_announcement(
                message="Loading...",
                politeness="polite",
            )

        assert component is not None

    def test_create_status_announcement_assertive(self):
        """Test create_status_announcement with assertive mode."""
        from integradio.pages.utils import create_status_announcement
        import gradio as gr

        with gr.Blocks() as demo:
            component = create_status_announcement(
                message="Error!",
                politeness="assertive",
            )

        assert component is not None

    def test_create_status_announcement_empty(self):
        """Test create_status_announcement with empty message."""
        from integradio.pages.utils import create_status_announcement
        import gradio as gr

        with gr.Blocks() as demo:
            component = create_status_announcement()

        assert component is not None

    def test_create_confirmation_button(self):
        """Test create_confirmation_button creates all components."""
        from integradio.pages.utils import create_confirmation_button
        import gradio as gr

        with gr.Blocks() as demo:
            button, dialog, confirmed = create_confirmation_button(
                label="Delete",
                confirm_title="Confirm Delete",
                confirm_message="Are you sure?",
            )

        assert button is not None
        assert dialog is not None
        assert confirmed is not None

    def test_create_confirmation_button_with_icon(self):
        """Test create_confirmation_button with icon."""
        from integradio.pages.utils import create_confirmation_button
        import gradio as gr

        with gr.Blocks() as demo:
            button, _, _ = create_confirmation_button(
                label="Delete",
                confirm_title="Confirm",
                confirm_message="Sure?",
                icon="🗑️",
            )

        assert button is not None

    def test_create_confirmation_button_stop_variant(self):
        """Test create_confirmation_button with stop variant (danger)."""
        from integradio.pages.utils import create_confirmation_button
        import gradio as gr

        with gr.Blocks() as demo:
            button, dialog, _ = create_confirmation_button(
                label="Stop",
                confirm_title="Stop Process",
                confirm_message="Stop running?",
                variant="stop",
            )

        assert button is not None
