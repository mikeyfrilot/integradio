"""
Settings Page - Configuration panel with organized sections.

Features:
- Grouped settings by category
- Toggle switches, dropdowns, sliders
- Save/reset functionality
- Import/export config
- Validation feedback
"""

from typing import Optional, Callable, Any
from dataclasses import dataclass, field

import gradio as gr

from ..components import semantic
from ..blocks import SemanticBlocks


@dataclass
class SettingItem:
    """Definition of a single setting."""
    key: str
    label: str
    type: str  # "text", "number", "toggle", "dropdown", "slider", "color"
    default: Any
    description: str = ""
    choices: list = field(default_factory=list)  # For dropdown
    min_value: float = 0  # For slider/number
    max_value: float = 100  # For slider/number
    step: float = 1  # For slider/number


@dataclass
class SettingsSection:
    """A group of related settings."""
    title: str
    icon: str = ""
    settings: list[SettingItem] = field(default_factory=list)


DEFAULT_SECTIONS = [
    SettingsSection(
        title="General",
        icon="⚙️",
        settings=[
            SettingItem("theme", "Theme", "dropdown", "System",
                       choices=["Light", "Dark", "System"],
                       description="Choose your preferred color theme"),
            SettingItem("language", "Language", "dropdown", "English",
                       choices=["English", "Spanish", "French", "German", "Japanese"],
                       description="Interface language"),
            SettingItem("auto_save", "Auto-save", "toggle", True,
                       description="Automatically save changes"),
        ]
    ),
    SettingsSection(
        title="Appearance",
        icon="🎨",
        settings=[
            SettingItem("font_size", "Font Size", "slider", 14,
                       min_value=10, max_value=24, step=1,
                       description="Text size in pixels"),
            SettingItem("accent_color", "Accent Color", "color", "#6366f1",
                       description="Primary accent color"),
            SettingItem("compact_mode", "Compact Mode", "toggle", False,
                       description="Reduce spacing for denser layout"),
        ]
    ),
    SettingsSection(
        title="AI Settings",
        icon="🤖",
        settings=[
            SettingItem("model", "Default Model", "dropdown", "gpt-4",
                       choices=["gpt-4", "gpt-3.5-turbo", "claude-3", "llama-3"],
                       description="Default AI model for conversations"),
            SettingItem("temperature", "Temperature", "slider", 0.7,
                       min_value=0, max_value=2, step=0.1,
                       description="Creativity level (0=focused, 2=creative)"),
            SettingItem("max_tokens", "Max Tokens", "number", 4096,
                       min_value=100, max_value=32000, step=100,
                       description="Maximum response length"),
            SettingItem("stream_responses", "Stream Responses", "toggle", True,
                       description="Show responses as they generate"),
        ]
    ),
    SettingsSection(
        title="Privacy",
        icon="🔒",
        settings=[
            SettingItem("save_history", "Save Chat History", "toggle", True,
                       description="Store conversations locally"),
            SettingItem("analytics", "Usage Analytics", "toggle", False,
                       description="Help improve the app with anonymous data"),
            SettingItem("crash_reports", "Crash Reports", "toggle", True,
                       description="Automatically send crash reports"),
        ]
    ),
    SettingsSection(
        title="Notifications",
        icon="🔔",
        settings=[
            SettingItem("enable_notifications", "Enable Notifications", "toggle", True,
                       description="Show desktop notifications"),
            SettingItem("sound_enabled", "Sound Effects", "toggle", True,
                       description="Play sounds for events"),
            SettingItem("notification_frequency", "Frequency", "dropdown", "Important Only",
                       choices=["All", "Important Only", "None"],
                       description="How often to show notifications"),
        ]
    ),
]


def create_settings_panel(
    sections: Optional[list[SettingsSection]] = None,
    on_save: Optional[Callable] = None,
    on_reset: Optional[Callable] = None,
) -> dict[str, Any]:
    """
    Create a settings panel with semantic-tracked components.

    Args:
        sections: List of SettingsSection definitions
        on_save: Callback when settings are saved
        on_reset: Callback when settings are reset

    Returns:
        Dict of component references
    """
    sections = sections or DEFAULT_SECTIONS
    components = {}

    # Header
    components["title"] = semantic(
        gr.Markdown("# ⚙️ Settings"),
        intent="displays settings page title",
        tags=["header"],
    )

    # Status message
    components["status"] = semantic(
        gr.Markdown("", visible=False),
        intent="shows save/error status messages",
        tags=["feedback", "status"],
    )

    # Create each section
    for section in sections:
        section_key = section.title.lower().replace(" ", "_")
        icon = f"{section.icon} " if section.icon else ""

        with gr.Accordion(f"{icon}{section.title}", open=True):
            for setting in section.settings:
                key = f"{section_key}_{setting.key}"

                with gr.Row():
                    with gr.Column(scale=2):
                        # Description
                        semantic(
                            gr.Markdown(f"**{setting.label}**\n\n{setting.description}"),
                            intent=f"describes {setting.label} setting purpose",
                            tags=["label", "description"],
                        )

                    with gr.Column(scale=3):
                        # Create appropriate input based on type
                        if setting.type == "toggle":
                            components[key] = semantic(
                                gr.Checkbox(
                                    value=setting.default,
                                    label="",
                                    elem_id=f"setting-{key}",
                                ),
                                intent=f"toggles {setting.label} on or off",
                                tags=["setting", "toggle", section_key],
                            )

                        elif setting.type == "dropdown":
                            components[key] = semantic(
                                gr.Dropdown(
                                    choices=setting.choices,
                                    value=setting.default,
                                    label="",
                                    elem_id=f"setting-{key}",
                                ),
                                intent=f"selects {setting.label} from options",
                                tags=["setting", "dropdown", section_key],
                            )

                        elif setting.type == "slider":
                            components[key] = semantic(
                                gr.Slider(
                                    minimum=setting.min_value,
                                    maximum=setting.max_value,
                                    value=setting.default,
                                    step=setting.step,
                                    label="",
                                    elem_id=f"setting-{key}",
                                ),
                                intent=f"adjusts {setting.label} value",
                                tags=["setting", "slider", section_key],
                            )

                        elif setting.type == "number":
                            components[key] = semantic(
                                gr.Number(
                                    value=setting.default,
                                    minimum=setting.min_value,
                                    maximum=setting.max_value,
                                    label="",
                                    elem_id=f"setting-{key}",
                                ),
                                intent=f"sets {setting.label} numeric value",
                                tags=["setting", "number", section_key],
                            )

                        elif setting.type == "color":
                            components[key] = semantic(
                                gr.ColorPicker(
                                    value=setting.default,
                                    label="",
                                    elem_id=f"setting-{key}",
                                ),
                                intent=f"picks color for {setting.label}",
                                tags=["setting", "color", section_key],
                            )

                        else:  # text
                            components[key] = semantic(
                                gr.Textbox(
                                    value=setting.default,
                                    label="",
                                    elem_id=f"setting-{key}",
                                ),
                                intent=f"enters text for {setting.label}",
                                tags=["setting", "text", section_key],
                            )

    # Action buttons
    gr.Markdown("---")
    with gr.Row():
        components["save_btn"] = semantic(
            gr.Button("Save Settings", variant="primary"),
            intent="saves all current settings",
            tags=["action", "primary", "save"],
        )

        components["reset_btn"] = semantic(
            gr.Button("Reset to Defaults", variant="secondary"),
            intent="resets all settings to default values",
            tags=["action", "reset", "destructive"],
        )

        components["export_btn"] = semantic(
            gr.Button("Export", variant="secondary"),
            intent="exports settings as downloadable file",
            tags=["action", "export"],
        )

        components["import_btn"] = semantic(
            gr.UploadButton("Import", variant="secondary"),
            intent="imports settings from uploaded file",
            tags=["action", "import"],
        )

    # Wire up handlers
    if on_save:
        setting_inputs = [v for k, v in components.items()
                        if k not in ["title", "status", "save_btn", "reset_btn",
                                    "export_btn", "import_btn"]]
        components["save_btn"].click(
            fn=on_save,
            inputs=setting_inputs,
            outputs=[components["status"]],
        )

    return components


class SettingsPage:
    """
    Complete settings page with SemanticBlocks integration.

    Usage:
        page = SettingsPage(sections=my_sections, on_save=save_handler)
        page.launch()
    """

    def __init__(
        self,
        title: str = "Settings",
        sections: Optional[list[SettingsSection]] = None,
        on_save: Optional[Callable] = None,
        on_reset: Optional[Callable] = None,
    ):
        self.title = title
        self.sections = sections
        self.on_save = on_save
        self.on_reset = on_reset
        self.components: dict[str, Any] = {}
        self.blocks: Optional[SemanticBlocks] = None

    def build(self) -> SemanticBlocks:
        """Build the settings interface."""
        self.blocks = SemanticBlocks(
            title=self.title,
            theme=gr.themes.Soft(),
        )

        with self.blocks:
            self.components = create_settings_panel(
                sections=self.sections,
                on_save=self.on_save,
                on_reset=self.on_reset,
            )

        return self.blocks

    def launch(self, **kwargs) -> None:
        """Build and launch the settings interface."""
        if self.blocks is None:
            self.build()
        self.blocks.launch(**kwargs)

    @staticmethod
    def render(
        sections: Optional[list[SettingsSection]] = None,
        on_save: Optional[Callable] = None,
    ) -> dict[str, Any]:
        """Render settings into existing Blocks context."""
        return create_settings_panel(sections=sections, on_save=on_save)
