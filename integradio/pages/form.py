"""
Form/Wizard Page - Multi-step forms with validation.

Features:
- Multi-step wizard flow
- Progress indicator
- Field validation
- Conditional fields
- Review step before submit
"""

from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

import gradio as gr

from ..components import semantic
from ..blocks import SemanticBlocks


class FieldType(Enum):
    TEXT = "text"
    EMAIL = "email"
    PASSWORD = "password"
    NUMBER = "number"
    TEXTAREA = "textarea"
    DROPDOWN = "dropdown"
    RADIO = "radio"
    CHECKBOX = "checkbox"
    DATE = "date"
    FILE = "file"


@dataclass
class FormField:
    """Definition of a form field."""
    name: str
    label: str
    type: FieldType = FieldType.TEXT
    required: bool = False
    placeholder: str = ""
    choices: list[str] = field(default_factory=list)
    default: Any = None
    validation_regex: str = ""
    help_text: str = ""


@dataclass
class FormStep:
    """A step in a multi-step form."""
    title: str
    description: str = ""
    fields: list[FormField] = field(default_factory=list)
    icon: str = ""


@dataclass
class FormConfig:
    """Configuration for form/wizard."""
    title: str = "Registration"
    description: str = "Complete the form below"
    steps: list[FormStep] = field(default_factory=lambda: [
        FormStep(
            title="Personal Info",
            description="Tell us about yourself",
            icon="👤",
            fields=[
                FormField("first_name", "First Name", required=True, placeholder="John"),
                FormField("last_name", "Last Name", required=True, placeholder="Doe"),
                FormField("email", "Email Address", FieldType.EMAIL, required=True,
                         placeholder="john@example.com"),
                FormField("phone", "Phone Number", placeholder="+1 (555) 000-0000"),
            ]
        ),
        FormStep(
            title="Account Setup",
            description="Create your account",
            icon="🔐",
            fields=[
                FormField("username", "Username", required=True, placeholder="johndoe"),
                FormField("password", "Password", FieldType.PASSWORD, required=True),
                FormField("confirm_password", "Confirm Password", FieldType.PASSWORD, required=True),
            ]
        ),
        FormStep(
            title="Preferences",
            description="Customize your experience",
            icon="⚙️",
            fields=[
                FormField("plan", "Select Plan", FieldType.DROPDOWN,
                         choices=["Free", "Pro", "Enterprise"], default="Free"),
                FormField("newsletter", "Subscribe to newsletter", FieldType.CHECKBOX, default=True),
                FormField("notifications", "Notification Preference", FieldType.RADIO,
                         choices=["All", "Important Only", "None"], default="Important Only"),
            ]
        ),
    ])
    show_progress: bool = True
    allow_skip: bool = False
    submit_text: str = "Submit"


def create_form_wizard(
    config: Optional[FormConfig] = None,
    on_submit: Optional[Callable] = None,
    on_step_change: Optional[Callable] = None,
) -> dict[str, Any]:
    """
    Create a multi-step form wizard with semantic-tracked components.

    Args:
        config: Form configuration
        on_submit: Form submission handler
        on_step_change: Step change handler

    Returns:
        Dict of component references
    """
    config = config or FormConfig()
    components = {}
    num_steps = len(config.steps)

    # Header
    components["title"] = semantic(
        gr.Markdown(f"# 📝 {config.title}"),
        intent="displays form wizard title",
        tags=["header"],
    )

    if config.description:
        components["description"] = semantic(
            gr.Markdown(config.description),
            intent="displays form description",
            tags=["header", "description"],
        )

    # Progress Indicator
    if config.show_progress:
        def make_progress_md(current_step: int) -> str:
            steps_display = []
            for i, step in enumerate(config.steps):
                icon = step.icon or f"{i+1}"
                if i < current_step:
                    steps_display.append(f"✅ ~~{icon} {step.title}~~")
                elif i == current_step:
                    steps_display.append(f"**🔵 {icon} {step.title}**")
                else:
                    steps_display.append(f"⚪ {icon} {step.title}")
            return " → ".join(steps_display)

        components["progress"] = semantic(
            gr.Markdown(make_progress_md(0)),
            intent="shows current progress through form steps",
            tags=["progress", "navigation"],
        )

    gr.Markdown("---")

    # Step State
    components["current_step"] = gr.State(0)

    # Create all step containers (only one visible at a time)
    for step_idx, step in enumerate(config.steps):
        with gr.Column(visible=(step_idx == 0)) as step_col:
            components[f"step_{step_idx}_col"] = step_col

            # Step header
            icon = f"{step.icon} " if step.icon else ""
            components[f"step_{step_idx}_title"] = semantic(
                gr.Markdown(f"## {icon}{step.title}"),
                intent=f"displays step {step_idx + 1} title: {step.title}",
                tags=["header", "step", f"step_{step_idx}"],
            )

            if step.description:
                components[f"step_{step_idx}_desc"] = semantic(
                    gr.Markdown(f"*{step.description}*"),
                    intent=f"describes step {step_idx + 1} purpose",
                    tags=["description", "step"],
                )

            # Step fields
            for field in step.fields:
                field_key = f"field_{field.name}"

                # Help text
                if field.help_text:
                    gr.Markdown(f"*{field.help_text}*")

                # Create appropriate input
                if field.type == FieldType.TEXT:
                    components[field_key] = semantic(
                        gr.Textbox(
                            label=f"{field.label}{'*' if field.required else ''}",
                            placeholder=field.placeholder,
                            value=field.default,
                            elem_id=f"form-{field.name}",
                        ),
                        intent=f"collects {field.label} input from user",
                        tags=["input", "text", f"step_{step_idx}"],
                    )

                elif field.type == FieldType.EMAIL:
                    components[field_key] = semantic(
                        gr.Textbox(
                            label=f"{field.label}{'*' if field.required else ''}",
                            placeholder=field.placeholder,
                            value=field.default,
                            elem_id=f"form-{field.name}",
                        ),
                        intent=f"collects {field.label} email address",
                        tags=["input", "email", f"step_{step_idx}"],
                    )

                elif field.type == FieldType.PASSWORD:
                    components[field_key] = semantic(
                        gr.Textbox(
                            label=f"{field.label}{'*' if field.required else ''}",
                            type="password",
                            value=field.default,
                            elem_id=f"form-{field.name}",
                        ),
                        intent=f"collects {field.label} securely",
                        tags=["input", "password", "sensitive", f"step_{step_idx}"],
                    )

                elif field.type == FieldType.NUMBER:
                    components[field_key] = semantic(
                        gr.Number(
                            label=f"{field.label}{'*' if field.required else ''}",
                            value=field.default,
                            elem_id=f"form-{field.name}",
                        ),
                        intent=f"collects numeric {field.label}",
                        tags=["input", "number", f"step_{step_idx}"],
                    )

                elif field.type == FieldType.TEXTAREA:
                    components[field_key] = semantic(
                        gr.Textbox(
                            label=f"{field.label}{'*' if field.required else ''}",
                            placeholder=field.placeholder,
                            lines=4,
                            value=field.default,
                            elem_id=f"form-{field.name}",
                        ),
                        intent=f"collects multi-line {field.label}",
                        tags=["input", "textarea", f"step_{step_idx}"],
                    )

                elif field.type == FieldType.DROPDOWN:
                    components[field_key] = semantic(
                        gr.Dropdown(
                            label=f"{field.label}{'*' if field.required else ''}",
                            choices=field.choices,
                            value=field.default,
                            elem_id=f"form-{field.name}",
                        ),
                        intent=f"selects {field.label} from options",
                        tags=["input", "dropdown", f"step_{step_idx}"],
                    )

                elif field.type == FieldType.RADIO:
                    components[field_key] = semantic(
                        gr.Radio(
                            label=f"{field.label}{'*' if field.required else ''}",
                            choices=field.choices,
                            value=field.default,
                            elem_id=f"form-{field.name}",
                        ),
                        intent=f"selects single {field.label} option",
                        tags=["input", "radio", f"step_{step_idx}"],
                    )

                elif field.type == FieldType.CHECKBOX:
                    components[field_key] = semantic(
                        gr.Checkbox(
                            label=field.label,
                            value=field.default or False,
                            elem_id=f"form-{field.name}",
                        ),
                        intent=f"toggles {field.label} option",
                        tags=["input", "checkbox", f"step_{step_idx}"],
                    )

                elif field.type == FieldType.FILE:
                    components[field_key] = semantic(
                        gr.File(
                            label=f"{field.label}{'*' if field.required else ''}",
                            elem_id=f"form-{field.name}",
                        ),
                        intent=f"uploads file for {field.label}",
                        tags=["input", "file", f"step_{step_idx}"],
                    )

    gr.Markdown("---")

    # Validation message
    components["validation_msg"] = semantic(
        gr.Markdown("", visible=False),
        intent="displays form validation errors",
        tags=["feedback", "validation", "error"],
    )

    # Navigation buttons
    with gr.Row():
        components["prev_btn"] = semantic(
            gr.Button("← Previous", variant="secondary", visible=False),
            intent="navigates to previous form step",
            tags=["navigation", "previous"],
        )

        components["next_btn"] = semantic(
            gr.Button("Next →", variant="primary"),
            intent="navigates to next form step",
            tags=["navigation", "next"],
        )

        components["submit_btn"] = semantic(
            gr.Button(config.submit_text, variant="primary", visible=False),
            intent="submits completed form",
            tags=["action", "submit", "primary"],
        )

    # Step navigation logic
    def go_to_step(current: int, direction: int):
        new_step = current + direction
        new_step = max(0, min(new_step, num_steps - 1))

        # Update visibility
        updates = []
        for i in range(num_steps):
            updates.append(gr.update(visible=(i == new_step)))

        # Update progress
        progress_md = make_progress_md(new_step) if config.show_progress else ""

        # Update button visibility
        show_prev = new_step > 0
        show_next = new_step < num_steps - 1
        show_submit = new_step == num_steps - 1

        return (
            new_step,
            *updates,
            gr.update(value=progress_md),
            gr.update(visible=show_prev),
            gr.update(visible=show_next),
            gr.update(visible=show_submit),
        )

    step_outputs = [components["current_step"]]
    step_outputs += [components[f"step_{i}_col"] for i in range(num_steps)]
    step_outputs += [
        components["progress"],
        components["prev_btn"],
        components["next_btn"],
        components["submit_btn"],
    ]

    components["next_btn"].click(
        fn=lambda curr: go_to_step(curr, 1),
        inputs=[components["current_step"]],
        outputs=step_outputs,
    )

    components["prev_btn"].click(
        fn=lambda curr: go_to_step(curr, -1),
        inputs=[components["current_step"]],
        outputs=step_outputs,
    )

    # Wire up submit
    if on_submit:
        # Gather all field inputs
        field_inputs = [
            components[f"field_{field.name}"]
            for step in config.steps
            for field in step.fields
        ]

        components["submit_btn"].click(
            fn=on_submit,
            inputs=field_inputs,
            outputs=[components["validation_msg"]],
        )

    return components


class FormPage:
    """
    Complete form wizard page with SemanticBlocks integration.

    Usage:
        page = FormPage(
            title="Registration",
            steps=[FormStep(...), ...],
            on_submit=handle_submit,
        )
        page.launch()
    """

    def __init__(
        self,
        title: str = "Form",
        steps: Optional[list[FormStep]] = None,
        on_submit: Optional[Callable] = None,
        **config_kwargs,
    ):
        self.config = FormConfig(title=title, **config_kwargs)
        if steps:
            self.config.steps = steps
        self.on_submit = on_submit
        self.components: dict[str, Any] = {}
        self.blocks: Optional[SemanticBlocks] = None

    def build(self) -> SemanticBlocks:
        """Build the form wizard."""
        self.blocks = SemanticBlocks(
            title=self.config.title,
            theme=gr.themes.Soft(),
        )

        with self.blocks:
            self.components = create_form_wizard(
                config=self.config,
                on_submit=self.on_submit,
            )

        return self.blocks

    def launch(self, **kwargs) -> None:
        """Build and launch the form wizard."""
        if self.blocks is None:
            self.build()
        self.blocks.launch(**kwargs)

    @staticmethod
    def render(config: Optional[FormConfig] = None) -> dict[str, Any]:
        """Render form wizard into existing Blocks context."""
        return create_form_wizard(config=config)
