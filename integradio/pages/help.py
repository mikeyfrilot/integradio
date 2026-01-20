"""
Help/Documentation Page - Searchable help center with FAQ.

Features:
- Searchable documentation
- FAQ accordion
- Category navigation
- Contact support form
- Video tutorials section
"""

from typing import Optional, Callable, Any
from dataclasses import dataclass, field

import gradio as gr

from ..components import semantic
from ..blocks import SemanticBlocks


@dataclass
class FAQItem:
    """A frequently asked question."""
    question: str
    answer: str
    category: str = "General"


@dataclass
class HelpArticle:
    """A help article."""
    title: str
    content: str
    category: str
    tags: list[str] = field(default_factory=list)


@dataclass
class HelpConfig:
    """Configuration for help center."""
    title: str = "Help Center"
    subtitle: str = "How can we help you today?"
    categories: list[str] = field(default_factory=lambda: [
        "Getting Started",
        "Account",
        "Features",
        "Billing",
        "Troubleshooting",
    ])
    faqs: list[FAQItem] = field(default_factory=lambda: [
        FAQItem("How do I get started?", "Follow our quick start guide to begin.", "Getting Started"),
        FAQItem("How do I reset my password?", "Click 'Forgot Password' on the login page.", "Account"),
        FAQItem("What payment methods do you accept?", "We accept all major credit cards and PayPal.", "Billing"),
        FAQItem("How do I contact support?", "Use the contact form below or email support@example.com.", "General"),
    ])
    show_contact_form: bool = True
    show_video_section: bool = True
    support_email: str = "support@example.com"


def create_help_center(
    config: Optional[HelpConfig] = None,
    on_search: Optional[Callable] = None,
    on_contact_submit: Optional[Callable] = None,
    articles: Optional[list[HelpArticle]] = None,
) -> dict[str, Any]:
    """
    Create a help center with semantic-tracked components.

    Args:
        config: Help center configuration
        on_search: Search handler
        on_contact_submit: Contact form submission handler
        articles: List of help articles

    Returns:
        Dict of component references
    """
    config = config or HelpConfig()
    components = {}

    # Header
    components["title"] = semantic(
        gr.Markdown(f"# 📚 {config.title}"),
        intent="displays help center page title",
        tags=["header"],
    )

    components["subtitle"] = semantic(
        gr.Markdown(f"*{config.subtitle}*"),
        intent="displays help center subtitle",
        tags=["header", "subtitle"],
    )

    # Search Bar (prominent)
    with gr.Row():
        with gr.Column(scale=5):
            components["search"] = semantic(
                gr.Textbox(
                    placeholder="Search for help articles, FAQs, guides...",
                    label="",
                    elem_id="help-search",
                    show_label=False,
                    lines=1,
                ),
                intent="searches help documentation by keywords",
                tags=["search", "primary-input"],
            )
        with gr.Column(scale=1):
            components["search_btn"] = semantic(
                gr.Button("Search", variant="primary"),
                intent="triggers help documentation search",
                tags=["action", "search"],
            )

    # Search Results (hidden by default)
    components["search_results"] = semantic(
        gr.Markdown("", visible=False),
        intent="displays search results from help documentation",
        tags=["results", "search"],
    )

    # Quick Links / Categories
    gr.Markdown("### 🏷️ Browse by Category")
    with gr.Row():
        for i, category in enumerate(config.categories):
            components[f"cat_{i}"] = semantic(
                gr.Button(category, variant="secondary", size="sm"),
                intent=f"filters help content to {category} category",
                tags=["filter", "category", "navigation"],
            )

    gr.Markdown("---")

    # Main Content in Tabs
    with gr.Tabs():
        # FAQ Tab
        with gr.Tab("❓ FAQ"):
            components["faq_title"] = semantic(
                gr.Markdown("## Frequently Asked Questions"),
                intent="introduces FAQ section",
                tags=["header", "section"],
            )

            # Group FAQs by category
            faq_by_cat: dict[str, list[FAQItem]] = {}
            for faq in config.faqs:
                if faq.category not in faq_by_cat:
                    faq_by_cat[faq.category] = []
                faq_by_cat[faq.category].append(faq)

            for cat, faqs in faq_by_cat.items():
                gr.Markdown(f"### {cat}")
                for j, faq in enumerate(faqs):
                    with gr.Accordion(faq.question, open=False):
                        key = f"faq_{cat}_{j}"
                        components[key] = semantic(
                            gr.Markdown(faq.answer),
                            intent=f"answers question: {faq.question[:50]}",
                            tags=["faq", "answer", cat.lower()],
                        )

        # Guides Tab
        with gr.Tab("📖 Guides"):
            components["guides_title"] = semantic(
                gr.Markdown("## Getting Started Guides"),
                intent="introduces guides section",
                tags=["header", "section"],
            )

            guides = [
                ("🚀 Quick Start", "Get up and running in 5 minutes"),
                ("⚙️ Configuration", "Customize your setup"),
                ("🔌 Integrations", "Connect with other tools"),
                ("🔒 Security Best Practices", "Keep your account safe"),
            ]

            for i, (title, desc) in enumerate(guides):
                with gr.Accordion(title, open=False):
                    components[f"guide_{i}"] = semantic(
                        gr.Markdown(f"**{title}**\n\n{desc}\n\n*Full guide content would go here...*"),
                        intent=f"provides {title} documentation",
                        tags=["guide", "documentation"],
                    )

        # Video Tutorials Tab
        if config.show_video_section:
            with gr.Tab("🎬 Videos"):
                components["videos_title"] = semantic(
                    gr.Markdown("## Video Tutorials"),
                    intent="introduces video tutorials section",
                    tags=["header", "section"],
                )

                gr.Markdown("*Video tutorials would be embedded here*")

                videos = [
                    ("Introduction to the Platform", "5:30"),
                    ("Advanced Features Deep Dive", "12:45"),
                    ("Tips & Tricks", "8:20"),
                ]

                with gr.Row():
                    for i, (title, duration) in enumerate(videos):
                        with gr.Column():
                            components[f"video_{i}"] = semantic(
                                gr.Markdown(f"### 🎥 {title}\n\n*Duration: {duration}*"),
                                intent=f"links to {title} tutorial video",
                                tags=["video", "tutorial"],
                            )

        # Contact Tab
        if config.show_contact_form:
            with gr.Tab("✉️ Contact Support"):
                components["contact_title"] = semantic(
                    gr.Markdown("## Get in Touch"),
                    intent="introduces contact support form",
                    tags=["header", "section"],
                )

                components["contact_info"] = semantic(
                    gr.Markdown(
                        f"Can't find what you're looking for? Send us a message and we'll get back to you within 24 hours.\n\n"
                        f"**Email:** {config.support_email}"
                    ),
                    intent="displays support contact information",
                    tags=["info", "contact"],
                )

                with gr.Row():
                    with gr.Column():
                        components["contact_name"] = semantic(
                            gr.Textbox(label="Your Name", elem_id="contact-name"),
                            intent="collects user name for support request",
                            tags=["input", "contact", "required"],
                        )

                    with gr.Column():
                        components["contact_email"] = semantic(
                            gr.Textbox(label="Email Address", elem_id="contact-email"),
                            intent="collects user email for support response",
                            tags=["input", "contact", "required", "email"],
                        )

                components["contact_category"] = semantic(
                    gr.Dropdown(
                        choices=["General Question", "Bug Report", "Feature Request", "Billing Issue", "Other"],
                        label="Category",
                        value="General Question",
                    ),
                    intent="categorizes support request type",
                    tags=["input", "contact", "category"],
                )

                components["contact_message"] = semantic(
                    gr.Textbox(
                        label="Message",
                        lines=5,
                        placeholder="Describe your issue or question...",
                        elem_id="contact-message",
                    ),
                    intent="collects detailed support request message",
                    tags=["input", "contact", "required", "message"],
                )

                components["contact_submit"] = semantic(
                    gr.Button("Send Message", variant="primary"),
                    intent="submits support request form",
                    tags=["action", "submit", "contact"],
                )

                components["contact_status"] = semantic(
                    gr.Markdown("", visible=False),
                    intent="displays contact form submission status",
                    tags=["feedback", "status"],
                )

                # Wire up contact form
                if on_contact_submit:
                    components["contact_submit"].click(
                        fn=on_contact_submit,
                        inputs=[
                            components["contact_name"],
                            components["contact_email"],
                            components["contact_category"],
                            components["contact_message"],
                        ],
                        outputs=[components["contact_status"]],
                    )

    # Wire up search
    if on_search:
        components["search_btn"].click(
            fn=on_search,
            inputs=[components["search"]],
            outputs=[components["search_results"]],
        )
        components["search"].submit(
            fn=on_search,
            inputs=[components["search"]],
            outputs=[components["search_results"]],
        )

    return components


class HelpPage:
    """
    Complete help center page with SemanticBlocks integration.

    Usage:
        page = HelpPage(
            title="Help Center",
            faqs=[FAQItem(...), ...],
        )
        page.launch()
    """

    def __init__(
        self,
        title: str = "Help Center",
        on_search: Optional[Callable] = None,
        on_contact_submit: Optional[Callable] = None,
        **config_kwargs,
    ):
        self.config = HelpConfig(title=title, **config_kwargs)
        self.on_search = on_search
        self.on_contact_submit = on_contact_submit
        self.components: dict[str, Any] = {}
        self.blocks: Optional[SemanticBlocks] = None

    def build(self) -> SemanticBlocks:
        """Build the help center."""
        self.blocks = SemanticBlocks(
            title=self.config.title,
            theme=gr.themes.Soft(),
        )

        with self.blocks:
            self.components = create_help_center(
                config=self.config,
                on_search=self.on_search,
                on_contact_submit=self.on_contact_submit,
            )

        return self.blocks

    def launch(self, **kwargs) -> None:
        """Build and launch the help center."""
        if self.blocks is None:
            self.build()
        self.blocks.launch(**kwargs)

    @staticmethod
    def render(config: Optional[HelpConfig] = None) -> dict[str, Any]:
        """Render help center into existing Blocks context."""
        return create_help_center(config=config)
