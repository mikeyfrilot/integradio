"""
Hero/Landing Page - Eye-catching landing page with CTAs.

Features:
- Large hero section with title/subtitle
- Feature highlights
- Call-to-action buttons
- Testimonials/social proof
- Footer with links
"""

from typing import Optional, Callable, Any
from dataclasses import dataclass, field

import gradio as gr

from ..components import semantic
from ..blocks import SemanticBlocks


@dataclass
class Feature:
    """A feature to highlight."""
    icon: str
    title: str
    description: str


@dataclass
class Testimonial:
    """A user testimonial."""
    quote: str
    author: str
    role: str = ""
    avatar: str = ""


@dataclass
class HeroConfig:
    """Configuration for hero/landing page."""
    title: str = "Welcome to Our App"
    subtitle: str = "The best solution for your needs"
    description: str = ""
    primary_cta: str = "Get Started"
    secondary_cta: str = "Learn More"
    hero_image: Optional[str] = None
    features: list[Feature] = field(default_factory=lambda: [
        Feature("🚀", "Fast", "Lightning-fast performance"),
        Feature("🔒", "Secure", "Enterprise-grade security"),
        Feature("🎨", "Beautiful", "Stunning user interface"),
        Feature("🤖", "AI-Powered", "Intelligent automation"),
    ])
    testimonials: list[Testimonial] = field(default_factory=list)
    footer_links: dict[str, str] = field(default_factory=lambda: {
        "Documentation": "#docs",
        "GitHub": "#github",
        "Discord": "#discord",
        "Twitter": "#twitter",
    })


def create_hero_section(
    config: Optional[HeroConfig] = None,
    on_primary_click: Optional[Callable] = None,
    on_secondary_click: Optional[Callable] = None,
) -> dict[str, Any]:
    """
    Create a hero/landing page with semantic-tracked components.

    Args:
        config: Hero page configuration
        on_primary_click: Primary CTA click handler
        on_secondary_click: Secondary CTA click handler

    Returns:
        Dict of component references
    """
    config = config or HeroConfig()
    components = {}

    # Hero Section
    with gr.Column(elem_classes=["hero-section"]):
        components["title"] = semantic(
            gr.Markdown(
                f"# {config.title}",
                elem_id="hero-title",
            ),
            intent="displays main landing page headline",
            tags=["header", "hero", "branding"],
        )

        components["subtitle"] = semantic(
            gr.Markdown(
                f"### {config.subtitle}",
                elem_id="hero-subtitle",
            ),
            intent="displays supporting tagline under headline",
            tags=["header", "hero", "tagline"],
        )

        if config.description:
            components["description"] = semantic(
                gr.Markdown(config.description),
                intent="provides detailed description of product/service",
                tags=["content", "hero"],
            )

        # CTA Buttons
        with gr.Row():
            components["primary_cta"] = semantic(
                gr.Button(
                    config.primary_cta,
                    variant="primary",
                    size="lg",
                    elem_id="primary-cta",
                ),
                intent="primary call-to-action for user conversion",
                tags=["cta", "primary", "conversion"],
            )

            components["secondary_cta"] = semantic(
                gr.Button(
                    config.secondary_cta,
                    variant="secondary",
                    size="lg",
                    elem_id="secondary-cta",
                ),
                intent="secondary action for users wanting more info",
                tags=["cta", "secondary"],
            )

        # Hero image
        if config.hero_image:
            components["hero_image"] = semantic(
                gr.Image(
                    value=config.hero_image,
                    show_label=False,
                    container=False,
                    elem_id="hero-image",
                ),
                intent="displays main visual for landing page",
                tags=["media", "hero"],
            )

    # Divider
    gr.Markdown("---")

    # Features Section
    if config.features:
        components["features_title"] = semantic(
            gr.Markdown("## ✨ Features"),
            intent="introduces features section",
            tags=["header", "section"],
        )

        # Feature grid
        feature_cols = min(len(config.features), 4)
        with gr.Row():
            for i, feature in enumerate(config.features):
                with gr.Column():
                    components[f"feature_{i}"] = semantic(
                        gr.Markdown(
                            f"### {feature.icon} {feature.title}\n\n{feature.description}"
                        ),
                        intent=f"highlights {feature.title} feature benefit",
                        tags=["feature", "benefit"],
                    )

    # Divider
    gr.Markdown("---")

    # Testimonials Section
    if config.testimonials:
        components["testimonials_title"] = semantic(
            gr.Markdown("## 💬 What People Say"),
            intent="introduces testimonials section",
            tags=["header", "section", "social-proof"],
        )

        with gr.Row():
            for i, testimonial in enumerate(config.testimonials):
                with gr.Column():
                    quote_md = f'> "{testimonial.quote}"\n\n'
                    quote_md += f"**{testimonial.author}**"
                    if testimonial.role:
                        quote_md += f"\n\n*{testimonial.role}*"

                    components[f"testimonial_{i}"] = semantic(
                        gr.Markdown(quote_md),
                        intent=f"displays testimonial from {testimonial.author}",
                        tags=["testimonial", "social-proof"],
                    )

    # Stats Section (optional - hardcoded example)
    gr.Markdown("---")
    components["stats_title"] = semantic(
        gr.Markdown("## 📊 By the Numbers"),
        intent="introduces statistics section",
        tags=["header", "section"],
    )

    with gr.Row():
        stats = [
            ("10K+", "Active Users"),
            ("99.9%", "Uptime"),
            ("50M+", "Requests/Day"),
            ("4.9⭐", "User Rating"),
        ]
        for i, (value, label) in enumerate(stats):
            with gr.Column():
                components[f"stat_{i}"] = semantic(
                    gr.Markdown(f"### {value}\n\n{label}"),
                    intent=f"displays {label} statistic",
                    tags=["stat", "social-proof"],
                )

    # Footer
    gr.Markdown("---")
    with gr.Row():
        link_md = " | ".join(
            f"[{name}]({url})" for name, url in config.footer_links.items()
        )
        components["footer"] = semantic(
            gr.Markdown(f"**Links:** {link_md}\n\n© 2026 Your Company. All rights reserved."),
            intent="displays footer navigation and copyright",
            tags=["footer", "navigation"],
        )

    # Wire up CTAs
    if on_primary_click:
        components["primary_cta"].click(fn=on_primary_click)
    if on_secondary_click:
        components["secondary_cta"].click(fn=on_secondary_click)

    return components


class HeroPage:
    """
    Complete hero/landing page with SemanticBlocks integration.

    Usage:
        page = HeroPage(
            title="My Awesome App",
            subtitle="Built for the future",
            features=[...],
        )
        page.launch()
    """

    def __init__(
        self,
        title: str = "Welcome",
        subtitle: str = "The best solution",
        on_primary_click: Optional[Callable] = None,
        on_secondary_click: Optional[Callable] = None,
        **config_kwargs,
    ):
        self.config = HeroConfig(title=title, subtitle=subtitle, **config_kwargs)
        self.on_primary_click = on_primary_click
        self.on_secondary_click = on_secondary_click
        self.components: dict[str, Any] = {}
        self.blocks: Optional[SemanticBlocks] = None

    def build(self) -> SemanticBlocks:
        """Build the hero page."""
        self.blocks = SemanticBlocks(
            title=self.config.title,
            theme=gr.themes.Soft(),
        )

        with self.blocks:
            self.components = create_hero_section(
                config=self.config,
                on_primary_click=self.on_primary_click,
                on_secondary_click=self.on_secondary_click,
            )

        return self.blocks

    def launch(self, **kwargs) -> None:
        """Build and launch the hero page."""
        if self.blocks is None:
            self.build()
        self.blocks.launch(**kwargs)

    @staticmethod
    def render(config: Optional[HeroConfig] = None) -> dict[str, Any]:
        """Render hero section into existing Blocks context."""
        return create_hero_section(config=config)
