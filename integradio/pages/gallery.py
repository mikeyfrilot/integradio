"""
Gallery Page - Image/media gallery with filtering and lightbox.

Features:
- Grid layout with responsive columns
- Category filtering
- Search functionality
- Lightbox preview
- Upload integration
- Metadata display
"""

from typing import Optional, Callable, Any
from dataclasses import dataclass, field

import gradio as gr

from ..components import semantic
from ..blocks import SemanticBlocks


@dataclass
class GalleryConfig:
    """Configuration for gallery interface."""
    title: str = "Gallery"
    columns: int = 4
    height: int = 600
    allow_upload: bool = True
    show_metadata: bool = True
    show_download: bool = True
    categories: list[str] = field(default_factory=lambda: ["All", "Photos", "AI Generated", "Favorites"])
    preview_size: tuple[int, int] = (800, 600)


def create_gallery_grid(
    config: Optional[GalleryConfig] = None,
    on_select: Optional[Callable] = None,
    on_upload: Optional[Callable] = None,
    initial_images: Optional[list] = None,
) -> dict[str, Any]:
    """
    Create a gallery interface with semantic-tracked components.

    Args:
        config: Gallery configuration
        on_select: Callback when image is selected
        on_upload: Callback when images are uploaded
        initial_images: Initial images to display

    Returns:
        Dict of component references
    """
    config = config or GalleryConfig()
    components = {}

    # Header
    components["title"] = semantic(
        gr.Markdown(f"# 🖼️ {config.title}"),
        intent="displays gallery page title",
        tags=["header"],
    )

    # Toolbar
    with gr.Row():
        with gr.Column(scale=3):
            components["search"] = semantic(
                gr.Textbox(
                    placeholder="Search images...",
                    label="",
                    elem_id="gallery-search",
                    show_label=False,
                ),
                intent="filters gallery by search text",
                tags=["filter", "search"],
            )

        with gr.Column(scale=2):
            components["category"] = semantic(
                gr.Dropdown(
                    choices=config.categories,
                    value="All",
                    label="Category",
                    elem_id="gallery-category",
                ),
                intent="filters gallery by category",
                tags=["filter", "category"],
            )

        with gr.Column(scale=1):
            components["sort"] = semantic(
                gr.Dropdown(
                    choices=["Newest", "Oldest", "Name A-Z", "Name Z-A"],
                    value="Newest",
                    label="Sort",
                ),
                intent="changes gallery sort order",
                tags=["filter", "sort"],
            )

        if config.allow_upload:
            with gr.Column(scale=1):
                components["upload_btn"] = semantic(
                    gr.UploadButton(
                        "Upload",
                        file_types=["image"],
                        file_count="multiple",
                        variant="primary",
                    ),
                    intent="uploads new images to gallery",
                    tags=["action", "upload"],
                )

    # Stats row
    components["stats"] = semantic(
        gr.Markdown("Showing **0** images"),
        intent="displays count of visible images",
        tags=["status", "count"],
    )

    # Main gallery grid
    with gr.Row():
        with gr.Column(scale=3):
            components["gallery"] = semantic(
                gr.Gallery(
                    value=initial_images,
                    label="",
                    columns=config.columns,
                    height=config.height,
                    object_fit="cover",
                    show_label=False,
                    elem_id="main-gallery",
                ),
                intent="displays image grid for browsing",
                tags=["display", "grid", "primary"],
            )

        # Preview panel (shown when image selected)
        with gr.Column(scale=1, visible=True) as preview_col:
            components["preview_col"] = preview_col

            components["preview_image"] = semantic(
                gr.Image(
                    label="Preview",
                    height=300,
                    elem_id="image-preview",
                ),
                intent="shows enlarged preview of selected image",
                tags=["display", "preview"],
            )

            if config.show_metadata:
                components["metadata"] = semantic(
                    gr.JSON(
                        label="Image Info",
                        elem_id="image-metadata",
                    ),
                    intent="displays selected image metadata",
                    tags=["display", "metadata"],
                )

            with gr.Row():
                components["favorite_btn"] = semantic(
                    gr.Button("❤️ Favorite", size="sm"),
                    intent="adds image to favorites collection",
                    tags=["action", "favorite"],
                )

                components["delete_btn"] = semantic(
                    gr.Button("🗑️ Delete", size="sm", variant="stop"),
                    intent="removes image from gallery",
                    tags=["action", "delete", "destructive"],
                )

    # Pagination
    with gr.Row():
        components["prev_btn"] = semantic(
            gr.Button("← Previous", size="sm"),
            intent="navigates to previous page of images",
            tags=["navigation", "pagination"],
        )

        components["page_info"] = semantic(
            gr.Markdown("Page 1 of 1", elem_id="page-info"),
            intent="shows current page number",
            tags=["status", "pagination"],
        )

        components["next_btn"] = semantic(
            gr.Button("Next →", size="sm"),
            intent="navigates to next page of images",
            tags=["navigation", "pagination"],
        )

    # Wire up selection handler
    def handle_select(evt: gr.SelectData, gallery_data):
        if evt is None or gallery_data is None:
            return None, {}

        selected_idx = evt.index
        if isinstance(gallery_data, list) and selected_idx < len(gallery_data):
            img = gallery_data[selected_idx]
            # Extract metadata if available
            metadata = {
                "index": selected_idx,
                "filename": getattr(img, "name", f"image_{selected_idx}"),
            }
            if hasattr(img, "size"):
                metadata["size"] = img.size
            return img, metadata
        return None, {}

    components["gallery"].select(
        fn=handle_select if not on_select else on_select,
        inputs=[components["gallery"]],
        outputs=[components["preview_image"], components["metadata"]]
        if config.show_metadata else [components["preview_image"]],
    )

    # Wire up upload handler
    if config.allow_upload and on_upload:
        components["upload_btn"].upload(
            fn=on_upload,
            inputs=[components["upload_btn"], components["gallery"]],
            outputs=[components["gallery"], components["stats"]],
        )

    return components


class GalleryPage:
    """
    Complete gallery page with SemanticBlocks integration.

    Usage:
        page = GalleryPage(title="My Photos", initial_images=images)
        page.launch()
    """

    def __init__(
        self,
        title: str = "Gallery",
        initial_images: Optional[list] = None,
        on_select: Optional[Callable] = None,
        on_upload: Optional[Callable] = None,
        **config_kwargs,
    ):
        self.config = GalleryConfig(title=title, **config_kwargs)
        self.initial_images = initial_images
        self.on_select = on_select
        self.on_upload = on_upload
        self.components: dict[str, Any] = {}
        self.blocks: Optional[SemanticBlocks] = None

    def build(self) -> SemanticBlocks:
        """Build the gallery interface."""
        self.blocks = SemanticBlocks(
            title=self.config.title,
            theme=gr.themes.Soft(),
        )

        with self.blocks:
            self.components = create_gallery_grid(
                config=self.config,
                on_select=self.on_select,
                on_upload=self.on_upload,
                initial_images=self.initial_images,
            )

        return self.blocks

    def launch(self, **kwargs) -> None:
        """Build and launch the gallery interface."""
        if self.blocks is None:
            self.build()
        self.blocks.launch(**kwargs)

    @staticmethod
    def render(
        config: Optional[GalleryConfig] = None,
        initial_images: Optional[list] = None,
    ) -> dict[str, Any]:
        """Render gallery into existing Blocks context."""
        return create_gallery_grid(config=config, initial_images=initial_images)
