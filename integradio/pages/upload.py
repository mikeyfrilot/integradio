"""
Upload/Media Page - File upload center with processing.

Features:
- Drag & drop upload
- Multi-file support
- Progress indicators
- File type validation
- Preview thumbnails
- Processing pipeline integration
"""

from typing import Optional, Callable, Any
from dataclasses import dataclass, field

import gradio as gr

from ..components import semantic
from ..blocks import SemanticBlocks


@dataclass
class UploadConfig:
    """Configuration for upload center."""
    title: str = "Upload Center"
    subtitle: str = "Drag & drop files or click to browse"
    allowed_types: list[str] = field(default_factory=lambda: ["image", "video", "audio", "document"])
    max_file_size: str = "100MB"
    max_files: int = 10
    show_preview: bool = True
    show_processing: bool = True
    auto_process: bool = False
    processing_options: list[str] = field(default_factory=lambda: [
        "Compress",
        "Convert Format",
        "Extract Metadata",
        "Generate Thumbnail",
    ])


@dataclass
class UploadedFile:
    """Representation of an uploaded file."""
    name: str
    size: str
    type: str
    status: str = "uploaded"  # "uploading", "uploaded", "processing", "done", "error"
    preview_url: Optional[str] = None
    metadata: dict = field(default_factory=dict)


def create_upload_center(
    config: Optional[UploadConfig] = None,
    on_upload: Optional[Callable] = None,
    on_process: Optional[Callable] = None,
    on_delete: Optional[Callable] = None,
) -> dict[str, Any]:
    """
    Create an upload center with semantic-tracked components.

    Args:
        config: Upload configuration
        on_upload: Upload handler
        on_process: Processing handler
        on_delete: Delete handler

    Returns:
        Dict of component references
    """
    config = config or UploadConfig()
    components = {}

    # Header
    components["title"] = semantic(
        gr.Markdown(f"# 📤 {config.title}"),
        intent="displays upload center page title",
        tags=["header"],
    )

    # Upload Zone
    with gr.Row():
        with gr.Column(scale=2):
            # Determine file types
            file_types = []
            if "image" in config.allowed_types:
                file_types.extend([".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"])
            if "video" in config.allowed_types:
                file_types.extend([".mp4", ".webm", ".mov", ".avi"])
            if "audio" in config.allowed_types:
                file_types.extend([".mp3", ".wav", ".ogg", ".flac"])
            if "document" in config.allowed_types:
                file_types.extend([".pdf", ".doc", ".docx", ".txt", ".csv", ".json"])

            components["upload_area"] = semantic(
                gr.File(
                    label=config.subtitle,
                    file_count="multiple",
                    file_types=file_types if file_types else None,
                    elem_id="main-upload",
                ),
                intent="accepts file uploads via drag-drop or browse",
                tags=["upload", "input", "primary"],
            )

            # Upload info
            components["upload_info"] = semantic(
                gr.Markdown(
                    f"**Allowed:** {', '.join(config.allowed_types)} | "
                    f"**Max size:** {config.max_file_size} | "
                    f"**Max files:** {config.max_files}"
                ),
                intent="displays upload constraints and limits",
                tags=["info", "constraints"],
            )

        # Quick upload buttons
        with gr.Column(scale=1):
            components["quick_title"] = semantic(
                gr.Markdown("### Quick Upload"),
                intent="introduces quick upload options",
                tags=["header", "section"],
            )

            if "image" in config.allowed_types:
                components["upload_image"] = semantic(
                    gr.UploadButton(
                        "🖼️ Images",
                        file_types=["image"],
                        file_count="multiple",
                        variant="secondary",
                    ),
                    intent="uploads image files specifically",
                    tags=["upload", "image", "quick"],
                )

            if "video" in config.allowed_types:
                components["upload_video"] = semantic(
                    gr.UploadButton(
                        "🎬 Videos",
                        file_types=["video"],
                        file_count="multiple",
                        variant="secondary",
                    ),
                    intent="uploads video files specifically",
                    tags=["upload", "video", "quick"],
                )

            if "audio" in config.allowed_types:
                components["upload_audio"] = semantic(
                    gr.UploadButton(
                        "🎵 Audio",
                        file_types=["audio"],
                        file_count="multiple",
                        variant="secondary",
                    ),
                    intent="uploads audio files specifically",
                    tags=["upload", "audio", "quick"],
                )

            if "document" in config.allowed_types:
                components["upload_doc"] = semantic(
                    gr.UploadButton(
                        "📄 Documents",
                        file_types=[".pdf", ".doc", ".docx", ".txt"],
                        file_count="multiple",
                        variant="secondary",
                    ),
                    intent="uploads document files specifically",
                    tags=["upload", "document", "quick"],
                )

    gr.Markdown("---")

    # Upload Queue / Progress
    components["queue_title"] = semantic(
        gr.Markdown("### 📋 Upload Queue"),
        intent="introduces upload queue section",
        tags=["header", "section"],
    )

    components["upload_status"] = semantic(
        gr.Markdown("No files uploaded yet"),
        intent="displays current upload queue status",
        tags=["status", "queue"],
    )

    # Progress bar (for active uploads)
    components["progress"] = semantic(
        gr.Markdown("", visible=False),
        intent="shows upload progress for current files",
        tags=["progress", "upload"],
    )

    gr.Markdown("---")

    # Processing Options (if enabled)
    if config.show_processing:
        with gr.Accordion("⚙️ Processing Options", open=False):
            components["process_options"] = semantic(
                gr.CheckboxGroup(
                    choices=config.processing_options,
                    label="Select processing operations",
                    value=[],
                ),
                intent="selects post-upload processing operations",
                tags=["config", "processing"],
            )

            with gr.Row():
                components["process_btn"] = semantic(
                    gr.Button("🔄 Process Selected", variant="primary"),
                    intent="starts processing on uploaded files",
                    tags=["action", "process"],
                )

                components["process_all_btn"] = semantic(
                    gr.Button("🔄 Process All", variant="secondary"),
                    intent="processes all uploaded files",
                    tags=["action", "process", "batch"],
                )

    # File List / Gallery
    components["files_title"] = semantic(
        gr.Markdown("### 📁 Uploaded Files"),
        intent="introduces uploaded files section",
        tags=["header", "section"],
    )

    # Preview gallery (for images/videos)
    if config.show_preview:
        components["preview_gallery"] = semantic(
            gr.Gallery(
                label="Previews",
                columns=4,
                height=200,
                object_fit="cover",
                show_label=False,
            ),
            intent="displays thumbnail previews of uploaded media",
            tags=["preview", "gallery", "thumbnails"],
        )

    # File list table
    components["file_list"] = semantic(
        gr.Dataframe(
            headers=["Name", "Size", "Type", "Status"],
            value=[],
            interactive=False,
            wrap=True,
        ),
        intent="displays list of all uploaded files with details",
        tags=["list", "files", "status"],
    )

    # Batch actions
    with gr.Row():
        components["select_all"] = semantic(
            gr.Button("☑️ Select All", size="sm", variant="secondary"),
            intent="selects all uploaded files",
            tags=["action", "selection"],
        )

        components["clear_selection"] = semantic(
            gr.Button("☐ Clear Selection", size="sm", variant="secondary"),
            intent="clears file selection",
            tags=["action", "selection"],
        )

        components["delete_selected"] = semantic(
            gr.Button("🗑️ Delete Selected", size="sm", variant="stop"),
            intent="deletes selected files",
            tags=["action", "delete", "destructive"],
        )

        components["download_all"] = semantic(
            gr.Button("📥 Download All", size="sm", variant="secondary"),
            intent="downloads all uploaded files as archive",
            tags=["action", "download", "batch"],
        )

    gr.Markdown("---")

    # Selected File Details
    with gr.Accordion("📄 File Details", open=False):
        with gr.Row():
            with gr.Column():
                components["file_preview"] = semantic(
                    gr.Image(
                        label="Preview",
                        height=300,
                        show_label=False,
                    ),
                    intent="displays preview of selected file",
                    tags=["preview", "detail"],
                )

            with gr.Column():
                components["file_metadata"] = semantic(
                    gr.JSON(label="Metadata"),
                    intent="displays metadata of selected file",
                    tags=["metadata", "detail"],
                )

                components["rename_input"] = semantic(
                    gr.Textbox(label="Rename File", placeholder="Enter new filename"),
                    intent="allows renaming of selected file",
                    tags=["input", "rename"],
                )

                components["rename_btn"] = semantic(
                    gr.Button("✏️ Rename", size="sm"),
                    intent="applies new filename to selected file",
                    tags=["action", "rename"],
                )

    # Storage info
    components["storage_info"] = semantic(
        gr.Markdown("**Storage:** 0 MB used"),
        intent="displays total storage usage",
        tags=["status", "storage"],
    )

    # Wire up handlers
    def handle_upload(files):
        if files is None:
            return "No files uploaded yet", [], []

        file_data = []
        for f in files:
            name = getattr(f, "name", str(f)).split("/")[-1].split("\\")[-1]
            file_data.append([name, "Unknown", "File", "✅ Uploaded"])

        status = f"**{len(files)}** file(s) uploaded successfully"
        return status, file_data, files

    components["upload_area"].change(
        fn=handle_upload if not on_upload else on_upload,
        inputs=[components["upload_area"]],
        outputs=[
            components["upload_status"],
            components["file_list"],
            components["preview_gallery"],
        ],
    )

    return components


class UploadPage:
    """
    Complete upload center page with SemanticBlocks integration.

    Usage:
        page = UploadPage(
            title="Media Upload",
            allowed_types=["image", "video"],
            on_upload=handle_upload,
        )
        page.launch()
    """

    def __init__(
        self,
        title: str = "Upload Center",
        on_upload: Optional[Callable] = None,
        on_process: Optional[Callable] = None,
        **config_kwargs,
    ):
        self.config = UploadConfig(title=title, **config_kwargs)
        self.on_upload = on_upload
        self.on_process = on_process
        self.components: dict[str, Any] = {}
        self.blocks: Optional[SemanticBlocks] = None

    def build(self) -> SemanticBlocks:
        """Build the upload center."""
        self.blocks = SemanticBlocks(
            title=self.config.title,
            theme=gr.themes.Soft(),
        )

        with self.blocks:
            self.components = create_upload_center(
                config=self.config,
                on_upload=self.on_upload,
                on_process=self.on_process,
            )

        return self.blocks

    def launch(self, **kwargs) -> None:
        """Build and launch the upload center."""
        if self.blocks is None:
            self.build()
        self.blocks.launch(**kwargs)

    @staticmethod
    def render(config: Optional[UploadConfig] = None) -> dict[str, Any]:
        """Render upload center into existing Blocks context."""
        return create_upload_center(config=config)
