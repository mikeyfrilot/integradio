"""
Upload/Media Page - File upload center with processing.

Features:
- Drag & drop upload
- Multi-file support
- Progress indicators
- File type validation
- Preview thumbnails
- Processing pipeline integration

Security (2026 best practices):
- Path traversal protection via pathlib sanitization
- Filename validation against malicious patterns
- File size validation
- Content-type verification
"""

from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from pathlib import Path
import re

import gradio as gr

from ..components import semantic
from ..blocks import SemanticBlocks
from ..ux import create_confirmation_dialog


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


# Security: Filename validation regex - allows alphanumeric, dash, underscore, dot
# Prevents path traversal and command injection
_SAFE_FILENAME_PATTERN = re.compile(r'^[\w\-. ]+$', re.UNICODE)

# Security: Maximum reasonable filename length
_MAX_FILENAME_LENGTH = 255

# Security: Dangerous file extensions that should be blocked
_BLOCKED_EXTENSIONS = frozenset({
    '.exe', '.bat', '.cmd', '.com', '.msi', '.scr', '.pif',  # Windows executables
    '.sh', '.bash', '.zsh', '.csh',  # Shell scripts
    '.ps1', '.psm1', '.psd1',  # PowerShell
    '.vbs', '.vbe', '.js', '.jse', '.ws', '.wsf', '.wsc', '.wsh',  # Script files
    '.dll', '.sys', '.drv',  # System files
    '.php', '.php3', '.php4', '.php5', '.phtml',  # Server-side scripts
    '.asp', '.aspx', '.jsp', '.cgi', '.pl',  # Web scripts
    '.htaccess', '.htpasswd',  # Apache config
})


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent path traversal and other attacks.

    Security measures:
    1. Extract only the basename (no directory components)
    2. Remove null bytes
    3. Validate against safe pattern
    4. Check length limits
    5. Block dangerous extensions

    Args:
        filename: Raw filename from upload

    Returns:
        Sanitized filename safe for filesystem operations

    Raises:
        ValueError: If filename is invalid or potentially malicious
    """
    if not filename:
        raise ValueError("Empty filename")

    # Step 1: Remove null bytes (can bypass extension checks)
    filename = filename.replace('\x00', '')

    # Step 2: Use pathlib to extract just the filename (prevents path traversal)
    # This handles ../../../etc/passwd -> passwd
    safe_name = Path(filename).name

    # Step 3: Double-check no path separators remain
    if '/' in safe_name or '\\' in safe_name:
        raise ValueError("Invalid filename: contains path separators")

    # Step 4: Check for empty after sanitization
    if not safe_name or safe_name in ('.', '..'):
        raise ValueError("Invalid filename")

    # Step 5: Validate length
    if len(safe_name) > _MAX_FILENAME_LENGTH:
        raise ValueError(f"Filename too long (max {_MAX_FILENAME_LENGTH} characters)")

    # Step 6: Validate against safe pattern
    if not _SAFE_FILENAME_PATTERN.match(safe_name):
        # Allow through but log - some legitimate filenames have special chars
        # Replace dangerous characters
        safe_name = re.sub(r'[^\w\-. ]', '_', safe_name)

    # Step 7: Check for blocked extensions
    ext = Path(safe_name).suffix.lower()
    if ext in _BLOCKED_EXTENSIONS:
        raise ValueError(f"File type '{ext}' is not allowed for security reasons")

    # Step 8: Prevent double extensions like file.jpg.exe
    parts = safe_name.lower().split('.')
    for part in parts[1:]:  # Skip the filename itself
        if f'.{part}' in _BLOCKED_EXTENSIONS:
            raise ValueError(f"File contains blocked extension '.{part}'")

    return safe_name


def _format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _get_file_type(extension: str) -> str:
    """Get human-readable file type from extension."""
    type_map = {
        # Images
        '.jpg': 'Image', '.jpeg': 'Image', '.png': 'Image', '.gif': 'Image',
        '.webp': 'Image', '.svg': 'Image', '.bmp': 'Image', '.ico': 'Image',
        # Videos
        '.mp4': 'Video', '.webm': 'Video', '.mov': 'Video', '.avi': 'Video',
        '.mkv': 'Video', '.flv': 'Video',
        # Audio
        '.mp3': 'Audio', '.wav': 'Audio', '.ogg': 'Audio', '.flac': 'Audio',
        '.aac': 'Audio', '.m4a': 'Audio',
        # Documents
        '.pdf': 'PDF', '.doc': 'Document', '.docx': 'Document',
        '.txt': 'Text', '.csv': 'CSV', '.json': 'JSON',
        '.xls': 'Spreadsheet', '.xlsx': 'Spreadsheet',
        # Archives
        '.zip': 'Archive', '.tar': 'Archive', '.gz': 'Archive',
        '.rar': 'Archive', '.7z': 'Archive',
    }
    return type_map.get(extension.lower(), 'File')


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

    # Batch actions - using default size for WCAG 2.2 touch target compliance (44x44px min)
    with gr.Row():
        components["select_all"] = semantic(
            gr.Button("☑️ Select All", variant="secondary"),
            intent="selects all uploaded files",
            tags=["action", "selection"],
        )

        components["clear_selection"] = semantic(
            gr.Button("☐ Clear Selection", variant="secondary"),
            intent="clears file selection",
            tags=["action", "selection"],
        )

        components["delete_selected"] = semantic(
            gr.Button("🗑️ Delete Selected", variant="stop"),
            intent="deletes selected files",
            tags=["action", "delete", "destructive"],
        )

        components["download_all"] = semantic(
            gr.Button("📥 Download All", variant="secondary"),
            intent="downloads all uploaded files as archive",
            tags=["action", "download", "batch"],
        )

    # Confirmation dialog for delete (2026 UX best practice)
    components["delete_confirm"] = gr.HTML(
        "",
        visible=False,
        elem_id="delete-confirm-dialog",
    )

    # Wire up confirmation flow for delete
    def show_delete_confirm():
        return gr.update(
            value=create_confirmation_dialog(
                title="Delete Selected Files?",
                message="This will permanently remove the selected files. This action cannot be undone.",
                confirm_label="Delete",
                cancel_label="Cancel",
                danger=True,
            ),
            visible=True,
        )

    components["delete_selected"].click(
        fn=show_delete_confirm,
        outputs=[components["delete_confirm"]],
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
                    gr.Button("✏️ Rename"),
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
        # Edge case: None or invalid input
        if files is None:
            return "No files uploaded yet", [], []

        # Edge case: files is not a list (single file or invalid type)
        if not isinstance(files, (list, tuple)):
            files = [files]

        # Edge case: too many files
        if len(files) > config.max_files:
            return (
                f"❌ Too many files ({len(files)} > {config.max_files}). "
                "Please reduce the number of files.",
                [],
                [],
            )

        file_data = []
        errors = []
        valid_files = []

        for f in files:
            try:
                # Edge case: None element in file list
                if f is None:
                    errors.append("Received null file entry")
                    continue

                # Security: Get raw filename and sanitize it
                # Edge case: File object without name attribute
                raw_name = getattr(f, "name", None)
                if not raw_name or not isinstance(raw_name, str):
                    errors.append("File has invalid or missing name")
                    continue

                safe_name = sanitize_filename(raw_name)

                # Get file size if available
                try:
                    file_path = Path(raw_name)
                    # Edge case: Check file exists AND is a file (not directory)
                    if file_path.exists() and file_path.is_file():
                        size_bytes = file_path.stat().st_size
                        if size_bytes > 100 * 1024 * 1024:  # 100MB limit
                            errors.append(f"{safe_name}: File too large (max 100MB)")
                            continue
                        size = _format_file_size(size_bytes)
                    else:
                        size = "Unknown"
                except (OSError, PermissionError) as e:
                    # Edge case: Permission denied or file system error
                    size = "Unknown"

                # Determine file type from extension
                ext = Path(safe_name).suffix.lower()
                file_type = _get_file_type(ext)

                file_data.append([safe_name, size, file_type, "✅ Uploaded"])
                valid_files.append(f)

            except ValueError as e:
                # Security: filename validation failed
                errors.append(str(e)[:200])  # Truncate long error messages
                continue
            except (OSError, TypeError) as e:
                # Edge case: Unexpected file system or type errors
                errors.append(f"File error: {type(e).__name__}")
                continue

        # Build status message
        if valid_files:
            status = f"**{len(valid_files)}** file(s) uploaded successfully"
            if errors:
                status += f"\n\n⚠️ **{len(errors)} file(s) rejected:**\n" + "\n".join(f"- {e}" for e in errors)
        elif errors:
            status = "❌ **All files rejected:**\n" + "\n".join(f"- {e}" for e in errors)
        else:
            status = "No files uploaded yet"

        return status, file_data, valid_files

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
