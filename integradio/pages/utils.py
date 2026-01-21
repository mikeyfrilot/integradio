"""
Page Utilities - Accessibility and UX Enhancements for Pages.

Provides reusable functions for adding accessibility features to pages:
- Skip navigation links
- ARIA landmarks
- Focus management
- Loading states
- Confirmation dialogs

Usage:
    from integradio.pages.utils import (
        add_page_accessibility,
        create_destructive_action,
        create_loading_wrapper,
    )
"""

from typing import Optional, Callable, Any
import gradio as gr

from ..ux import (
    create_skip_link,
    get_all_ux_css,
    create_confirmation_dialog,
    create_skeleton,
    create_spinner,
    announce_to_screen_reader,
    AriaLive,
)


def get_page_css() -> str:
    """
    Get combined CSS for all page UX components.

    Includes:
    - Loading states (skeletons, spinners, progress)
    - Form validation feedback
    - Toast notifications
    - Accessibility (skip links, focus styles)
    - Confirmation dialogs

    Returns:
        Combined CSS string
    """
    return get_all_ux_css()


def add_skip_navigation(main_content_id: str = "main-content") -> gr.HTML:
    """
    Add skip navigation link for keyboard users.

    Should be called at the very beginning of your Blocks context.

    Args:
        main_content_id: ID of the main content element to skip to

    Returns:
        Gradio HTML component with skip link

    Example:
        with gr.Blocks(css=get_page_css()) as demo:
            add_skip_navigation()
            # ... rest of page
    """
    return gr.HTML(
        create_skip_link(target_id=main_content_id),
        visible=True,
        elem_id="skip-nav",
    )


def add_main_content_landmark(content_id: str = "main-content") -> gr.HTML:
    """
    Add main content landmark for accessibility.

    Call this before your main content section.

    Args:
        content_id: ID for the main content landmark

    Returns:
        Gradio HTML component marking start of main content
    """
    return gr.HTML(
        f'<div id="{content_id}" tabindex="-1" role="main" aria-label="Main content">',
        visible=True,
    )


def close_main_content_landmark() -> gr.HTML:
    """Close the main content landmark div."""
    return gr.HTML('</div>', visible=True)


def create_confirmation_button(
    label: str,
    confirm_title: str,
    confirm_message: str,
    confirm_label: str = "Confirm",
    cancel_label: str = "Cancel",
    variant: str = "stop",
    icon: str = "",
) -> tuple[gr.Button, gr.HTML, gr.State]:
    """
    Create a button that shows confirmation before action.

    Args:
        label: Button label
        confirm_title: Confirmation dialog title
        confirm_message: Confirmation dialog message
        confirm_label: Confirm button text
        cancel_label: Cancel button text
        variant: Button variant (primary, secondary, stop)
        icon: Optional emoji/icon prefix

    Returns:
        Tuple of (button, dialog_html, confirmed_state)

    Example:
        delete_btn, dialog, confirmed = create_confirmation_button(
            "Delete All",
            "Confirm Delete",
            "This will permanently delete all items. Continue?",
        )

        delete_btn.click(fn=show_dialog, outputs=[dialog])
        # Handle confirmation separately in JS or with state
    """
    button_label = f"{icon} {label}".strip() if icon else label

    button = gr.Button(button_label, variant=variant)
    dialog_html = gr.HTML("", visible=False)
    confirmed = gr.State(False)

    dialog_content = create_confirmation_dialog(
        title=confirm_title,
        message=confirm_message,
        confirm_label=confirm_label,
        cancel_label=cancel_label,
        danger=(variant == "stop"),
    )

    def show_dialog():
        return gr.update(value=dialog_content, visible=True)

    def hide_dialog():
        return gr.update(value="", visible=False)

    return button, dialog_html, confirmed


def create_loading_section(
    label: str = "Content",
    lines: int = 3,
    show_skeleton: bool = True,
) -> tuple[gr.HTML, gr.HTML]:
    """
    Create a section with loading skeleton support.

    Args:
        label: Accessible label for the section
        lines: Number of skeleton lines when loading
        show_skeleton: Start in loading state

    Returns:
        Tuple of (skeleton_component, content_wrapper)

    Example:
        skeleton, content = create_loading_section("Search Results")
        # Show skeleton while loading
        # Hide skeleton and show content when ready
    """
    skeleton_html = create_skeleton(lines=lines)

    skeleton = gr.HTML(
        skeleton_html if show_skeleton else "",
        visible=show_skeleton,
        elem_id=f"{label.lower().replace(' ', '-')}-skeleton",
    )

    content = gr.HTML(
        "",
        visible=not show_skeleton,
        elem_id=f"{label.lower().replace(' ', '-')}-content",
    )

    return skeleton, content


def create_status_announcement(
    message: str = "",
    politeness: str = "polite",
) -> gr.HTML:
    """
    Create a live region for screen reader announcements.

    Args:
        message: Initial message (usually empty)
        politeness: "polite" or "assertive"

    Returns:
        Gradio HTML component for announcements

    Example:
        announcement = create_status_announcement()

        def on_save():
            return gr.update(value=announce_to_screen_reader("Settings saved"))

        save_btn.click(fn=on_save, outputs=[announcement])
    """
    live = AriaLive.ASSERTIVE if politeness == "assertive" else AriaLive.POLITE

    return gr.HTML(
        announce_to_screen_reader(message, live) if message else "",
        visible=True,
        elem_id="status-announcement",
    )


# Standard empty state messages
EMPTY_STATES = {
    "no_files": """
        <div class="empty-state" role="status">
            <div class="empty-state-icon">📁</div>
            <h3>No files uploaded</h3>
            <p>Drag and drop files here, or click to browse</p>
        </div>
    """,
    "no_results": """
        <div class="empty-state" role="status">
            <div class="empty-state-icon">🔍</div>
            <h3>No results found</h3>
            <p>Try adjusting your search or filters</p>
        </div>
    """,
    "no_data": """
        <div class="empty-state" role="status">
            <div class="empty-state-icon">📊</div>
            <h3>No data available</h3>
            <p>Data will appear here once available</p>
        </div>
    """,
    "no_messages": """
        <div class="empty-state" role="status">
            <div class="empty-state-icon">💬</div>
            <h3>No messages yet</h3>
            <p>Start a conversation by typing a message below</p>
        </div>
    """,
    "no_images": """
        <div class="empty-state" role="status">
            <div class="empty-state-icon">🖼️</div>
            <h3>No images in gallery</h3>
            <p>Upload images to see them here</p>
        </div>
    """,
}


def get_empty_state(state_type: str) -> str:
    """
    Get HTML for an empty state message.

    Args:
        state_type: One of "no_files", "no_results", "no_data", "no_messages", "no_images"

    Returns:
        HTML string for empty state
    """
    return EMPTY_STATES.get(state_type, EMPTY_STATES["no_data"])


# CSS for empty states
EMPTY_STATE_CSS = """
/* Empty state styling */
.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    color: #6b7280;
}

.empty-state-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
    opacity: 0.5;
}

.empty-state h3 {
    margin: 0 0 0.5rem 0;
    color: #374151;
    font-size: 1.125rem;
}

[data-theme="dark"] .empty-state h3,
.dark .empty-state h3 {
    color: #e5e7eb;
}

.empty-state p {
    margin: 0;
    font-size: 0.875rem;
}
"""


def get_enhanced_page_css() -> str:
    """
    Get all CSS for enhanced page UX including empty states.

    Returns:
        Combined CSS string
    """
    return get_all_ux_css() + EMPTY_STATE_CSS
