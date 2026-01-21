"""
Loading States - Skeleton Screens, Progress Indicators, Spinners

Best practices (2026):
- Use skeleton screens when content structure is known (< 10 seconds)
- Use progress bars for operations > 10 seconds with known duration
- Use spinners for short operations (1-3 seconds) or unknown duration
- Implement 300ms delay to prevent flicker on fast operations
- Show estimated time for long operations

References:
- NN/g: https://www.nngroup.com/articles/skeleton-screens/
- Carbon Design: https://carbondesignsystem.com/patterns/loading-pattern/
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import gradio as gr


class ProgressType(Enum):
    """Types of progress indicators."""
    DETERMINATE = "determinate"  # Known progress percentage
    INDETERMINATE = "indeterminate"  # Unknown duration


class LoadingState(Enum):
    """Loading state for components."""
    IDLE = "idle"
    LOADING = "loading"
    SUCCESS = "success"
    ERROR = "error"


@dataclass
class LoadingConfig:
    """Configuration for loading behavior."""
    # Delay before showing loading indicator (prevents flicker)
    delay_ms: int = 300
    # Minimum display time (prevents jarring fast transitions)
    min_display_ms: int = 500
    # Show estimated time for long operations
    show_estimated_time: bool = True
    # Use skeleton vs spinner
    use_skeleton: bool = True
    # Animation type for skeleton
    skeleton_animation: str = "pulse"  # "pulse", "wave", or "none"


# CSS for skeleton screens and loading states
LOADING_CSS = """
/* Skeleton Screen Styles */
.skeleton {
    background: linear-gradient(90deg, #e0e0e0 25%, #f0f0f0 50%, #e0e0e0 75%);
    background-size: 200% 100%;
    border-radius: 4px;
}

.skeleton-pulse {
    animation: skeleton-pulse 1.5s ease-in-out infinite;
}

.skeleton-wave {
    animation: skeleton-wave 1.5s linear infinite;
}

@keyframes skeleton-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

@keyframes skeleton-wave {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

/* Dark mode skeleton */
[data-theme="dark"] .skeleton,
.dark .skeleton {
    background: linear-gradient(90deg, #374151 25%, #4b5563 50%, #374151 75%);
    background-size: 200% 100%;
}

/* Skeleton text lines */
.skeleton-text {
    height: 1em;
    margin-bottom: 0.5em;
}

.skeleton-text:last-child {
    width: 60%;
}

/* Skeleton card */
.skeleton-card {
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #e5e7eb;
}

[data-theme="dark"] .skeleton-card,
.dark .skeleton-card {
    border-color: #374151;
}

/* Spinner Styles */
.spinner {
    width: 24px;
    height: 24px;
    border: 3px solid #e5e7eb;
    border-top-color: #3b82f6;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}

.spinner-sm { width: 16px; height: 16px; border-width: 2px; }
.spinner-lg { width: 32px; height: 32px; border-width: 4px; }

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Progress bar */
.progress-bar {
    height: 4px;
    background: #e5e7eb;
    border-radius: 2px;
    overflow: hidden;
}

.progress-bar-fill {
    height: 100%;
    background: #3b82f6;
    transition: width 0.3s ease;
}

.progress-bar-indeterminate .progress-bar-fill {
    width: 30%;
    animation: progress-indeterminate 1.5s ease-in-out infinite;
}

@keyframes progress-indeterminate {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(400%); }
}

/* Loading overlay */
.loading-overlay {
    position: absolute;
    inset: 0;
    background: rgba(255, 255, 255, 0.8);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 10;
}

[data-theme="dark"] .loading-overlay,
.dark .loading-overlay {
    background: rgba(0, 0, 0, 0.6);
}

/* Accessible loading announcement */
.sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
}
"""


def create_skeleton(
    lines: int = 3,
    animation: str = "pulse",
    width: str = "100%",
) -> str:
    """
    Create skeleton screen HTML for content loading.

    Args:
        lines: Number of skeleton text lines
        animation: Animation type ("pulse", "wave", or "none")
        width: Container width

    Returns:
        HTML string for skeleton placeholder

    Example:
        gr.HTML(create_skeleton(lines=4))
    """
    anim_class = f"skeleton-{animation}" if animation != "none" else ""

    skeleton_lines = "\n".join(
        f'<div class="skeleton skeleton-text {anim_class}" '
        f'style="width: {width if i < lines - 1 else "60%"};"></div>'
        for i in range(lines)
    )

    return f"""
    <div class="skeleton-container" style="width: {width};" role="status" aria-label="Loading content">
        {skeleton_lines}
        <span class="sr-only">Loading...</span>
    </div>
    """


def create_skeleton_text(
    words: int = 10,
    animation: str = "pulse",
) -> str:
    """
    Create skeleton text placeholder.

    Args:
        words: Approximate number of words to simulate
        animation: Animation type

    Returns:
        HTML string for skeleton text
    """
    # Estimate width: ~6 characters per word, ~8px per character
    width = min(words * 48, 400)
    anim_class = f"skeleton-{animation}" if animation != "none" else ""

    return f"""
    <span class="skeleton {anim_class}"
          style="display: inline-block; width: {width}px; height: 1em; vertical-align: middle;"
          role="status" aria-label="Loading text">
        <span class="sr-only">Loading...</span>
    </span>
    """


def create_skeleton_card(
    show_image: bool = True,
    show_title: bool = True,
    lines: int = 2,
    animation: str = "pulse",
) -> str:
    """
    Create skeleton card placeholder.

    Args:
        show_image: Include image placeholder
        show_title: Include title placeholder
        lines: Number of body text lines
        animation: Animation type

    Returns:
        HTML string for skeleton card
    """
    anim_class = f"skeleton-{animation}" if animation != "none" else ""

    parts = []

    if show_image:
        parts.append(
            f'<div class="skeleton {anim_class}" '
            f'style="width: 100%; height: 150px; margin-bottom: 1rem;"></div>'
        )

    if show_title:
        parts.append(
            f'<div class="skeleton skeleton-text {anim_class}" '
            f'style="width: 70%; height: 1.5em; margin-bottom: 0.75rem;"></div>'
        )

    for i in range(lines):
        width = "100%" if i < lines - 1 else "80%"
        parts.append(
            f'<div class="skeleton skeleton-text {anim_class}" '
            f'style="width: {width};"></div>'
        )

    return f"""
    <div class="skeleton-card" role="status" aria-label="Loading card">
        {"".join(parts)}
        <span class="sr-only">Loading card content...</span>
    </div>
    """


def create_spinner(
    size: str = "md",
    label: str = "Loading",
    show_label: bool = True,
) -> str:
    """
    Create a spinner loading indicator.

    Args:
        size: Spinner size ("sm", "md", "lg")
        label: Accessible label
        show_label: Show visible label text

    Returns:
        HTML string for spinner
    """
    size_class = {
        "sm": "spinner-sm",
        "md": "",
        "lg": "spinner-lg",
    }.get(size, "")

    label_html = f'<span style="margin-left: 0.5rem;">{label}</span>' if show_label else ""

    return f"""
    <div style="display: flex; align-items: center; justify-content: center;"
         role="status" aria-label="{label}">
        <div class="spinner {size_class}"></div>
        {label_html}
        <span class="sr-only">{label}</span>
    </div>
    """


def create_progress_indicator(
    progress: float = 0,
    progress_type: ProgressType = ProgressType.DETERMINATE,
    label: str = "Progress",
    show_percentage: bool = True,
    estimated_time: Optional[str] = None,
) -> str:
    """
    Create a progress bar indicator.

    Args:
        progress: Progress percentage (0-100) for determinate type
        progress_type: DETERMINATE or INDETERMINATE
        label: Accessible label
        show_percentage: Show percentage text
        estimated_time: Optional estimated time remaining

    Returns:
        HTML string for progress bar
    """
    is_indeterminate = progress_type == ProgressType.INDETERMINATE
    indeterminate_class = "progress-bar-indeterminate" if is_indeterminate else ""

    percentage_text = ""
    if show_percentage and not is_indeterminate:
        percentage_text = f'<span style="margin-left: 0.5rem;">{progress:.0f}%</span>'

    time_text = ""
    if estimated_time:
        time_text = f'<span style="margin-left: 0.5rem; color: #6b7280;">({estimated_time})</span>'

    width = "100%" if is_indeterminate else f"{progress}%"

    return f"""
    <div role="progressbar"
         aria-label="{label}"
         aria-valuenow="{progress if not is_indeterminate else ''}"
         aria-valuemin="0"
         aria-valuemax="100"
         style="width: 100%;">
        <div style="display: flex; align-items: center; margin-bottom: 0.25rem;">
            <span>{label}</span>
            {percentage_text}
            {time_text}
        </div>
        <div class="progress-bar {indeterminate_class}">
            <div class="progress-bar-fill" style="width: {width};"></div>
        </div>
    </div>
    """


def get_loading_css() -> str:
    """Get CSS styles for loading components."""
    return LOADING_CSS


def create_loading_overlay(
    loading: bool = True,
    message: str = "Loading...",
) -> str:
    """
    Create a loading overlay for containers.

    Args:
        loading: Whether to show the overlay
        message: Loading message

    Returns:
        HTML string for overlay
    """
    if not loading:
        return ""

    return f"""
    <div class="loading-overlay" role="status" aria-live="polite">
        <div style="text-align: center;">
            {create_spinner(show_label=False)}
            <div style="margin-top: 0.5rem;">{message}</div>
        </div>
    </div>
    """
