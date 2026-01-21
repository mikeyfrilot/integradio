"""
Feedback Components - Toasts, Confirmations, Status Badges

Best practices (2026):
- Use toasts for transient, non-critical messages
- Require confirmation for destructive actions
- Provide clear status indicators with semantic meaning
- Support dismissal and auto-dismiss with appropriate timing
- Ensure feedback is accessible to screen readers

References:
- Carbon Design: https://carbondesignsystem.com/components/notification/usage/
- Material Design: https://m3.material.io/components/snackbar/overview
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable
import gradio as gr
import time


class ToastType(Enum):
    """Types of toast notifications."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


class StatusType(Enum):
    """Types of status indicators."""
    DEFAULT = "default"
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    PENDING = "pending"


@dataclass
class ToastConfig:
    """Configuration for toast notifications."""
    # Auto-dismiss delay in milliseconds (0 = no auto-dismiss)
    auto_dismiss_ms: int = 5000
    # Show dismiss button
    dismissible: bool = True
    # Position on screen
    position: str = "bottom-right"  # top-left, top-right, bottom-left, bottom-right
    # Maximum toasts to show at once
    max_visible: int = 3
    # Stack direction
    stack_direction: str = "up"  # up or down


@dataclass
class Toast:
    """A toast notification."""
    message: str
    toast_type: ToastType = ToastType.INFO
    title: Optional[str] = None
    action_label: Optional[str] = None
    action_callback: Optional[Callable] = None
    created_at: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: f"toast-{int(time.time() * 1000)}")


# CSS for feedback components
FEEDBACK_CSS = """
/* Toast container */
.toast-container {
    position: fixed;
    z-index: 1000;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    max-width: 400px;
    padding: 1rem;
}

.toast-container.top-left { top: 0; left: 0; }
.toast-container.top-right { top: 0; right: 0; }
.toast-container.bottom-left { bottom: 0; left: 0; }
.toast-container.bottom-right { bottom: 0; right: 0; }

.toast-container.stack-up { flex-direction: column-reverse; }

/* Toast notification */
.toast {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    background: white;
    animation: toast-slide-in 0.3s ease;
    max-width: 100%;
}

[data-theme="dark"] .toast,
.dark .toast {
    background: #1f2937;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
}

@keyframes toast-slide-in {
    from {
        transform: translateX(100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

.toast.dismissing {
    animation: toast-slide-out 0.2s ease forwards;
}

@keyframes toast-slide-out {
    to {
        transform: translateX(100%);
        opacity: 0;
    }
}

/* Toast icon */
.toast-icon {
    flex-shrink: 0;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: bold;
}

.toast.info .toast-icon { background: #3b82f6; color: white; }
.toast.success .toast-icon { background: #10b981; color: white; }
.toast.warning .toast-icon { background: #f59e0b; color: white; }
.toast.error .toast-icon { background: #ef4444; color: white; }

/* Toast content */
.toast-content {
    flex: 1;
    min-width: 0;
}

.toast-title {
    font-weight: 600;
    margin-bottom: 0.25rem;
}

.toast-message {
    color: #6b7280;
    font-size: 0.875rem;
}

[data-theme="dark"] .toast-message,
.dark .toast-message {
    color: #9ca3af;
}

/* Toast dismiss button */
.toast-dismiss {
    flex-shrink: 0;
    background: none;
    border: none;
    color: #9ca3af;
    cursor: pointer;
    padding: 0.25rem;
    border-radius: 4px;
    line-height: 1;
}

.toast-dismiss:hover {
    background: #f3f4f6;
    color: #374151;
}

[data-theme="dark"] .toast-dismiss:hover,
.dark .toast-dismiss:hover {
    background: #374151;
    color: #e5e7eb;
}

/* Toast action button */
.toast-action {
    background: none;
    border: none;
    color: #3b82f6;
    font-weight: 600;
    cursor: pointer;
    padding: 0.25rem 0;
    font-size: 0.875rem;
}

.toast-action:hover {
    text-decoration: underline;
}

/* Confirmation dialog */
.confirmation-dialog {
    position: fixed;
    inset: 0;
    z-index: 1001;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(0, 0, 0, 0.5);
    animation: fade-in 0.2s ease;
}

@keyframes fade-in {
    from { opacity: 0; }
    to { opacity: 1; }
}

.confirmation-content {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    max-width: 400px;
    width: 90%;
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
    animation: scale-in 0.2s ease;
}

[data-theme="dark"] .confirmation-content,
.dark .confirmation-content {
    background: #1f2937;
}

@keyframes scale-in {
    from { transform: scale(0.95); opacity: 0; }
    to { transform: scale(1); opacity: 1; }
}

.confirmation-icon {
    width: 48px;
    height: 48px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 1rem;
    font-size: 24px;
}

.confirmation-icon.warning { background: #fef3c7; color: #f59e0b; }
.confirmation-icon.danger { background: #fee2e2; color: #ef4444; }
.confirmation-icon.info { background: #dbeafe; color: #3b82f6; }

.confirmation-title {
    text-align: center;
    font-size: 1.125rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.confirmation-message {
    text-align: center;
    color: #6b7280;
    margin-bottom: 1.5rem;
}

.confirmation-actions {
    display: flex;
    gap: 0.75rem;
    justify-content: center;
}

.confirmation-btn {
    padding: 0.5rem 1rem;
    border-radius: 6px;
    font-weight: 500;
    cursor: pointer;
    transition: background-color 0.2s;
}

.confirmation-btn.cancel {
    background: #f3f4f6;
    border: 1px solid #e5e7eb;
    color: #374151;
}

.confirmation-btn.cancel:hover {
    background: #e5e7eb;
}

.confirmation-btn.confirm {
    background: #ef4444;
    border: none;
    color: white;
}

.confirmation-btn.confirm:hover {
    background: #dc2626;
}

.confirmation-btn.confirm.primary {
    background: #3b82f6;
}

.confirmation-btn.confirm.primary:hover {
    background: #2563eb;
}

/* Status badge */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.375rem;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 500;
}

.status-badge.default { background: #f3f4f6; color: #374151; }
.status-badge.info { background: #dbeafe; color: #1d4ed8; }
.status-badge.success { background: #d1fae5; color: #047857; }
.status-badge.warning { background: #fef3c7; color: #b45309; }
.status-badge.error { background: #fee2e2; color: #b91c1c; }
.status-badge.pending { background: #e0e7ff; color: #4338ca; }

.status-badge-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
}

.status-badge.pending .status-badge-dot {
    animation: pulse-dot 1.5s ease-in-out infinite;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
"""


def create_toast(
    message: str,
    toast_type: ToastType = ToastType.INFO,
    title: Optional[str] = None,
    dismissible: bool = True,
    action_label: Optional[str] = None,
) -> str:
    """
    Create a toast notification HTML.

    Args:
        message: Toast message
        toast_type: Type of toast (info, success, warning, error)
        title: Optional title
        dismissible: Show dismiss button
        action_label: Optional action button label

    Returns:
        HTML string for toast
    """
    type_class = toast_type.value
    icon = {
        ToastType.INFO: "i",
        ToastType.SUCCESS: "\\2713",
        ToastType.WARNING: "!",
        ToastType.ERROR: "\\00D7",
    }.get(toast_type, "i")

    title_html = f'<div class="toast-title">{title}</div>' if title else ""
    dismiss_html = '<button class="toast-dismiss" aria-label="Dismiss">&times;</button>' if dismissible else ""
    action_html = f'<button class="toast-action">{action_label}</button>' if action_label else ""

    return f"""
    <div class="toast {type_class}" role="alert" aria-live="polite">
        <div class="toast-icon">{icon}</div>
        <div class="toast-content">
            {title_html}
            <div class="toast-message">{message}</div>
            {action_html}
        </div>
        {dismiss_html}
    </div>
    """


def create_toast_container(
    position: str = "bottom-right",
    stack_direction: str = "up",
) -> str:
    """
    Create a toast container HTML.

    Args:
        position: Container position (top-left, top-right, bottom-left, bottom-right)
        stack_direction: Stack direction (up or down)

    Returns:
        HTML string for toast container
    """
    stack_class = f"stack-{stack_direction}"
    return f'<div class="toast-container {position} {stack_class}" aria-live="polite"></div>'


def create_confirmation_dialog(
    title: str,
    message: str,
    confirm_label: str = "Confirm",
    cancel_label: str = "Cancel",
    danger: bool = False,
    icon_type: str = "warning",
) -> str:
    """
    Create a confirmation dialog HTML.

    Args:
        title: Dialog title
        message: Dialog message
        confirm_label: Confirm button text
        cancel_label: Cancel button text
        danger: Use danger styling for confirm button
        icon_type: Icon type (warning, danger, info)

    Returns:
        HTML string for confirmation dialog
    """
    icon = {
        "warning": "!",
        "danger": "\\00D7",
        "info": "?",
    }.get(icon_type, "!")

    confirm_class = "confirm" if not danger else "confirm"

    return f"""
    <div class="confirmation-dialog" role="dialog" aria-modal="true" aria-labelledby="confirm-title">
        <div class="confirmation-content">
            <div class="confirmation-icon {icon_type}">{icon}</div>
            <h2 class="confirmation-title" id="confirm-title">{title}</h2>
            <p class="confirmation-message">{message}</p>
            <div class="confirmation-actions">
                <button class="confirmation-btn cancel">{cancel_label}</button>
                <button class="confirmation-btn {confirm_class}">{confirm_label}</button>
            </div>
        </div>
    </div>
    """


def create_status_badge(
    label: str,
    status_type: StatusType = StatusType.DEFAULT,
    show_dot: bool = True,
) -> str:
    """
    Create a status badge HTML.

    Args:
        label: Badge text
        status_type: Type of status
        show_dot: Show status dot indicator

    Returns:
        HTML string for status badge
    """
    type_class = status_type.value
    dot_html = '<span class="status-badge-dot"></span>' if show_dot else ""

    return f"""
    <span class="status-badge {type_class}">
        {dot_html}
        {label}
    </span>
    """


def create_inline_status(
    message: str,
    status_type: StatusType = StatusType.INFO,
) -> str:
    """
    Create inline status message HTML.

    Args:
        message: Status message
        status_type: Type of status

    Returns:
        HTML string for inline status
    """
    icons = {
        StatusType.INFO: "\\2139",
        StatusType.SUCCESS: "\\2713",
        StatusType.WARNING: "\\26A0",
        StatusType.ERROR: "\\2717",
        StatusType.PENDING: "\\23F3",
        StatusType.DEFAULT: "",
    }
    icon = icons.get(status_type, "")
    type_class = status_type.value

    return f"""
    <div class="inline-status {type_class}" role="status">
        <span class="inline-status-icon">{icon}</span>
        <span class="inline-status-message">{message}</span>
    </div>
    """


def get_feedback_css() -> str:
    """Get CSS styles for feedback components."""
    return FEEDBACK_CSS


# Gradio-compatible toast system using state
class ToastManager:
    """
    Manager for toast notifications in Gradio apps.

    Example:
        toast_mgr = ToastManager()

        with gr.Blocks() as demo:
            toast_html = gr.HTML(toast_mgr.render())

            def show_success():
                toast_mgr.add("Saved successfully!", ToastType.SUCCESS)
                return toast_mgr.render()

            btn.click(show_success, outputs=[toast_html])
    """

    def __init__(self, config: Optional[ToastConfig] = None):
        self.config = config or ToastConfig()
        self.toasts: list[Toast] = []

    def add(
        self,
        message: str,
        toast_type: ToastType = ToastType.INFO,
        title: Optional[str] = None,
    ) -> Toast:
        """Add a toast notification."""
        toast = Toast(message=message, toast_type=toast_type, title=title)
        self.toasts.append(toast)

        # Limit visible toasts
        if len(self.toasts) > self.config.max_visible:
            self.toasts = self.toasts[-self.config.max_visible:]

        return toast

    def remove(self, toast_id: str):
        """Remove a toast by ID."""
        self.toasts = [t for t in self.toasts if t.id != toast_id]

    def clear(self):
        """Clear all toasts."""
        self.toasts = []

    def render(self) -> str:
        """Render all toasts to HTML."""
        if not self.toasts:
            return create_toast_container(
                self.config.position,
                self.config.stack_direction,
            )

        toasts_html = "".join(
            create_toast(
                t.message,
                t.toast_type,
                t.title,
                self.config.dismissible,
            )
            for t in self.toasts
        )

        return f"""
        <div class="toast-container {self.config.position} stack-{self.config.stack_direction}">
            {toasts_html}
        </div>
        """
