"""
Integradio UX Utilities - 2026 Best Practices

This module provides UI/UX components and patterns following modern best practices:

Loading States:
- Skeleton screens for content loading
- Progress indicators with estimated time
- Streaming feedback for real-time operations

Form Validation:
- Real-time inline validation
- Field-level error messages
- Accessibility-compliant error states

Feedback:
- Toast notifications for transient messages
- Confirmation dialogs for destructive actions
- Status indicators with proper semantics

Accessibility:
- ARIA attribute helpers
- Keyboard navigation support
- Screen reader announcements

Usage:
    from integradio.ux import (
        # Loading
        create_skeleton,
        create_progress_indicator,
        LoadingState,
        # Validation
        FieldValidator,
        ValidationResult,
        create_inline_error,
        # Feedback
        Toast,
        ToastType,
        create_confirmation_dialog,
        # Accessibility
        aria_attrs,
        announce_to_screen_reader,
    )

References:
- NN/g Skeleton Screens: https://www.nngroup.com/articles/skeleton-screens/
- Smashing Magazine Inline Validation: https://www.smashingmagazine.com/2022/09/inline-validation-web-forms-ux/
- WCAG 2.2 Guidelines: https://www.w3.org/WAI/WCAG22/quickref/
"""

from .loading import (
    LoadingState,
    LoadingConfig,
    create_skeleton,
    create_skeleton_text,
    create_skeleton_card,
    create_progress_indicator,
    create_spinner,
    ProgressType,
)

from .validation import (
    FieldValidator,
    ValidationResult,
    ValidationRule,
    create_inline_error,
    validate_email,
    validate_required,
    validate_min_length,
    validate_max_length,
    validate_pattern,
    validate_number_range,
    validate_password_strength,
    create_password_strength_meter,
)

from .feedback import (
    Toast,
    ToastType,
    ToastConfig,
    create_toast,
    create_confirmation_dialog,
    create_status_badge,
    StatusType,
)

from .accessibility import (
    aria_attrs,
    AriaRole,
    AriaLive,
    announce_to_screen_reader,
    create_skip_link,
    create_focus_trap,
    keyboard_shortcut,
    visible_focus_styles,
    get_accessibility_css,
)

from .loading import get_loading_css
from .validation import get_validation_css
from .feedback import get_feedback_css, ToastManager


def get_all_ux_css() -> str:
    """
    Get all CSS styles for UX components.

    Returns combined CSS for loading states, validation, feedback,
    and accessibility components. Use in Gradio Blocks:

        with gr.Blocks(css=get_all_ux_css()) as demo:
            ...

    Returns:
        Combined CSS string for all UX components
    """
    return "\n".join([
        get_loading_css(),
        get_validation_css(),
        get_feedback_css(),
        get_accessibility_css(),
    ])


__all__ = [
    # Loading
    "LoadingState",
    "LoadingConfig",
    "create_skeleton",
    "create_skeleton_text",
    "create_skeleton_card",
    "create_progress_indicator",
    "create_spinner",
    "ProgressType",
    # Validation
    "FieldValidator",
    "ValidationResult",
    "ValidationRule",
    "create_inline_error",
    "validate_email",
    "validate_required",
    "validate_min_length",
    "validate_max_length",
    "validate_pattern",
    "validate_number_range",
    "validate_password_strength",
    "create_password_strength_meter",
    # Feedback
    "Toast",
    "ToastType",
    "ToastConfig",
    "create_toast",
    "create_confirmation_dialog",
    "create_status_badge",
    "StatusType",
    # Accessibility
    "aria_attrs",
    "AriaRole",
    "AriaLive",
    "announce_to_screen_reader",
    "create_skip_link",
    "create_focus_trap",
    "keyboard_shortcut",
    "visible_focus_styles",
    "get_accessibility_css",
    # CSS
    "get_loading_css",
    "get_validation_css",
    "get_feedback_css",
    "get_all_ux_css",
    # Managers
    "ToastManager",
]
