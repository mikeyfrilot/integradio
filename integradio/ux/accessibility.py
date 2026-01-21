"""
Accessibility Utilities - ARIA, Keyboard Navigation, Screen Reader Support

Best practices (2026):
- Use semantic HTML elements
- Add ARIA attributes for dynamic content
- Support keyboard navigation for all interactive elements
- Provide skip links for main content
- Announce dynamic changes to screen readers
- Ensure sufficient color contrast (WCAG 2.2 AA)
- Support reduced motion preferences

References:
- WCAG 2.2: https://www.w3.org/WAI/WCAG22/quickref/
- WAI-ARIA 1.2: https://www.w3.org/TR/wai-aria-1.2/
- TetraLogical: https://tetralogical.com/blog/2024/10/21/foundations-form-validation-and-error-messages/
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AriaRole(Enum):
    """Common ARIA roles."""
    ALERT = "alert"
    ALERTDIALOG = "alertdialog"
    BUTTON = "button"
    CHECKBOX = "checkbox"
    DIALOG = "dialog"
    FORM = "form"
    GRID = "grid"
    LINK = "link"
    LISTBOX = "listbox"
    MENU = "menu"
    MENUITEM = "menuitem"
    NAVIGATION = "navigation"
    PROGRESSBAR = "progressbar"
    REGION = "region"
    SEARCH = "search"
    SLIDER = "slider"
    STATUS = "status"
    TAB = "tab"
    TABLIST = "tablist"
    TABPANEL = "tabpanel"
    TEXTBOX = "textbox"
    TOOLTIP = "tooltip"
    TREE = "tree"


class AriaLive(Enum):
    """ARIA live region politeness levels."""
    OFF = "off"
    POLITE = "polite"  # Wait for user to be idle
    ASSERTIVE = "assertive"  # Interrupt immediately


@dataclass
class AriaAttrs:
    """ARIA attributes for an element."""
    role: Optional[AriaRole] = None
    label: Optional[str] = None
    labelledby: Optional[str] = None
    describedby: Optional[str] = None
    live: Optional[AriaLive] = None
    atomic: Optional[bool] = None
    busy: Optional[bool] = None
    controls: Optional[str] = None
    current: Optional[str] = None
    disabled: Optional[bool] = None
    expanded: Optional[bool] = None
    haspopup: Optional[str] = None
    hidden: Optional[bool] = None
    invalid: Optional[bool] = None
    pressed: Optional[bool] = None
    required: Optional[bool] = None
    selected: Optional[bool] = None
    valuenow: Optional[float] = None
    valuemin: Optional[float] = None
    valuemax: Optional[float] = None
    valuetext: Optional[str] = None

    def to_html_attrs(self) -> str:
        """Convert to HTML attribute string."""
        attrs = []

        if self.role:
            attrs.append(f'role="{self.role.value}"')
        if self.label:
            attrs.append(f'aria-label="{self.label}"')
        if self.labelledby:
            attrs.append(f'aria-labelledby="{self.labelledby}"')
        if self.describedby:
            attrs.append(f'aria-describedby="{self.describedby}"')
        if self.live:
            attrs.append(f'aria-live="{self.live.value}"')
        if self.atomic is not None:
            attrs.append(f'aria-atomic="{str(self.atomic).lower()}"')
        if self.busy is not None:
            attrs.append(f'aria-busy="{str(self.busy).lower()}"')
        if self.controls:
            attrs.append(f'aria-controls="{self.controls}"')
        if self.current:
            attrs.append(f'aria-current="{self.current}"')
        if self.disabled is not None:
            attrs.append(f'aria-disabled="{str(self.disabled).lower()}"')
        if self.expanded is not None:
            attrs.append(f'aria-expanded="{str(self.expanded).lower()}"')
        if self.haspopup:
            attrs.append(f'aria-haspopup="{self.haspopup}"')
        if self.hidden is not None:
            attrs.append(f'aria-hidden="{str(self.hidden).lower()}"')
        if self.invalid is not None:
            attrs.append(f'aria-invalid="{str(self.invalid).lower()}"')
        if self.pressed is not None:
            attrs.append(f'aria-pressed="{str(self.pressed).lower()}"')
        if self.required is not None:
            attrs.append(f'aria-required="{str(self.required).lower()}"')
        if self.selected is not None:
            attrs.append(f'aria-selected="{str(self.selected).lower()}"')
        if self.valuenow is not None:
            attrs.append(f'aria-valuenow="{self.valuenow}"')
        if self.valuemin is not None:
            attrs.append(f'aria-valuemin="{self.valuemin}"')
        if self.valuemax is not None:
            attrs.append(f'aria-valuemax="{self.valuemax}"')
        if self.valuetext:
            attrs.append(f'aria-valuetext="{self.valuetext}"')

        return " ".join(attrs)


def aria_attrs(**kwargs) -> str:
    """
    Create ARIA attribute string from keyword arguments.

    Args:
        **kwargs: ARIA attribute names (without aria- prefix) and values

    Returns:
        HTML attribute string

    Example:
        aria_attrs(label="Search", expanded=False, controls="search-results")
        # Returns: 'aria-label="Search" aria-expanded="false" aria-controls="search-results"'
    """
    attrs = []
    for key, value in kwargs.items():
        if value is None:
            continue

        # Handle role specially (no aria- prefix)
        if key == "role":
            if isinstance(value, AriaRole):
                attrs.append(f'role="{value.value}"')
            else:
                attrs.append(f'role="{value}"')
            continue

        # Convert key from snake_case to aria-kebab-case
        aria_key = f"aria-{key.replace('_', '-')}"

        # Convert boolean values
        if isinstance(value, bool):
            attrs.append(f'{aria_key}="{str(value).lower()}"')
        elif isinstance(value, AriaLive):
            attrs.append(f'{aria_key}="{value.value}"')
        else:
            attrs.append(f'{aria_key}="{value}"')

    return " ".join(attrs)


def announce_to_screen_reader(
    message: str,
    politeness: AriaLive = AriaLive.POLITE,
    atomic: bool = True,
) -> str:
    """
    Create HTML for screen reader announcement.

    This creates a visually hidden element that will be read by screen readers.
    Use aria-live="polite" for most announcements, "assertive" for urgent ones.

    Args:
        message: Message to announce
        politeness: How urgently to announce
        atomic: Whether to announce full content or just changes

    Returns:
        HTML string for announcement element

    Example:
        gr.HTML(announce_to_screen_reader("Form submitted successfully"))
    """
    return f"""
    <div class="sr-only" aria-live="{politeness.value}" aria-atomic="{str(atomic).lower()}">
        {message}
    </div>
    """


def create_skip_link(
    target_id: str = "main-content",
    label: str = "Skip to main content",
) -> str:
    """
    Create a skip link for keyboard navigation.

    Skip links allow keyboard users to bypass repetitive navigation.
    Should be the first focusable element on the page.

    Args:
        target_id: ID of the element to skip to
        label: Link text

    Returns:
        HTML string for skip link
    """
    return f"""
    <a href="#{target_id}" class="skip-link">{label}</a>
    """


def create_focus_trap(content_html: str, trap_id: str = "focus-trap") -> str:
    """
    Create a focus trap container for modals/dialogs.

    Args:
        content_html: HTML content to wrap
        trap_id: ID for the trap container

    Returns:
        HTML string with focus trap
    """
    return f"""
    <div id="{trap_id}" class="focus-trap" tabindex="-1">
        <span class="focus-trap-start" tabindex="0" aria-hidden="true"></span>
        {content_html}
        <span class="focus-trap-end" tabindex="0" aria-hidden="true"></span>
    </div>
    """


def keyboard_shortcut(
    key: str,
    modifier: Optional[str] = None,
    description: str = "",
) -> str:
    """
    Create keyboard shortcut display HTML.

    Args:
        key: Main key (e.g., "s", "Enter", "Escape")
        modifier: Modifier key (e.g., "Ctrl", "Alt", "Shift")
        description: Description of what the shortcut does

    Returns:
        HTML string for keyboard shortcut display
    """
    keys = []
    if modifier:
        keys.append(f'<kbd class="kbd">{modifier}</kbd>')
    keys.append(f'<kbd class="kbd">{key}</kbd>')

    shortcut_html = " + ".join(keys)

    if description:
        return f'<span class="keyboard-shortcut">{shortcut_html} <span class="shortcut-desc">{description}</span></span>'

    return f'<span class="keyboard-shortcut">{shortcut_html}</span>'


def visible_focus_styles() -> str:
    """
    Get CSS for visible focus indicators.

    Returns:
        CSS string for focus styles
    """
    return """
    /* Focus visible styles */
    :focus-visible {
        outline: 2px solid #3b82f6;
        outline-offset: 2px;
    }

    /* Remove default focus for mouse users */
    :focus:not(:focus-visible) {
        outline: none;
    }

    /* High contrast focus for inputs */
    input:focus-visible,
    textarea:focus-visible,
    select:focus-visible,
    button:focus-visible {
        outline: 2px solid #3b82f6;
        outline-offset: 2px;
        box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.2);
    }
    """


# CSS for accessibility components
ACCESSIBILITY_CSS = """
/* Screen reader only (visually hidden) */
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

/* Skip link */
.skip-link {
    position: fixed;
    top: -100px;
    left: 50%;
    transform: translateX(-50%);
    background: #1f2937;
    color: white;
    padding: 0.75rem 1.5rem;
    border-radius: 0 0 8px 8px;
    z-index: 10000;
    text-decoration: none;
    font-weight: 500;
    transition: top 0.2s ease;
}

.skip-link:focus {
    top: 0;
    outline: none;
}

/* Focus trap */
.focus-trap-start,
.focus-trap-end {
    position: absolute;
    width: 1px;
    height: 1px;
    overflow: hidden;
}

/* Keyboard shortcut display */
.keyboard-shortcut {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    font-size: 0.875rem;
}

.kbd {
    display: inline-block;
    padding: 0.125rem 0.5rem;
    background: #f3f4f6;
    border: 1px solid #e5e7eb;
    border-radius: 4px;
    font-family: ui-monospace, monospace;
    font-size: 0.75rem;
    color: #374151;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
}

[data-theme="dark"] .kbd,
.dark .kbd {
    background: #374151;
    border-color: #4b5563;
    color: #e5e7eb;
}

.shortcut-desc {
    color: #6b7280;
    margin-left: 0.5rem;
}

/* Reduced motion */
@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}

/* High contrast mode */
@media (prefers-contrast: high) {
    :focus-visible {
        outline: 3px solid currentColor;
        outline-offset: 3px;
    }

    .skip-link:focus {
        outline: 3px solid white;
    }
}

/* Focus visible styles */
:focus-visible {
    outline: 2px solid #3b82f6;
    outline-offset: 2px;
}

:focus:not(:focus-visible) {
    outline: none;
}

input:focus-visible,
textarea:focus-visible,
select:focus-visible,
button:focus-visible {
    outline: 2px solid #3b82f6;
    outline-offset: 2px;
    box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.2);
}

/* Link focus */
a:focus-visible {
    outline: 2px solid #3b82f6;
    outline-offset: 2px;
    border-radius: 2px;
}
"""


def get_accessibility_css() -> str:
    """Get CSS styles for accessibility components."""
    return ACCESSIBILITY_CSS


def create_landmark(
    content_html: str,
    role: AriaRole,
    label: str,
    id: Optional[str] = None,
) -> str:
    """
    Create a landmark region.

    Args:
        content_html: HTML content
        role: ARIA landmark role
        label: Accessible label
        id: Optional ID attribute

    Returns:
        HTML string with landmark wrapper
    """
    id_attr = f'id="{id}"' if id else ""
    return f"""
    <div {id_attr} role="{role.value}" aria-label="{label}">
        {content_html}
    </div>
    """


def create_loading_region(
    content_html: str,
    loading: bool = False,
    label: str = "Content area",
) -> str:
    """
    Create a region that announces loading state.

    Args:
        content_html: HTML content
        loading: Whether currently loading
        label: Accessible label

    Returns:
        HTML string with loading announcement
    """
    busy_attr = 'aria-busy="true"' if loading else 'aria-busy="false"'

    return f"""
    <div role="region" aria-label="{label}" aria-live="polite" {busy_attr}>
        {content_html}
        {announce_to_screen_reader("Loading..." if loading else "", AriaLive.POLITE) if loading else ""}
    </div>
    """


def format_field_description(
    description: str,
    error: Optional[str] = None,
    hint: Optional[str] = None,
) -> tuple[str, str]:
    """
    Format field description for accessibility.

    Returns both the visible text and the aria-describedby IDs.

    Args:
        description: Main description
        error: Error message
        hint: Additional hint

    Returns:
        Tuple of (HTML, aria-describedby value)
    """
    parts_html = []
    describedby_ids = []

    if description:
        parts_html.append(f'<span id="desc">{description}</span>')
        describedby_ids.append("desc")

    if hint:
        parts_html.append(f'<span id="hint" class="field-hint">{hint}</span>')
        describedby_ids.append("hint")

    if error:
        parts_html.append(f'<span id="error" class="field-error" role="alert">{error}</span>')
        describedby_ids.append("error")

    html = " ".join(parts_html)
    describedby = " ".join(describedby_ids)

    return html, describedby
