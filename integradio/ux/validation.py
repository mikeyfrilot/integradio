"""
Form Validation - Real-time Inline Validation

Best practices (2026):
- Validate on blur (after user leaves field), not on every keystroke
- Show errors inline, close to the field
- Use clear, specific error messages
- Mark invalid fields with aria-invalid="true"
- Validate empty required fields only on submit
- Provide positive feedback for valid fields

References:
- Smashing Magazine: https://www.smashingmagazine.com/2022/09/inline-validation-web-forms-ux/
- NN/g: https://www.nngroup.com/articles/errors-forms-design-guidelines/
- Baymard: https://baymard.com/blog/inline-form-validation
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Any
import gradio as gr


class ValidationState(Enum):
    """Validation state for a field."""
    PRISTINE = "pristine"  # Not yet interacted
    VALID = "valid"
    INVALID = "invalid"
    VALIDATING = "validating"  # Async validation in progress


@dataclass
class ValidationResult:
    """Result of field validation."""
    valid: bool
    message: str = ""
    field_name: str = ""
    state: ValidationState = ValidationState.PRISTINE

    def __bool__(self) -> bool:
        return self.valid


@dataclass
class ValidationRule:
    """A single validation rule."""
    validator: Callable[[Any], bool]
    message: str
    # Only validate if field has value (for optional fields)
    skip_empty: bool = False


@dataclass
class FieldValidator:
    """
    Validator for a form field with multiple rules.

    Example:
        validator = FieldValidator(
            field_name="email",
            rules=[
                ValidationRule(validate_required, "Email is required"),
                ValidationRule(validate_email, "Please enter a valid email"),
            ]
        )
        result = validator.validate("test@example.com")
    """
    field_name: str
    rules: list[ValidationRule] = field(default_factory=list)
    # Custom async validator (e.g., check username availability)
    async_validator: Optional[Callable[[Any], bool]] = None
    async_message: str = ""

    def validate(self, value: Any, check_required_only: bool = False) -> ValidationResult:
        """
        Validate a field value against all rules.

        Args:
            value: The field value to validate
            check_required_only: Only check required rule (for submit-time)

        Returns:
            ValidationResult with valid status and error message
        """
        for rule in self.rules:
            # Skip empty check if rule allows it
            if rule.skip_empty and not value:
                continue

            # For check_required_only, only run required validation
            if check_required_only and rule.validator != validate_required:
                continue

            if not rule.validator(value):
                return ValidationResult(
                    valid=False,
                    message=rule.message,
                    field_name=self.field_name,
                    state=ValidationState.INVALID,
                )

        return ValidationResult(
            valid=True,
            field_name=self.field_name,
            state=ValidationState.VALID,
        )

    def add_rule(self, validator: Callable[[Any], bool], message: str, skip_empty: bool = False):
        """Add a validation rule."""
        self.rules.append(ValidationRule(validator, message, skip_empty))
        return self  # Allow chaining


# Common validators
def validate_required(value: Any) -> bool:
    """Check if value is not empty."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def validate_email(value: str) -> bool:
    """Check if value is a valid email address."""
    if not value:
        return True  # Empty handled by required validator
    # RFC 5322 simplified pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, value))


def validate_min_length(min_len: int) -> Callable[[str], bool]:
    """Create a minimum length validator."""
    def validator(value: str) -> bool:
        if not value:
            return True  # Empty handled by required validator
        return len(value) >= min_len
    return validator


def validate_max_length(max_len: int) -> Callable[[str], bool]:
    """Create a maximum length validator."""
    def validator(value: str) -> bool:
        if not value:
            return True
        return len(value) <= max_len
    return validator


def validate_pattern(pattern: str, flags: int = 0) -> Callable[[str], bool]:
    """Create a regex pattern validator."""
    compiled = re.compile(pattern, flags)
    def validator(value: str) -> bool:
        if not value:
            return True
        return bool(compiled.match(value))
    return validator


def validate_number_range(
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> Callable[[Any], bool]:
    """Create a numeric range validator."""
    def validator(value: Any) -> bool:
        if value is None or value == "":
            return True
        try:
            num = float(value)
            if min_val is not None and num < min_val:
                return False
            if max_val is not None and num > max_val:
                return False
            return True
        except (ValueError, TypeError):
            return False
    return validator


def validate_url(value: str) -> bool:
    """Check if value is a valid URL."""
    if not value:
        return True
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return bool(re.match(pattern, value, re.IGNORECASE))


def validate_phone(value: str) -> bool:
    """Check if value is a valid phone number (flexible format)."""
    if not value:
        return True
    # Remove common separators
    cleaned = re.sub(r'[\s\-\.\(\)]', '', value)
    # Check for 10-15 digits, optionally starting with +
    pattern = r'^\+?\d{10,15}$'
    return bool(re.match(pattern, cleaned))


def validate_password_strength(value: str) -> bool:
    """Check if password meets strength requirements."""
    if not value:
        return True
    # At least 8 chars, 1 uppercase, 1 lowercase, 1 digit
    if len(value) < 8:
        return False
    if not re.search(r'[A-Z]', value):
        return False
    if not re.search(r'[a-z]', value):
        return False
    if not re.search(r'\d', value):
        return False
    return True


# CSS for inline validation feedback
VALIDATION_CSS = """
/* Field validation states */
.field-valid input,
.field-valid textarea,
.field-valid select {
    border-color: #10b981 !important;
}

.field-invalid input,
.field-invalid textarea,
.field-invalid select {
    border-color: #ef4444 !important;
}

/* Inline error message */
.field-error {
    color: #ef4444;
    font-size: 0.875rem;
    margin-top: 0.25rem;
    display: flex;
    align-items: center;
    gap: 0.25rem;
}

.field-error::before {
    content: "!";
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 16px;
    height: 16px;
    background: #ef4444;
    color: white;
    border-radius: 50%;
    font-size: 0.75rem;
    font-weight: bold;
}

/* Success indicator */
.field-success {
    color: #10b981;
    font-size: 0.875rem;
    margin-top: 0.25rem;
}

.field-success::before {
    content: "\\2713";
    margin-right: 0.25rem;
}

/* Character counter */
.char-counter {
    font-size: 0.75rem;
    color: #6b7280;
    text-align: right;
    margin-top: 0.25rem;
}

.char-counter.warning {
    color: #f59e0b;
}

.char-counter.error {
    color: #ef4444;
}

/* Password strength meter */
.password-strength {
    height: 4px;
    background: #e5e7eb;
    border-radius: 2px;
    margin-top: 0.5rem;
    overflow: hidden;
}

.password-strength-fill {
    height: 100%;
    transition: width 0.3s, background-color 0.3s;
}

.password-strength-fill.weak {
    width: 25%;
    background: #ef4444;
}

.password-strength-fill.fair {
    width: 50%;
    background: #f59e0b;
}

.password-strength-fill.good {
    width: 75%;
    background: #3b82f6;
}

.password-strength-fill.strong {
    width: 100%;
    background: #10b981;
}

/* Required field indicator */
.field-required::after {
    content: " *";
    color: #ef4444;
}

/* Focus visible for accessibility */
input:focus-visible,
textarea:focus-visible,
select:focus-visible {
    outline: 2px solid #3b82f6;
    outline-offset: 2px;
}
"""


def create_inline_error(
    message: str,
    field_id: str = "",
    visible: bool = True,
) -> str:
    """
    Create inline error message HTML.

    Args:
        message: Error message text
        field_id: Associated field ID for aria-describedby
        visible: Whether to show the error

    Returns:
        HTML string for error message
    """
    if not visible or not message:
        return ""

    id_attr = f'id="{field_id}-error"' if field_id else ""
    return f"""
    <div class="field-error" {id_attr} role="alert" aria-live="polite">
        {message}
    </div>
    """


def create_success_indicator(
    message: str = "Valid",
    visible: bool = True,
) -> str:
    """Create success indicator HTML."""
    if not visible:
        return ""
    return f'<div class="field-success">{message}</div>'


def create_char_counter(
    current: int,
    max_chars: int,
    warning_threshold: float = 0.8,
) -> str:
    """
    Create character counter HTML.

    Args:
        current: Current character count
        max_chars: Maximum allowed characters
        warning_threshold: Show warning when exceeding this ratio

    Returns:
        HTML string for character counter
    """
    remaining = max_chars - current
    ratio = current / max_chars if max_chars > 0 else 0

    css_class = "char-counter"
    if ratio >= 1:
        css_class += " error"
    elif ratio >= warning_threshold:
        css_class += " warning"

    return f'<div class="{css_class}">{current}/{max_chars}</div>'


def create_password_strength_meter(password: str) -> str:
    """
    Create password strength meter HTML.

    Args:
        password: Password to evaluate

    Returns:
        HTML string for strength meter
    """
    if not password:
        return '<div class="password-strength"><div class="password-strength-fill"></div></div>'

    strength = 0
    if len(password) >= 8:
        strength += 1
    if re.search(r'[A-Z]', password):
        strength += 1
    if re.search(r'[a-z]', password):
        strength += 1
    if re.search(r'\d', password):
        strength += 1
    if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        strength += 1

    strength_class = {
        0: "",
        1: "weak",
        2: "fair",
        3: "good",
        4: "good",
        5: "strong",
    }.get(strength, "weak")

    strength_text = {
        0: "",
        1: "Weak",
        2: "Fair",
        3: "Good",
        4: "Good",
        5: "Strong",
    }.get(strength, "")

    return f"""
    <div>
        <div class="password-strength" role="meter" aria-label="Password strength: {strength_text}"
             aria-valuenow="{strength}" aria-valuemin="0" aria-valuemax="5">
            <div class="password-strength-fill {strength_class}"></div>
        </div>
        <div style="font-size: 0.75rem; color: #6b7280; margin-top: 0.25rem;">
            {strength_text}
        </div>
    </div>
    """


def get_validation_css() -> str:
    """Get CSS styles for validation components."""
    return VALIDATION_CSS


def create_form_validator(fields: dict[str, FieldValidator]) -> Callable:
    """
    Create a form validation function.

    Args:
        fields: Dictionary of field name to FieldValidator

    Returns:
        Function that validates all fields and returns results

    Example:
        validator = create_form_validator({
            "email": FieldValidator("email", [
                ValidationRule(validate_required, "Required"),
                ValidationRule(validate_email, "Invalid email"),
            ]),
            "name": FieldValidator("name", [
                ValidationRule(validate_required, "Required"),
            ]),
        })

        results = validator({"email": "test@example.com", "name": ""})
    """
    def validate_all(values: dict[str, Any]) -> dict[str, ValidationResult]:
        results = {}
        for field_name, validator in fields.items():
            value = values.get(field_name)
            results[field_name] = validator.validate(value)
        return results

    return validate_all
